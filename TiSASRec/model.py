from modules import *


class Model():
    def __init__(self, usernum, itemnum, timenum, available_items, maxlen, hidden_units, l2_emb, time_span,
                 dropout_rate, num_blocks, num_heads, lr, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.time_matrix = tf.placeholder(tf.int32, shape=(None, maxlen, maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        self.time_matrix = tf.reshape(self.time_matrix, [tf.shape(self.input_seq)[0], maxlen, maxlen])
        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            absolute_pos_K = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=maxlen,
                num_units=hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=l2_emb,
                scope="abs_pos_K",
                reuse=reuse,
                with_t=False
            )
            absolute_pos_V = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=maxlen,
                num_units=hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=l2_emb,
                scope="abs_pos_V",
                reuse=reuse,
                with_t=False
            )
            # Time Encoding
            time_matrix_emb_K = embedding(
                self.time_matrix,
                vocab_size=time_span + 1,
                num_units=hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=l2_emb,
                scope="dec_time_K",
                reuse=reuse,
                with_t=False
            )
            time_matrix_emb_V = embedding(
                self.time_matrix,
                vocab_size=time_span + 1,
                num_units=hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=l2_emb,
                scope="dec_time_V",
                reuse=reuse,
                with_t=False
            )

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            time_matrix_emb_K = tf.layers.dropout(time_matrix_emb_K,
                                                  rate=dropout_rate,
                                                  training=tf.convert_to_tensor(self.is_training))
            time_matrix_emb_V = tf.layers.dropout(time_matrix_emb_V,
                                                  rate=dropout_rate,
                                                  training=tf.convert_to_tensor(self.is_training))
            absolute_pos_K = tf.layers.dropout(absolute_pos_K,
                                               rate=dropout_rate,
                                               training=tf.convert_to_tensor(self.is_training))
            absolute_pos_V = tf.layers.dropout(absolute_pos_V,
                                               rate=dropout_rate,
                                               training=tf.convert_to_tensor(self.is_training))
            # Build blocks

            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   time_matrix_K=time_matrix_emb_K,
                                                   time_matrix_V=time_matrix_emb_V,
                                                   absolute_pos_K=absolute_pos_K,
                                                   absolute_pos_V=absolute_pos_V,
                                                   num_units=hidden_units,
                                                   num_heads=num_heads,
                                                   dropout_rate=dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention",
                                                   )

                    # Feed forward

                    self.seq = feedforward(normalize(self.seq), num_units=[hidden_units, hidden_units],
                                           dropout_rate=dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(len(available_items)))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], maxlen, len(available_items)])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, time_matrix, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.time_matrix: time_matrix, self.test_item: item_idx,
                         self.is_training: False})
