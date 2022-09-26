import os
import time
import argparse
import tensorflow as tf
from model import Model
from util import *
import heapq
import numpy as np

import pickle
from tqdm import tqdm
import random


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--selected_year', required=True, type=int)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--maxlen', type=int, default = 5)
parser.add_argument('--hidden_units', type=int, default = 32)
parser.add_argument('--l2_emb', type=float, default = 0.0001)
parser.add_argument('--time_span', type=int, default = 128)
parser.add_argument('--num_blocks', type=int, default = 3)
parser.add_argument('--num_heads', type=int, default = 1)
parser.add_argument('--learning_rate', type=float, default = 0.001)
parser.add_argument('--dropout_rate', type=float, default = 0.2)





args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

dataset_name = args.data
data_path = args.data_path
selected_year = args.selected_year
maxlen = args.maxlen
hidden_units = args.hidden_units
l2_emb = args.l2_emb
time_span = args.time_span
num_blocks = args.num_blocks
num_heads = args.num_heads
lr = args.learning_rate
dropout_rate = args.dropout_rate


path = data_path + dataset_name+'/'




report_folder = 'tisasrec_report/'
checkpoints_folder = 'tisasrec_checkpoints/'

dataset = data_partition(path, selected_year)
[user_train, user_valid, usernum, itemnum, timenum, num_ratings, available_items] = dataset
print(dataset_name, selected_year, len(user_train), len(user_valid), usernum, itemnum, num_ratings,
      len(available_items))

tf.reset_default_graph()
model = Model(usernum, itemnum, timenum, available_items, maxlen, hidden_units, l2_emb, time_span, dropout_rate,
              num_blocks, num_heads, lr)
saver = tf.train.Saver()
batch_size = 256
num_batch = int(len(user_train) / batch_size)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
relation_matrix = Relation(user_train, usernum, maxlen, time_span)
sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=batch_size, maxlen=maxlen, n_workers=4)

T = 0.0
num_epochs = 100
display = 20
best_loss = np.inf

for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    total_loss = []
    for s_batch in range(num_batch):
        u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch()
        auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                {model.u: u, model.input_seq: seq, model.time_matrix: time_matrix, model.pos: pos,
                                 model.neg: neg,
                                 model.is_training: True})
        total_loss.append(loss)
    print(epoch, np.mean(total_loss))
    if np.mean(total_loss) < best_loss:
        best_loss = np.mean(total_loss)
        saver.save(sess, checkpoints_folder + 'best_model_' + dataset_name + '_' + str(selected_year), global_step=0)

saver.restore(sess, checkpoints_folder + 'best_model_' + dataset_name + '_' + str(selected_year) + '-0')
for u in user_valid:
    seq = np.zeros([maxlen], dtype=np.int32)
    time_seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1

    for i in reversed(user_train[u]):
        seq[idx] = i[0]
        time_seq[idx] = i[1]
        idx -= 1
        if idx == -1: break

    item_idx = list(available_items)
    time_matrix = computeRePos(time_seq, time_span)

    predictions = model.predict(sess, [u], [seq], [time_matrix], item_idx)
    predictions = predictions[0]
    preds = dict(zip(item_idx, predictions))
    k_values = [5, 10, 20]
    for k in k_values:
        recommended = heapq.nlargest(k, preds, key=preds.get)
        with open(report_folder + dataset_name + '_' + str(selected_year) + '_at_' + str(k) + '.txt', 'a') as file:
            file.write(str(u))
            for rec_item in recommended:
                file.write('\t')
                file.write(str(rec_item))
            file.write('\n')