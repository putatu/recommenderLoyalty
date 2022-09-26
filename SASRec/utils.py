import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        users = list(user_train.keys())
        user = random.sample(users, k=1)[0]
        # np.random.randint(1, usernum + 1)
        # while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(path, mode, selected_year):
    num_user = 0
    num_item = 0
    num_ratings = 0
    user_train = {}
    user_test = {}
    available_items = set()
    if mode == 'train':
        data_filename = path + 'data.data.RATING'
        train_filename = path + 'train.train.RATING'
        test_filename = path + 'test.train.RATING'
    else:
        data_filename = path + 'data.data.RATING'
        train_filename = path + 'train.test.RATING'
        test_filename = path + 'test.test.RATING'

    # assume user/item index starting from 1
    with open(data_filename, 'r') as file:
        for line in file:
            arr = line.split('\t')
            user, item, timestamp, year = int(arr[0]), int(arr[1]) + 1, int(arr[3]), int(arr[4])
            if (year < selected_year):
                if user in user_train:
                    user_train[user].append(item)
                else:
                    user_train[user] = [item]
                num_ratings += 1
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)

    with open(train_filename, 'r') as file:
        for line in file:
            arr = line.split('\t')
            user, item, timestamp, year = int(arr[0]), int(arr[1]) + 1, int(arr[3]), int(arr[4])
            if (year == selected_year):
                if user in user_train:
                    user_train[user].append(item)
                else:
                    user_train[user] = [item]
                num_ratings += 1
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)

    with open(test_filename, 'r') as file:
        for line in file:
            arr = line.split('\t')
            user, item, timestamp, year = int(arr[0]), int(arr[1]) + 1, int(arr[3]), int(arr[4])
            if (year == selected_year):
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                user_test[user] = [item]

    train_data = {}
    i = 0
    num_item += 1
    for user in user_train.keys():
        if len(user_train[user]) > 1:
            train_data = user_train[user].copy()

    return [user_train, user_test, num_user, num_item, available_items, num_ratings]


import heapq


def evaluate(model, dataset, maxlen):
    [user_train, user_test, usernum, itemnum, available_items, num_ratings] = copy.deepcopy(dataset)
    k = 10
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    HR = []
    NDCG = []
    for u in user_test:
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = list(available_items)
        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        preds = dict(zip(list(available_items), predictions))
        recommended = heapq.nlargest(k, preds, key=preds.get)
        if user_test[u] in recommended:
            HR.append(1)
        else:
            HR.append(0)

    return np.mean(HR)


def evaluate_valid(model, dataset, maxlen):
    [user_train, user_test, usernum, itemnum, available_items, num_ratings] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    for u in user_test:

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = [user_test[u][0]]
        item_idx += list(available_items - set(user_test[u].copy()))
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user