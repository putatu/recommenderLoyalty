import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
import heapq
import numpy as np


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in user_train:  # tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    def sample(user):

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]

        idx = maxlen - 1
        ts = set(map(lambda x: x[0], user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = random.sample(user_train.keys(), 1)[0]
            # while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
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


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def data_partition(path, selected_year):
    data_filename = path + 'data.data.RATING'
    train_filename = path + 'train.test.RATING'
    test_filename = path + 'test.test.RATING'

    data = {}
    num_user = 0
    num_item = 0
    num_ratings = 0
    available_items = set()
    time_set = set()
    with open(data_filename, 'r') as file:
        for line in file:
            arr = line.split('\t')
            user, item, timestamp, year = int(arr[0]), int(arr[1]) + 1, int(arr[3]), int(arr[4])
            if (year < selected_year):
                if user in data:
                    data[user].append([item, timestamp])
                else:
                    data[user] = [[item, timestamp]]
                num_ratings += 1
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)
                time_set.add(timestamp)

    with open(train_filename, 'r') as file:
        for line in file:
            arr = line.split('\t')
            user, item, timestamp, year = int(arr[0]), int(arr[1]) + 1, int(arr[3]), int(arr[4])
            if (year == selected_year):
                if user in data:
                    data[user].append([item, timestamp])
                else:
                    data[user] = [[item, timestamp]]
                num_ratings += 1
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                available_items.add(item)
                time_set.add(timestamp)

    # read test sequence

    test_users = set()
    with open(test_filename, 'r') as file:
        for line in file:
            arr = line.split('\t')
            user, item, timestamp, year = int(arr[0]), int(arr[1]) + 1, int(arr[3]), int(arr[4])
            if (year == selected_year):
                num_user = max(num_user, user)
                num_item = max(num_item, item)
                if user in data:
                    data[user].append([item, timestamp])
                else:
                    data[user] = [[item, timestamp]]
                test_users.add(user)
                # test[user] = [item, timestamp,year]
                time_set.add(timestamp)
    time_map = timeSlice(time_set)
    data, timenum = cleanAndsort(data, time_map)

    train = {}
    test = {}
    for user in data:
        if user in test_users:
            train[user] = data[user][:-1].copy()
            test[user] = data[user][-1].copy()
        else:
            if len(data[user]) > 1:
                train[user] = data[user].copy()

    return [train, test, num_user, num_item, timenum, num_ratings, available_items]


def cleanAndsort(User, time_map):
    for user, items in User.items():
        User[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User.items():
        User_res[user] = list(map(lambda x: [x[0], time_map[x[1]]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, max(time_max)