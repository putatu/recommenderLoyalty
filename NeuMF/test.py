'''
Created on Apr 15, 2016

An Example to run the MFbpr

@author: hexiangnan
'''
from dataloader import load_data
import numpy as np
import argparse
import torch.optim as optim
import time
import torch
from NeuMF import NeuMF
import heapq
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--selected_year', type=int, default=5, help='selected_year', required=True)
parser.add_argument('--data', type=str, default='movielens', required=True)
parser.add_argument('--data_path', type=str, default='data/', required=True)
parser.add_argument('--mf_embedding_size', type=int, default=64)
parser.add_argument('--mlp_embedding_size', type=int, default=64)
parser.add_argument('--mlp_hidden_size', type=int, default=64)
parser.add_argument('--dropout_prob', type=float, default=0.2)
parser.add_argument('--learning_rate', type=float, default=0.0001)




parser.add_argument('--gpu', default='4', help='gpu')

args = parser.parse_args()





dataset_name = args.data
path = args.data_path+dataset_name+'/'
selected_year = args.selected_year
mf_embedding_size = args.mf_embedding_size
mlp_embedding_size = args.mlp_embedding_size
mlp_hidden_size = [args.mlp_hidden_size]
dropout_prob = args.dropout_prob
learning_rate = args.learning_rate

splitter = "\t"
hold_k_out = 1

device = 'cuda:' + args.gpu

report_folder = 'neumf_report/'
checkpoints_folder = 'neumf_checkpoints/'
if not os.path.isdir(report_folder):
    os.mkdir(report_folder)
if not os.path.isdir(checkpoints_folder):
    os.mkdir(checkpoints_folder)

train, test, available_items, num_user, num_item, num_ratings = load_data(path, splitter, hold_k_out,selected_year)


def get_train_instances(train, available_items):
    user_input, item_input, labels = [], [], []
    num_negatives = 1
    for user in train:
        sampled_negatives = random.choices(list(available_items - set(train[user])), k=len(train[user]) * num_negatives)
        i = 0
        for item in train[user]:
            user_input.append(user)
            item_input.append(item)
            labels.append(1)
            for n_neg in range(num_negatives):
                user_input.append(user)
                item_input.append(sampled_negatives[i])
                labels.append(0)
                i += 1

    return user_input, item_input, labels


model = NeuMF(mf_embedding_size, mlp_embedding_size, mlp_hidden_size, dropout_prob, num_user, num_item).to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)
best_loss = np.inf
num_epochs=200
i = 0
for epoch in range(1,num_epochs+1):
    model.train()
    users_set, poss_set, labels_set = get_train_instances(train, available_items)
    total_loss = []
    time_0 = time.time()
    batch_size = 1024

    for s in range(np.int(num_ratings*2 / batch_size)+1):
        user = torch.tensor(users_set[s*batch_size:(s+1)*batch_size],dtype=torch.long).to(device)
        pos = torch.tensor(poss_set[s*batch_size:(s+1)*batch_size],dtype=torch.long).to(device)
        label = torch.tensor(labels_set[s*batch_size:(s+1)*batch_size],dtype=torch.float).to(device)
        optimizer.zero_grad()

        loss = model.calculate_loss(user, pos,label)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    if np.mean(total_loss) < best_loss:
        best_loss = np.mean(total_loss)
        torch.save(model.state_dict(), checkpoints_folder+dataset_name+'_'+str(selected_year)+'.pt')
    print(epoch, np.mean(total_loss), time.time() - time_0,users_set[0], users_set[-1])

model = NeuMF(mf_embedding_size,mlp_embedding_size, mlp_hidden_size, dropout_prob, num_user, num_item).to(device)
model.load_state_dict(torch.load(checkpoints_folder+dataset_name+'_'+str(selected_year)+'.pt'))
model.eval()
k_values = [5, 10, 20]
for k in k_values:
    if os.path.isfile(report_folder + dataset_name + '_' + str(selected_year) + '_at_' + str(k) + '.txt'):
        os.remove(report_folder + dataset_name + '_' + str(selected_year) + '_at_' + str(k) + '.txt')

for test_i in range(0, len(test)):
    user_test = test[test_i][0]
    item_test = test[test_i][1]
    negative_items = list(available_items)
    scores = model.predict(torch.tensor([user_test]*len(negative_items)).to(device), torch.tensor(negative_items).to(device))

    scores = scores.detach().cpu().numpy()
    preds = dict(zip(negative_items, scores))

    for k in k_values:
        recommended = heapq.nlargest(k, preds, key = preds.get)
        with open(report_folder+dataset_name+'_'+str(selected_year)+'_at_'+str(k)+'.txt', 'a') as file:
            file.write(str(user_test))
            for rec_item in recommended:
                file.write('\t')
                file.write(str(rec_item))
            file.write('\n')

