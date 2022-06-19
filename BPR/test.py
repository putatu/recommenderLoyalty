
from dataloader import load_data
import numpy as np
import argparse
from BPR import BPR
import torch.optim as optim
import torch
import heapq
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='movielens', help='dataset name', required=True)
parser.add_argument('--data_path', type=str, default='data/', help='dataset path', required=True)
parser.add_argument('--selected_year', type=int, default=6, help='selected_year', required=True)
parser.add_argument('--gpu', type=str, default="0", help='gpu', required=True)
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate', required=True)
parser.add_argument('--regularization_coefficient', type=float, default=0.0001, help='regularization coefficient', required=True)
parser.add_argument('--embedding_size', type=int, default=128, help='latent dimension size', required=True)


args = parser.parse_args()

dataset_name = args.data
path = args.data_path+dataset_name+'/'
selected_year = args.selected_year
learning_rate = args.learning_rate
reg = args.regularization_coefficient
embedding_size = args.embedding_size
device = 'cuda:' + args.gpu



report_folder = "bpr_report/"
checkpoints_folder = "bpr_checkpoints/"

if not os.path.isdir(report_folder):
    os.mkdir(report_folder)
if not os.path.isdir(checkpoints_folder):
    os.mkdir(checkpoints_folder)



splitter = '\t'
hold_k_out = 1
train, test, available_items, num_user, num_item, num_ratings = load_data(path,splitter,hold_k_out,selected_year)

display_step = 20
best_loss = np.inf
model = BPR(embedding_size, num_user, num_item).to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)


i = 0
num_epochs = 200


class trainDataset(Dataset):
    def __init__(self, train, alr_train, available_items):
        self.train = train
        self.alr_train = alr_train
        self.available_items = list(available_items)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        user = self.train[idx][0]
        item = self.train[idx][1]
        # bpr_boolean = self.train[idx][2]
        # past_items = self.alr_train[user]

        neg_item = random.sample(self.available_items, 1)[0]
        while neg_item in self.alr_train[user]:
            neg_item = random.sample(self.available_items, 1)[0]
        return user, item, neg_item

train_data = []
for user in train:
    for item in train[user]:
        train_data.append((user, item))

train_dataset = trainDataset(train_data, train, available_items)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)

best_loss = np.inf
for epoch in range(num_epochs):
    model.train()
    total_loss = []
    for idx, (user, pos, neg) in enumerate(train_loader):
        optimizer.zero_grad()
        user = user.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        loss = model.calculate_loss(user, pos, neg, reg)  # + RegLoss(model.parameters())
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    print(epoch, np.mean(total_loss))
    if np.mean(total_loss) < best_loss:
        best_loss = np.mean(total_loss)
        torch.save(model.state_dict(), checkpoints_folder + dataset_name + '_' + str(selected_year) + '.pt')

model = BPR(embedding_size, num_user, num_item).to(device)
model.load_state_dict(torch.load(checkpoints_folder + dataset_name + '_' + str(selected_year) + '.pt'))
model.eval()

k_values = [5, 10, 20]
for k in k_values:
    if os.path.isfile(report_folder + dataset_name + '_' + str(selected_year) + '_at_' + str(k) + '.txt'):
        os.remove(report_folder + dataset_name + '_' + str(selected_year) + '_at_' + str(k) + '.txt')

for test_i in range(0, len(test)):
    user_test = test[test_i][0]
    item_test = test[test_i][1]
    negative_items = list(
        available_items)
    scores = model.full_sort_predict(torch.tensor(user_test).to(device),
                                     torch.tensor(negative_items).to(device))
    scores = scores.detach().cpu().numpy()
    preds = dict(zip(negative_items, scores))

    for k in k_values:
        recommended = heapq.nlargest(k, preds, key=preds.get)
        with open(report_folder + dataset_name + '_' + str(selected_year) + '_at_' + str(k) + '.txt',
                  'a') as file:
            file.write(str(user_test))
            for rec_item in recommended:
                file.write('\t')
                file.write(str(rec_item))
            file.write('\n')
