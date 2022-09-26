import os
import time
import torch
import argparse
import random
from model import SASRec
from tqdm import tqdm
from utils import *
import heapq


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default = 'data/',help='data path', required=True)
parser.add_argument('--data', type=str, default = 'movielens',help='data to be tested', required=True)
parser.add_argument('--gpu', default= '4', help='gpu', required=True)
parser.add_argument('--selected_year', default= 6, type=int,help='year for testing', required=True)
parser.add_argument('--batch_size', default= 128, type=int, help='batch size')
parser.add_argument('--dropout_rate', default= 0.2, type=float, help='dropout rate')
parser.add_argument('--l2_emb', default= 0, type=float, help='regularization coefficient')
parser.add_argument('--learning_rate', default= 0.001, type=float, help='learning rate')
parser.add_argument('--num_heads', default= 1, type=int, help='number of heads in the self-attention network')
parser.add_argument('--hidden_units', default= 64, type=int, help='hidden units in the self-attention network')
parser.add_argument('--maxlen', default= 20, type=int, help='maximum number of past interactions to be modelled in the self-attention network')
parser.add_argument('--num_blocks', default= 1, type=int, help='number of layers in the self-attention network')



args = parser.parse_args()

dataset_name = args.data
path = args.data_path + dataset_name+'/'
mode = 'test'
device = 'cuda:'+args.gpu
selected_year = args.selected_year
batch_size = args.batch_size
dropout_rate = args.dropout_rate
l2_emb = args.l2_emb
lr = args.learning_rate
num_heads = args.num_heads
hidden_units = args.hidden_units
maxlen = args.maxlen
num_blocks = args.num_blocks


checkpoints_folder = 'sasrec_checkpoints/'
report_folder = 'sasrec_report/'



dataset = data_partition(path, mode, selected_year)
[user_train, user_test, usernum, itemnum, available_items, num_ratings] = dataset

print(dataset_name, selected_year, len(user_train), len(user_test), usernum, itemnum, num_ratings, len(available_items))




num_batch = len(user_train) // batch_size
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=2)
model = SASRec(usernum, itemnum, hidden_units, maxlen, dropout_rate, num_heads, device, num_blocks).to(device) # no ReLU activation in original SASRec implementation?


for name, param in model.named_parameters():
	try:
		torch.nn.init.xavier_normal_(param.data)
	except:
		pass # just ignore those failed init layers

model.train() # enable model training

epoch_start_idx = 1
bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

T = 0.0
num_epochs = 100
best_loss = np.inf
for epoch in range(epoch_start_idx, num_epochs + 1):
    #if args.inference_only: break # just to decrease identition
	t0 = time.time()
	total_loss = []
	for step in range(num_batch):
		u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
		u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
		pos_logits, neg_logits = model(u, seq, pos, neg)
		pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
		adam_optimizer.zero_grad()
		indices = np.where(pos != 0)
		loss = bce_criterion(pos_logits[indices], pos_labels[indices])
		loss += bce_criterion(neg_logits[indices], neg_labels[indices])
		for param in model.item_emb.parameters(): loss += l2_emb * torch.norm(param)
		loss.backward()
		adam_optimizer.step()
		total_loss.append(loss.item())

	if np.mean(total_loss) < best_loss:
		best_loss = np.mean(total_loss)
		torch.save(model.state_dict(), checkpoints_folder+dataset_name+'_'+str(selected_year)+'.pt')


	print("Dataset {} Selected_year {} loss in epoch {}: {}, in time {}".format(dataset_name, selected_year, epoch, loss.item(), time.time() - t0))


model = SASRec(usernum, itemnum, hidden_units, maxlen, dropout_rate, num_heads, device, num_blocks).to(device)
model.load_state_dict(torch.load(checkpoints_folder+dataset_name+'_'+str(selected_year)+'.pt'))
model.eval()


k_values = [5,10,20]
for u in user_test:
	seq = np.zeros([maxlen], dtype=np.int32)
	idx = maxlen - 1
	for i in reversed(user_train[u]):
		seq[idx] = i
		idx -= 1
		if idx == -1: break
	item_idx = [user_test[u][0]]
	item_idx += list(available_items - set(user_test[u].copy()))
	predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
	predictions = predictions[0]

	preds = dict(zip(item_idx, predictions.cpu().detach().numpy()))
	for k in k_values:
		recommended = heapq.nlargest(k, preds, key = preds.get)
		with open(report_folder+dataset_name+'_'+str(selected_year)+'_at_'+str(k)+'.txt', 'a') as file:
			file.write(str(u))
			for rec_item in recommended:
				file.write('\t')
				file.write(str(rec_item))
			file.write('\n')

