
# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn
from model.loss import BPRLoss


class BPR(nn.Module):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """

    # input_type = InputType.PAIRWISE

    def __init__(self, embedding_size, n_users, n_items):
        super(BPR, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size
        self.n_users = n_users
        self.n_items = n_items

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        nn.init.normal_(self.user_embedding.weight ,0, 0.01)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        nn.init.normal_(self.item_embedding.weight, 0, 0.01)
        self.loss = BPRLoss()

    def get_user_embedding(self, user):
        """ Get a batch of user embedding tensor according to input user's id.
        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]
        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        """ Get a batch of item embedding tensor according to input item's id.
        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]
        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e



    def calculate_loss(self, user, pos_item, neg_item, reg):

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        regloss = torch.sum \
            (torch.sum(torch.mul(user_e, user_e) ,dim=-1) + torch.sum(torch.mul(pos_e, pos_e) ,dim=-1) +torch.sum(
                torch.mul(neg_e, neg_e), dim=-1))
        loss = self.loss(pos_item_score, neg_item_score) + reg * regloss
        return loss

    def full_sort_predict(self, user, items):
        user_e = self.get_user_embedding(user)
        all_item_e = self.get_item_embedding(items)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
