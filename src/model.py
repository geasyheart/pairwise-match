# -*- coding: utf8 -*-
#
import torch
from torch import nn
from torch import sigmoid
from torch.functional import F

from src.g import ptm_model


class PairwiseMatchingModel(nn.Module):
    def __init__(self):
        super(PairwiseMatchingModel, self).__init__()

        self.ptm = ptm_model
        self.dropout = nn.Dropout(self.ptm.config.hidden_dropout_prob)
        self.similar = nn.Linear(self.ptm.config.hidden_size, 1)

    def forward(self, pos_input, neg_input):
        pos_output = self.ptm(**pos_input)
        neg_output = self.ptm(**neg_input)
        pos_cls_embedding = self.dropout(pos_output[0][:, 0, :])
        neg_cls_embedding = self.dropout(neg_output[0][:, 0, :])
        pos_sim = sigmoid(self.similar(pos_cls_embedding))
        neg_sim = sigmoid(self.similar(neg_cls_embedding))

        labels = torch.ones(pos_sim.shape[0], dtype=torch.long, device=pos_sim.device)
        loss = F.margin_ranking_loss(
            pos_sim,
            neg_sim,
            target=labels,
            margin=0.2
        )
        return loss

    def predict(self, input):
        output = self.ptm(**input)
        cls_embedding = self.dropout(output[0][:, 0, :])
        score = sigmoid(self.similar(cls_embedding))
        return score.cpu().numpy()
