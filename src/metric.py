# -*- coding: utf8 -*-
#
from typing import Dict

import numpy as np
from sklearn.metrics import roc_curve, auc
from utrainer.metric import Metric


class AUC(Metric):
    def __init__(self):
        self.y_true = []
        self.y_score = []

    def step(self, inputs):
        y_true, y_pred = inputs
        self.y_true.append(y_true)
        self.y_score.append(y_pred)

    def score(self) -> float:
        fpr, tpr, thresholds = roc_curve(np.concatenate(self.y_true), np.concatenate(self.y_score), pos_label=1)
        return auc(fpr, tpr)

    def report(self) -> Dict:
        return {}
