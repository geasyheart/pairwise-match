# -*- coding: utf8 -*-
#
import numpy as np
from sklearn.metrics import roc_curve, auc
from paddle.metric.metrics import Auc


def sklearn_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    return auc(fpr, tpr)


def paddle_auc(y_true, y_pred):
    metric = Auc()
    y_pred1 = 1 - y_pred
    preds = np.concatenate((y_pred1, y_pred), axis=1)
    metric.update(preds=preds, labels=y_true)
    return metric.accumulate()


if __name__ == '__main__':
    y_pred = np.random.random(size=(11111, 1))
    y_true = np.random.randint(2, size=(11111, 1))
    print(sklearn_auc(y_true=y_true, y_pred=y_pred))
    print(paddle_auc(y_true=y_true, y_pred=y_pred))
