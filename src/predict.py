# -*- coding: utf8 -*-
#
from typing import List, Dict

from src.conf import DATA_PATH
from src.model import PairwiseMatchingModel
import torch
from torch import nn
from src.eval_transform import PairWiseEvalDL


class PairWiseMatchInfer(object):
    def __init__(self):
        self.model = PairwiseMatchingModel()
        self.load_weights(save_path=DATA_PATH.joinpath('savepoint').joinpath('dev_0.9311_epoch_2.pt'))

    def load_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            self.model.load_state_dict(torch.load(save_path))
        else:
            self.model.module.load_state_dict(torch.load(save_path))
        self.model.eval()

    @torch.no_grad()
    def infer(self, datas: List[Dict]):
        for input_data, _ in PairWiseEvalDL(datas=datas).to_dl(batch_size=len(datas), shuffle=False):
            output = self.model.predict(input=input_data)
            return output

if __name__ == '__main__':

    samples = [
        {"query": "我长的帅不帅", "title": "我长的就问帅不帅", "label": 0},  # label是假的哦，这里省事了～
        {"query": "嚼口香糖能减肥吗", "title": "嚼口香糖会减肥吗？", "label": 0},

        {"query": "孕妇能用护肤品吗", "title": "哪些护肤品孕妇能用？", "label": 0},
        {"query": "桂林山水在哪个省", "title": "桂林山水在哪个市", "label": 0},
        {"query": "动物园好玩么", "title": "这个时候去动物园好玩吗", "label": 0},
        {"query": "肆虐是什么意思？", "title": "肆虐的肆是什么意思", "label": 0}

    ]
    infer = PairWiseMatchInfer()
    output = infer.infer(datas=samples)
    print(output)
    # array([[0.94228584],
    #        [0.90933144],
    #        [0.748481  ],
    #        [0.34310436],
    #        [0.8954417 ],
    #        [0.33769682]], dtype=float32)


