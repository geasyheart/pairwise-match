# -*- coding: utf8 -*-
#
import json
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.conf import DATA_PATH
from src.g import tokenizer


class PairWiseTrainDL(Dataset):
    def __init__(self, datas: List[Dict], device='cpu'):
        self.datas = datas
        self.device = torch.device(device=device)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas[item]
        pos_input = tokenizer.encode_plus(text=data['query'], text_pair=data['title'], max_length=512)
        neg_input = tokenizer.encode_plus(text=data['negative_query'], text_pair=data['title'], max_length=512)
        return {"pos_input": pos_input, "neg_input": neg_input}

    def collate_fn(self, batch_data):
        batch_pos_input, batch_neg_input = [], []
        for data in batch_data:
            batch_pos_input.append(data['pos_input'])
            batch_neg_input.append(data['neg_input'])
        return self.convert_to_model_input(batch_pos_input), self.convert_to_model_input(batch_neg_input)

    def convert_to_model_input(self, batch_data):
        batch_input_ids, batch_token_type_ids, batch_attention_mask = [], [], []
        for input_data in batch_data:
            batch_input_ids.append(torch.tensor(input_data['input_ids'], dtype=torch.long, device=self.device))
            batch_token_type_ids.append(
                torch.tensor(input_data['token_type_ids'], dtype=torch.long, device=self.device)
            )
            batch_attention_mask.append(
                torch.tensor(input_data['attention_mask'], dtype=torch.bool, device=self.device)
            )
        return {
            "input_ids": pad_sequence(batch_input_ids, batch_first=True),
            "token_type_ids": pad_sequence(batch_token_type_ids, batch_first=True),
            "attention_mask": pad_sequence(batch_attention_mask, batch_first=True),
        }

    def to_dl(self, batch_size=32, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


if __name__ == '__main__':
    train_data = []
    with open(DATA_PATH.joinpath('lock-train.jsonl'), 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    for data in PairWiseDL(datas=train_data).to_dl(batch_size=32, shuffle=False):
        print()
