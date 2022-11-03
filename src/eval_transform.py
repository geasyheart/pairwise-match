# -*- coding: utf8 -*-
#
from typing import List, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.g import tokenizer


class PairWiseEvalDL(Dataset):
    def __init__(self, datas: List[Dict], device='cpu'):
        self.datas = datas
        self.device = torch.device(device=device)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas[item]
        pos_input = tokenizer.encode_plus(text=data['query'], text_pair=data['title'], max_length=512)
        return {"input": pos_input, "label": data['label']}

    def collate_fn(self, batch_data):
        batch_input_ids, batch_token_type_ids, batch_attention_mask = [], [], []
        batch_labels = []
        for input_data in batch_data:
            batch_input_ids.append(
                torch.tensor(input_data['input']['input_ids'], dtype=torch.long, device=self.device)
            )
            batch_token_type_ids.append(
                torch.tensor(input_data['input']['token_type_ids'], dtype=torch.long, device=self.device)
            )
            batch_attention_mask.append(
                torch.tensor(input_data['input']['attention_mask'], dtype=torch.bool, device=self.device)
            )
            batch_labels.append(np.array(input_data['label']))
        return {
            "input_ids": pad_sequence(batch_input_ids, batch_first=True),
            "token_type_ids": pad_sequence(batch_token_type_ids, batch_first=True),
            "attention_mask": pad_sequence(batch_attention_mask, batch_first=True),
        }, np.stack(batch_labels)

    def to_dl(self, batch_size=32, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
