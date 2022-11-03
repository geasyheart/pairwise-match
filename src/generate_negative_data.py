# -*- coding: utf8 -*-
#
import functools
import json
import os
import random

import tqdm
from torch.utils.data import Dataset

from src.conf import DATA_PATH


class PairwiseTransform(Dataset):
    def __init__(self, datas, ):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        pass


def build_negative_data(sample, title_map):
    negative_samples = []

    queries = title_map.get(sample['title'], [])

    for query in queries:
        if query['query'] != sample['query'] and query['label'] == 0:
            negative_samples.append(
                {"query": sample["query"], "title": sample['title'], "negative_query": query['query']}
            )

    # 再随机增加几个负样本
    for title in random.sample(title_map.keys(), 5):
        queries = title_map[title]
        if title != sample['title']:
            if queries[0]['query'] != sample['query']:
                negative_samples.append(
                    {"query": sample["query"], "title": sample['title'], "negative_query": queries[0]['query']}
                )
    return negative_samples


def cache_file(func):
    @functools.wraps(func)
    def w(file_path):
        abs_path = os.path.join(os.path.dirname(file_path), f"lock-{os.path.basename(file_path)}")
        if os.path.exists(abs_path):
            datas = []
            with open(abs_path, 'r') as f:
                for line in f:
                    datas.append(json.loads(line))
            return datas
        else:
            datas = func(file_path=file_path)
            with open(abs_path, 'w') as f:
                for line in datas:
                    line = json.dumps(line, ensure_ascii=False)
                    f.write(line + '\n')
            return datas

    return w


@cache_file
def load_data(file_path):
    datas = []
    with open(file_path, 'r') as f:
        for line in f:
            datas.append(json.loads(line))

    title_map = {}
    for data in datas:
        title_map.setdefault(data['title'], []).append(data)

    final_datas = []
    for data in tqdm.tqdm(datas, total=len(datas)):
        # 注意这里哦，如果label为0,那也没法找正样本做排序
        if data['label'] == 1:
            final_datas.extend(build_negative_data(sample=data, title_map=title_map))
    return final_datas


if __name__ == '__main__':
    load_data(DATA_PATH.joinpath('train.jsonl'))
    load_data(DATA_PATH.joinpath('dev.jsonl'))
    # 这条没有
    # load_data(DATA_PATH.joinpath('test.jsonl'))
