# -*- coding: utf8 -*-
#

# 啊，恶心啦，训练的数据和评估的数据格式是不一样的啊～
import json
import math
import os
from typing import Union, Type, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import set_seed, get_linear_schedule_with_warmup
from utrainer import Metric

from src.conf import DATA_PATH
from src.eval_transform import PairWiseEvalDL
from src.metric import AUC
from src.model import PairwiseMatchingModel
from src.train_transform import PairWiseTrainDL


class Trainer(object):
    def __init__(self):
        set_seed(1000)
        self.save_path = os.path.join(os.path.expanduser('~'), '.u-trainer', type(self).__name__)
        os.makedirs(self.save_path, exist_ok=True)

        self.tb_writer = SummaryWriter(os.path.join(self.save_path, 'tb_log'))

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self._model.to(self.device)

    def build_optimizer(
            self,
            warmup_steps: Union[float, int],
            num_training_steps: int,
            lr=1e-5,
            weight_decay=0.01
    ):
        if warmup_steps <= 1:
            warmup_steps = int(num_training_steps * warmup_steps)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def fit(
            self,
            train_dl: dataloader.DataLoader,
            eval_train_dl: dataloader.DataLoader,
            eval_dev_dl: dataloader.DataLoader,
            epochs: int = 30,
            lr: Union[int, float] = 1e-5,
            warmup_steps: Union[int, float] = 0.1,
            metric_cls: Type[Metric] = None,
            fine_tune_model=None
    ):

        optimizer, scheduler = self.build_optimizer(
            warmup_steps=warmup_steps,
            num_training_steps=len(train_dl) * epochs,
            lr=lr
        )
        if fine_tune_model:
            self.load_weights(fine_tune_model)

        min_loss = math.inf

        for epoch in range(epochs):
            train_loss = self._fit_dataloader(
                train_dl=train_dl,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch
            )
            self.tb_writer.add_scalar('train_loss', train_loss, epoch)

            train_metric = self._evaluate_dataloader(
                dl=eval_train_dl,
                metric=metric_cls(),
                epoch=epoch
            )
            train_metric_score = train_metric.score()
            train_metric_report = train_metric.report() or {}

            self.tb_writer.add_scalar('train_metric_score', train_metric_score, epoch)
            if train_metric_report:
                self.tb_writer.add_scalars('train_metric_report', train_metric_report, epoch)

            dev_metric = self._evaluate_dataloader(
                dl=eval_dev_dl,
                metric=metric_cls(),
                epoch=epoch
            )
            dev_metric_score = dev_metric.score()
            dev_metric_report = dev_metric.report() or {}

            self.tb_writer.add_scalar('dev_metric_score', dev_metric_score, epoch)
            if dev_metric_report:
                self.tb_writer.add_scalars('dev_metric_report', dev_metric_report, epoch)

            # savepoint
            if train_loss < min_loss:
                self.save_weights(f'loss_{train_loss:.4f}_epoch_{epoch}.pt')
                min_loss = train_loss
            self.save_weights(f'dev_{dev_metric_score:.4f}_epoch_{epoch}.pt')

    @torch.no_grad()
    def _evaluate_dataloader(self, dl: dataloader.DataLoader, metric: Metric, epoch):
        self.model.eval()
        for batch_idx, batch_data in tqdm(enumerate(dl), desc=f'eval[{epoch}]'):
            out = self.evaluate_steps(batch_idx, batch_data)
            metric.step(out)
        return metric

    def _fit_dataloader(
            self,
            train_dl: dataloader.DataLoader,
            optimizer,
            scheduler,
            epoch
    ) -> float:
        self.model.train()
        total_loss = 0
        for batch_idx, batch_data in tqdm(enumerate(train_dl), desc=f'fit[{epoch}]'):
            train_info = self.train_steps(batch_idx, batch_data)
            loss = train_info['loss']
            total_loss += loss.item()

            detail_loss = train_info.get('detail_loss', {})
            loss.backward()
            self._step(optimizer=optimizer, scheduler=scheduler)

            if detail_loss:
                detail_loss = {key: getattr(value, "item", lambda: value)() for key, value in detail_loss.items()}
                self.tb_writer.add_scalars('detail_loss', detail_loss, epoch * len(train_dl) + batch_idx)
            # 删除
            del train_info
        return total_loss / len(train_dl)

    def train_steps(self, batch_idx, batch_data) -> Dict:
        pos_input, neg_input = batch_data
        loss = self.model(pos_input=pos_input, neg_input=neg_input)
        return {"loss": loss}

    def evaluate_steps(self, batch_idx, batch_data):
        input, labels = batch_data
        y_scores = self.model.predict(input=input)
        return labels, y_scores

    def _step(self, optimizer, scheduler):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    def save_weights(self, save_name):
        dir_path = os.path.join(self.save_path, 'savepoint')
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, save_name)

        if not isinstance(self.model, nn.DataParallel):
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model.module.state_dict(), save_path)

    def load_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            self.model.load_state_dict(torch.load(save_path))
        else:
            self.model.module.load_state_dict(torch.load(save_path))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.model = PairwiseMatchingModel()
    with open(DATA_PATH.joinpath('lock-train.jsonl'), 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    train_dl = PairWiseTrainDL(datas=train_data, device=trainer.device).to_dl(batch_size=32, shuffle=True)

    # eval
    with open(DATA_PATH.joinpath('train.jsonl'), 'r', encoding='utf-8') as f:
        eval_train_data = [json.loads(line) for line in f]
    with open(DATA_PATH.joinpath('dev.jsonl'), 'r', encoding='utf-8') as f:
        eval_dev_data = [json.loads(line) for line in f]

    eval_train_dl = PairWiseEvalDL(eval_train_data, device=trainer.device).to_dl(batch_size=32, shuffle=False)
    eval_dev_dl = PairWiseEvalDL(eval_dev_data, device=trainer.device).to_dl(batch_size=32, shuffle=False)
    trainer.fit(
        train_dl=train_dl,
        eval_train_dl=eval_train_dl,
        eval_dev_dl=eval_dev_dl,
        metric_cls=AUC
    )
