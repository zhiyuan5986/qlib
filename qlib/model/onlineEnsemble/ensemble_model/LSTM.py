# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import pickle

import warnings
warnings.filterwarnings("ignore")

class LSTM(Model):
    """LSTM Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluate metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2,
        early_stop=20,
        optimizer="adam",
        GPU=0,
        seed=None,
        n_models=20,
        pred_size=800,
        **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("LSTM")
        print("LSTM pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.pred_size = pred_size
        with open('./pkl_data/label.pkl', 'rb') as f: self.label = pickle.load(f)

        print(
            "LSTM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.lstm_model = LSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            outsize=n_models
        )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.lstm_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.lstm_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        
        x, y = self.ensemble(pred, label)
        return torch.mean((y - x)**2)
    
    def ensemble(self, pred, label):
        return (pred * label[:, 1:]).sum(dim=1), label[:, 0]
    
    def correlate(self, x, y):
        return ((x-x.mean()) * (y-y.mean()) / (x.std()*y.std())).mean()
    
    def ic_loss(self, pred, label):
        x, y = self.ensemble(pred, label)
        
        loss = 1 - self.correlate(x, y)
        return loss
    
    def rank_loss(self, pred, label):
        x, y = self.ensemble(pred, label)
        x1 = x.repeat(len(x))
        x2 = x.repeat(len(x), 1).T.flatten()
        y1 = y.repeat(len(y))
        y2 = y.repeat(len(y), 1).T.flatten()
        return torch.mean(F.relu(-(y1-y2).sign()*(x1-x2)))
    
    def mean_inc(self, pred, label, topk=20):
        x, y = self.ensemble(pred, label)
        return torch.nanmean(y[(-x).argsort()[:topk]])

    def loss_fn(self, preds, labels):
        masks = [torch.isfinite(label).all(axis=1) & torch.isfinite(pred).all(axis=1) for pred, label in zip(preds, labels)]
        
        return sum([self.ic_loss(pred[mask], label[mask]) for pred, label, mask in zip(preds, labels, masks)]) / len(preds)
    
        
    def metric_fn(self, preds, labels):
        # return -self.loss_fn(preds, labels)
        
        masks = [torch.isfinite(label).all(axis=1) & torch.isfinite(pred).all(axis=1) for pred, label in zip(preds, labels)]
        return torch.nanmean(torch.tensor([self.mean_inc(pred[mask], label[mask]) for pred, label, mask in zip(preds, labels, masks)]))

    def train_epoch(self, x_train, y_train):

        x_train_values = np.array([x_train.loc[d].values for d in self.dates_train])
        y_train_values = np.array([y_train.loc[d].values for d in self.dates_train])

        self.lstm_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            features = [torch.from_numpy(x).float().to(self.device) for x in x_train_values[indices[i : i + self.batch_size]]]
            labels = [torch.from_numpy(y).float().to(self.device) for y in y_train_values[indices[i : i + self.batch_size]]]

            preds = [self.lstm_model(feature) for feature in features]
            loss = self.loss_fn(preds, labels)
            if not torch.isfinite(loss): continue

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch0(self, data_x, data_y):

        # prepare training data
        dates = list(set([x[0] for x in data_x.index]))
        x_values = np.array([data_x.loc[d].values for d in dates])
        y_values = np.array([data_y.loc[d].values for d in dates])

        self.lstm_model.eval()

        scores = []

        indices = np.arange(len(x_values))
 
        for i in range(len(indices))[::self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            features = [torch.from_numpy(x).float().to(self.device) for x in x_values[indices[i : i + self.batch_size]]]
            labels = [torch.from_numpy(y).float().to(self.device) for y in y_values[indices[i : i + self.batch_size]]]

            preds = [self.lstm_model(feature) for feature in features]

            score = self.metric_fn(preds, labels)
            if torch.isfinite(score): scores.append(score.item())

        return np.mean(scores)
    
    def test_epoch(self, data_x, data_y):
        
        X = data_y.values[:, 1:]
        weight = self.predict(data_x.values)
        back_test_data = (X * weight).sum(axis=1)
        back_test_data_df = pd.DataFrame(back_test_data, index=data_y.index)
        inc = []
        for d in self.dates_valid:
            y = self.label.loc[d].values
            x = back_test_data_df.loc[d].values.squeeze()
            inc.append(y[(-x).argsort()[:20]].mean())
        return np.nanmean(inc)


    def fit(
        self,
        data,
        evals_result=dict(),
        save_path=None,
    ):

        df_train, df_valid, df_test = data
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        # y_valid['label'] = self.label.loc[y_valid.index]
        # y_train['label'] = (y_train['label'] - y_train['label'].rank(pct=True))
        self.dates_train = sorted(list(set([x[0] for x in df_train.index])))
        self.dates_valid = sorted(list(set([x[0] for x in df_valid.index])))

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["valid"] = []
        
        val_score = self.test_epoch(x_valid, y_valid)
        print("valid %.6f" % (val_score))

        # train
        print("\ntraining...")
        self.fitted = True

        for step in range(self.n_epochs):
            print("\nEpoch%d:" % step)
            print("training...")
            self.train_epoch(x_train, y_train)
            print("evaluating...")
            val_score = self.test_epoch(x_valid, y_valid)
            print("valid %.6f" % (val_score))
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.lstm_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.lstm_model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def finetune( self, df_train, ):

        df_train = df_train.dropna()
        x_train, y_train = df_train["feature"], df_train["label"]
        dates_train = sorted(list(set([x[0] for x in df_train.index])))

        if self.use_gpu: torch.cuda.empty_cache()
        
        x_train_values = np.array([x_train.loc[d].values for d in dates_train])
        y_train_values = np.array([y_train.loc[d].values for d in dates_train])

        self.lstm_model.train()

        features = [torch.from_numpy(x).float().to(self.device) for x in x_train_values]
        labels = [torch.from_numpy(y).float().to(self.device) for y in y_train_values]

        preds = [self.lstm_model(feature) for feature in features]
        loss = self.loss_fn(preds, labels)
        if not torch.isfinite(loss): return

        self.train_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
        self.train_optimizer.step()

    def predict(self, x_values):
        
        self.lstm_model.eval()
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.pred_size]:
            if sample_num - begin < self.pred_size:
                end = sample_num
            else:
                end = begin + self.pred_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.lstm_model(x_batch).detach().cpu().numpy()
                pred[np.isnan(pred)] = 0
            preds.append(pred)

        return np.concatenate(preds)


class LSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, outsize=1):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, outsize)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        # return F.gumbel_softmax(self.fc_out(out[:, -1, :]), dim=1, tau=0.2)#.squeeze()
        return F.softmax(self.fc_out(out[:, -1, :]), dim=1)# .squeeze()
        
