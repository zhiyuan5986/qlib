# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

PATH = '/kaggle/working/'

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

class WeightGenerator(Model):

    def __init__(
        self,
        d_feat=5,
        hidden_size=64,
        embedding_size=32,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        task_batch_size=10,
        data_batch_size=800,
        early_stop=20,
        optimizer="adam",
        GPU=0,
        seed=None,
        pred_size=800,
        feature_size=32,
        model_count=20,
        **kwargs
    ):

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = task_batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.pred_size = pred_size
        self.feature_size = feature_size
        # with open('./pkl_data/label_all.pkl', 'rb') as f: self.label = pickle.load(f)
        self.tested = False

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
                task_batch_size,
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

        self.reweight_model = ReweightModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            embedding_size=embedding_size,
            batch_size=data_batch_size,
            feature_size=feature_size,
            model_count=model_count
        )
        
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.reweight_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.reweight_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.reweight_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        
        x, y = self.ensemble(pred, label)
        return 0.5*torch.mean((y - x)**2)
    
    def loss_fn(self, x, sig):
        if torch.abs(torch.det(sig)) < 1e-8: sig += torch.eye(len(sig)).to(sig.device) * 1e-8
        sig_rev = torch.inverse(sig)
        logp = - ( torch.log(torch.det(sig)) + torch.sum((x @ sig_rev) * x, dim=1).mean() )
        return -logp
    
        
    def metric_fn(self, error, sig):
        loss = self.loss_fn(error, sig)
        sig_rev = torch.inverse(sig)
        w = sig_rev.sum(dim=1, keepdim=True)
        w /= w.sum()
        wtmp = np.repeat(w.cpu().detach().numpy().T, len(error), 0)
        self.weight_tmp = wtmp if type(self.weight_tmp) == type(None) else np.vstack((self.weight_tmp, wtmp))
        score = ((error @ w) ** 2).mean() * 0.5
        avg_score = ((error @ (torch.ones_like(w)/w.shape[0]).to(error.device)) ** 2).mean() * 0.5
        
        return -score, loss, avg_score
        
        
    def train_epoch(self, tasks):
        
        torch.cuda.empty_cache()
        self.reweight_model.train()

        indices = np.arange(len(tasks))
        np.random.shuffle(indices)

        for i in tqdm(range(len(indices))[::self.batch_size][:-1]):
            batch_loss = 0; cnt = 0
    
            if len(indices) - i < self.batch_size:
                break
            for k in range(i, i+self.batch_size):
                support_set = torch.from_numpy(self.data.loc[tasks[k][0]].values.astype(np.float32)).to(self.device)
                query_set_data = torch.from_numpy(self.data['label'].loc[tasks[k][1]].values.astype(np.float32)).to(self.device)
                normParam = self.reweight_model(support_set)
                
                loss = self.loss_fn(query_set_data, normParam)
                if torch.isfinite(loss): batch_loss += loss; cnt += 1
                del support_set, query_set_data, normParam, loss
                torch.cuda.empty_cache()
            
            if not cnt: continue
            loss = batch_loss / cnt
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.reweight_model.parameters(), 3.0)
            self.train_optimizer.step()
            del loss
            torch.cuda.empty_cache()

    def test_epoch(self, tasks):

        # prepare training data
        self.weight_tmp = None

        self.reweight_model.eval()

        scores = []
        losses = []
        avg_scores = []

        indices = np.arange(len(tasks))
 
        for i in tqdm(range(len(indices))[::self.batch_size]):

            for k in range(i, min(i+self.batch_size, len(indices))):
                support_set = torch.from_numpy(self.data.loc[tasks[k][0]].values.astype(np.float32)).to(self.device)
                query_set_data = torch.from_numpy(self.data['label'].loc[tasks[k][1 if k == len(indices)-1 else 2]].values.astype(np.float32)).to(self.device)
                normParam = self.reweight_model(support_set)

                score, loss, avg_score = self.metric_fn(query_set_data, normParam)
                if torch.isfinite(score): scores.append(score.item())
                if torch.isfinite(loss): losses.append(loss.item())
                if torch.isfinite(avg_score): avg_scores.append(avg_score.item())
                del support_set, query_set_data, normParam
                torch.cuda.empty_cache()

        self.tested = True
        return np.mean(scores), np.mean(losses), np.mean(avg_scores)


    def fit(
        self,
        tasks,
        data,
    ):
        train_tasks, valid_tasks, test_tasks = tasks
        self.data = data
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        
        val_score, val_loss, avg_score = self.test_epoch(valid_tasks)
        print("valid score: %.6f, valid loss: %.6f, avg score: %.6f" % (val_score, val_loss, avg_score))

        # train
        print("\ntraining...")
        self.fitted = True

        for step in range(self.n_epochs):
            print("\nEpoch%d:" % step)
            print("training...")
            self.train_epoch(train_tasks)
            print("evaluating...")
            val_score, val_loss, avg_score = self.test_epoch(valid_tasks)
            print("valid score: %.6f, valid loss: %.6f, avg score: %.6f" % (val_score, val_loss, avg_score))

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.reweight_model.state_dict())
                self.weight = self.weight_tmp
                with open(os.path.join(PATH, 'best_params.pkl'), 'wb') as f: pickle.dump(best_param, f)
                with open(os.path.join(PATH, 'backtest_weight.pkl'), 'wb') as f: pickle.dump(self.weight, f)
                
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.reweight_model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()


    def predict(self, x_values):
        
        self.reweight_model.eval()
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.pred_size]:
            if sample_num - begin < self.pred_size:
                end = sample_num
            else:
                end = begin + self.pred_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.reweight_model(x_batch).detach().cpu().numpy()
                pred[np.isnan(pred)] = 0
            preds.append(pred)

        return np.concatenate(preds)


class ReweightModel(nn.Module):
    def __init__(self, d_feat=5, hidden_size=64, num_layers=2, dropout=0.0, 
                 embedding_size=32, feature_size=32, sim_size=32, model_count=20, batch_size=800):
        super().__init__()
        self.model_count = model_count
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.sim_size = sim_size

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.reweighter = nn.Linear(sim_size, 1)
        self.attn = nn.MultiheadAttention(self.embedding_size+self.model_count, 2)
        # self.weight_extractor = nn.Linear()
        # self.feature_extractor = nn.Linear(embedding_size, feature_size)

        self.d_feat = d_feat
    
    def normParam(self, data, weight):
        miu_w = weight.T @ data
        # miu = data.mean(dim=0, keepdim=True)
        sig_w = (data-miu_w).T @ ((data-miu_w) * weight)
        return sig_w
    
    def reweight(self, embedding, error):
        embedding = torch.hstack((torch.ones((len(embedding), 1)).to(embedding.device), embedding))
        n = len(embedding) // 2 // self.sim_size * self.sim_size
        assert n > 0
        xx = embedding[len(embedding)-n:].reshape((self.sim_size, -1, (self.embedding_size+1)))
        yy = error[len(embedding)-n:].reshape((self.sim_size, -1, self.model_count))
        ww = torch.stack([torch.inverse(x.T @ x) @ x.T @ y for x, y in zip(xx, yy)])
        preds = torch.matmul(embedding, ww)
        reweight_feature = 0.5 * ((preds-error)**2).mean(axis=-1).T
        # weight = F.softmax(self.reweighter(reweight_feature), dim=0)
        weight = F.sigmoid(self.reweighter(reweight_feature))
        
        return weight / weight.sum()
    
    def reweight_attn(self, embedding, error):
        x = torch.hstack((embedding, error)).reshape((1, -1, self.embedding_size + self.model_count))
        

    def forward(self, data):
        torch.cuda.empty_cache()
        x, error = torch.hsplit(data, [data.shape[1]-self.model_count])
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        embedding = F.tanh(self.linear(out[:, -1, :]))
        # feature = self.feature_extractor(embedding)
        # processed_data = torch.hstack((feature, error))
        # if not return_normParam: return processed_data
        
        weight = self.reweight(embedding, error)
        del out, embedding
        return self.normParam(error, weight)

class SelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x) # N x F
        key = self.key(x) # N x F
        value = self.value(x) # N x F

        # N x N
        attention_matrix = torch.matmul(query, key.transpose(1, 2))
        attention_matrix = self.softmax(attention_matrix)

        # N x F
        output = torch.matmul(attention_matrix, value)
        return output

        
