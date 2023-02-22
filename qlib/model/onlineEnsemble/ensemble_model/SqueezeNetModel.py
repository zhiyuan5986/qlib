import sys
sys.path.append('./reweight')

import numpy as np
import pandas as pd
import copy
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
from ensemble_model.squeezeNet import SqueezeNet1d
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import warnings
warnings.filterwarnings("ignore")

class SqueezeNetModel(Model):


    def __init__(
        self,
        d_feat=21,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        n_models=20,
        pred_size=800,
        **kwargs
    ):

        # set hyper-parameters.
        self.d_feat = d_feat
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.pred_size = pred_size
        with open('./pkl_data/label.pkl', 'rb') as f: self.label = pickle.load(f)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.squeeze_net = SqueezeNet1d(
            in_channels=self.d_feat,
            outsize=n_models
        )
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.squeeze_net.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.squeeze_net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.squeeze_net.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        
        x, y = self.ensemble(pred, label)
        return torch.mean((y - x)**2)
    
    def correlate(self, x, y):
        return ((x-x.mean()) * (y-y.mean()) / (x.std()*y.std())).mean()
    
    def ensemble(self, pred, label):
        return (pred * label[:, 1:]).sum(dim=1), label[:, 0]
    
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
    

    def loss_fn(self, preds, labels):
        masks = [torch.isfinite(label).all(axis=1) & torch.isfinite(pred).all(axis=1) for pred, label in zip(preds, labels)]
        
        return sum([self.ic_loss(pred[mask], label[mask]) for pred, label, mask in zip(preds, labels, masks)]) / len(preds)
    
    def score_fn(self, pred, label):
        x = (pred * label[:, 1:]).sum(dim=1)
        y = label[:, 0]
        return self.correlate(x, y)
        

    def metric_fn(self, preds, labels):
        return -self.loss_fn(preds, labels)
        
        masks = [torch.isfinite(label).all(axis=1) & torch.isfinite(pred).all(axis=1) for pred, label in zip(preds, labels)]
        return sum([self.score_fn(pred[mask], label[mask]) for pred, label, mask in zip(preds, labels, masks)]) / len(preds)

    def train_epoch(self, x_train, y_train):

        x_train_values = np.array([x_train.loc[d].values for d in self.dates_train])
        y_train_values = np.array([y_train.loc[d].values for d in self.dates_train])

        self.squeeze_net.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:

            if len(indices) - i < self.batch_size:
                break

            features = [torch.from_numpy(x).float().to(self.device) for x in x_train_values[indices[i : i + self.batch_size]]]
            labels = [torch.from_numpy(y).float().to(self.device) for y in y_train_values[indices[i : i + self.batch_size]]]

            preds = [self.squeeze_net(feature) for feature in features]
            loss = self.loss_fn(preds, labels)
            if not torch.isfinite(loss): continue

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.squeeze_net.parameters(), 3.0)
            self.train_optimizer.step()

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
        # y_train['label'] = (y_train['label'] - y_train['label'].rank(pct=True))
        self.dates_train = sorted(list(set([x[0] for x in df_train.index])))
        self.dates_valid = sorted(list(set([x[0] for x in df_valid.index])))

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        
        val_score = self.test_epoch(x_valid, y_valid)
        print("valid %.6f" % (val_score))

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
                best_param = copy.deepcopy(self.squeeze_net.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.squeeze_net.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()
    

    def predict(self, x_values):
        
        self.squeeze_net.eval()
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.pred_size]:
            if sample_num - begin < self.pred_size:
                end = sample_num
            else:
                end = begin + self.pred_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.squeeze_net(x_batch).detach().cpu().numpy()
                pred[np.isnan(pred)] = 0
            preds.append(pred)

        return np.concatenate(preds)