# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from qlib.model.base import Model
import pickle

import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from pprint import pprint

class XGBoost(Model):

    def __init__(
        self,
        n_models=20,
        epochs = 20,
        early_stop = 5,
        params={
            'colsample_bytree': 0.8879,
            'nthread':20,
            'eta': 0.05,
            'max_depth': 8,
            'n_estimators': 647,
            'subsample': 0.8789,
            'objective': 'multi:softprob',
            'disable_default_eval_metric': 1,
            'df_seasonality': None
        },
    ):
        # Set logger.

        # set hyper-parameters.
        params['num_class'] = n_models
        self.params = params
        self.epochs = epochs
        self.early_stop = early_stop
        with open('./pkl_data/label.pkl', 'rb') as f: self.label = pickle.load(f)

        pprint(params)

        self.xgb_model = None
        with open('./pkl_data/label_all.pkl', 'rb') as f: self.label = pickle.load(f)
    
    def fobj_mse(self, predt: np.ndarray, dtrain: xgb.DMatrix):

        y = dtrain.get_label().astype(int)
        n_train = len(y)
        # predt in softmax
        preds = np.reshape(predt, self.errors[y, :].shape)
        tmp = np.exp(preds)
        preds = tmp / tmp.sum(axis=1, keepdims=True)
        weighted_avg_loss_func = (preds * self.errors[y, :]).sum(axis=1, keepdims=True)

        grad = preds * (self.errors[y, :] - weighted_avg_loss_func)
        hess = preds * (self.errors[y,:] * (1.0 - preds) - grad)

        return grad.flatten(), hess.flatten()
    
    def correlate(self, x, y):
        return ((x-x.mean()) * (y-y.mean()) / (x.std()*y.std())).sum()
    
    def loss(self, preds_raw, y):
        tmp = torch.exp(preds_raw)
        preds = tmp / tmp.sum(axis=1, keepdim=True)
        y_hat = (preds * self.y_all[y, 1:]).sum(axis=1)
        label = self.y_all[y, 0]
        loss = -self.correlate(y_hat, label)
        
        return loss

    def fobj(self, predt: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label().astype(int)
        preds_raw = torch.tensor(predt, requires_grad=True)
        loss = self.loss(preds_raw, y)
        # weighted_avg_loss_func = (preds * torch.tensor(self.errors[y, :])).sum(axis=1)
        grad = torch.autograd.grad(loss, preds_raw)[0].numpy()
        
        hess = 1e-2 * np.ones_like(grad)
        # hess = preds.detach().numpy() * (self.errors[y,:] * (1.0 - preds.detach().numpy()) - grad)

        return grad.flatten(), hess.flatten()

    def feval(self, predt: np.ndarray, dtrain: xgb.DMatrix):
        """
        """
        y = dtrain.get_label().astype(int)
        preds_raw = torch.tensor(predt, requires_grad=True)
        loss = self.loss(preds_raw, y)

        return 'FFORMA-loss', loss/len(y)

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
        y_all = pd.concat((y_train, y_valid), axis=0)
        y_all['label'] = self.label.loc[y_all.index]
        self.y_all = torch.from_numpy(y_all.values)

        # self.errors = np.vstack((
        #     (y_train.values - y_train['label'].values.reshape((-1, 1)))[:, 1:] ** 2 * 0.5,
        #     (y_valid.values - y_valid['label'].values.reshape((-1, 1)))[:, 1:] ** 2 * 0.5
        # ))
        
        dtrain = xgb.DMatrix(data=x_train, label=np.arange(len(x_train)))
        dvalid = xgb.DMatrix(data=x_valid, label=np.arange(len(x_train), len(x_train)+len(x_valid)))
        
        self.xgb_model = xgb.train(
            params = self.params,
            dtrain = dtrain,
            obj = self.fobj,
            num_boost_round = self.epochs,
            early_stopping_rounds = self.early_stop,
            feval = self.feval,
            # verbose_eval = 5,
            evals=[(dtrain, "train"), (dvalid, "valid")],
        )
        
    
    def predict(self, x_values):
        
        return self.xgb_model.predict(xgb.DMatrix(x_values))
        
