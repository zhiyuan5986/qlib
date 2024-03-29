import numpy as np
import pandas as pd
import copy
from typing import Optional, List, Tuple, Union, Text
import tqdm
import pprint as pp
import pickle
import os

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

import qlib
from qlib.utils import init_instance_by_config
from qlib.data.dataset import Dataset, DataHandlerLP
from qlib.contrib.data.dataset import TSDataSampler
from qlib.workflow.record_temp import SigAnaRecord, PortAnaRecord
from qlib.workflow import R, Experiment
from qlib.workflow.task.utils import TimeAdjuster

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', benchmark = 'SH000300', market = 'csi300', only_backtest = False):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.benchmark = benchmark
        self.market = market
        self.infer_exp_name = f"{self.market}_MASTER_alpha158_horizon4_step{self.basic_config['dataset']['kwargs']['step_len']}_backtest"


        self.fitted = False

        self.model = None
        self.train_optimizer = None
        self.data_dir = 'cn_data'
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.only_backtest = only_backtest


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def load_model(self, param_path):
        try:
            self.model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def load_data(self):
        # with open(f'/data/liuqiaoan/{self.market}/{self.market}_dl_train.pkl', 'rb') as f:
        #     self.dl_train = pickle.load(f)
        # with open(f'/data/liuqiaoan/{self.market}/{self.market}_dl_valid.pkl', 'rb') as f:
        #     self.dl_valid = pickle.load(f)
        # with open(f'/data/liuqiaoan/{self.market}/{self.market}_dl_test.pkl', 'rb') as f:
        #     self.dl_test = pickle.load(f)
        # print("Data Loaded.")
        
        ds = init_instance_by_config(self.basic_config['dataset'], accept_types=Dataset)
        self.dl_train = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        print(self.dl_train.get_index())
        self.dl_valid = ds.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        print(self.dl_valid.get_index())
        self.dl_test = ds.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        print(self.dl_test.get_index())

        ta = TimeAdjuster(future=True)
        segments = self.basic_config["dataset"]["kwargs"]["segments"]
        test_begin, test_end = ta.align_seg(segments["test"])
        print('Test segment:', test_begin, test_end)
        ds = init_instance_by_config(self.basic_config["dataset"], accept_types=Dataset)
        label_all = ds.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_R, only_label = True)
        if isinstance(label_all, TSDataSampler):
            label_all = pd.DataFrame({"label": label_all.data_arr[:-1][:, 0]}, index=label_all.data_index)
            label_all = label_all.loc[test_begin:test_end]
        self.label_all = label_all.dropna(axis=0)

    def train_epoch(self, data_loader, use_prompts = True):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            assert not torch.any(torch.isnan(label))

            if use_prompts:
                [pred, cos_result] = self.model(feature.float(), use_prompts)
                # print(cos_result)
                loss = self.loss_fn(pred, label)+self.model.lamb * cos_result
            else:
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)

            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader, use_prompts = True):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            assert not torch.any(torch.isnan(label))
            if use_prompts:
                (pred, _) = self.model(feature.float(), use_prompts)
            else:
                pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def fit(self, use_pretrained = True):
        train_loader = self._init_data_loader(self.dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(self.dl_valid, shuffle=False, drop_last=True)

        self.fitted = True
        best_param = None
        best_val_loss = 1e3

        # freeze some parameters
        # for name, param in self.model.master.named_parameters():
        #     if "tatten" in name or "satten" in name or 'feature_gate' in name or 'x2y' in name:
        #         param.requires_grad = False

        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (step, train_loss, val_loss))
            if best_val_loss > val_loss:
                best_param = copy.deepcopy(self.model.state_dict())
                best_val_loss = val_loss

            if train_loss <= self.train_stop_loss_thred:
                break
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(f'{self.save_path}/mprompts{self.m_prompts}_nprompts{self.n_prompts}_lenprompts{self.len_prompts}_lamb{self.lamb}'):
            os.makedirs(f'{self.save_path}/mprompts{self.m_prompts}_nprompts{self.n_prompts}_lenprompts{self.len_prompts}_lamb{self.lamb}')

        torch.save(best_param, f'{self.save_path}/mprompts{self.m_prompts}_nprompts{self.n_prompts}_lenprompts{self.len_prompts}_lamb{self.lamb}/{self.save_prefix}promptmaster_{self.seed}.pkl')

    def backtest(self):
        backtest_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {"signal": "<PRED>", "topk": 30, "n_drop": 30},
            },
            "backtest": {
                "start_time": "2017-01-01",
                "end_time": "2020-08-01",
                "account": 100000000,
                "benchmark": self.benchmark,
                "exchange_kwargs": {
                    # "limit_threshold":  0.095,
                    "deal_price": "close",
                    # "open_cost": 0.0005,
                    # "close_cost": 0.0015,
                    # "min_cost": 5,
                },
            },
        }
        # backtest_config = {
        #     "strategy": {
        #         "class": "TopkDropoutStrategy",
        #         "module_path": "qlib.contrib.strategy",
        #         "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
        #     },
        #     "backtest": {
        #         "start_time": None,
        #         "end_time": None,
        #         "account": 100000000,
        #         "benchmark": self.benchmark,
        #         "exchange_kwargs": {
        #             "limit_threshold": None if self.data_dir == "us_data" else 0.095,
        #             "deal_price": "close",
        #             "open_cost": 0.0005,
        #             "close_cost": 0.0015,
        #             "min_cost": 5,
        #         },
        #     },
        # }
        rec = R.get_exp(experiment_name=self.infer_exp_name).list_recorders(rtype=Experiment.RT_L)[0]
        # mse = ((predictions.to_numpy() - labels.to_numpy()) ** 2).mean()
        # mae = np.abs(predictions.to_numpy() - labels.to_numpy()).mean()
        # print('mse:', mse, 'mae', mae)
        # rec.log_metrics(mse=mse, mae=mae)
        SigAnaRecord(recorder=rec, skip_existing=False).generate()
        PortAnaRecord(recorder=rec, config=backtest_config, skip_existing=True).generate()
        print(f"Your evaluation results can be found in the experiment named `{self.infer_exp_name}`.")
        return rec

    def predict(self, use_pretrained = True, use_prompts = True):
        if use_pretrained:
            self.load_param(f'{self.save_path}/mprompts{self.m_prompts}_nprompts{self.n_prompts}_lenprompts{self.len_prompts}_lamb{self.lamb}/{self.save_prefix}promptmaster_{self.seed}.pkl')
            
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_loader = self._init_data_loader(self.dl_test, shuffle=False, drop_last=False)

        pred_all = []
        with R.start(experiment_name=self.infer_exp_name):

            self.model.eval()
            for data in test_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, 0:-1].to(self.device)
                with torch.no_grad():
                    if use_prompts:
                        pred, _ = self.model(feature.float(), use_prompts)
                        pred = pred.detach().cpu().numpy()
                    else:
                        pred = self.model(feature.float()).detach().cpu().numpy()
                pred_all.append(pred.ravel())
                # labels.append(label.ravel())


            pred_all = pd.DataFrame(np.concatenate(pred_all), index=self.dl_test.get_index())
            pred_all = pred_all.loc[self.label_all.index]
            # labels = pd.DataFrame(np.concatenate(labels), index=self.dl_test.get_index())

            
            # print(metrics)
            # print(predictions)
            # print(predictions.shape)
            # print(labels)
            # print(labels.shape)
            R.save_objects(**{"pred.pkl": pred_all, "label.pkl": self.label_all})
        rec = self.backtest()
        return rec

        # return predictions, metrics

    def run_all(self):
        all_metrics = {
            k: []
            for k in [
                # 'mse', 'mae',
                "IC",
                "ICIR",
                "Rank IC",
                "Rank ICIR",
                "1day.excess_return_without_cost.annualized_return",
                "1day.excess_return_without_cost.information_ratio",
                # "1day.excess_return_with_cost.max_drawdown",
            ]
        }
        self.load_data()
        self.fit()
        rec = self.predict()
        metrics = rec.list_metrics()
        print(metrics)
        for k in all_metrics.keys():
            all_metrics[k].append(metrics[k])
        pp.pprint(all_metrics)