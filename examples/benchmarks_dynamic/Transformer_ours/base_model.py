import numpy as np
import pandas as pd
import copy
import typing
from typing import Optional, List, Tuple, Union, Text, OrderedDict
from tqdm import tqdm
import pprint as pp
import pickle
import yaml
import os

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

import qlib
from qlib.utils import init_instance_by_config
from qlib.utils.data import deepcopy_basic_type
from qlib.data.dataset import Dataset, DataHandlerLP
from qlib.contrib.data.dataset import TSDataSampler
from qlib.model.meta.model import MetaTaskModel
from qlib.contrib.meta.incremental import MetaDatasetInc
from qlib.contrib.meta.incremental.utils import preprocess
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

class MetaModelRolling(MetaTaskModel):
    def __init__(
        self,
        n_epochs,
        lr,
        online_lr: dict = None,
        GPU=None, 
        seed=None, 
        train_stop_loss_thred=None, 
        begin_valid_epoch = 0,
        over_patience = 8,
        save_path = 'model/', 
        save_prefix= '',
        benchmark = 'SH000300', 
        market = 'csi300',
        only_backtest = False,
        **kwargs
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.online_lr = online_lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        self.begin_valid_epoch = begin_valid_epoch
        self.over_patience = over_patience
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.benchmark = benchmark
        self.market = market
        self.only_backtest = only_backtest
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.fitted = False
        with open("./workflow_config_transformer_Alpha158.yaml", 'r') as f:
            self.basic_config = yaml.safe_load(f)
        self.basic_config['market'] = self.market
        self.basic_config['benchmark'] = self.benchmark

        self.infer_exp_name = self.meta_exp_name+"_backtest"

        self.framework = None
        self.train_optimizer = None

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    @property
    def meta_exp_name(self):
        return f"{self.market}_transformer_alpha158_seed{self.seed}"

    def load_model(self, param_path):
        try:
            self.framework.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.") 

    def state_dict(self, destination: typing.OrderedDict[str, torch.Tensor]=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module and the state of the optimizer.

        Returns:
            dict:
                a dictionary containing a whole state of the module and the state of the optimizer.
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['framework'] = self.framework.state_dict()
        # destination['framework_opt'] = self.framework.opt.state_dict()
        destination['opt'] = self.train_optimizer.state_dict()
        return destination

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor],):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and the optimizer.

        Args:
            dict:
                a dict containing parameters and persistent buffers.
        """
        self.framework.load_state_dict(state_dict['framework'])
        # self.framework.opt.load_state_dict(state_dict['framework_opt'])
        self.train_optimizer.load_state_dict(state_dict['opt'])
    def init_model(self):
        if self.framework is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.framework.parameters(), self.lr)
        self.framework.to(self.device)

    def override_online_lr_(self):
        if self.online_lr is not None:
            if 'lr' in self.online_lr:
                self.lr = self.online_lr['lr']
                self.train_optimizer.param_groups[0]['lr'] = self.online_lr['lr']

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def load_data(self):
        segments = self.basic_config["dataset"]["kwargs"]["segments"] # Benchmark.basic_task()['dataset']['kwargs']['segments']
        t = deepcopy_basic_type(self.basic_config) # copy.deepcopy(self.basic_task)
        t["dataset"]["kwargs"]["segments"]["train"] = (
            segments["train"][0],
            segments["test"][1],
        ) # set the train segment to be the union of train and test segments
        ds = init_instance_by_config(t["dataset"], accept_types=Dataset) # get the dataset by the configuration
        data = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L) # get the training data (indeed it conclude all the data for self.basic_task)
        ta = TimeAdjuster(future=True, end_time=segments['test'][1]) # TimeAdjuster is a tool to adjust the time. future: whether including future trading day.

        if isinstance(data, TSDataSampler):
            assert ta.align_seg(t["dataset"]["kwargs"]["segments"]["train"])[0] == data.data_index[0][0] # ensure the date of the first record is aligned
        else:
            assert ta.align_seg(t["dataset"]["kwargs"]["segments"]["train"])[0] == data.index[0][0] # ensure the date of the first record is aligned

        rolling_task = deepcopy_basic_type(self.basic_config)
        self.factor_num = 20       
    
        self.horizon = 4
        self.data_dir = "cn_data"
        self.step = int(self.basic_config["dataset"]["kwargs"]["step_len"])
        self.use_extra = False
        trunc_days = self.horizon if self.data_dir == "us_data" else (self.horizon + 1) # set the trunc_days according to the horizon, days to be truncated based on the test start
        segments = rolling_task["dataset"]["kwargs"]["segments"]
        train_begin = segments["train"][0]
        train_end = ta.get(ta.align_idx(train_begin) + self.step - 1)
        test_begin = ta.get(ta.align_idx(train_begin) + self.step - 1 + trunc_days)
        test_end = rolling_task["dataset"]["kwargs"]["segments"]["valid"][1]
        extra_begin = ta.get(ta.align_idx(train_end) + 1)
        extra_end = ta.get(ta.align_idx(test_begin) - 1)
        test_end = ta.get(ta.align_idx(test_end) - self.step)
        seperate_point = str(rolling_task["dataset"]["kwargs"]["segments"]["valid"][0])
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "train": (train_begin, train_end),
            "test": (test_begin, test_end),
        } 
        if self.use_extra:
            rolling_task["dataset"]["kwargs"]["segments"]["extra"] = (extra_begin, extra_end)

        kwargs = dict(
            task_tpl=rolling_task,
            step=self.step,
            segments=seperate_point,
            task_mode="train",
        )

        # if self.forecast_model == "MLP" and self.alpha == 158:
        #     kwargs.update(task_mode="test")
        # md_offline contains a list named `meta_task_l`, each element of which is a MetaTaskInc instance containing the training and testing data.
        md_offline = MetaDatasetInc(data=data, **kwargs)
        # assert isinstance(md_offline.meta_task_l[0].get_meta_input()['X_train'], TSDataSampler), type(md_offline.meta_task_l[0].get_meta_input()['X_train'])

        # preprocess the meta_task_l, doing what?
        self.alpha = 158
        md_offline.meta_task_l = preprocess(
            md_offline.meta_task_l,
            factor_num=self.factor_num,
            is_mlp=False,
            alpha=self.alpha,
            step=self.step,
            H=self.horizon if self.data_dir == "us_data" else (1 + self.horizon),
            not_sequence=False,
            to_tensor=False
        )

        train_begin = segments["valid"][0]
        train_end = ta.get(ta.align_idx(train_begin) + self.step - 1)
        test_begin = ta.get(ta.align_idx(train_begin) + self.step - 1 + trunc_days)
        extra_begin = ta.get(ta.align_idx(train_end) + 1)
        extra_end = ta.get(ta.align_idx(test_begin) - 1)
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "train": (train_begin, train_end),
            "test": (test_begin, segments["test"][1]),
        }
        if self.use_extra:
            rolling_task["dataset"]["kwargs"]["segments"]["extra"] = (extra_begin, extra_end)

        kwargs.update(task_tpl=rolling_task, segments=0.0)
        data_I = None
        md_online = MetaDatasetInc(data=data, data_I=data_I, **kwargs)

        # assert isinstance(md_online.meta_task_l[0].get_meta_input()['X_train'], TSDataSampler)
        md_online.meta_task_l = preprocess(
            md_online.meta_task_l,
            factor_num=self.factor_num,
            is_mlp=False,
            alpha=self.alpha,
            step=self.step,
            H=self.horizon if self.data_dir == "us_data" else (1 + self.horizon),
            not_sequence=False,
            to_tensor=False
        )

        ta = TimeAdjuster(future=True)
        segments = self.basic_config["dataset"]["kwargs"]["segments"]
        test_begin, test_end = ta.align_seg(segments["test"])
        print('Test segment:', test_begin, test_end)
        ds = init_instance_by_config(self.basic_config["dataset"], accept_types=Dataset)
        label_all = ds.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
        if isinstance(label_all, TSDataSampler):
            label_all = pd.DataFrame({"label": label_all.data_arr[:-1][:, 0]}, index=label_all.data_index)
            label_all = label_all.loc[test_begin:test_end]
        label_all = label_all.dropna(axis=0)

        self.md_offline = md_offline
        self.md_online = md_online
        self.label_all = label_all

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def run_task(self, meta_input):
        """ A single task for """

        # self.train_optimizer.zero_grad()
        train_loader = self._init_data_loader(meta_input['d_train'], shuffle=True, drop_last=True)
        test_loader = self._init_data_loader(meta_input['d_test'], shuffle=False, drop_last=True)
        
        # X_train_data_loader = DataLoader(meta_input['X_train'], batch_size=300, shuffle=False, drop_last=False)
        # y_train_data_loader = DataLoader(meta_input['y_train'], batch_size=300, shuffle=False, drop_last=False)
        self.framework.train()
        losses = 0

        for data in train_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 20 selected factors + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.framework(feature.float())
            loss = self.loss_fn(pred, label)
            losses += loss

        # update once in every task 
        self.train_optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_value_(self.framework.parameters(), 3.0)
        self.train_optimizer.step()

        preds = []
        labels = []
        with torch.no_grad():
            for data in test_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].detach().numpy()

                pred = self.framework(feature.float()).detach().cpu().numpy()
                # assert np.sum(pred == np.nan) == 0

                preds.append(pred)
                labels.append(label)
        pred = np.concatenate(preds)
        label = np.concatenate(labels)
        return pred, label

    def run_epoch(self, phase, task_list, tqdm_show=False):
        pred_y_all, mse_all = [], 0
        self.phase = phase

        indices = np.arange(len(task_list))
        if phase == 'train':
            np.random.shuffle(indices)
        else:
            if phase == "test":
                checkpoint = copy.deepcopy(self.state_dict())
            lr = self.lr
            self.override_online_lr_()

        for i in tqdm(indices, desc=phase) if tqdm_show else indices:
            # torch.cuda.empty_cache()
            meta_input = task_list[i].get_meta_input()
            # if not isinstance(meta_input['X_train'], torch.Tensor):
            #     meta_input = {
            #         k: torch.tensor(v, device=self.framework.device, dtype=torch.float32) if 'idx' not in k else v
            #         for k, v in meta_input.items()
            #     }
            pred, label = self.run_task(meta_input)
            if phase != "train":
                test_idx = meta_input["test_idx"]
                # print(pred, test_idx)
                pred_y_all.append(
                    pd.DataFrame(
                        {
                            "pred": pd.Series(pred, index=test_idx),
                            "label": pd.Series(label, index=test_idx),
                        }
                    )
                )
        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
        if phase == "test":
            self.lr = lr
            self.load_state_dict(checkpoint)
            ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
            # print(ic)
            return pred_y_all, ic
        return pred_y_all, None

    def fit(self):
        phases = ["train", "test"]
        meta_tasks_l = self.md_offline.prepare_tasks(phases)

        self.cnt = 0
        self.framework.train()
        torch.set_grad_enabled(True)

        best_ic, patience = -1e3, 8
        best_checkpoint = copy.deepcopy(self.framework.state_dict())
        # run 100 epoch
        for epoch in tqdm(range(self.n_epochs), desc="epoch"):
            for phase, task_list in zip(phases, meta_tasks_l):
                if phase == "test":
                    if epoch < self.begin_valid_epoch:
                        continue
                pred_y, ic = self.run_epoch(phase, task_list)
                if phase == "test":
                    if ic < best_ic:
                        patience -= 1
                    else:
                        best_ic = ic
                        print("best ic:", best_ic)
                        patience = self.over_patience
                        best_checkpoint = copy.deepcopy(self.framework.state_dict())
            if patience <= 0:
                # R.save_objects(**{"model.pkl": self.tn})
                break
        self.fitted = True
        self.framework.load_state_dict(best_checkpoint)
        torch.save(best_checkpoint, f'{self.save_path}{self.save_prefix}transformer_{self.seed}.pkl')
    
    def backtest(self):
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
        rec = R.get_exp(experiment_name=self.infer_exp_name).list_recorders(rtype=Experiment.RT_L)[0]
        # mse = ((pred_y_all['pred'].to_numpy() - pred_y_all['label'].to_numpy()) ** 2).mean()
        # mae = np.abs(pred_y_all['pred'].to_numpy() - pred_y_all['label'].to_numpy()).mean()
        # print('mse:', mse, 'mae', mae)
        # rec.log_metrics(mse=mse, mae=mae)
        SigAnaRecord(recorder=rec, skip_existing=True).generate()
        PortAnaRecord(recorder=rec, config=backtest_config, skip_existing=True).generate()
        print(f"Your evaluation results can be found in the experiment named `{self.infer_exp_name}`.")
        return rec

    def inference(self):
        meta_tasks_test = self.md_online.prepare_tasks("test")
        self.framework.train()
        pred_y_all, ic = self.run_epoch("online", meta_tasks_test, tqdm_show=True)
        return pred_y_all, ic 

    def online_training(self):
        with R.start(experiment_name=self.infer_exp_name):

            pred_y_all, ic = self.inference()
            # print(pred_y_all)
            pred_y_all = pred_y_all[['pred']]
            # print(pred_y_all)
            pred_y_all = pred_y_all.loc[self.label_all.index]
            # print('lr_model:', meta_model.lr_model, 'lr_ma:', meta_model.framework.opt.param_groups[0]['lr'],
            #       'lr_da:', meta_model.opt.param_groups[0]['lr'])
            print('lr:', self.lr)
            R.save_objects(**{"pred.pkl": pred_y_all, "label.pkl": self.label_all})
        rec = self.backtest()
        return rec


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', benchmark = 'SH000300', market = 'csi300'):
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

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

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
        self.dl_test = ds.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        print(self.dl_test.get_index())

    def load_model(self, param_path):
        try:
            self.model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.")


    def train_epoch(self, data_loader):
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

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            assert not torch.any(torch.isnan(label))
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

    def fit(self, use_pretrained = False, param_path = None):
        if use_pretrained:
            try:
                self.load_param(param_path)
                print(f"Model Loaded from {param_path}.")
                return
            except ValueError("Model not found."):
                print(f"Model not found. Start training from scratch.")
        train_loader = self._init_data_loader(self.dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(self.dl_valid, shuffle=False, drop_last=True)

        self.fitted = True
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (step, train_loss, val_loss))
            best_param = copy.deepcopy(self.model.state_dict())

            if train_loss <= self.train_stop_loss_thred:
                break
        torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')

    def backtest(self, predictions, labels):
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
        rec = R.get_exp(experiment_name=self.infer_exp_name).list_recorders(rtype=Experiment.RT_L)[0]
        mse = ((predictions.to_numpy() - labels.to_numpy()) ** 2).mean()
        mae = np.abs(predictions.to_numpy() - labels.to_numpy()).mean()
        print('mse:', mse, 'mae', mae)
        rec.log_metrics(mse=mse, mae=mae)
        SigAnaRecord(recorder=rec, skip_existing=False).generate()
        PortAnaRecord(recorder=rec, config=backtest_config, skip_existing=True).generate()
        print(f"Your evaluation results can be found in the experiment named `{self.infer_exp_name}`.")
        return rec

    def predict(self, use_pretrained = True):
        if use_pretrained:
            self.load_param(f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_loader = self._init_data_loader(self.dl_test, shuffle=False, drop_last=False)

        preds = []
        labels = []
        ic = []
        ric = []
        with R.start(experiment_name=self.infer_exp_name):
            self.model.eval()
            for data in test_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].detach().numpy()
                with torch.no_grad():
                    pred = self.model(feature.float()).detach().cpu().numpy()
                preds.append(pred.ravel())
                labels.append(label.ravel())

                daily_ic, daily_ric = calc_ic(pred, label)
                ic.append(daily_ic)
                ric.append(daily_ric)

            predictions = pd.DataFrame(np.concatenate(preds), index=self.dl_test.get_index())
            labels = pd.DataFrame(np.concatenate(labels), index=self.dl_test.get_index())

            mask = labels.isnull()
            predictions = predictions[~mask]
            labels = labels[~mask]
            
            metrics = {
                'IC': np.mean(ic),
                'ICIR': np.mean(ic)/np.std(ic),
                'RIC': np.mean(ric),
                'RICIR': np.mean(ic)/np.std(ric)
            }
            # print(metrics)
            # print(predictions)
            # print(predictions.shape)
            # print(labels)
            # print(labels.shape)
            R.save_objects(**{"pred.pkl": predictions, "label.pkl": labels})
        rec = self.backtest(predictions, labels)
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
        # self.load_model(param_path=f"../../benchmarks/MASTER/model/csi300master_{self.seed}.pkl")
        self.fit()
        rec = self.predict()
        metrics = rec.list_metrics()
        print(metrics)
        for k in all_metrics.keys():
            all_metrics[k].append(metrics[k])
        pp.pprint(all_metrics)