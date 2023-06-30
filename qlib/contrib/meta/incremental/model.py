import copy
from collections import defaultdict

import numpy as np
from qlib.model.meta import MetaTaskDataset

from qlib.model.meta.model import MetaTaskModel

from tqdm import tqdm
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
import higher
from . import higher_optim  # IMPORTANT, DO NOT DELETE

from .utils import override_state, has_rnn
from .dataset import MetaDatasetInc
from .net import DoubleAdapt, ForecastModel, CoG



class MetaModelInc(MetaTaskModel):
    def __init__(
        self,
        task_config,
        lr_model=0.001,
        first_order=True,
        x_dim=None,
        alpha=360,
        pretrained_model=None,
        begin_valid_epoch=0,
        **kwargs
    ):
        self.task_config = task_config
        self.lr_model = lr_model
        self.first_order = first_order
        self.begin_valid_epoch = begin_valid_epoch
        self.framework = self._init_framework(task_config, x_dim, lr_model, need_permute=int(alpha) == 360,
                                              model=pretrained_model, **kwargs)
        self.opt = self._init_meta_optimizer(**kwargs)
        self.has_rnn = has_rnn(self.framework)

    def _init_framework(self, task_config, x_dim=None, lr_model=0.001, need_permute=False, model=None, **kwargs):
        return ForecastModel(task_config, x_dim=x_dim, lr=lr_model, need_permute=need_permute, model=model)

    def _init_meta_optimizer(self, **kwargs):
        return self.framework.opt

    def fit(self, meta_dataset: MetaDatasetInc):

        phases = ["train", "test"]
        meta_tasks_l = meta_dataset.prepare_tasks(phases)

        self.cnt = 0
        self.framework.train()
        torch.set_grad_enabled(True)
        # run training
        best_ic, over_patience = -1e3, 8
        patience = over_patience
        best_checkpoint = copy.deepcopy(self.framework.state_dict())
        for epoch in tqdm(range(300), desc="epoch"):
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
                        patience = over_patience
                        best_checkpoint = copy.deepcopy(self.framework.state_dict())
            if patience <= 0:
                # R.save_objects(**{"model.pkl": self.tn})
                break
        self.fitted = True
        self.framework.load_state_dict(best_checkpoint)

    def run_epoch(self, phase, task_list, tqdm_show=False):
        pred_y_all, mse_all = [], 0
        indices = np.arange(len(task_list))
        if phase == "test":
            checkpoint = copy.deepcopy(self.framework.state_dict())
            checkpoint_opt = copy.deepcopy(self.framework.opt.state_dict())
            checkpoint_opt_meta = copy.deepcopy(self.opt.state_dict())
        elif phase == "train":
            np.random.shuffle(indices)
        self.phase = phase
        for i in tqdm(indices, desc=phase) if tqdm_show else indices:
            # torch.cuda.empty_cache()
            meta_input = task_list[i].get_meta_input()
            if not isinstance(meta_input['X_train'], torch.Tensor):
                meta_input = {
                    k: torch.tensor(v, device=self.framework.device, dtype=torch.float32) if 'idx' not in k else v
                    for k, v in meta_input.items()
                }
            pred = self.run_task(meta_input, phase)
            if phase != "train":
                test_idx = meta_input["test_idx"]
                pred_y_all.append(
                    pd.DataFrame(
                        {
                            "pred": pd.Series(pred, index=test_idx),
                            "label": pd.Series(meta_input["y_test"], index=test_idx),
                        }
                    )
                )
        if phase == "test":
            self.framework.load_state_dict(checkpoint)
            self.framework.opt.load_state_dict(checkpoint_opt)
            self.opt.load_state_dict(checkpoint_opt_meta)
        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
        if phase == "test":
            ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
            print(ic)
            return pred_y_all, ic
        return pred_y_all, None

    def run_task(self, meta_input, phase):
        """ Naive incremental learning """
        self.framework.opt.zero_grad()
        y_hat = self.framework(meta_input["X_train"].to(self.framework.device), None, transform=False)
        loss = self.framework.criterion(y_hat, meta_input["y_train"].to(self.framework.device))
        loss.backward()
        self.framework.opt.step()
        self.framework.opt.zero_grad()
        with torch.no_grad():
            pred = self.framework(meta_input["X_test"].to(self.framework.device), None, transform=False)
        return pred.detach().cpu().numpy()

    def inference(self, meta_dataset: MetaTaskDataset):
        meta_tasks_test = meta_dataset.prepare_tasks("test")
        self.framework.train()
        pred_y_all, ic = self.run_epoch("online", meta_tasks_test, tqdm_show=True)
        return pred_y_all, ic


class DoubleAdaptManager(MetaModelInc):
    def __init__(
        self,
        task_config,
        lr_model=0.001,
        lr_da=0.01,
        lr_ma=0.001,
        reg=0.5,
        adapt_x=True,
        adapt_y=True,
        first_order=True,
        factor_num=6,
        x_dim=360,
        alpha=360,
        num_head=8,
        temperature=10,
        pretrained_model=None,
        begin_valid_epoch=0,
    ):
        super(DoubleAdaptManager, self).__init__(task_config, x_dim=x_dim, lr_model=lr_model,
                                                 first_order=first_order, alpha=alpha,
                                                 pretrained_model=pretrained_model,
                                                 begin_valid_epoch=begin_valid_epoch)
        self.lr_da = lr_da
        self.lr_ma = lr_ma
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.sigma = 1 ** 2 * 2
        self.factor_num = factor_num
        self.lamda = 0.5
        self.num_head = num_head
        self.temperature = temperature
        self.begin_valid_epoch = begin_valid_epoch

    def _init_framework(self, task_config, x_dim=None, lr_model=0.001, need_permute=False, model=None,
                        num_head=8, temperature=10, factor_num=6, **kwargs):
        return DoubleAdapt(
            task_config, x_dim=x_dim, lr=lr_model, need_permute=need_permute, model=model,
            factor_num=factor_num, num_head=num_head, temperature=temperature,
        )

    def _init_meta_optimizer(self, lr_da=0.01, **kwargs):
        return optim.Adam(self.framework.meta_params, lr=lr_da)    # To optimize the data adapter
        # return optim.Adam([{'params': self.tn.teacher_y.parameters(), 'lr': self.lr_y},
        #                    {'params': self.tn.teacher_x.parameters()}], lr=self.lr)

    def run_task(self, meta_input, phase):

        self.framework.opt.zero_grad()
        self.opt.zero_grad()

        """ Incremental data adaptation & Model adaptation """
        X = meta_input["X_train"].to(self.framework.device)
        with higher.innerloop_ctx(
            self.framework.model,
            self.framework.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
            override={'lr': [self.lr_model]}
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                y_hat, _ = self.framework(X, model=fmodel, transform=self.adapt_x)
        y = meta_input["y_train"].to(self.framework.device)
        if self.adapt_y:
            raw_y = y
            y = self.framework.teacher_y(X, raw_y, inverse=False)
        loss2 = self.framework.criterion(y_hat, y)
        diffopt.step(loss2)

        """ Online inference """
        if phase != "train" and "X_extra" in meta_input and meta_input["X_extra"].shape[0] > 0:
            X_test = torch.cat([meta_input["X_extra"].to(self.framework.device), meta_input["X_test"].to(self.framework.device), ], 0, )
            y_test = torch.cat([meta_input["y_extra"].to(self.framework.device), meta_input["y_test"].to(self.framework.device), ], 0, )
        else:
            X_test = meta_input["X_test"].to(self.framework.device)
            y_test = meta_input["y_test"].to(self.framework.device)
        pred, X_test_adapted = self.framework(X_test, model=fmodel, transform=self.adapt_x)
        mask_y = meta_input.get("mask_y")
        if self.adapt_y:
            pred = self.framework.teacher_y(X_test, pred, inverse=True)
        if phase != "train":
            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            meta_end = test_begin + meta_input["meta_end"]
            output = pred[test_begin:].detach().cpu().numpy()
            X_test = X_test[:meta_end]
            X_test_adapted = X_test_adapted[:meta_end]
            if mask_y is not None:
                pred = pred[mask_y]
                meta_end = sum(mask_y[:meta_end])
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()

        """ Optimization of meta-learners """
        loss = self.framework.criterion(pred, y_test)
        if self.adapt_y:
            if not self.first_order:
                y = self.framework.teacher_y(X, raw_y, inverse=False)
            loss_y = F.mse_loss(y, raw_y)
            if self.first_order:
                """ Please refer to Appendix C in https://arxiv.org/pdf/2306.09862.pdf """
                with torch.no_grad():
                    pred2, _ = self.framework(X_test_adapted, model=None, transform=False, )
                    pred2 = self.framework.teacher_y(X_test, pred2, inverse=True).detach()
                    loss_old = self.framework.criterion(pred2.view_as(y_test), y_test)
                loss_y = (loss_old.item() - loss.item()) / self.sigma * loss_y + loss_y * self.reg
            else:
                loss_y = loss_y * self.reg
            loss_y.backward()
        loss.backward()
        if self.adapt_x or self.adapt_y:
            self.opt.step()
        self.framework.opt.step()
        return output


class MetaCoG(MetaModelInc):
    def __init__(self, task_config, **kwargs):
        super().__init__(task_config, **kwargs)
        self.gamma = 0.2

    def _init_framework(self, task_config, x_dim=None, lr_model=0.001, need_permute=False, model=None, **kwargs):
        return CoG(task_config, x_dim=x_dim, lr=lr_model, need_permute=need_permute, model=model)

    def _init_meta_optimizer(self, lr, **kwargs):
        return optim.Adam(self.framework.meta_params, lr=lr)

    def run_task(self, meta_input, phase):

        self.framework.opt.zero_grad()
        self.opt.zero_grad()
        X = meta_input["X_train"].to(self.framework.device)
        fmodel = higher.monkeypatch(self.framework.model, copy_initial_weights=True, track_higher_grads=not self.first_order, )
        fmask = higher.monkeypatch(self.framework.mask, copy_initial_weights=False, track_higher_grads=not self.first_order, )
        diffopt = higher.optim.get_diff_optim(
            self.opt, self.framework.mask.parameters(), fmodel=fmask, track_higher_grads=not self.first_order,
        )
        fmodel.update_params(list(self.framework.model.parameters()))
        with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
            y_hat = self.framework(X, fmodel=fmodel, fmask=fmask)
        y = meta_input["y_train"].to(self.framework.device)

        loss2 = self.framework.criterion(y_hat, y)
        if not self.first_order:
            loss2 += sum([torch.norm(p, 1) for p in fmask.fast_params]) * self.gamma
        diffopt.step(loss2)

        if phase == "test" and "X_extra" in meta_input and meta_input["X_extra"].shape[0] > 0:
            X_test = torch.cat([meta_input["X_extra"].to(self.framework.device), meta_input["X_test"].to(self.framework.device), ], 0, )
            y_test = torch.cat([meta_input["y_extra"].to(self.framework.device), meta_input["y_test"].to(self.framework.device), ], 0, )
        else:
            X_test = meta_input["X_test"].to(self.framework.device)
            y_test = meta_input["y_test"].to(self.framework.device)

        fmodel = higher.monkeypatch(self.framework.model, copy_initial_weights=True, track_higher_grads=not self.first_order, )
        pred = self.framework(X_test, fmodel=fmodel, fmask=fmask)
        if phase != "train":
            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            meta_end = test_begin + meta_input["meta_end"]
            output = pred[test_begin:].detach().cpu().numpy()
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()
        loss = self.framework.criterion(pred, y_test)
        loss.backward()
        self.opt.step()
        return output, None


class CMAML(MetaModelInc):
    def __init__(self, task_config, sample_num=5000, **kwargs):
        super().__init__(task_config, **kwargs)
        self.task_config = task_config
        self.batch_size = 1
        self.sample_num = sample_num
        self.gamma = 0.000
        self.lamda = 0.5
        self.buffer_size = self.sample_num * 2
        self.begin_valid_epoch = 10

    def fit(self, meta_dataset: MetaDatasetInc):

        phases = ["train", "test"]
        meta_tasks_train, meta_tasks_valid = meta_dataset.prepare_tasks(phases)

        self.cnt = 0
        self.framework.train()
        torch.set_grad_enabled(True)

        # run training
        best_ic, over_patience = -1e3, 8
        patience = over_patience
        best_checkpoint = copy.deepcopy(self.framework.state_dict())
        for epoch in tqdm(range(300), desc="epoch"):
            self.meta_train_epoch(meta_tasks_train)
            if epoch < self.begin_valid_epoch:
                continue
            pred_y, ic = self.meta_valid_epoch(meta_tasks_valid)
            if ic < best_ic:
                patience -= 1
            else:
                best_ic = ic
                print("best ic:", best_ic)
                patience = over_patience
                best_checkpoint = copy.deepcopy(self.framework.state_dict())
            if patience <= 0:
                break
        self.fitted = True
        self.framework.load_state_dict(best_checkpoint)

    def meta_train_epoch(self, task_list):
        context_indices = np.arange(len(task_list))
        np.random.shuffle(context_indices)
        i = 0
        while i < len(context_indices):
            # torch.cuda.empty_cache()
            loss = 0
            for j in context_indices[i : i + self.batch_size]:
                meta_input = task_list[j].get_meta_input()
                # indices = np.arange(len(meta_input['y_test']))
                # sample_idx = np.random.choice(indices, min(self.sample_num * 2, len(indices)), replace=False)
                X = meta_input["X_train"].to(self.framework.device)
                y = meta_input["y_train"].to(self.framework.device)
                # k = self.sample_num
                with higher.innerloop_ctx(
                    self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=not self.first_order,
                ) as (fmodel, diffopt):
                    with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                        y_hat = self.framework(X, model=fmodel)
                        diffopt.step(self.framework.criterion(y_hat, y))

                X = meta_input["X_test"].to(self.framework.device)
                y = meta_input["y_test"].to(self.framework.device)
                y_hat = self.framework(X, model=fmodel)
                loss += self.framework.criterion(y_hat, y)
            self.framework.opt.zero_grad()
            loss.backward()
            self.framework.opt.step()
            i += self.batch_size

    def meta_valid_epoch(self, task_list):
        pred_y_all = []
        for task in task_list:
            # torch.cuda.empty_cache()
            loss = 0
            meta_input = task.get_meta_input()
            self.framework.opt.zero_grad()
            X = meta_input["X_train"].to(self.framework.device)
            with higher.innerloop_ctx(
                self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=False,
            ) as (fmodel, diffopt):
                with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                    y_hat = self.framework(X, model=fmodel)
            y = meta_input["y_train"].to(self.framework.device)
            loss2 = self.framework.criterion(y_hat, y)
            diffopt.step(loss2)

            with torch.no_grad():
                X_test = meta_input["X_test"].to(self.framework.device)
                pred = self.framework(X_test, model=fmodel)
                output = pred.detach().cpu().numpy()

            test_idx = meta_input["test_idx"]
            pred_y_all.append(
                pd.DataFrame(
                    {
                        "pred": pd.Series(output, index=test_idx),
                        "label": pd.Series(meta_input["y_test"], index=test_idx),
                    }
                )
            )
        pred_y_all = pd.concat(pred_y_all)
        ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
        # R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})
        # nloss = -sum(losses) / len(losses)
        print(ic)
        return pred_y_all, ic

    def run_online_task(self, meta_input):

        self.framework.opt.zero_grad()
        begin_point = 0
        end_point = meta_input["meta_end"]
        X = meta_input["X_test"].to(self.framework.device)
        y = meta_input["y_test"][:end_point].to(self.framework.device)
        if "X_train" in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.framework.device), X])
            y = torch.cat([meta_input["y_train"].to(self.framework.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(
                self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=not self.first_order,
            ) as (fmodel, diffopt):
                self.fast_model = fmodel
                self.fast_opt = diffopt
            pred = self.framework(X, model=None)
            output = pred[begin_point:].detach().cpu().numpy()
            return output

        with torch.backends.cudnn.flags(enabled=True):
            with torch.no_grad():
                pred = self.framework(X, model=self.fast_model)
                output = pred[begin_point:].detach().cpu().numpy()

        X = X[:end_point]

        if len(self.buffer_x) == 0:
            self.buffer_x = X
            self.buffer_y = y
            return output

        with higher.innerloop_ctx(
            self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=not self.first_order,
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                diffopt.step(self.framework.criterion(self.framework(self.buffer_x, model=fmodel), self.buffer_y))
        self.fast_model = fmodel
        self.fast_opt = diffopt

        y_hat = self.framework(X, model=fmodel)
        loss2 = self.framework.criterion(y_hat, y)
        self.framework.opt.zero_grad()
        # smoothing_weight = (1 - torch.exp(-self.lamda * loss3.detach()))
        loss2.backward()
        self.framework.opt.step()

        self.buffer_x = X
        self.buffer_y = y
        return output

    def run_cmaml_task(self, meta_input):
        self.framework.opt.zero_grad()
        begin_point = 0
        end_point = meta_input["meta_end"]
        X = meta_input["X_test"].to(self.framework.device)
        y = meta_input["y_test"][:end_point].to(self.framework.device)
        if "X_train" in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.framework.device), X])
            y = torch.cat([meta_input["y_train"].to(self.framework.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(
                self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=not self.first_order,
            ) as (fmodel, diffopt):
                self.fast_model = fmodel
                self.fast_opt = diffopt
            pred = self.framework(X, model=None)
            output = pred[begin_point:].detach().cpu().numpy()
            return output

        with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
            pred = self.framework(X, model=self.fast_model)
            output = pred[begin_point:].detach().cpu().numpy()
            loss1 = self.framework.criterion(pred[:end_point], y)

        X = X[:end_point]

        with higher.innerloop_ctx(
            self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=not self.first_order,
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                diffopt.step(self.framework.criterion(self.framework(X, model=fmodel), y))

        self.fast_model = fmodel
        self.fast_opt = diffopt

        with torch.no_grad():
            y_hat = self.framework(X, model=fmodel)
            loss2 = self.framework.criterion(y_hat, y)

        print(loss1 - loss2)
        if loss1 - loss2 < self.gamma:
            self.framework.opt.zero_grad()
            smoothing_weight = 1 - torch.exp(-self.lamda * loss1.detach())
            (smoothing_weight * loss1).backward()
            # loss1.backward()
            self.framework.opt.step()
        return output

    def run_cmaml_pap_task(self, meta_input):

        self.framework.opt.zero_grad()
        begin_point = 0
        end_point = meta_input["meta_end"]
        X = meta_input["X_test"].to(self.framework.device)
        y = meta_input["y_test"][:end_point].to(self.framework.device)
        if "X_train" in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.framework.device), X])
            y = torch.cat([meta_input["y_train"].to(self.framework.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(
                self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=False,
            ) as (fmodel, diffopt):
                self.fast_model = fmodel
                self.fast_opt = diffopt
            pred = self.framework(X, model=None)
            output = pred[begin_point:].detach().cpu().numpy()
            X = X[:end_point]
            self.buffer_x = X.detach().cpu()
            self.buffer_y = y.detach().cpu()
            return output

        # with torch.no_grad():
        with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
            pred = self.framework(X, model=self.fast_model)
            output = pred[begin_point:].detach().cpu().numpy()
        loss1 = self.framework.criterion(pred[:end_point], y)

        X = X[:end_point]

        with higher.innerloop_ctx(
            self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=False,
        ) as (fmodel, diffopt):
            diffopt.step(self.framework.criterion(self.framework(X, model=fmodel), y), first_order=True)

        with torch.no_grad():
            y_hat = self.framework(X, model=fmodel)
            loss2 = self.framework.criterion(y_hat, y)

        # print(loss1 - loss2)
        if loss1 - loss2 < self.gamma:
            self.fast_opt.step(loss1)
            self.fast_model.update_params([p.detach().requires_grad_() for p in self.fast_model.fast_params])
            self.buffer_x = torch.cat([self.buffer_x[-self.buffer_size + len(X):], X.cpu()], 0)
            self.buffer_y = torch.cat([self.buffer_y[-self.buffer_size + len(X):], y.cpu()], 0)
        else:
            self.consolidate()
            with higher.innerloop_ctx(
                self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=False,
            ) as (fmodel, diffopt):
                diffopt.step(self.framework.criterion(self.framework(X, model=fmodel), y), first_order=True)
                self.fast_model = fmodel
                # self.fast_model.update_params([p.detach().requires_grad_() for p in self.fast_model.fast_params])
                self.fast_opt = diffopt

            self.buffer_x = X.detach().cpu()
            self.buffer_y = y.detach().cpu()
        return output

    def consolidate(self):
        sample_num = min(self.sample_num * 2, len(self.buffer_x))
        # indices = np.random.choice(np.arange(len(self.buffer_x)), sample_num, replace=False)
        # sample_x = self.buffer_x[indices].to(self.tn.device)
        # sample_y = self.buffer_y[indices].to(self.tn.device)
        sample_x = self.buffer_x[-sample_num:].to(self.framework.device)
        sample_y = self.buffer_y[-sample_num:].to(self.framework.device)
        with higher.innerloop_ctx(
            self.framework.model, self.framework.opt, copy_initial_weights=False, track_higher_grads=not self.first_order,
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                diffopt.step(
                    self.framework.criterion(
                        self.framework(sample_x[: len(sample_x) // 2], model=fmodel), sample_y[: len(sample_x) // 2],
                    )
                )
        loss3 = self.framework.criterion(
            self.framework(sample_x[len(sample_x) // 2:], model=fmodel), sample_y[len(sample_x) // 2:],
        )
        self.framework.opt.zero_grad()
        smoothing_weight = 1 - torch.exp(-self.lamda * loss3.detach())
        (loss3 * smoothing_weight).backward()
        # loss3.backward()
        self.framework.opt.step()

    def inference(self, meta_dataset: MetaTaskDataset):
        meta_tasks_test = meta_dataset.prepare_tasks("test")
        self.framework.train()
        self.buffer_x, self.buffer_y = [], []
        pred_y_all = []
        self.fast_model = None
        for task in meta_tasks_test:
            meta_input = task.get_meta_input()
            pred = self.run_cmaml_pap_task(meta_input)
            test_idx = meta_input["test_idx"]
            pred_y_all.append(
                pd.DataFrame(
                    {"pred": pd.Series(pred, index=test_idx), "label": pd.Series(meta_input["y_test"], index=test_idx),}
                )
            )
        pred_y_all = pd.concat(pred_y_all)
        return pred_y_all, None
