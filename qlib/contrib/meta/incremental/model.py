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

from .utils import override_state
from .dataset import MetaDatasetInc
from .net import TeacherNet, ForecastModel, CoG


class MetaModelInc(MetaTaskModel):
    def __init__(
        self,
        task_config,
        lr=0.01,
        lr_y=0.01,
        lr_model=0.001,
        lr_ma=0.001,
        reg=0.2,
        adapt_x=True,
        adapt_y=True,
        first_order=True,
        is_rnn=False,
        d_feat=6,
        L=60,
        alpha=360,
        num_head=6,
        temperature=4,
        pretrained_model=None,
        naive=False,
        begin_valid_epoch=0,
    ):
        self.task_config = task_config
        self.lr = lr
        self.lr_y = lr_y
        self.lr_model = lr_model
        self.lr_ma = lr_ma
        self.first_order = first_order
        self.naive = naive
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.sigma = 1 ** 2 * 2
        self.is_rnn = is_rnn
        self.d_feat = d_feat
        self.alpha = alpha
        self.lamda = 0.5
        self.num_head = num_head
        self.temperature = temperature
        self.tn = TeacherNet(
            task_config,
            num_head=num_head,
            temperature=temperature,
            dim=self.d_feat,
            need_permute=self.alpha == 360,
            model=pretrained_model,
            seq_len=L,
            lr=lr_ma,
        )
        self.opt = optim.Adam(self.tn.meta_params, lr=self.lr)
        # self.opt = optim.Adam([{'params': self.tn.teacher_y.parameters(), 'lr': self.lr_y},
        #                        {'params': self.tn.teacher_x.parameters()}], lr=self.lr)
        self.begin_valid_epoch = begin_valid_epoch

    def fit(self, meta_dataset: MetaDatasetInc):

        phases = ["train", "test"]
        meta_tasks_l = meta_dataset.prepare_tasks(phases)

        self.cnt = 0
        self.tn.train()
        torch.set_grad_enabled(True)
        # run training
        best_ic, over_patience = -1e3, 8
        patience = over_patience
        best_checkpoint = copy.deepcopy(self.tn.state_dict())
        for epoch in tqdm(range(300), desc="epoch"):
            # self.opt = optim.Adam(self.tn.meta_params, lr=self.lr)
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
                        best_checkpoint = copy.deepcopy(self.tn.state_dict())
                    # pre_ic = ic
            if patience <= 0:
                # R.save_objects(**{"model.pkl": self.tn})
                break
        self.fitted = True
        self.tn.load_state_dict(best_checkpoint)

    def run_epoch(self, phase, task_list):
        pred_y_all, mse_all = [], 0
        indices = np.arange(len(task_list))
        if phase == "test":
            checkpoint = copy.deepcopy(self.tn.state_dict())
            checkpoint_opt = copy.deepcopy(self.tn.opt.state_dict())
            checkpoint_opt_meta = copy.deepcopy(self.opt.state_dict())
        elif phase == "train":
            np.random.shuffle(indices)
        self.phase = phase
        for i in indices:
            # torch.cuda.empty_cache()
            meta_input = task_list[i].get_meta_input()
            run_task_func = self.run_task_naive if self.naive else self.run_task
            pred, mse = run_task_func(meta_input, phase)
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
                # if phase == 'test':
                #     mse_all += mse
        if phase == "test":
            self.tn.load_state_dict(checkpoint)
            self.tn.opt.load_state_dict(checkpoint_opt)
            self.opt.load_state_dict(checkpoint_opt_meta)
        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
        if phase == "test":
            ic = (
                pred_y_all.groupby("datetime")
                .apply(lambda df: df["pred"].corr(df["label"], method="pearson"))
                .mean()
            )
            print(ic)
            return pred_y_all, ic
            # print(-mse_all)
            return pred_y_all, -mse_all
        return pred_y_all, None

    def run_task(self, meta_input, phase):

        self.tn.opt.zero_grad()
        self.opt.zero_grad()
        X = meta_input["X_train"].to(self.tn.device)
        with higher.innerloop_ctx(
            self.tn.model,
            self.tn.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
            # override={'lr': [self.lr_model]}
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(
                enabled=self.first_order or not self.is_rnn
            ):
                y_hat = self.tn(
                    X,
                    # meta_input['train_mu'], meta_input['train_std'],
                    # meta_input['train_date_id'],
                    model=fmodel,
                    transform=self.adapt_x,
                )
        y = meta_input["y_train"].to(self.tn.device)
        if self.adapt_y:
            raw_y = y
            y = self.tn.teacher_y(X, raw_y, inverse=False)
        loss2 = self.tn.criterion(y_hat, y)
        diffopt.step(loss2)

        if (
            phase != "train"
            and "X_extra" in meta_input
            and meta_input["X_extra"].shape[0] > 0
        ):
            X_test = torch.cat(
                [
                    meta_input["X_extra"].to(self.tn.device),
                    meta_input["X_test"].to(self.tn.device),
                ],
                0,
            )
            y_test = torch.cat(
                [
                    meta_input["y_extra"].to(self.tn.device),
                    meta_input["y_test"].to(self.tn.device),
                ],
                0,
            )
        else:
            X_test = meta_input["X_test"].to(self.tn.device)
            y_test = meta_input["y_test"].to(self.tn.device)
        pred = self.tn(
            X_test,
            # meta_input['test_mu'], meta_input['test_std'],
            # meta_input['test_date_id'],
            model=fmodel,
            transform=self.adapt_x,
        )
        mask_y = meta_input.get("mask_y")
        if self.adapt_y:
            pred = self.tn.teacher_y(X_test, pred, inverse=True)
        if phase != "train":
            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            meta_end = test_begin + meta_input["meta_end"]
            output = pred[test_begin:].detach().cpu().numpy()
            X_test = X_test[:meta_end]
            if mask_y is not None:
                pred = pred[mask_y]
                meta_end = sum(mask_y[:meta_end])
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()
        loss = self.tn.criterion(pred, y_test)
        if self.adapt_y:
            if not self.first_order:
                y = self.tn.teacher_y(X, raw_y, inverse=False)
            loss_y = F.mse_loss(y, raw_y)
            if self.first_order:
                with torch.no_grad():
                    pred2 = self.tn(
                        X_test if phase == "test" else X_test,
                        model=None,
                        transform=self.adapt_x,
                    )
                    pred2 = self.tn.teacher_y(
                        X_test, pred2.detach(), inverse=True
                    ).detach()
                    loss_old = self.tn.criterion(pred2.view_as(y_test), y_test)
                loss_y = (
                    loss_old.item() - loss.item()
                ) / self.sigma * loss_y + loss_y * self.reg
            else:
                loss_y = loss_y * self.reg
            loss_y.backward()
        # smoothing_weight = (1 - torch.exp(-self.lamda * loss.detach()))
        # (loss * smoothing_weight).backward()
        loss.backward()
        if self.adapt_x or self.adapt_y:
            self.opt.step()
        self.tn.opt.step()
        # self.tn.model.load_state_dict(fmodel.state_dict())
        # self.tn.opt.state = override_state(self.tn.opt.param_groups, diffopt)
        return output, None

    def run_task_naive(self, meta_input, phase):
        # if phase == 'train':
        self.tn.opt.zero_grad()
        y_hat = self.tn(meta_input["X_train"].to(self.tn.device), None, transform=False)
        loss2 = self.tn.criterion(y_hat, meta_input["y_train"].to(self.tn.device))
        loss2.backward()
        # torch.nn.utils.clip_grad_value_(self.tn.model.parameters(), 3.0)
        self.tn.opt.step()
        self.tn.opt.zero_grad()
        # else:
        pred = self.tn(meta_input["X_test"].to(self.tn.device), None, transform=False)
        with torch.no_grad():
            mse = self.tn.criterion(
                pred, meta_input["y_test"].to(self.tn.device)
            ).item()
        return pred.detach().cpu().numpy(), mse

    def inference(self, meta_dataset: MetaTaskDataset):
        meta_tasks_test = meta_dataset.prepare_tasks("test")
        self.tn.train()
        pred_y_all, ic = self.run_epoch("online", meta_tasks_test)
        return pred_y_all, ic


class MetaCoG(MetaModelInc):
    def __init__(
        self,
        task_config,
        lr=0.001,
        first_order=True,
        is_rnn=False,
        d_feat=6,
        alpha=360,
        pretrained_model=None,
    ):
        self.task_config = task_config
        self.lr = lr
        self.first_order = first_order
        self.is_seq = is_rnn
        self.d_feat = d_feat
        self.alpha = alpha
        self.tn = CoG(
            self.task_config,
            dim=self.d_feat,
            need_permute=self.alpha == 360,
            model=pretrained_model,
            lr=0.001,
        )
        self.opt = optim.Adam(self.tn.meta_params, lr=self.lr)
        self.naive = False
        self.gamma = 0.2
        self.begin_valid_epoch = 0

    def run_task(self, meta_input, phase):

        self.tn.opt.zero_grad()
        self.opt.zero_grad()
        X = meta_input["X_train"].to(self.tn.device)
        fmodel = higher.monkeypatch(
            self.tn.model,
            copy_initial_weights=True,
            track_higher_grads=not self.first_order,
        )
        fmask = higher.monkeypatch(
            self.tn.mask,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
        )
        diffopt = higher.optim.get_diff_optim(
            self.opt,
            self.tn.mask.parameters(),
            fmodel=fmask,
            track_higher_grads=not self.first_order,
        )
        fmodel.update_params(list(self.tn.model.parameters()))
        with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
            y_hat = self.tn(X, fmodel=fmodel, fmask=fmask)
        y = meta_input["y_train"].to(self.tn.device)

        loss2 = self.tn.criterion(y_hat, y)
        if not self.first_order:
            loss2 += sum([torch.norm(p, 1) for p in fmask.fast_params]) * self.gamma
        diffopt.step(loss2)

        if (
            phase == "test"
            and "X_extra" in meta_input
            and meta_input["X_extra"].shape[0] > 0
        ):
            X_test = torch.cat(
                [
                    meta_input["X_extra"].to(self.tn.device),
                    meta_input["X_test"].to(self.tn.device),
                ],
                0,
            )
            y_test = torch.cat(
                [
                    meta_input["y_extra"].to(self.tn.device),
                    meta_input["y_test"].to(self.tn.device),
                ],
                0,
            )
        else:
            X_test = meta_input["X_test"].to(self.tn.device)
            y_test = meta_input["y_test"].to(self.tn.device)

        fmodel = higher.monkeypatch(
            self.tn.model,
            copy_initial_weights=True,
            track_higher_grads=not self.first_order,
        )
        pred = self.tn(
            X_test, fmodel=fmodel, fmask=fmask, mask_y=meta_input.get("mask_y")
        )
        if phase != "train":
            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            meta_end = test_begin + meta_input["meta_end"]
            output = pred[test_begin:].detach().cpu().numpy()
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()
        loss = self.tn.criterion(pred, y_test)
        loss.backward()
        self.opt.step()
        return output, None


class CMAML:
    def __init__(
        self,
        task_config,
        lr=0.01,
        first_order=True,
        is_seq=False,
        d_feat=6,
        sample_num=5000,
        alpha=360,
        num_head=6,
        temperature=4,
        pretrained_model=None,
    ):
        self.task_config = task_config
        self.lr = lr
        self.first_order = first_order
        self.is_seq = is_seq
        self.d_feat = d_feat
        self.alpha = alpha
        self.num_head = num_head
        self.temperature = temperature
        self.tn = ForecastModel(
            self.task_config,
            dim=self.d_feat,
            need_permute=self.alpha == 360,
            model=pretrained_model,
            lr=0.001,
        )
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
        self.tn.train()
        torch.set_grad_enabled(True)

        # run training
        best_ic, over_patience = -1e3, 8
        patience = over_patience
        best_checkpoint = copy.deepcopy(self.tn.state_dict())
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
                best_checkpoint = copy.deepcopy(self.tn.state_dict())
            if patience <= 0:
                break
        self.fitted = True
        self.tn.load_state_dict(best_checkpoint)

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
                X = meta_input["X_train"].to(self.tn.device)
                y = meta_input["y_train"].to(self.tn.device)
                # k = self.sample_num
                with higher.innerloop_ctx(
                    self.tn.model,
                    self.tn.opt,
                    copy_initial_weights=False,
                    track_higher_grads=not self.first_order,
                ) as (fmodel, diffopt):
                    with torch.backends.cudnn.flags(
                        enabled=self.first_order or not self.is_seq
                    ):
                        y_hat = self.tn(X, model=fmodel)
                        diffopt.step(self.tn.criterion(y_hat, y))

                X = meta_input["X_test"].to(self.tn.device)
                y = meta_input["y_test"].to(self.tn.device)
                y_hat = self.tn(X, model=fmodel)
                loss += self.tn.criterion(y_hat, y)
            self.tn.opt.zero_grad()
            loss.backward()
            self.tn.opt.step()
            i += self.batch_size

    def meta_valid_epoch(self, task_list):
        pred_y_all = []
        for task in task_list:
            # torch.cuda.empty_cache()
            loss = 0
            meta_input = task.get_meta_input()
            self.tn.opt.zero_grad()
            X = meta_input["X_train"].to(self.tn.device)
            with higher.innerloop_ctx(
                self.tn.model,
                self.tn.opt,
                copy_initial_weights=False,
                track_higher_grads=False,
            ) as (fmodel, diffopt):
                with torch.backends.cudnn.flags(
                    enabled=self.first_order or not self.is_seq
                ):
                    y_hat = self.tn(X, model=fmodel)
            y = meta_input["y_train"].to(self.tn.device)
            loss2 = self.tn.criterion(y_hat, y)
            diffopt.step(loss2)

            with torch.no_grad():
                X_test = meta_input["X_test"].to(self.tn.device)
                pred = self.tn(X_test, model=fmodel)
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
        ic = (
            pred_y_all.groupby("datetime")
            .apply(lambda df: df["pred"].corr(df["label"], method="pearson"))
            .mean()
        )
        # R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})
        # nloss = -sum(losses) / len(losses)
        print(ic)
        return pred_y_all, ic

    def run_online_task(self, meta_input):

        self.tn.opt.zero_grad()
        begin_point = 0
        end_point = meta_input["meta_end"]
        X = meta_input["X_test"].to(self.tn.device)
        y = meta_input["y_test"][:end_point].to(self.tn.device)
        if "X_train" in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.tn.device), X])
            y = torch.cat([meta_input["y_train"].to(self.tn.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(
                self.tn.model,
                self.tn.opt,
                copy_initial_weights=False,
                track_higher_grads=not self.first_order,
            ) as (fmodel, diffopt):
                self.fast_model = fmodel
                self.fast_opt = diffopt
            pred = self.tn(X, model=None)
            output = pred[begin_point:].detach().cpu().numpy()
            return output

        with torch.backends.cudnn.flags(enabled=True):
            with torch.no_grad():
                pred = self.tn(X, model=self.fast_model)
                output = pred[begin_point:].detach().cpu().numpy()

        X = X[:end_point]

        if len(self.buffer_x) == 0:
            self.buffer_x = X
            self.buffer_y = y
            return output

        with higher.innerloop_ctx(
            self.tn.model,
            self.tn.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(
                enabled=self.first_order or not self.is_seq
            ):
                diffopt.step(
                    self.tn.criterion(
                        self.tn(self.buffer_x, model=fmodel), self.buffer_y
                    )
                )
        self.fast_model = fmodel
        self.fast_opt = diffopt

        y_hat = self.tn(X, model=fmodel)
        loss2 = self.tn.criterion(y_hat, y)
        self.tn.opt.zero_grad()
        # smoothing_weight = (1 - torch.exp(-self.lamda * loss3.detach()))
        loss2.backward()
        self.tn.opt.step()

        self.buffer_x = X
        self.buffer_y = y
        return output

    def run_cmaml_task(self, meta_input):
        self.tn.opt.zero_grad()
        begin_point = 0
        end_point = meta_input["meta_end"]
        X = meta_input["X_test"].to(self.tn.device)
        y = meta_input["y_test"][:end_point].to(self.tn.device)
        if "X_train" in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.tn.device), X])
            y = torch.cat([meta_input["y_train"].to(self.tn.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(
                self.tn.model,
                self.tn.opt,
                copy_initial_weights=False,
                track_higher_grads=not self.first_order,
            ) as (fmodel, diffopt):
                self.fast_model = fmodel
                self.fast_opt = diffopt
            pred = self.tn(X, model=None)
            output = pred[begin_point:].detach().cpu().numpy()
            return output

        with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
            pred = self.tn(X, model=self.fast_model)
            output = pred[begin_point:].detach().cpu().numpy()
            loss1 = self.tn.criterion(pred[:end_point], y)

        X = X[:end_point]

        with higher.innerloop_ctx(
            self.tn.model,
            self.tn.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(
                enabled=self.first_order or not self.is_seq
            ):
                diffopt.step(self.tn.criterion(self.tn(X, model=fmodel), y))

        self.fast_model = fmodel
        self.fast_opt = diffopt

        with torch.no_grad():
            y_hat = self.tn(X, model=fmodel)
            loss2 = self.tn.criterion(y_hat, y)

        print(loss1 - loss2)
        if loss1 - loss2 < self.gamma:
            self.tn.opt.zero_grad()
            smoothing_weight = 1 - torch.exp(-self.lamda * loss1.detach())
            (smoothing_weight * loss1).backward()
            # loss1.backward()
            self.tn.opt.step()
        return output

    def run_cmaml_pap_task(self, meta_input):

        self.tn.opt.zero_grad()
        begin_point = 0
        end_point = meta_input["meta_end"]
        X = meta_input["X_test"].to(self.tn.device)
        y = meta_input["y_test"][:end_point].to(self.tn.device)
        if "X_train" in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.tn.device), X])
            y = torch.cat([meta_input["y_train"].to(self.tn.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(
                self.tn.model,
                self.tn.opt,
                copy_initial_weights=False,
                track_higher_grads=False,
            ) as (fmodel, diffopt):
                self.fast_model = fmodel
                self.fast_opt = diffopt
            pred = self.tn(X, model=None)
            output = pred[begin_point:].detach().cpu().numpy()
            X = X[:end_point]
            self.buffer_x = X.detach().cpu()
            self.buffer_y = y.detach().cpu()
            return output

        # with torch.no_grad():
        with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
            pred = self.tn(X, model=self.fast_model)
            output = pred[begin_point:].detach().cpu().numpy()
        loss1 = self.tn.criterion(pred[:end_point], y)

        X = X[:end_point]

        with higher.innerloop_ctx(
            self.tn.model,
            self.tn.opt,
            copy_initial_weights=False,
            track_higher_grads=False,
        ) as (fmodel, diffopt):
            diffopt.step(
                self.tn.criterion(self.tn(X, model=fmodel), y), first_order=True
            )

        with torch.no_grad():
            y_hat = self.tn(X, model=fmodel)
            loss2 = self.tn.criterion(y_hat, y)

        # print(loss1 - loss2)
        if loss1 - loss2 < self.gamma:
            self.fast_opt.step(loss1)
            self.fast_model.update_params(
                [p.detach().requires_grad_() for p in self.fast_model.fast_params]
            )
            self.buffer_x = torch.cat([self.buffer_x, X.cpu()], 0)[-self.buffer_size :]
            self.buffer_y = torch.cat([self.buffer_y, y.cpu()], 0)[-self.buffer_size :]
        else:
            # if True:
            # sample_idx = np.random.choice(np.arange(len(self.buffer_x)), min(self.sample_num * 2, len(self.buffer_x)),
            #                               replace=False)
            # sample_idx = torch.tensor(sample_idx)
            # sample_x, sample_y = self.buffer_x[sample_idx].to(self.tn.device), self.buffer_y[sample_idx].to(self.tn.device)
            self.consolidate()
            with higher.innerloop_ctx(
                self.tn.model,
                self.tn.opt,
                copy_initial_weights=False,
                track_higher_grads=False,
            ) as (fmodel, diffopt):
                diffopt.step(
                    self.tn.criterion(self.tn(X, model=fmodel), y), first_order=True
                )
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
        sample_x = self.buffer_x[-sample_num:].to(self.tn.device)
        sample_y = self.buffer_y[-sample_num:].to(self.tn.device)
        with higher.innerloop_ctx(
            self.tn.model,
            self.tn.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(
                enabled=self.first_order or not self.is_seq
            ):
                diffopt.step(
                    self.tn.criterion(
                        self.tn(sample_x[: len(sample_x) // 2], model=fmodel),
                        sample_y[: len(sample_x) // 2],
                    )
                )
        loss3 = self.tn.criterion(
            self.tn(sample_x[len(sample_x) // 2 :], model=fmodel),
            sample_y[len(sample_x) // 2 :],
        )
        self.tn.opt.zero_grad()
        smoothing_weight = 1 - torch.exp(-self.lamda * loss3.detach())
        (loss3 * smoothing_weight).backward()
        # loss3.backward()
        self.tn.opt.step()

    def inference(self, meta_dataset: MetaTaskDataset):
        meta_tasks_test = meta_dataset.prepare_tasks("test")
        self.tn.train()
        self.buffer_x, self.buffer_y = [], []
        pred_y_all = []
        self.fast_model = None
        for task in meta_tasks_test:
            meta_input = task.get_meta_input()
            pred = self.run_cmaml_pap_task(meta_input)
            test_idx = meta_input["test_idx"]
            pred_y_all.append(
                pd.DataFrame(
                    {
                        "pred": pd.Series(pred, index=test_idx),
                        "label": pd.Series(meta_input["y_test"], index=test_idx),
                    }
                )
            )
        pred_y_all = pd.concat(pred_y_all)
        # ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="spearman")).mean()
        # R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})
        # nloss = -sum(losses) / len(losses)
        # print(ic)
        return pred_y_all, None
