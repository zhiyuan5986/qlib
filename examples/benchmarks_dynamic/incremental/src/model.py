import copy
from collections import defaultdict

import numpy as np

from tqdm import tqdm
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
import higher

from .net import TeacherNet


def override_state(groups, new_opt):
    saved_groups = new_opt.param_groups
    id_map = {old_id: p for old_id, p in zip(range(len(saved_groups[0]['params'])), groups[0]['params'])}
    state = defaultdict(dict)
    for k, v in new_opt.state[0].items():
        if k in id_map:
            param = id_map[k]
            for _k, _v in v.items():
                state[param][_k] = _v.detach() if isinstance(_v, torch.Tensor) else _v
        else:
            state[k] = v
    return state


class MetaModelDS:
    def __init__(
            self,
            task_config,
            lr=0.01, lr_y=0.01, lr_model=0.001, lr_ma=0.001, reg=0.2,
            adapt_x=True, adapt_y=True, first_order=True, is_rnn=False,
            d_feat=6, L=60, alpha=360,
            num_head=6, temperature=4,
            pretrained_model=None,
            naive=False,
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
            task_config, num_head=num_head, temperature=temperature, dim=self.d_feat,
            need_permute=self.alpha == 360,
            model=pretrained_model, seq_len=L, lr=lr_ma,
        )
        self.opt = optim.Adam(self.tn.meta_params, lr=self.lr)
        # self.opt = optim.Adam([{'params': self.tn.teacher_y.parameters(), 'lr': self.lr_y},
        #                        {'params': self.tn.teacher_x.parameters()}], lr=self.lr)
        self.begin_valid_epoch = 10

    def fit(self, meta_tasks_train, meta_tasks_valid):

        phases = ["train", "test"]
        meta_tasks_l = [meta_tasks_train, meta_tasks_valid]

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
                if phase == 'test':
                    if epoch < self.begin_valid_epoch:
                        continue
                pred_y, ic = self.run_epoch(phase, task_list)
                if phase == 'test':
                    if ic < best_ic:
                        patience -= 1
                    else:
                        best_ic = ic
                        print('best ic:', best_ic)
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
        if phase == 'test':
            checkpoint = copy.deepcopy(self.tn.state_dict())
            checkpoint_opt = copy.deepcopy(self.tn.opt.state_dict())
            checkpoint_opt_meta = copy.deepcopy(self.opt.state_dict())
        elif phase == 'train':
            np.random.shuffle(indices)
        self.phase = phase
        for i in indices:
            # torch.cuda.empty_cache()
            meta_input = task_list[i].get_meta_input()
            run_task_func = self.run_task_naive if self.naive else self.run_task
            pred, mse = run_task_func(meta_input, phase)
            if phase != 'train':
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
        if phase == 'test':
            self.tn.load_state_dict(checkpoint)
            self.tn.opt.load_state_dict(checkpoint_opt)
            self.opt.load_state_dict(checkpoint_opt_meta)
        if phase != 'train':
            pred_y_all = pd.concat(pred_y_all)
        if phase == 'test':
            ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
            print(ic)
            return pred_y_all, ic
            # print(-mse_all)
            return pred_y_all, -mse_all
        return pred_y_all, None

    def run_task(self, meta_input, phase):

        self.tn.opt.zero_grad()
        self.opt.zero_grad()
        X = meta_input["X_train"].to(self.tn.device)
        with higher.innerloop_ctx(self.tn.model, self.tn.opt,
                                  copy_initial_weights=False,
                                  track_higher_grads=not self.first_order,
                                  # override={'lr': [self.lr_model]}
                                  ) as (
        fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_rnn):
                y_hat = self.tn(X,
                                # meta_input['train_mu'], meta_input['train_std'],
                                # meta_input['train_date_id'],
                                model=fmodel, transform=self.adapt_x)
        y = meta_input['y_train'].to(self.tn.device)
        if self.adapt_y:
            raw_y = y
            y = self.tn.teacher_y(X, raw_y, inverse=False)
        loss2 = self.tn.criterion(y_hat, y)
        diffopt.step(loss2)

        if phase != 'train' and 'X_extra' in meta_input and meta_input['X_extra'].shape[0] > 0:
            X_test = torch.cat([meta_input['X_extra'].to(self.tn.device), meta_input["X_test"].to(self.tn.device)], 0)
            y_test = torch.cat([meta_input['y_extra'].to(self.tn.device), meta_input["y_test"].to(self.tn.device)], 0)
        else:
            X_test = meta_input["X_test"].to(self.tn.device)
            y_test = meta_input["y_test"].to(self.tn.device)
        pred = self.tn(X_test,
                                # meta_input['test_mu'], meta_input['test_std'],
                                # meta_input['test_date_id'],
                                model=fmodel, transform=self.adapt_x)
        mask_y = meta_input.get('mask_y')
        if self.adapt_y:
            pred = self.tn.teacher_y(X_test, pred, inverse=True)
        if phase != 'train':
            test_begin = len(meta_input['y_extra']) if 'y_extra' in meta_input else 0
            meta_end = test_begin + meta_input['meta_end']
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
                    pred2 = self.tn(X_test if phase == 'test' else X_test, model=None, transform=self.adapt_x)
                    pred2 = self.tn.teacher_y(X_test, pred2.detach(), inverse=True).detach()
                    loss_old = self.tn.criterion(pred2.view_as(y_test), y_test)
                loss_y = (loss_old.item() - loss.item()) / self.sigma * loss_y + loss_y * self.reg
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
            mse = self.tn.criterion(pred, meta_input['y_test'].to(self.tn.device)).item()
        return pred.detach().cpu().numpy(), mse

    def infer(self, meta_tasks_test):
        self.tn.train()
        pred_y_all, ic = self.run_epoch('online', meta_tasks_test)
        return pred_y_all, ic

