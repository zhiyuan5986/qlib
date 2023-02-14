import copy
from collections import defaultdict
from pathlib import Path

import numpy as np

from tqdm import tqdm
import pandas as pd
import torch
import higher

import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
from net import ForecastModel


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
            lr=0.01,
            first_order=True,
            is_seq=False,
            d_feat=6,
            sample_num=5000,
            alpha=360, num_head=6, temperature=4,
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
            self.task_config, dim=self.d_feat, need_permute=self.alpha == 360, model=pretrained_model, lr=0.001,
        )
        self.batch_size = 1
        self.sample_num = sample_num
        self.gamma = 0.000
        self.lamda = 0.5
        self.buffer_size = self.sample_num * 2

    def fit(self, meta_tasks_train, meta_tasks_valid):
        self.cnt = 0
        self.tn.train()
        torch.set_grad_enabled(True)

        # run training
        best_ic, over_patience = -1e3, 8
        patience = over_patience
        best_checkpoint = copy.deepcopy(self.tn.state_dict())
        for epoch in tqdm(range(300), desc="epoch"):
            self.meta_train_epoch(meta_tasks_train)
            pred_y, ic = self.meta_valid_epoch(meta_tasks_valid)
            if ic < best_ic:
                patience -= 1
            else:
                best_ic = ic
                print('best ic:', best_ic)
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
            for j in context_indices[i: i + self.batch_size]:
                meta_input = task_list[j].get_meta_input()
                # indices = np.arange(len(meta_input['y_test']))
                # sample_idx = np.random.choice(indices, min(self.sample_num * 2, len(indices)), replace=False)
                X = meta_input['X_train'].to(self.tn.device)
                y = meta_input['y_train'].to(self.tn.device)
                # k = self.sample_num
                with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                          track_higher_grads=not self.first_order) as (fmodel, diffopt):
                    with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
                        y_hat = self.tn(X, model=fmodel)
                        diffopt.step(self.tn.criterion(y_hat, y))

                X = meta_input['X_test'].to(self.tn.device)
                y = meta_input['y_test'].to(self.tn.device)
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
            with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False, track_higher_grads=False) as (
            fmodel, diffopt):
                with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
                    y_hat = self.tn(X,model=fmodel)
            y = meta_input['y_train'].to(self.tn.device)
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
        ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
        # R.log_metrics(**{f"ic/{phase}": ic, "step": epoch})
        # nloss = -sum(losses) / len(losses)
        print(ic)
        return pred_y_all, ic


    def run_maml_task(self, meta_input):

        self.tn.opt.zero_grad()
        X = meta_input["X_train"].to(self.tn.device)
        with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False, track_higher_grads=not self.first_order) as (
        fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
                y_hat = self.tn(X,model=fmodel)
        y = meta_input['y_train'].to(self.tn.device)
        loss2 = self.tn.criterion(y_hat, y)
        diffopt.step(loss2)

        X_test = meta_input["X_test"].to(self.tn.device)
        y_test = meta_input["y_test"].to(self.tn.device)
        pred = self.tn(X_test,
                                # meta_input['test_mu'], meta_input['test_std'],
                                # meta_input['test_date_id'],
                                model=fmodel)
        meta_end = meta_input['meta_end']
        output = pred.detach().cpu().numpy()
        pred = pred[:meta_end]
        y_test = y_test[:meta_end]
        loss = self.tn.criterion(pred, y_test)
        loss.backward()
        self.tn.opt.step()
        # self.tn.model.load_state_dict(fmodel.state_dict())
        # self.tn.opt.state = override_state(self.tn.opt.param_groups, diffopt)
        return output

    def run_online_task(self, meta_input):

        self.tn.opt.zero_grad()
        begin_point = 0
        end_point = meta_input['meta_end']
        X = meta_input["X_test"].to(self.tn.device)
        y = meta_input["y_test"][:end_point].to(self.tn.device)
        if 'X_train' in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.tn.device), X])
            y = torch.cat([meta_input["y_train"].to(self.tn.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                      track_higher_grads=not self.first_order) as (
                    fmodel, diffopt):
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

        with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                  track_higher_grads=not self.first_order) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
                diffopt.step(self.tn.criterion(self.tn(self.buffer_x, model=fmodel),
                                               self.buffer_y))
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
        end_point = meta_input['meta_end']
        X = meta_input["X_test"].to(self.tn.device)
        y = meta_input["y_test"][:end_point].to(self.tn.device)
        if 'X_train' in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.tn.device), X])
            y = torch.cat([meta_input["y_train"].to(self.tn.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                      track_higher_grads=not self.first_order) as (
                    fmodel, diffopt):
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

        with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                  track_higher_grads=not self.first_order) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
                diffopt.step(self.tn.criterion(self.tn(X, model=fmodel), y))

        self.fast_model = fmodel
        self.fast_opt = diffopt

        with torch.no_grad():
            y_hat = self.tn(X, model=fmodel)
            loss2 = self.tn.criterion(y_hat, y)

        print(loss1 - loss2)
        if loss1 - loss2 < self.gamma:
            self.tn.opt.zero_grad()
            smoothing_weight = (1 - torch.exp(-self.lamda * loss1.detach()))
            (smoothing_weight * loss1).backward()
            # loss1.backward()
            self.tn.opt.step()
        return output

    def run_cmaml_pap_task(self, meta_input):

        self.tn.opt.zero_grad()
        begin_point = 0
        end_point = meta_input['meta_end']
        X = meta_input["X_test"].to(self.tn.device)
        y = meta_input["y_test"][:end_point].to(self.tn.device)
        if 'X_train' in meta_input:
            begin_point = len(meta_input["X_train"])
            end_point += begin_point
            X = torch.cat([meta_input["X_train"].to(self.tn.device), X])
            y = torch.cat([meta_input["y_train"].to(self.tn.device), y])

        if self.fast_model is None:
            with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                      track_higher_grads=False) as (
                    fmodel, diffopt):
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

        with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                  track_higher_grads=False) as (fmodel, diffopt):
            diffopt.step(self.tn.criterion(self.tn(X, model=fmodel), y), first_order=True)

        with torch.no_grad():
            y_hat = self.tn(X, model=fmodel)
            loss2 = self.tn.criterion(y_hat, y)

        # print(loss1 - loss2)
        if loss1 - loss2 < self.gamma:
            self.fast_opt.step(loss1)
            self.fast_model.update_params([p.detach().requires_grad_() for p in self.fast_model.fast_params])
            self.buffer_x = torch.cat([self.buffer_x, X.cpu()], 0)[-self.buffer_size:]
            self.buffer_y = torch.cat([self.buffer_y, y.cpu()], 0)[-self.buffer_size:]
        else:
        # if True:
            # sample_idx = np.random.choice(np.arange(len(self.buffer_x)), min(self.sample_num * 2, len(self.buffer_x)),
            #                               replace=False)
            # sample_idx = torch.tensor(sample_idx)
            # sample_x, sample_y = self.buffer_x[sample_idx].to(self.tn.device), self.buffer_y[sample_idx].to(self.tn.device)
            self.consolidate()
            with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                      track_higher_grads=False) as (fmodel, diffopt):
                diffopt.step(self.tn.criterion(self.tn(X, model=fmodel), y), first_order=True)
                self.fast_model = fmodel
                # self.fast_model.update_params([p.detach().requires_grad_() for p in self.fast_model.fast_params])
                self.fast_opt = diffopt

            self.buffer_x = X.detach().cpu()
            self.buffer_y = y.detach().cpu()
        return output

    def consolidate(self):
        sample_num = min(self.sample_num * 2, len(self.buffer_x))
        indices = np.random.choice(np.arange(len(self.buffer_x)), sample_num, replace=False)
        sample_x = self.buffer_x[indices].to(self.tn.device)
        sample_y = self.buffer_y[indices].to(self.tn.device)
        with higher.innerloop_ctx(self.tn.model, self.tn.opt, copy_initial_weights=False,
                                  track_higher_grads=not self.first_order) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
                diffopt.step(self.tn.criterion(self.tn(sample_x[:len(sample_x) // 2], model=fmodel),
                                               sample_y[:len(sample_x) // 2]))
        loss3 = self.tn.criterion(self.tn(sample_x[len(sample_x) // 2:], model=fmodel), sample_y[len(sample_x) // 2:])
        self.tn.opt.zero_grad()
        smoothing_weight = (1 - torch.exp(-self.lamda * loss3.detach()))
        (loss3 * smoothing_weight).backward()
        # loss3.backward()
        self.tn.opt.step()

    def infer(self, meta_tasks_test):
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

