import copy
from tqdm import tqdm
import torch
from torch import optim
import higher

from pathlib import Path
import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent.parent))
from examples.benchmarks_dynamic.incremental.src import model
from net import ForecastModel

class MetaModelDS(model.MetaModelDS):
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
        self.tn = ForecastModel(
            self.task_config, dim=self.d_feat, need_permute=self.alpha == 360, model=pretrained_model, lr=0.001,
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
            track_higher_grads=not self.first_order
        )
        fmask = higher.monkeypatch(
            self.tn.mask,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order
        )
        diffopt = higher.optim.get_diff_optim(
            self.opt,
            self.tn.mask.parameters(),
            fmodel=fmask,
            track_higher_grads=not self.first_order
        )
        fmodel.update_params(list(self.tn.model.parameters()))
        with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_seq):
            y_hat = self.tn(X, fmodel=fmodel, fmask=fmask)
        y = meta_input['y_train'].to(self.tn.device)

        loss2 = self.tn.criterion(y_hat, y)
        if not self.first_order:
            loss2 += sum([torch.norm(p, 1) for p in fmask.fast_params]) * self.gamma
        diffopt.step(loss2)

        if phase == 'test' and 'X_extra' in meta_input and meta_input['X_extra'].shape[0] > 0:
            X_test = torch.cat([meta_input['X_extra'].to(self.tn.device), meta_input["X_test"].to(self.tn.device)], 0)
            y_test = torch.cat([meta_input['y_extra'].to(self.tn.device), meta_input["y_test"].to(self.tn.device)], 0)
        else:
            X_test = meta_input["X_test"].to(self.tn.device)
            y_test = meta_input["y_test"].to(self.tn.device)

        fmodel = higher.monkeypatch(
            self.tn.model,
            copy_initial_weights=True,
            track_higher_grads=not self.first_order
        )
        pred = self.tn(X_test, fmodel=fmodel, fmask=fmask, mask_y=meta_input.get('mask_y'))
        if phase != 'train':
            test_begin = len(meta_input['y_extra']) if 'y_extra' in meta_input else 0
            meta_end = test_begin + meta_input['meta_end']
            output = pred[test_begin:].detach().cpu().numpy()
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()
        loss = self.tn.criterion(pred, y_test)
        loss.backward()
        self.opt.step()
        return output, None