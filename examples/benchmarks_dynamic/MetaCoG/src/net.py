import collections
import torch
from qlib.model import Model

from qlib.utils import init_instance_by_config

from torch import nn
import higher
from examples.benchmarks_dynamic.incremental.src import optim

class ForecastModel(nn.Module):
    def __init__(self, task_config, dim=None, lr=0.001, need_permute=False, model=None):
        super().__init__()
        self.lr = lr
        # self.lr = task_config["model"]['kwargs']['lr']
        self.criterion = nn.MSELoss()
        if task_config['model']['class'] == 'LinearModel':
            self.model = nn.Linear(dim, 1)
            self.model.load_state_dict(collections.OrderedDict({'weight': torch.from_numpy(model.coef_).unsqueeze(0), 'bias': torch.tensor([model.intercept_])}))
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            self.device = None
        else:
            if model is None:
                model = init_instance_by_config(task_config["model"], accept_types=Model)
                self.opt = None
            else:
                self.opt = model.train_optimizer
            for child in model.__dict__.values():
                if isinstance(child, nn.Module):
                    self.model = child
                    break
            if hasattr(model, 'device'):
                self.device = model.device
            else:
                self.device = None
            self.need_permute = need_permute
            # self.opt = model.train_optimizer
        self.dim = dim
        self.mask = nn.ParameterList([nn.Parameter(torch.ones_like(param.data) * 3) for param in self.model.parameters()])
        if self.device is not None:
            self.to(self.device)
        self.meta_params = self.mask.parameters()
        if self.opt is None:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, X, fmodel=None, fmask=None, mask_y=None):
        new_params = []
        if fmodel is None:
            fmodel = self.model
        else:
            for i in range(len(fmask)):
                new_params.append(fmodel.fast_params[i] * torch.sigmoid(fmask[i]))
            fmodel.update_params(new_params)
        y_hat = fmodel(X.permute(0, 2, 1).reshape(len(X), -1) if self.need_permute else X.reshape(len(X), -1))
        y_hat = y_hat.view(-1)
        if mask_y:
            y_hat = y_hat[mask_y]
        return y_hat
