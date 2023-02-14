import collections
import torch
from qlib.model import Model

from qlib.utils import init_instance_by_config

from torch import nn
import higher
import optim

# import sys
# DIRNAME = Path(__file__).absolute().resolve().parent
# sys.path.append(str(DIRNAME.parent.parent.parent.parent))
# from examples.benchmarks_dynamic.incremental.src import optim

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
            else:
                device = torch.device('cuda')
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
        if self.device is not None:
            self.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, X, model=None, transform=False, mask_y=None):
        if model is None:
            model = self.model
        if X.dim() == 3:
            X = X.permute(0, 2, 1).reshape(len(X), -1) if self.need_permute else X.reshape(len(X), -1)
        y_hat = model(X)
        y_hat = y_hat.view(-1)
        # TODO: fix for MLP on Alpha 158
        if mask_y:
            y_hat = y_hat[mask_y]
        return y_hat
