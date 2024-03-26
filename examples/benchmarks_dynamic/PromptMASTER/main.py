import pickle
from typing import Optional, List, Tuple, Union, Text
# from base_model import SequenceModel
import yaml
import tqdm
import fire
import sys
from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import qlib
from qlib.utils import init_instance_by_config
from qlib.data.dataset import Dataset, DataHandlerLP
from qlib.contrib.data.dataset import TSDataSampler
from qlib.workflow.record_temp import SigAnaRecord, PortAnaRecord

import torch
from torch.utils.data import DataLoader, Sampler

from master import PromptMASTERModel
import argparse
# import logging
# logging.basicConfig()


# Please install qlib first before load the data.
# with open(f'data/{universe}/{universe}_dl_train.pkl', 'rb') as f:
#     dl_train = pickle.load(f)
# with open(f'data/{universe}/{universe}_dl_valid.pkl', 'rb') as f:
#     dl_valid = pickle.load(f)
# with open(f'data/{universe}/{universe}_dl_test.pkl', 'rb') as f:
#     dl_test = pickle.load(f)
# print("Data Loaded.")

# class MASTERManager(SequenceModel):
#     def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
#                  gate_input_start_index=158, gate_input_end_index=221, beta=None, m_prompts = 10, n_prompts = 5, len_prompts = 5, lamb = 0.5, **kwargs):
#                 #  n_epochs = 40, lr = 8e-6, GPU = 3, seed = 0, train_stop_loss_thred = 0.95, benchmark = 'SH000300', market = 'csi300'):
#         self.d_feat = d_feat
#         self.d_model = d_model
#         self.t_nhead = t_nhead
#         self.s_nhead = s_nhead
#         self.T_dropout_rate = T_dropout_rate
#         self.S_dropout_rate = S_dropout_rate
#         self.gate_input_start_index = gate_input_start_index
#         self.gate_input_end_index = gate_input_end_index
#         self.beta = beta
#         self.m_prompts = m_prompts
#         self.n_prompts = n_prompts
#         self.len_prompts = len_prompts
#         self.lamb = lamb
#         with open("./workflow_config_master_Alpha158.yaml", 'r') as f:
#             self.basic_config = yaml.safe_load(f)

#         super.__init__(self, **kwargs)

    # def load_data(self) -> Tuple[TSDataSampler]:
    #     ds = init_instance_by_config(self.basic_config['dataset'], accept_types=Dataset)
    #     self.train_data = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    #     self.valid_data = ds.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    #     self.test_data = ds.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)


    # def train(self):
        


    
    # def test(self):
    
def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default='csi300',
        help="which market to choose")
    parser.add_argument("--mprompts", type=int, default=10,
        help="num of prompts in the prompt pool")
    parser.add_argument("--nprompts", type=int, default=5,
        help="num of prompts to be chosen when inference")
    parser.add_argument("--lenprompts", type=int, default=5,
        help="length of prompts")
    parser.add_argument("--online_lr", type=float, default=1e-4,
        help="online training learning rate")
    # parser.add_argument("--only_backtest", type=bool, default=False,
    #     help="whether only backtest or not")
    parser.add_argument("--only_backtest", action="store_true", help="whether only backtest or not")
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
    d_feat = 158
    d_model = 256
    t_nhead = 4
    s_nhead = 2
    dropout = 0.5
    gate_input_start_index=158
    gate_input_end_index = 221

    universe = args.universe # or 'csi500'
    if universe == 'csi300':
        beta = 10
        benchmark = 'SH000300'
    elif universe == 'csi500':
        beta = 5
        benchmark = 'SH000905'

    n_epochs = 40
    lr = 8e-6
    GPU = 3
    seed = 0
    train_stop_loss_thred = 0.95
    
    m_prompts = args.mprompts
    n_prompts = args.nprompts
    len_prompts = args.lenprompts
    lamb = 0.5
    use_prompts = True
    online_lr = args.online_lr
    only_backtest = args.only_backtest

    master = PromptMASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epochs, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        m_prompts = m_prompts, n_prompts = n_prompts, len_prompts = len_prompts, lamb = lamb, use_prompts = use_prompts,
        save_path='model/', save_prefix=universe, benchmark = benchmark, market = universe,
        online_lr = {'lr': online_lr}, only_backtest = only_backtest
    )

    master.run_all()

    # print(sys.argv)
    # fire.Fire(MASTERModel)

    # # Train
    # model.fit(dl_train, dl_valid)
    # print("Model Trained.")

    # # Test
    # predictions, metrics = model.predict(dl_test)
    # print(metrics)

    # Load and Test
    #param_path = f'model/{universe}master_0.pkl.'
    #print(f'Model Loaded from {param_path}')
    #model.load_param(param_path)
    #predictions, metrics = model.predict(dl_test)
    #print(metrics)



