import pickle
from typing import Optional, List, Tuple, Union, Text
# from base_model import SequenceModel
import yaml
import tqdm
import fire
import sys
from pathlib import Path
import argparse

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

from master import MASTERModel

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpronmpts", type=int, default=10,
        help="num of prompts in the prompt pool")
    parser.add_argument("--nprompts", type=int, default=5,
        help="num of prompts to be chosen when inference")
    parser.add_argument("--lenprompts", type=int, default=5,
        help="length of prompts")
    parser.add_argument("--horizon", type=int, default=1,
        help="horizon of label")
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

    universe = 'csi300' # or 'csi800'
    if universe == 'csi300':
        beta = 10
    elif universe == 'csi800':
        beta = 5
    benchmark = 'SH000300'

    n_epoch = 40
    lr = 8e-6
    GPU = 3
    seed = 0
    train_stop_loss_thred = 0.95
    horizon = args.horizon

    master = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path='model/', save_prefix=universe, benchmark = benchmark, market = universe, horizon = args.horizon
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



