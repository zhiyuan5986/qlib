import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
import yaml
import numpy as np
import pprint as pp

from base_model import MetaModelRolling

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()


class TransformerModel(MetaModelRolling):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        **kwargs
    ):
        self.d_model = d_model
        self.d_feat = d_feat
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        super(TransformerModel, self).__init__(**kwargs)
        self.init_model()

    def init_model(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.framework = Transformer(self.d_feat, self.d_model, self.nhead, self.num_layers, self.dropout, self.device)
        if self.only_backtest:
            self.load_model(f"./model/{self.market}transformer_{self.seed}.pkl")
        else:
            self.load_model(f"../../benchmarks/Transformer_ours/model/{self.market}transformer_{self.seed}.pkl")
        super(TransformerModel, self).init_model()
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
        seed = self.seed
        for s in range(seed, seed+10):
            self.seed = s
            print("--------------------")
            print("seed: ", self.seed)
            self.init_model()

            if not self.only_backtest:
                self.fit()
            rec = self.online_training()

            metrics = rec.list_metrics()
            print(metrics)
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pp.pprint(all_metrics)
        for k in all_metrics.keys():
            print(f"{k}: {np.mean(all_metrics[k])} +- {np.std(all_metrics[k])}")

# class MASTERManager(SequenceModel):
    # def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
    #              gate_input_start_index=158, gate_input_end_index=221, beta=None, m_prompts = 10, n_prompts = 5, len_prompts = 5, lamb = 0.5, **kwargs):
    #             #  n_epochs = 40, lr = 8e-6, GPU = 3, seed = 0, train_stop_loss_thred = 0.95, benchmark = 'SH000300', market = 'csi300'):
    #     self.d_feat = d_feat
    #     self.d_model = d_model
    #     self.t_nhead = t_nhead
    #     self.s_nhead = s_nhead
    #     self.T_dropout_rate = T_dropout_rate
    #     self.S_dropout_rate = S_dropout_rate
    #     self.gate_input_start_index = gate_input_start_index
    #     self.gate_input_end_index = gate_input_end_index
    #     self.beta = beta
    #     with open("./workflow_config_master_Alpha158.yaml", 'r') as f:
    #         self.basic_config = yaml.safe_load(f)

    #     super.__init__(self, **kwargs)