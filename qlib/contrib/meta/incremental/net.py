import collections
import math

import torch

import qlib
from qlib.model import Model

from qlib.utils import init_instance_by_config

from torch import nn
from torch.nn import functional as F, init
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

def cosine(x1, x2, eps=1e-8):
    x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
    x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
    return x1 @ x2.transpose(0, 1)


# class LabelAdaptHead(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty(1))
#         self.bias = nn.Parameter(torch.ones(1) / 8)
#         init.uniform_(self.weight, 0.75, 1.25)
#
#     def forward(self, y, inverse=False):
#         if inverse:
#             return (y - self.bias) / (self.weight + 1e-9)
#         else:
#             return (self.weight + 1e-9) * y + self.bias

class LabelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias

class LabelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        # self.heads = nn.ModuleList([LabelAdaptHead() for _ in range(num_head)])
        self.heads = LabelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        # return sum([gate[:, i] * self.heads[i](y, inverse=inverse) for i in range(self.num_head)])
        return (gate * self.heads(y, inverse=inverse)).sum(-1)


class FiLM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(in_dim))
        nn.init.uniform_(self.scale, 0.75, 1.25)

    def forward(self, x):
        return x * self.scale


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim, num_head=4, temperature=4):
        super().__init__()
        self.num_head = num_head
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)])
        self.temperature = temperature

    def forward(self, x):
        s_hat = torch.cat(
            [torch.cosine_similarity(x, self.P[i], dim=-1).unsqueeze(-1) for i in range(self.num_head)], -1,
        )
        # s_hat = cosine(x, self.P)
        s = torch.softmax(s_hat / self.temperature, -1).unsqueeze(-1)
        return x + sum([s[..., i, :] * self.heads[i](x) for i in range(self.num_head)])


class ForecastModel(nn.Module):
    def __init__(self, task_config, x_dim=None, lr=0.001, weight_decay=0, need_permute=False, model=None):
        super().__init__()
        self.lr = lr
        # self.lr = task_config["model"]['kwargs']['lr']
        self.criterion = nn.MSELoss()
        if task_config["model"]["class"] == "LinearModel":
            if model is not None:
                if isinstance(model, qlib.contrib.model.LinearModel):
                    self.model = nn.Linear(x_dim, 1)
                    self.model.load_state_dict(
                        collections.OrderedDict(
                            {"weight": torch.from_numpy(model.coef_).unsqueeze(0), "bias": torch.tensor([model.intercept_]),}
                        )
                    )
                elif isinstance(model, nn.Linear):
                    self.model = model
            else:
                self.model = nn.Linear(x_dim, 1, bias=False)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
            self.device = torch.device("cuda")
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
            if hasattr(model, "device"):
                self.device = model.device
            else:
                self.device = torch.device("cuda")
            self.need_permute = need_permute
        if self.opt is None:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None):
        if model is None:
            model = self.model
        if X.dim() == 3:
            X = X.permute(0, 2, 1).reshape(len(X), -1) if self.need_permute else X.reshape(len(X), -1)
        y_hat = model(X)
        y_hat = y_hat.view(-1)
        return y_hat


class DoubleAdapt(ForecastModel):
    def __init__(
        self, task_config, factor_num, x_dim=None, lr=0.001, weight_decay=0,
            need_permute=False, model=None, num_head=8, temperature=10,
    ):
        super().__init__(
            task_config=task_config, x_dim=x_dim, lr=lr, weight_decay=weight_decay,
            need_permute=need_permute, model=model,
        )
        self.teacher_x = FeatureAdapter(factor_num, num_head, temperature)
        self.teacher_y = LabelAdapter(factor_num if x_dim is None else x_dim, num_head, temperature)
        self.meta_params = list(self.teacher_x.parameters()) + list(self.teacher_y.parameters())
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, model=None, transform=False):
        if transform:
            X = self.teacher_x(X)
        return super().forward(X, model), X


class CoG(ForecastModel):
    def __init__(self, task_config, x_dim=None, lr=0.001, need_permute=False, model=None):
        super().__init__(
            task_config=task_config, x_dim=x_dim, lr=lr, need_permute=need_permute, model=model,
        )
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.randn_like(param.data) ** 2 + 1) for param in self.model.parameters()]
        )
        self.meta_params = self.mask.parameters()
        if self.device is not None:
            self.to(self.device)

    def forward(self, X, fmodel=None, fmask=None):
        new_params = []
        if fmodel is None:
            fmodel = self.model
        else:
            for i in range(len(fmask)):
                new_params.append(fmodel.fast_params[i] * torch.sigmoid(fmask[i]))
            fmodel.update_params(new_params)
        return super().forward(X, model=fmodel)

#######################################################################################################################
# lqa: MASTER and prompt pool

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # print("After TAttention:", x)
        # print("After TAttention: torch.any(torch.isnan(x))", torch.any(torch.isnan(x)))
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            arr = (torch.matmul(qh, kh.transpose(1, 2)) / self.temperature).detach().cpu().numpy()
            # print(arr.max(axis=-1))
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            # print("SAttention's vh", vh)
            # print("torch.matmul(qh, kh.transpose(1, 2)) / self.temperature", torch.matmul(qh, kh.transpose(1, 2)) / self.temperature)
            # print("SAttention's atten_ave_matrixh", atten_ave_matrixh)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # print("After pe: ", x)
        # print("After pe: torch.any(torch.isnan(x))", torch.any(torch.isnan(x)))
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        # print("TAttention", att_output)
        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        # nn.init.uniform_(self.trans.weight, 5e5, 1e6)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # print("After SAttention:", z)
        h = self.trans(z) # [N, T, D]
        # print(h.shape)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.x2y = nn.Linear(d_feat, d_model)
        self.pe = PositionalEncoding(d_model)
        self.tatten = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temporalatten = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)
        # self.layers = nn.Sequential(
        #     # feature layer
        #     nn.Linear(d_feat, d_model),
        #     PositionalEncoding(d_model),
        #     # intra-stock aggregation
        #     TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
        #     # inter-stock aggregation
        #     SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
        #     TemporalAttention(d_model=d_model),
        #     # decoder
        #     nn.Linear(d_model, 1)
        # )


    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        # print("torch.any(torch.isnan(src)):", torch.any(torch.isnan(src)))
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]

        # print("gate_input", gate_input)
        # print("torch.any(torch.isnan(gate_input)):", torch.any(torch.isnan(gate_input)))
        # print("feature_gate.trans.weight", self.feature_gate.trans.weight)
        # print("feature_gate", self.feature_gate(gate_input))
        # print("torch.any(torch.isnan(feature_gate)):", torch.any(torch.isnan(self.feature_gate(gate_input))))
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        # output_tmp = self.layers(src)
        # print(output_tmp)
        # output = self.decoder(output_tmp).squeeze(-1)
        x = self.x2y(src)
        x = self.pe(x)
        x = self.tatten(x)
        x = self.satten(x)
        x = self.temporalatten(x)
        output = self.decoder(x).squeeze(-1)
        # output = self.layers(src).squeeze(-1)

        return output

class PromptMASTER(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None, m_prompts = 10, n_prompts = 5, len_prompts = 5, lamb = 0.5):
        super(PromptMASTER, self).__init__()
        # # market
        # self.gate_input_start_index = gate_input_start_index
        # self.gate_input_end_index = gate_input_end_index
        # self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        
        # master
        self.master = MASTER(d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=T_dropout_rate, S_dropout_rate=S_dropout_rate,
                             gate_input_start_index=gate_input_start_index, gate_input_end_index=gate_input_end_index, beta=beta)
        
        # self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        # self.x2y = nn.Linear(d_feat, d_model)
        # self.pe = PositionalEncoding(d_model)
        # self.tatten = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        # self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        # self.temporalatten = TemporalAttention(d_model=d_model)
        # self.decoder = nn.Linear(d_model, 1)

        self.keys = nn.Parameter(torch.zeros((m_prompts, d_feat)))
        nn.init.uniform_(self.keys, a = 0, b = 0.01)
        self.prompts = nn.Parameter(torch.zeros((m_prompts, len_prompts, d_feat)))
        nn.init.uniform_(self.prompts, a = 0, b = 0.01)
        self.n_prompts = n_prompts
        self.lamb = lamb

    def forward(self, x, use_prompts = False):
        src = x[:, :, :self.master.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.master.gate_input_start_index:self.master.gate_input_end_index]
        market_infos = self.master.feature_gate(gate_input)
        src = src * torch.unsqueeze(market_infos, dim=1)

        # if use_prompts == True:
        if use_prompts:
            cos_result = torch.norm(self.keys-market_infos[-1,:], dim=1) / self.keys.norm(dim=1) / market_infos.norm()
            topk = torch.topk(cos_result, self.n_prompts)
            selected_prompts = self.prompts[topk.indices,:,:]
            prompts = selected_prompts.flatten(start_dim=0, end_dim = 1)
            prompts = prompts.unsqueeze(dim=0).repeat(src.shape[0], 1, 1)
            src = torch.cat([src, prompts], dim=1)

        x = self.master.x2y(src)
        x = self.master.pe(x)
        x = self.master.tatten(x)
        x = self.master.satten(x)
        x = self.master.temporalatten(x)
        output = self.master.decoder(x).squeeze(-1)
        
        if use_prompts:
            return output, torch.sum(cos_result[topk.indices])
        else:
            return output
