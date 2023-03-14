# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import dgl

class HomographModel(nn.Module):
    def __init__(self, base_model, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.g = None

    def get_attention(self, graph):
        raise ValueError("please implement cal_attention() in the specific graph model")

    def forward(self, x, index=None):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_rnn(self, x):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_graph(self, x, index=None, return_subgraph=False):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_predictor(self, x0, x):
        raise ValueError("please implement forward() in the specific graph model ")

    def set_graph(self, rel_encoding, device):
        if len(rel_encoding.shape) == 3:
            rel_encoding = rel_encoding.sum(axis=-1)  # [N, N]
        idx = rel_encoding.nonzero()
        self.g = dgl.graph((idx[0], idx[1])).to(device)

    def predict_on_graph(self, g):
        x = g.ndata['nfeat']
        origin_graph = self.g
        self.g = g
        h = self.forward_graph(x)
        pred = self.forward_predictor(x, h)
        self.g = origin_graph
        return pred

class HeterographModel(nn.Module):
    def __init__(self, base_model, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.g = None
        self.target_type = 's' # for stock
        self.none_feature = nn.Parameter(torch.FloatTensor(size=(1, hidden_size)), requires_grad=False)

    def get_attention(self, x, index):
        raise ValueError("please implement cal_attention() in the specific graph model")

    def forward(self, x, index=None):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_rnn(self, x):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_graph(self, x, index=None, return_subgraph = True):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_predictor(self, x0, x):
        raise ValueError("please implement forward() in the specific graph model ")

    def set_graph(self, rel_encoding, device):
        nr = rel_encoding.shape[-1]
        edges = {('none',str(nr),'none'):([0],[0])} # 'none' type has a virtual node to fit in dgl heterogeneous graph
        for i in range(nr):
            idx = rel_encoding[:,:,i].nonzero()
            edges[(self.target_type, str(i), self.target_type)] = (idx[0], idx[1])

        self.g = dgl.heterograph(edges).to(device)

    def predict_on_graph(self, g):
        x = {}
        for tp in g.ntypes:
            x[tp] = g.nodes[tp].data['nfeat']

        origin_graph = self.g
        self.g = g
        pred = self.forward(x)
        self.g = origin_graph
        return pred










