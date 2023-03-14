from __future__ import division
from __future__ import print_function

import torch
import dgl

from qlib.model.base import GraphExplainer


class AttentionX(GraphExplainer):
    def __init__(self, graph_model, num_layers, device):
        super(AttentionX, self).__init__(graph_model, num_layers, device)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)



    def explain(self, full_model, graph, stkid, top_k=3):
        if self.graph_model == 'homograph':
            dataloader = dgl.dataloading.NodeDataLoader(graph,
                                                     torch.Tensor([stkid]).type(torch.int64).to(self.device),
                                                     self.sampler,
                                                    batch_size=1, shuffle=False, drop_last=False)

            neighbors = None
            for neighbors, _, _ in dataloader:
                break

            target_id = stkid

            g_c = dgl.node_subgraph(graph, neighbors) # induce the computation graph
            attn = full_model.get_attention(g_c)

            #print('g_c', g_c.num_edges())

            new_target_id = (g_c.ndata['_ID'].tolist()).index(target_id)

            g_c = g_c.to(self.device)
            for l in range(self.num_layers):
                attn[l] = torch.mean(attn[l], dim=1).squeeze(dim=1).tolist() # summarize multi-head attention

            node_attn = {self.num_layers: {new_target_id: 1}}

            for l in range(self.num_layers - 1, -1, -1):  # assign attention to nodes iteratively from top layer
                node_attn[l] = {}  # src: attn_score
                src, dst = g_c.edges()
                for j, (ss, dd) in enumerate(zip(src.tolist(), dst.tolist())):
                    if (dd in node_attn[l + 1]):  # connected to top layer nodes
                        node_attn[l][ss] = attn[l][j] * node_attn[l + 1][dd]

            # sum up for each node
            ne_attn = {}
            new2old = {}
            for j, n in enumerate(neighbors.tolist()):
                ne_attn[n] = 0
                new2old[j] = n
            for l in range(self.num_layers - 1, -1, -1):  # from top
                for newn in node_attn[l]:
                    ne_attn[new2old[newn]] += node_attn[l][newn]
            sorted_ne = sorted(zip(ne_attn.keys(), ne_attn.values()), key=lambda x: x[1], reverse=True)
            return sorted_ne[:top_k]
        else:
            return NotImplementedError

    def explanation_to_graph(self, explanation, subgraph, stkid):
        g_m_nodes = [i[0] for i in explanation]
        if not stkid in g_m_nodes:
            g_m_nodes.append(stkid)
        g_m = dgl.node_subgraph(subgraph, g_m_nodes)
        new_stkid = (g_m.ndata['_ID'].tolist()).index(stkid)
        #print('g_m', g_m.num_edges())
        return g_m, new_stkid
