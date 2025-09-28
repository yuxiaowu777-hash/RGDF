import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch_scatter import scatter
from collections import defaultdict

# Compute edge attention weights (alpha) and optionally sample important edges; 
# aggregate neighbor messages to update node representations; optionally sample important nodes, 
# and return the updated nodes and representations.
class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_ent, n_node_topk=-1, n_edge_topk=-1, tau=1.0,
                 act=lambda x: x):
        super(GNNLayer, self).__init__()
        #
        self.n_rel = n_rel
        self.n_ent = n_ent
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.n_node_topk = n_node_topk
        self.n_edge_topk = n_edge_topk
        self.tau = tau
        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.W_samp = nn.Linear(in_dim, 1, bias=False)

        # Learnable weights for structural features
        self.wt = nn.Parameter(torch.tensor(1.0))
        self.wc = nn.Parameter(torch.tensor(1.0))

        # Stage 1: Local Interaction Module
        self.local_interact = nn.Sequential(
            nn.Linear(in_dim * 2, attn_dim, bias=False),  # Take hs and hr as input
        )
        # Stage 2: Global Modulation Module — using structural coefficients
        self.global_modulate = nn.Sequential(
            nn.Linear(attn_dim * 2 + 1, attn_dim),  # Take the local results, hqr, and structural coefficients as input
            nn.ReLU(),
            nn.Linear(attn_dim, attn_dim)
        )
        self.norm = nn.LayerNorm(attn_dim)

    def compute_structural_coefficient(self, degrees, triangles, cycles_4, eps=1e-8):
        # Ensure the input is a 2D tensor
        if degrees.dim() == 1:
            degrees = degrees.unsqueeze(-1)
        if triangles.dim() == 1:
            triangles = triangles.unsqueeze(-1)
        if cycles_4.dim() == 1:
            cycles_4 = cycles_4.unsqueeze(-1)

        # 2 * (wt * triangles_i + wc * 4-cycles_i)
        numerator = 2.0 * (self.wt * triangles + self.wc * cycles_4)

        # D_i^(ℓ) * (D_i^(ℓ) - 1) + eps
        denominator = degrees * (degrees - 1) + eps

        # Compute the coefficients
        coefficient = numerator / denominator

        return coefficient

    def get_structural_weights(self):
        return self.wt.item(), self.wc.item()

    def set_structural_weights(self, wt=None, wc=None):
        if wt is not None:
            with torch.no_grad():
                self.wt.fill_(wt)
        if wc is not None:
            with torch.no_grad():
                self.wc.fill_(wc)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.training and self.tau > 0:
            self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x: F.softmax(x, dim=1)
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize, node_degrees=None,
                node_triangles=None, node_cycles_4=None):
        # edges: [N_edge_of_all_batch, 6]
        # with (batch_idx, head, rela, tail, head_idx, tail_idx)
        # note that head_idx and tail_idx are relative index
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        hs = hidden[sub]
        hr = self.rela_embed(rel)
        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]
        n_node = nodes.shape[0]
        message = hs + hr

        # Stage 1: Local Interaction (hs and hr)
        local_input = torch.cat([hs, hr], dim=-1)  # [N_edge, in_dim * 2]
        local_feature = self.local_interact(local_input)  # [N_edge, attn_dim]

        if node_degrees is not None and node_triangles is not None and node_cycles_4 is not None:
            edge_degrees = node_degrees[obj].unsqueeze(-1)  # [N_edge, 1]
            edge_triangles = node_triangles[obj].unsqueeze(-1)  # [N_edge, 1]
            edge_cycles = node_cycles_4[obj].unsqueeze(-1)  # [N_edge, 1]
        else:
            # Compatibility
            degrees = scatter(torch.ones_like(edges[:, 5]), edges[:, 5], dim=0, dim_size=nodes.shape[0], reduce='sum')
            edge_degrees = degrees[obj].unsqueeze(-1)  # [N_edge, 1]
            edge_triangles = torch.zeros_like(edge_degrees)  # [N_edge, 1]
            edge_cycles = torch.zeros_like(edge_degrees)  # [N_edge, 1]

        # Stage 2: Global Modulation (local features + h_qr + degree + triangles + 4-cycles)
        structural_coeff = self.compute_structural_coefficient(edge_degrees, edge_triangles, edge_cycles)

        global_input = torch.cat([local_feature, self.Wqr_attn(h_qr), structural_coeff],
                                 dim=-1)  # [N_edge, attn_dim * 2 + 1]
        attn_input = self.global_modulate(global_input)  # [N_edge, attn_dim]
        attn_input = self.norm(attn_input)

        # sample edges w.r.t. alpha
        if self.n_edge_topk > 0:
            alpha = self.w_alpha(nn.ReLU()(attn_input)).squeeze(-1)
            edge_prob = lambda x: F.gumbel_softmax(alpha, tau=self.tau)
            topk_index = torch.topk(edge_prob, self.n_edge_topk, dim=0).indices  # 使用 topk 替代 argsort
            edge_prob_hard = torch.zeros_like(alpha, device=alpha.device).scatter_(0, topk_index, 1)
            alpha = alpha * (edge_prob_hard - edge_prob.detach() + edge_prob)
            alpha = torch.sigmoid(alpha).unsqueeze(-1)

        else:
            alpha = torch.sigmoid(self.w_alpha(
                nn.ReLU()(attn_input)))  # [N_edge_of_all_batch, 1]

        # aggregate message and then propagate
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg))  # [n_node, dim]
        hidden_new = hidden_new.clone()

        # forward without node sampling
        if self.n_node_topk <= 0:
            return hidden_new

        # forward with node sampling
        # indexing sampling operation
        valid_old_nodes = old_nodes_new_idx[old_nodes_new_idx < n_node]
        bool_diff_node_idx = ~torch.isin(torch.arange(n_node, device=hidden_new.device), valid_old_nodes)
        diff_node = nodes[bool_diff_node_idx]
        diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1)

        # save logit to node_scores for later indexing
        node_scores = torch.ones((batchsize, self.n_ent)).cuda() * float('-inf')
        node_scores[diff_node[:, 0], diff_node[:, 1]] = diff_node_logit

        # select top-k nodes
        # (train mode) self.softmax == F.concrete_sample
        # (eval mode)  self.softmax == F.softmax
        node_scores = self.softmax(node_scores)  # [batchsize, n_ent]
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batchidx = torch.arange(batchsize).repeat(self.n_node_topk, 1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batchsize, self.n_ent)).cuda()
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        # get sampled nodes' relative index
        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx.cuda()
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx

        # update node embeddings
        diff_node_prob_hard = batch_topk_nodes[diff_node[:, 0], diff_node[:, 1]]
        diff_node_prob = node_scores[diff_node[:, 0], diff_node[:, 1]]
        hidden_new[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)

        # extract sampled nodes an their embeddings
        new_nodes = nodes[bool_same_node_idx]
        hidden_new = hidden_new[bool_same_node_idx]

        return hidden_new, new_nodes, bool_same_node_idx


class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_ent = params.n_ent
        self.n_rel = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.n_ent, \
                                            n_node_topk=i_n_node_topk, n_edge_topk=self.n_edge_topk, tau=params.tau,
                                            act=act))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

        # Graph-enhanced residual gated fusion module — using the structural coefficient formula
        self.res_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim + 1, self.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 4, self.hidden_dim),
                nn.Sigmoid()  # Output the gating weights
            ) for _ in range(self.n_layer)
        ])
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):
        def freeze(m):
            m.requires_grad = False

        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def forward(self, subs, rels, mode='train'):
        n = len(subs)  # n == B (Batchsize)
        q_sub = torch.LongTensor(subs).cuda()  # [B]
        q_rel = torch.LongTensor(rels).cuda()  # [B]
        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()  # [1, B, dim]
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)],
                          1)  # [B, 2] with (batch_idx, node_idx)
        hidden = torch.zeros(n, self.hidden_dim).cuda()  # [B, dim]
        prev_hidden = None  # Previous layer's hidden state
        for i in range(self.n_layer):
            # layers with sampling
            # nodes (of i-th layer): [k1, 2]
            # edges (of i-th layer): [k2, 6]
            # old_nodes_new_idx (of previous layer): [k1']
            # node_degrees, node_triangles, node_cycles_4 (of current layer): [k1, 1] each
            nodes, edges, old_nodes_new_idx, node_degrees, node_triangles, node_cycles_4 = self.loader.get_neighbors(
                nodes, n, mode=mode)
            n_node = nodes.size(0)

            # GNN forward -> get hidden representation at i-th layer
            # hidden: [k1, dim]
            hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx,
                                                                  n, node_degrees, node_triangles, node_cycles_4)
            # Graph-enhanced residual gating fusion
            if prev_hidden is not None:
                # Use precomputed graph structure features
                degrees = node_degrees[sampled_nodes_idx].unsqueeze(-1)  # [n_node, 1]
                node_triangles = node_triangles[sampled_nodes_idx].unsqueeze(-1)  # [n_node, 1]
                node_cycles = node_cycles_4[sampled_nodes_idx].unsqueeze(-1)  # [n_node, 1]

                # Adjust prev_hidden size
                prev_padded = torch.zeros(n_node, self.hidden_dim).cuda()
                prev_padded[old_nodes_new_idx[:prev_hidden.shape[0]]] = prev_hidden
                prev_padded = prev_padded[sampled_nodes_idx]
                hidden = self.dropout(hidden)
                # Compute gating weights with enhanced graph structure features using the formula
                structural_coeff = self.gnn_layers[i].compute_structural_coefficient(degrees, node_triangles,
                                                                                     node_cycles)
                gate_input = torch.cat([hidden, structural_coeff], dim=-1)  # [n_node, hidden_dim + 1]
                gate = self.res_gate[i](gate_input)  # [n_node, hidden_dim]
                # Residual gating fusion
                hidden = gate * hidden + (1 - gate) * prev_padded
                hidden = self.norm(hidden)

            prev_hidden = hidden.clone()

        # readout
        # [K, 2] (batch_idx, node_idx) K is w.r.t. n_nodes
        scores = self.W_final(hidden).squeeze(-1)
        # non-visited entities have 0 scores
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        # [B, n_all_nodes]
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all

    def get_all_structural_weights(self):
        weights = []
        for i, layer in enumerate(self.gnn_layers):
            wt, wc = layer.get_structural_weights()
            weights.append((i, wt, wc))
        return weights

    def set_all_structural_weights(self, weights):
        for layer_idx, wt, wc in weights:
            if 0 <= layer_idx < len(self.gnn_layers):
                self.gnn_layers[layer_idx].set_structural_weights(wt, wc)