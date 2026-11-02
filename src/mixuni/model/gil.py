"""Graph interaction layer (GIL) from GeoMix.

Passes messages between the components of a mixture. Each component carries
an embedding ``h``, a position ``x`` (mean of its atoms) and a vector state
``v``; vector messages are rotated into the receiver's local frame. With
``num_experts > 1`` every edge is routed to one low-rank adapter.

MixUni itself runs without GIL; it is used by the single-task GeoMix baselines.
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import BaseMLP, scatter_mean

ADAPTER_RANK = 16


def graph_pairs(g2b):
    """Directed edges between every two components of the same mixture."""
    n = g2b.numel()
    empty = torch.empty((2, 0), dtype=torch.long, device=g2b.device)
    if n <= 1:
        return empty
    ids = torch.arange(n, device=g2b.device)
    same = (g2b.unsqueeze(0) == g2b.unsqueeze(1)) & (ids.unsqueeze(0) != ids.unsqueeze(1))
    row, col = torch.triu(same, diagonal=1).nonzero(as_tuple=True)
    if row.numel() == 0:
        return empty
    return torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)


def _edge_features(h_i, h_j, x_i, x_j, v_i, v_j):
    """[h_i, h_j, |x_i|, |x_j|, |v_i|, |v_j|] for every edge i <- j."""
    return torch.cat([h_i, h_j] + [t.norm(dim=-1, keepdim=True) for t in (x_i, x_j, v_i, v_j)], dim=-1)


def gate_entropy_gap(probs):
    """log(K) minus the entropy of the mean routing distribution (0 when usage is uniform)."""
    mean = probs.mean(dim=0)
    entropy = -(mean * mean.clamp_min(1e-8).log()).sum()
    return probs.new_tensor(float(probs.size(-1))).log() - entropy


class GIL(nn.Module):
    def __init__(self, hidden_dim, num_experts=1):
        super().__init__()
        self.num_experts = num_experts
        self.msg_mlp_1 = BaseMLP(2 * hidden_dim + 4, hidden_dim, hidden_dim, last_act=True)
        self.msg_mlp_2 = BaseMLP(2 * hidden_dim + 5, hidden_dim, hidden_dim, last_act=True)
        self.x_mlp = BaseMLP(hidden_dim, hidden_dim, 1)
        self.v_mlp = BaseMLP(hidden_dim, hidden_dim, 1)
        self.h_mlp = BaseMLP(2 * hidden_dim, hidden_dim, hidden_dim, last_act=True)
        self.rot_9d = BaseMLP(hidden_dim, hidden_dim, 9, last_act=True)
        self.t_mlp = BaseMLP(hidden_dim, hidden_dim, 3, last_act=True)
        if num_experts > 1:
            self.expert_down = nn.ModuleList([nn.Linear(hidden_dim, ADAPTER_RANK) for _ in range(num_experts)])
            self.expert_up = nn.ModuleList([nn.Linear(ADAPTER_RANK, hidden_dim) for _ in range(num_experts)])
            self.expert_v_scale = nn.ParameterList([nn.Parameter(torch.ones(1, 3)) for _ in range(num_experts)])
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_experts)
            )

    def _message(self, feats, x_j, v_j, f_i, f_j):
        msg = self.msg_mlp_1(feats)
        rot = self.rot_9d(msg).view(-1, 3, 3)
        t = self.t_mlp(msg)
        msg = self.msg_mlp_2(torch.cat([feats, t.norm(dim=-1, keepdim=True)], dim=-1))
        msg_v = self.x_mlp(msg) * x_j + self.v_mlp(msg) * v_j
        msg_v = rot @ f_j @ msg_v.unsqueeze(-1) + t.unsqueeze(-1)
        return msg, (f_i.transpose(-1, -2) @ msg_v).squeeze(-1)

    def _route(self, feats, msg, msg_v):
        """Send each edge to its top-1 adapter; returns the routed messages and the entropy gap."""
        logits = self.gate(feats)
        top_logit, expert = logits.topk(1, dim=-1)
        weight = torch.softmax(top_logit, dim=-1)
        out, out_v = torch.zeros_like(msg), torch.zeros_like(msg_v)
        for k in range(self.num_experts):
            chosen = expert == k
            if not bool(chosen.any()):
                continue
            edges = chosen.any(dim=1)
            ids = edges.nonzero(as_tuple=True)[0]
            m = msg[edges]
            w = (weight[edges] * chosen[edges].to(weight.dtype)).sum(dim=1, keepdim=True)
            adapter = self.expert_up[k](torch.nn.functional.silu(self.expert_down[k](m)))
            out.index_add_(0, ids, (w * (m + adapter)).to(out.dtype))
            out_v.index_add_(0, ids, (w * (msg_v[edges] * self.expert_v_scale[k])).to(out_v.dtype))
        return out, out_v, gate_entropy_gap(torch.softmax(logits, dim=-1))

    def forward(self, h, x, v, edge_index, frames):
        if edge_index.numel() == 0:
            return h, v, h.new_zeros(())
        row, col = edge_index
        ends = (h[row], h[col], x[row], x[col], v[row], v[col])
        x_j, v_j = ends[3], ends[5]
        msg, msg_v = self._message(_edge_features(*ends), x_j, v_j, frames[row], frames[col])
        if self.num_experts > 1:
            # the gate gets its own copy of the edge features, as in the original implementation
            msg, msg_v, aux = self._route(_edge_features(*ends), msg, msg_v)
        else:
            aux = msg.new_zeros(())
        v = v + scatter_mean(msg_v, row, v.size(0))
        h = self.h_mlp(torch.cat([h, scatter_mean(msg, row, h.size(0))], dim=-1))
        return h, v, aux
