"""Shared heads over all properties: a mixture of experts, or one MLP per property.

Both predict the z-scored target of the queried property.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .heads import MLPHead, inv_t


def load_balance_loss(logits):
    """K * sum_k (mean routing probability of expert k)^2; 1 when balanced, K when collapsed."""
    usage = F.softmax(logits, dim=-1).mean(dim=0)
    return logits.size(-1) * (usage * usage).sum()


class Router(nn.Module):
    """Noisy top-k gate on (h, T, 1/T, c, T^2, c^2, property embedding)."""

    def __init__(self, hidden_dim, property_dim, num_experts, top_k):
        super().__init__()
        self.top_k = min(top_k, num_experts)
        in_dim = hidden_dim + 5 + property_dim
        self.gate_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_experts))
        self.noise_linear = nn.Linear(in_dim, num_experts)

    def forward(self, h, T, c, p):
        x = torch.cat(
            [h, T.unsqueeze(1), inv_t(T).unsqueeze(1), c.unsqueeze(1), (T ** 2).unsqueeze(1), (c ** 2).unsqueeze(1), p],
            dim=1,
        )
        logits = self.gate_net(x)
        if self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(self.noise_linear(x))
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)
        return F.softmax(top_logits, dim=-1), top_idx, logits


class SharedMoE(nn.Module):
    """K MLP experts shared by all properties; the router sees the property embedding."""

    def __init__(self, hidden_dim, num_properties, property_dim, num_experts, top_k):
        super().__init__()
        self.prop_embedding = nn.Embedding(num_properties, property_dim)
        self.experts = nn.ModuleList([MLPHead(hidden_dim, property_dim) for _ in range(num_experts)])
        self.gate = Router(hidden_dim, property_dim, num_experts, top_k)

    def forward(self, h, T, c, property_id):
        p = self.prop_embedding(property_id)
        weights, chosen, logits = self.gate(h, T, c, p)
        pred = h.new_zeros(h.size(0))
        for k, expert in enumerate(self.experts):
            hit = chosen == k
            if not bool(hit.any()):
                continue
            rows = hit.any(dim=1)
            w = (weights[rows] * hit[rows].to(weights.dtype)).sum(dim=1)
            pred.index_add_(0, rows.nonzero(as_tuple=True)[0], w * expert(h[rows], T[rows], c[rows], p[rows]))
        return pred, load_balance_loss(logits)


class PerPropertyHeads(nn.Module):
    """No sharing: a separate 3-layer MLP on (h, T, 1/T, c) for each property."""

    def __init__(self, hidden_dim, num_properties, property_dim):
        super().__init__()
        self.prop_embedding = nn.Embedding(num_properties, property_dim)  # unused; kept so checkpoints load
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_properties)
        ])

    def forward(self, h, T, c, property_id):
        x = torch.cat([h, T.unsqueeze(1), inv_t(T).unsqueeze(1), c.unsqueeze(1)], dim=1)
        pred = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        for i, head in enumerate(self.heads):
            mask = property_id == i
            if mask.any():
                pred[mask] = head(x[mask]).squeeze(-1).to(pred.dtype)  # cast: AMP may return half
        return pred, h.new_zeros(())
