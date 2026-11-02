"""Small building blocks: MLPs, scatter helpers and the transformer pooling."""

from __future__ import annotations

import itertools

import torch
from torch import nn

try:
    from torch_scatter import scatter as _scatter
except ImportError:  # plain-torch fallback below
    _scatter = None


class BaseMLP(nn.Module):
    """Linear -> SiLU -> Linear (-> SiLU)."""

    def __init__(self, input_dim, hidden_dim, output_dim, last_act=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU() if last_act else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """``num_layers`` linear layers with SiLU in between."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def scatter_mean(src, index, dim_size):
    if _scatter is not None:
        return _scatter(src=src, index=index, dim=0, dim_size=dim_size, reduce="mean")
    out = src.new_zeros((dim_size,) + src.shape[1:]).index_add_(0, index, src)
    counts = src.new_zeros(dim_size).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / counts.clamp_min(1.0).view((dim_size,) + (1,) * (src.dim() - 1))


def to_sequences(features, sample_index, batch_size, seq_len):
    """Pack per-graph features into zero-padded per-sample sequences, keeping graph order."""
    counts = torch.bincount(sample_index, minlength=batch_size)
    if counts.numel() and int(counts.max()) > seq_len:
        raise ValueError(f"a mixture has {int(counts.max())} components, more than max_components={seq_len}")
    seq = features.new_zeros(batch_size, seq_len, features.size(-1))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=features.device)
    n = sample_index.numel()
    if n == 0:
        return seq, mask
    # slot of each graph within its sample, in order of appearance
    order = torch.argsort(sample_index.long() * n + torch.arange(n, device=sample_index.device))
    starts = counts.cumsum(0) - counts
    slots = torch.empty_like(sample_index)
    slots[order] = torch.arange(n, device=sample_index.device) - torch.repeat_interleave(starts, counts)
    flat = sample_index * seq_len + slots
    seq.view(-1, features.size(-1))[flat] = features
    mask.view(-1)[flat] = True
    return seq, mask


class MixtureTransformer(nn.Module):
    """Transformer over the components of each mixture, then a masked mean."""

    def __init__(self, hidden_dim, num_layers, num_heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=4 * hidden_dim,
            dropout=0.0, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.output_norm(pooled)


def principal_axes_frame(pos, index):
    """Per-graph frame from the principal axes of the centred coordinates.

    The sign of each axis is fixed by a majority vote over atoms (the sign
    pattern that puts most atoms at the smallest summed fractional coordinate).
    """
    n_graph = int(index.max()) + 1
    counts = torch.zeros(n_graph, dtype=pos.dtype, device=pos.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=pos.dtype))
    counts = counts.clamp_min(1.0)
    sums = torch.zeros(n_graph, 3, dtype=pos.dtype, device=pos.device)
    sums.scatter_add_(0, index.unsqueeze(1).expand(-1, 3), pos)
    centred = pos - (sums / counts.unsqueeze(1))[index]

    cov = torch.zeros(n_graph, 3, 3, dtype=pos.dtype, device=pos.device)
    for r in range(3):
        for c in range(3):
            cov[:, r, c].scatter_add_(0, index, centred[:, r] * centred[:, c])
    cov = cov / counts.view(-1, 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, dim=1, descending=True)
    bases = eigvecs.gather(2, order.unsqueeze(1).expand(-1, 3, -1))

    signs = torch.tensor(list(itertools.product([1, -1], repeat=3)), dtype=pos.dtype, device=pos.device)
    signed = bases.unsqueeze(1).repeat(1, 8, 1, 1) * signs.view(1, 8, 3, 1).expand(n_graph, -1, -1, 3)
    frac = torch.matmul(centred.unsqueeze(1).unsqueeze(-2), torch.inverse(signed)[index]).squeeze(-2)
    best = frac.sum(-1).min(-1).indices
    choice = torch.zeros(n_graph, dtype=torch.long, device=pos.device)
    for g in range(n_graph):
        votes = best[index == g]
        if votes.numel():
            vals, cnt = votes.unique(return_counts=True)
            choice[g] = vals[cnt.argmax()]
    return signed[torch.arange(n_graph, device=pos.device), choice]
