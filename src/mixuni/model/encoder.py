"""Mixture encoder: one EGNN per component type, then a transformer over components.

    solvent graphs --EGNN--+
                           +--> mean over atoms --> [GIL] --> transformer --> mixture embedding
    salt graphs ----EGNN---+

Each atom carries its component's mole fraction as a feature (set during
preprocessing). During training the EGNN also runs on noisy coordinates and is
asked to recover the clean ones (coordinate denoising loss).
"""

from __future__ import annotations

import torch
from torch import nn

from .gil import GIL, graph_pairs
from .layers import BaseMLP, MixtureTransformer, principal_axes_frame, scatter_mean, to_sequences

NODE_DIM = 16       # atom features written by scripts/prepare_data.py
EDGE_DIM = 1        # bond feature
GIL_LAYERS = 3
GIL_ENTROPY_WEIGHT = 0.01
FRAME_NOISE = 0.3   # Å, jitter on the coordinates used to build GIL frames (also at eval time)


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp_msg = BaseMLP(2 * hidden_dim + EDGE_DIM + 1, hidden_dim, hidden_dim, last_act=True)
        self.mlp_pos = BaseMLP(hidden_dim, hidden_dim, 1)
        self.mlp_node_feat = BaseMLP(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index
        diff = pos[row] - pos[col]
        diff = diff / (diff.norm(dim=-1, keepdim=True) + 1)
        dist = diff.norm(dim=-1, keepdim=True) ** 2
        msg = self.mlp_msg(torch.cat([h[row], h[col], edge_attr, dist], dim=-1))
        pos = pos + scatter_mean(diff * self.mlp_pos(msg), row, h.size(0))
        h = self.mlp_node_feat(torch.cat([h, scatter_mean(msg, row, h.size(0))], dim=-1))
        return h, pos


class EGNN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Linear(NODE_DIM, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, pos, edge_index, edge_attr):
        h = self.embedding(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
        return h, pos


class MixtureEncoder(nn.Module):
    def __init__(self, *, hidden_dim, egnn_layers, mixture_layers, mixture_heads, max_components,
                 use_gil, gil_experts, denoise_sigma, denoise_weight):
        super().__init__()
        self.max_components = max_components
        self.use_gil = use_gil
        self.denoise_sigma = denoise_sigma
        self.denoise_weight = denoise_weight
        self.solvent_encoder = EGNN(hidden_dim, egnn_layers)
        self.salt_encoder = EGNN(hidden_dim, egnn_layers)
        self.gil_layers = nn.ModuleList(
            [GIL(hidden_dim, num_experts=gil_experts) for _ in range(GIL_LAYERS)] if use_gil else []
        )
        # The two extra inputs are |v| and <v, x> of the GIL vector state (zero without GIL).
        self.vector_project = nn.Linear(hidden_dim + 2, hidden_dim)
        self.transformer = MixtureTransformer(hidden_dim, mixture_layers, mixture_heads)

    def _denoising(self):
        return self.training and torch.is_grad_enabled() and self.denoise_weight > 0 and self.denoise_sigma > 0

    def _encode(self, egnn, graphs):
        """Atom embeddings and coordinates; plus the denoising loss in training."""
        pos = graphs.pos
        edge_attr = graphs.edge_attr.view(-1, EDGE_DIM)
        if not self._denoising():
            h, x = egnn(graphs.x, pos, graphs.edge_index, edge_attr)
            return h, x, None
        # One pass over [clean copy, noisy copy] of the batch.
        n = pos.size(0)
        noisy = pos + torch.randn_like(pos) * self.denoise_sigma
        h, x = egnn(
            torch.cat([graphs.x, graphs.x]),
            torch.cat([pos, noisy]),
            torch.cat([graphs.edge_index, graphs.edge_index + n], dim=1),
            torch.cat([edge_attr, edge_attr]),
        )
        loss = torch.norm(torch.nan_to_num(x[n:] - pos), p="fro") / max(n, 1) if n else h.new_zeros(())
        return h[:n], x[:n], loss

    def forward(self, batch):
        graphs, salt_graphs = batch["graphs"], batch["salt_graphs"]
        n2g, g2b = batch["n2g"], batch["g2b"]
        num_graphs = g2b.numel()
        batch_size = int(g2b.max()) + 1

        h_solv, x_solv, dn_solv = self._encode(self.solvent_encoder, graphs)
        h_salt, x_salt, dn_salt = self._encode(self.salt_encoder, salt_graphs)
        h_atom = torch.cat([h_solv, h_salt])
        x_atom = torch.cat([x_solv, x_salt])
        aux = (dn_solv + dn_salt) * self.denoise_weight if dn_solv is not None else h_atom.new_zeros(())

        h_graph = scatter_mean(h_atom, n2g, num_graphs)
        x_graph = scatter_mean(x_atom, n2g, num_graphs)
        v_graph = torch.zeros_like(x_graph)
        if self.use_gil:
            # GIL passes messages between the components of a mixture, in a
            # local frame per component built from its input coordinates. The
            # frames are jittered in eval mode as well, as in the reported runs,
            # so GIL predictions vary slightly from call to call.
            pos = torch.cat([graphs.pos, salt_graphs.pos])
            frames = principal_axes_frame(pos + torch.randn_like(pos) * FRAME_NOISE, n2g).detach()
            pairs = graph_pairs(g2b)
            entropy = []
            for layer in self.gil_layers:
                h_graph, v_graph, ent = layer(h_graph, x_graph, v_graph, pairs, frames)
                entropy.append(ent)
            aux = aux + sum(entropy[1:], entropy[0]) * GIL_ENTROPY_WEIGHT

        v_norm = v_graph.norm(dim=-1, keepdim=True)
        v_inner = (v_graph * x_graph).sum(dim=-1, keepdim=True)
        h_graph = self.vector_project(torch.cat([h_graph, v_norm, v_inner], dim=-1))
        seq, mask = to_sequences(h_graph, g2b, batch_size, self.max_components)
        return self.transformer(seq, mask), aux
