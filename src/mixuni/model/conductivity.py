"""Physics-informed conductivity head.

For conductivity properties the shared head's prediction is blended with a
physics head (e.g. stretched Arrhenius). The blend is done in a "conductivity
space", z-scored log1p(sigma), where both predictions are on the same scale:

    z = w_base(h, T, c, p) * z_base + w_phys(h, T, c, p) * z_phys

The gate starts at ~98% base / 2% physics so that the randomly initialised
physics head does not dominate early training. The loss for conductivity
samples is computed in this space; metrics are reported back in the usual
z-scored sigma space. Other properties pass through unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .heads import PHYSICS_HEADS
from .moe import Router

CLIP = 8.0            # bound on conductivity-space values
GATE_INIT = (2.0, -2.0)  # initial gate logits (base, physics)


class ConductivityHead(nn.Module):
    def __init__(self, base, physics, *, hidden_dim, num_properties, property_dim, conductivity_ids):
        super().__init__()
        self.base = base
        self.prop_embedding = nn.Embedding(num_properties, property_dim)
        self.blend_gate = Router(hidden_dim, property_dim, num_experts=2, top_k=2)
        self.physics = PHYSICS_HEADS[physics](hidden_dim, property_dim)

        is_cond = torch.zeros(num_properties, dtype=torch.bool)
        is_cond[torch.as_tensor(list(conductivity_ids), dtype=torch.long)] = True
        self.register_buffer("is_conductivity", is_cond)
        # Training-split statistics; set by set_statistics() or loaded from a checkpoint.
        self.register_buffer("prop_mean", torch.zeros(num_properties))
        self.register_buffer("prop_std", torch.ones(num_properties))
        self.register_buffer("cond_log_mean", torch.zeros(num_properties))
        self.register_buffer("cond_log_std", torch.ones(num_properties))

        last = self.blend_gate.gate_net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(GATE_INIT))
            self.blend_gate.noise_linear.weight.zero_()
            self.blend_gate.noise_linear.bias.zero_()

    @torch.no_grad()
    def set_statistics(self, stats):
        self.prop_mean.copy_(torch.as_tensor(stats["prop_mean"], dtype=torch.float32))
        self.prop_std.copy_(torch.as_tensor(stats["prop_std"], dtype=torch.float32).clamp_min(1e-8))
        self.cond_log_mean.copy_(torch.as_tensor(stats["cond_log_mean"], dtype=torch.float32))
        self.cond_log_std.copy_(torch.as_tensor(stats["cond_log_std"], dtype=torch.float32).clamp_min(1e-8))

    @staticmethod
    def _clip(z):
        return torch.nan_to_num(z, nan=0.0, posinf=CLIP, neginf=-CLIP).clamp(-CLIP, CLIP)

    def _sigma_to_z(self, sigma, ids):
        log = torch.log1p(sigma.clamp_min(0.0))
        return self._clip((log - self.cond_log_mean[ids]) / self.cond_log_std[ids])

    def _z_to_norm(self, z, ids):
        """Conductivity space -> z-scored sigma."""
        log = self._clip(z) * self.cond_log_std[ids] + self.cond_log_mean[ids]
        sigma = torch.expm1(log.clamp(-20.0, 20.0)).clamp_min(0.0)
        return (sigma - self.prop_mean[ids]) / self.prop_std[ids]

    def loss_targets(self, property_id, y, y_raw):
        mask = self.is_conductivity[property_id]
        if not mask.any():
            return y
        y = y.clone()
        y[mask] = self._sigma_to_z(y_raw[mask], property_id[mask])
        return y

    def forward(self, h, T, c, property_id):
        pred, aux = self.base(h, T, c, property_id)
        mask = self.is_conductivity[property_id]
        if not mask.any():
            return pred, pred, aux

        ids = property_id[mask]
        h_c, T_c, c_c = h[mask], T[mask], c[mask]
        p = self.prop_embedding(ids)
        _, _, logits = self.blend_gate(h_c, T_c, c_c, p)
        z_base = self._sigma_to_z(pred[mask] * self.prop_std[ids] + self.prop_mean[ids], ids)
        z_phys = self._sigma_to_z(torch.exp(self.physics(h_c, T_c, c_c, p).clamp(-20.0, 20.0)), ids)
        w = F.softmax(logits, dim=-1)

        out = pred.clone()
        out[mask] = self._clip(w[:, 0] * z_base + w[:, 1] * z_phys)
        for_metrics = out.clone()
        for_metrics[mask] = self._z_to_norm(out[mask], ids)
        return out, for_metrics, aux
