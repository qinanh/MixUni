"""Prediction heads.

All heads take the mixture embedding ``h``, temperature ``T`` (K), salt mole
fraction ``c`` and a learned property embedding ``p``.

``MLPHead`` is the expert used in the shared mixture-of-experts. The physics
heads predict log-conductivity through a temperature law whose parameters are
read from (h, p, c):

    arrhenius            log sigma = log A - B / T
    vtf                  log sigma = log A - B / (T - T0)
    stretched_arrhenius  log sigma = log A - B / T**gamma,   gamma in (0.5, 1.5)
    free_mlp             log sigma = log softplus(MLP(h, p, T, 1/T, c))   (no law; control)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import DeepMLP


def inv_t(T):
    return 1.0 / T.clamp_min(1e-6)


class MLPHead(nn.Module):
    def __init__(self, hidden_dim, property_dim):
        super().__init__()
        self.net = DeepMLP(hidden_dim + property_dim + 3, hidden_dim, 1)

    def forward(self, h, T, c, p):
        x = torch.cat([h, p, T.unsqueeze(1), inv_t(T).unsqueeze(1), c.unsqueeze(1)], dim=-1)
        return self.net(x).squeeze(-1)


class FreeMLPHead(MLPHead):
    def forward(self, h, T, c, p):
        sigma = F.softplus(super().forward(h, T, c, p)).clamp_min(1e-8)
        return sigma.log().clamp(-20.0, 20.0)


class ArrheniusHead(nn.Module):
    """log A and B > 0 from an MLP of (h, p, c); subclasses change the temperature term."""

    def __init__(self, hidden_dim, property_dim):
        super().__init__()
        self.shared = DeepMLP(hidden_dim + property_dim + 1, hidden_dim, hidden_dim)
        self.head_logA = nn.Linear(hidden_dim, 1)
        self.head_B = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def exponent(self, z, B, T):
        return B / T.clamp_min(1e-6)

    def forward(self, h, T, c, p):
        z = self.shared(torch.cat([h, p, c.unsqueeze(1)], dim=-1))
        log_a = self.head_logA(z).squeeze(-1)
        B = self.head_B(z).squeeze(-1)
        return (log_a - self.exponent(z, B, T)).clamp(-20.0, 20.0)


class VTFHead(ArrheniusHead):
    T0_MAX = 180.0     # K
    DENOM_MIN = 10.0   # K, keeps T - T0 away from zero

    def __init__(self, hidden_dim, property_dim):
        super().__init__(hidden_dim, property_dim)
        self.head_T0 = nn.Linear(hidden_dim, 1)

    def exponent(self, z, B, T):
        T0 = torch.sigmoid(self.head_T0(z).squeeze(-1)) * self.T0_MAX
        return B / torch.clamp(T - T0, min=self.DENOM_MIN)


class StretchedArrheniusHead(ArrheniusHead):
    GAMMA_MIN, GAMMA_MAX = 0.5, 1.5

    def __init__(self, hidden_dim, property_dim):
        super().__init__(hidden_dim, property_dim)
        self.head_alpha = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())  # gamma in the paper

    def exponent(self, z, B, T):
        gamma = self.GAMMA_MIN + self.head_alpha(z).squeeze(-1) * (self.GAMMA_MAX - self.GAMMA_MIN)
        return B / T.clamp_min(1e-6).pow(gamma)


PHYSICS_HEADS = {
    "arrhenius": ArrheniusHead,
    "vtf": VTFHead,
    "stretched_arrhenius": StretchedArrheniusHead,
    "free_mlp": FreeMLPHead,
}
