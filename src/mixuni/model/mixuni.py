"""MixUni = mixture encoder + shared head (+ physics-informed conductivity head)."""

from __future__ import annotations

import torch
from torch import nn

from ..config import conductivity_ids
from .conductivity import ConductivityHead
from .encoder import MixtureEncoder
from .moe import PerPropertyHeads, SharedMoE


class MixUni(nn.Module):
    def __init__(self, model_cfg: dict, properties: list[str]):
        super().__init__()
        m = model_cfg
        self.encoder = MixtureEncoder(
            hidden_dim=m["hidden_dim"],
            egnn_layers=m["egnn_layers"],
            mixture_layers=m["mixture_layers"],
            mixture_heads=m["mixture_heads"],
            max_components=m["max_components"],
            use_gil=m["use_gil"],
            gil_experts=m["gil_experts"],
            denoise_sigma=m["denoise_sigma"],
            denoise_weight=m["denoise_weight"],
        )
        n = len(properties)
        if m["shared_head"] == "moe":
            head = SharedMoE(m["hidden_dim"], n, m["property_dim"], m["num_experts"], m["top_k"])
        else:
            head = PerPropertyHeads(m["hidden_dim"], n, m["property_dim"])
        if m["physics_head"] != "none":
            head = ConductivityHead(
                head, m["physics_head"], hidden_dim=m["hidden_dim"], num_properties=n,
                property_dim=m["property_dim"], conductivity_ids=conductivity_ids(properties),
            )
        self.head = head

    def forward(self, batch) -> dict:
        """Returns the prediction used by the loss (``pred``), the prediction in
        z-scored target units (``pred_for_metrics``) and the auxiliary losses."""
        h, aux = self.encoder(batch)
        out = self.head(h, batch["T"], batch["c"], batch["property_id"])
        if isinstance(self.head, ConductivityHead):
            pred, for_metrics, head_aux = out
        else:
            (pred, head_aux), for_metrics = out, out[0]
        return {"pred": pred, "pred_for_metrics": for_metrics, "aux_loss": aux + head_aux}

    def loss_targets(self, batch):
        if isinstance(self.head, ConductivityHead):
            return self.head.loss_targets(batch["property_id"], batch["y"], batch["y_raw"])
        return batch["y"]


def build_model(config: dict, statistics: dict | None = None) -> MixUni:
    """Build from a config; ``statistics`` (training-split target stats) is needed
    when training from scratch and is restored from the weights otherwise."""
    model = MixUni(config["model"], config["properties"])
    if statistics is not None and isinstance(model.head, ConductivityHead):
        model.head.set_statistics(statistics)
    return model


def load_model(path, device="cpu"):
    """Load a checkpoint written by scripts/train.py; returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
