"""Regression metrics, overall and per property. Non-finite entries are ignored."""

from __future__ import annotations

import math

import torch


def _pearson(x, y):
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    return float((x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12))


def _ranks(x):
    """Ordinal ranks (ties broken by position, not averaged)."""
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[torch.argsort(x)] = torch.arange(x.numel(), dtype=torch.float64)
    return ranks


def regression_metrics(pred, target) -> dict:
    pred, target = pred.reshape(-1).float(), target.reshape(-1).float()
    ok = torch.isfinite(pred) & torch.isfinite(target)
    pred, target = pred[ok], target[ok]
    n = pred.numel()
    if n == 0:
        return {"count": 0, "mae": math.nan, "rmse": math.nan, "pearson": math.nan, "spearman": math.nan}
    err = pred - target
    return {
        "count": n,
        "mae": float(err.abs().mean()),
        "rmse": float(torch.sqrt((err ** 2).mean())),
        "pearson": _pearson(pred, target) if n >= 2 else math.nan,
        "spearman": _pearson(_ranks(pred), _ranks(target)) if n >= 2 else math.nan,
    }


def summarize(pred, target, property_id, property_names) -> dict:
    per_property = {
        property_names[i]: regression_metrics(pred[property_id == i], target[property_id == i])
        for i in torch.unique(property_id).tolist()
    }
    maes = [m["mae"] for m in per_property.values() if m["count"] > 0 and math.isfinite(m["mae"])]
    return {
        "overall": regression_metrics(pred, target),
        "per_property": per_property,
        "macro_mae": sum(maes) / len(maes) if maes else math.nan,
    }
