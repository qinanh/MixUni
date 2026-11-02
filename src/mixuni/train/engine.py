"""Training and evaluation loops."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from .losses import PCGrad
from .metrics import summarize

mse = nn.MSELoss()


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v.to(device) for k, v in batch.items()}


def autocast(amp, device):
    if amp and torch.device(device).type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def compute_loss(model, batch, *, weighting, num_properties, aux_weight):
    """Loss on one batch, or None if the batch has no usable (finite) sample.

    ``weighting`` is None for the plain pooled MSE, otherwise a module that
    combines the per-property MSEs (see losses.py).
    """
    out = model(batch)
    pred = out["pred"].reshape(-1)
    target = model.loss_targets(batch).reshape(-1)
    metric_pred = out["pred_for_metrics"].reshape(-1)
    metric_target = batch["y"].reshape(-1)
    valid = torch.isfinite(pred) & torch.isfinite(target) & torch.isfinite(metric_pred) & torch.isfinite(metric_target)
    if not valid.any():
        return None

    aux = out["aux_loss"]
    aux = (aux if torch.isfinite(aux) else pred.new_zeros(())) * float(aux_weight)
    if weighting is None or num_properties <= 1:
        pred_loss = mse(pred[valid], target[valid])
    else:
        pid = batch["property_id"]
        losses, active = [], []
        for i in range(num_properties):
            mask = valid & (pid == i)
            active.append(bool(mask.any()))
            losses.append(mse(pred[mask], target[mask]) if active[-1] else pred.sum() * 0.0)
        pred_loss = weighting(losses, active)
        if pred_loss is None:
            return None

    loss = pred_loss + aux
    if not torch.isfinite(loss):
        return None
    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "aux_loss": aux.detach(),
        "valid": valid,
        "pred": metric_pred.detach(),
        "target": metric_target.detach(),
    }


class _Tally:
    """Accumulates losses and predictions over an epoch."""

    def __init__(self, property_names):
        self.names = property_names
        self.n = 0
        self.sums = {"loss": 0.0, "pred_loss": 0.0, "aux_loss": 0.0}
        self.pred, self.target, self.pid = [], [], []

    def add(self, info, batch):
        valid = info["valid"].detach().cpu()
        k = int(valid.sum())
        self.n += k
        for key in self.sums:
            self.sums[key] += float(info[key].detach().cpu()) * k
        self.pred.append(info["pred"].cpu()[valid])
        self.target.append(info["target"].cpu()[valid])
        self.pid.append(batch["property_id"].detach().cpu().reshape(-1)[valid])

    def summary(self):
        cat = lambda xs, dtype: torch.cat(xs) if xs else torch.empty(0, dtype=dtype)  # noqa: E731
        metrics = summarize(cat(self.pred, torch.float32), cat(self.target, torch.float32),
                            cat(self.pid, torch.long), self.names)
        for key, total in self.sums.items():
            metrics["overall"][key] = total / self.n if self.n else math.nan
        return metrics


def train_epoch(model, loader, optimizer, *, device, cfg, weighting, scaler, property_names):
    model.train()
    tally = _Tally(property_names)
    n_prop = len(property_names)
    kw = dict(weighting=weighting, num_properties=n_prop, aux_weight=cfg["aux_weight"])
    pcgrad = PCGrad(optimizer) if cfg["pcgrad"] else None
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if pcgrad is not None:
            # One forward/backward per property, then the projected step.
            def task_loss(i):
                out = model(batch)
                pred, target = out["pred"].reshape(-1), model.loss_targets(batch).reshape(-1)
                mask = (batch["property_id"] == i) & torch.isfinite(pred) & torch.isfinite(target)
                return mse(pred[mask], target[mask]) if mask.any() else None

            def aux_loss():
                aux = model(batch)["aux_loss"]
                return (aux if torch.isfinite(aux) else aux.new_zeros(())) * float(cfg["aux_weight"])

            pcgrad.step(task_loss, n_prop, model.parameters(), grad_clip=cfg["grad_clip"], extra_loss=aux_loss)
            with torch.no_grad():
                info = compute_loss(model, batch, **kw)
        else:
            with autocast(cfg["amp"], device):
                info = compute_loss(model, batch, **kw)
            if info is None:
                continue
            if scaler is not None:
                scaler.scale(info["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                info["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
        if info is not None:
            tally.add(info, batch)
    return tally.summary()


@torch.no_grad()
def evaluate(model, loader, *, device, cfg, weighting, property_names):
    model.eval()
    tally = _Tally(property_names)
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(cfg["amp"], device):
            info = compute_loss(model, batch, weighting=weighting, num_properties=len(property_names),
                                aux_weight=cfg["aux_weight"])
        if info is not None:
            tally.add(info, batch)
    return tally.summary()


def _lookup(record, path):
    for part in path.split("/"):
        if not isinstance(record, dict) or part not in record:
            return None
        record = record[part]
    return record


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, weighting, epoch, metrics, best, config, statistics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
        "statistics": statistics,
        "metrics": metrics,
        "best": best,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if weighting is not None:
        payload["weighting_state_dict"] = weighting.state_dict()
    torch.save(payload, path)


def fit(model, loaders, optimizer, scheduler, *, device, config, statistics, weighting, out_dir,
        monitor="val/overall/loss", monitor_mode="min", start_epoch=0, best=None, scaler=None):
    """Train for ``config["train"]["epochs"]`` epochs. Every epoch is evaluated on
    val and test; ``best.pt`` follows ``monitor`` and ``latest.pt`` is the last epoch."""
    cfg = config["train"]
    names = config["properties"]
    ckpt_dir = Path(out_dir) / "checkpoints"
    best = best or {"metric": monitor, "mode": monitor_mode, "value": None, "epoch": None}
    history = []
    for epoch in range(start_epoch, cfg["epochs"]):
        record = {"epoch": epoch}
        record["train"] = train_epoch(model, loaders["train"], optimizer, device=device, cfg=cfg,
                                      weighting=weighting, scaler=scaler, property_names=names)
        record["val"] = evaluate(model, loaders["val"], device=device, cfg=cfg, weighting=weighting,
                                 property_names=names)
        if scheduler is not None:
            scheduler.step()
        record["test"] = evaluate(model, loaders["test"], device=device, cfg=cfg, weighting=weighting,
                                  property_names=names)

        state = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, weighting=weighting,
                     epoch=epoch, metrics=record, config=config, statistics=statistics)
        value = _lookup(record, monitor)
        if value is not None and (best["value"] is None
                                  or (value < best["value"] if monitor_mode == "min" else value > best["value"])):
            best = {"metric": monitor, "mode": monitor_mode, "value": float(value), "epoch": epoch}
            save_checkpoint(ckpt_dir / "best.pt", best=best, **state)
        save_checkpoint(ckpt_dir / "latest.pt", best=best, **state)

        history.append(record)
        log = {"epoch": epoch, "best_epoch": best["epoch"]}
        for split in ("train", "val", "test"):
            log[f"{split}_loss"] = round(record[split]["overall"]["loss"], 4)
            log[f"{split}_pearson"] = {k: round(v["pearson"], 3) for k, v in record[split]["per_property"].items()}
        print(json.dumps(log), flush=True)
    return {"history": history, "best": best}
