#!/usr/bin/env python
"""Train a model from a config.

    python scripts/train.py configs/mixuni.yaml
    python scripts/train.py configs/mixuni.yaml --seed 43     # another training seed
    python scripts/train.py configs/mixuni.yaml --dry-run     # print the full config and exit

Outputs go to artifacts/runs/<config name>[_seed<N>]/: config.yaml, metrics.json
(per-epoch train/val/test metrics) and checkpoints/{best,latest}.pt.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from mixuni.config import load_config, save_config  # noqa: E402
from mixuni.data import MixtureDataset, make_loader, make_splits, normalize_targets, target_statistics  # noqa: E402
from mixuni.model import build_model  # noqa: E402
from mixuni.train import build_weighting, fit  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, help="override train.seed; appends _seed<N> to the run name")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--out", help="output directory (default artifacts/runs/<run name>)")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--monitor", default="val/overall/loss", help="metric that selects best.pt")
    parser.add_argument("--monitor-mode", default="min", choices=["min", "max"])
    parser.add_argument("--resume", action="store_true", help="continue from <out>/checkpoints/latest.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
        cfg["name"] += f"_seed{args.seed}"
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    out = Path(args.out) if args.out else ROOT / "artifacts/runs" / cfg["name"]
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False), end="")
        print(f"# output: {out}")
        return

    t = cfg["train"]
    seed_everything(t["seed"])
    save_config(cfg, out / "config.yaml")

    dataset = MixtureDataset(cfg["properties"], args.data_dir)
    splits = make_splits(dataset, cfg["data"]["split"])
    stats = target_statistics(splits["train"], cfg["properties"])
    normalize_targets(dataset, stats)

    model = build_model(cfg, stats)
    weighting = build_weighting(t["loss_weighting"], len(cfg["properties"]))
    if t["pcgrad"] and weighting is not None:
        raise ValueError("pcgrad needs loss_weighting: plain")
    params = list(model.parameters()) + (list(weighting.parameters()) if weighting is not None else [])
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"], eta_min=1e-5)
                 if t["scheduler"] == "cosine" else None)
    loaders = {
        name: make_loader(subset, batch_size=cfg["data"]["batch_size"], train=name == "train",
                          balanced=cfg["data"]["balanced_sampling"], num_workers=args.num_workers)
        for name, subset in splits.items()
    }

    device = torch.device(args.device)
    model.to(device)
    if weighting is not None:
        weighting.to(device)
    scaler = torch.amp.GradScaler("cuda") if t["amp"] and device.type == "cuda" else None

    start_epoch, best = 0, None
    if args.resume and (out / "checkpoints/latest.pt").exists():
        ckpt = torch.load(out / "checkpoints/latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for name, obj in (("scheduler", scheduler), ("scaler", scaler), ("weighting", weighting)):
            if obj is not None and f"{name}_state_dict" in ckpt:
                obj.load_state_dict(ckpt[f"{name}_state_dict"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"resuming at epoch {start_epoch}", file=sys.stderr)

    result = fit(model, loaders, optimizer, scheduler, device=device, config=cfg, statistics=stats,
                 weighting=weighting, out_dir=out, monitor=args.monitor, monitor_mode=args.monitor_mode,
                 start_epoch=start_epoch, best=best, scaler=scaler)
    result["statistics"] = stats
    result["split_sizes"] = {name: len(subset) for name, subset in splits.items()}
    with open(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"run": cfg["name"], "best": result["best"], "output": str(out)}))


if __name__ == "__main__":
    main()
