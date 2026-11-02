#!/usr/bin/env python
"""Evaluate a checkpoint on its own split and print per-property metrics.

    python scripts/evaluate.py checkpoints/mixuni/best.pt
    python scripts/evaluate.py artifacts/runs/mixuni/checkpoints/best.pt --splits val test --out metrics.json

Metrics are computed on z-scored targets (training-split mean/std), as during
training; Pearson and Spearman are unaffected by the scaling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mixuni.data import MixtureDataset, make_loader, make_splits, normalize_targets, target_statistics  # noqa: E402
from mixuni.model import load_model  # noqa: E402
from mixuni.train import build_weighting, evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out", help="also write the metrics as JSON")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    cfg = ckpt["config"]
    dataset = MixtureDataset(cfg["properties"], args.data_dir)
    splits = make_splits(dataset, cfg["data"]["split"])
    normalize_targets(dataset, target_statistics(splits["train"], cfg["properties"]))

    weighting = build_weighting(cfg["train"]["loss_weighting"], len(cfg["properties"]))
    if weighting is not None:
        if "weighting_state_dict" in ckpt:
            weighting.load_state_dict(ckpt["weighting_state_dict"])
        weighting.to(args.device)

    print(f"{cfg['name']}  epoch {ckpt['epoch']}  ({args.checkpoint})")
    results = {}
    for split in args.splits:
        loader = make_loader(splits[split], batch_size=cfg["data"]["batch_size"], num_workers=args.num_workers)
        results[split] = evaluate(model, loader, device=args.device, cfg=cfg["train"], weighting=weighting,
                                  property_names=cfg["properties"])
        print(f"\n[{split}]  {'property':<18}{'n':>8}{'MAE':>9}{'RMSE':>9}{'Pearson':>9}{'Spearman':>9}")
        rows = list(results[split]["per_property"].items()) + [("overall", results[split]["overall"])]
        for name, m in rows:
            print(f"{'':10}{name:<18}{m['count']:>8}{m['mae']:>9.4f}{m['rmse']:>9.4f}"
                  f"{m['pearson']:>9.4f}{m['spearman']:>9.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "epoch": ckpt["epoch"], **results},
                                             indent=1))


if __name__ == "__main__":
    main()
