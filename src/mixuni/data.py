"""Processed datasets, splits, target normalization and batching.

Each property lives in its own processed file, ``<data_dir>/<property>.pt``
(written by ``scripts/prepare_data.py``). A sample is a dict holding the
component graphs of a mixture (solvents in ``graphs``, salts in
``salt_graphs``), temperature ``T``, salt mole fraction ``c`` and the raw
target ``y_raw``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

from .config import ROOT, conductivity_ids

DEFAULT_T = 298.15  # K, used when a sample has no temperature
DEFAULT_C = 0.0     # salt mole fraction for salt-free mixtures
RANDOM_SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
RANDOM_SPLIT_SEED = 42


class MixtureDataset(Dataset):
    """All samples of the requested properties, concatenated in config order."""

    def __init__(self, properties, data_dir="data/processed"):
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        self.properties = list(properties)
        self.samples = []
        for prop_id, name in enumerate(self.properties):
            payload = torch.load(data_dir / f"{name.lower()}.pt", map_location="cpu", weights_only=False)
            for sample in payload["samples"]:
                if sample["property_name"] != name:
                    raise ValueError(f"{name.lower()}.pt contains a {sample['property_name']} sample")
                self.samples.append({**sample, "property_id": prop_id, "y": sample["y_raw"]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _iter_samples(subset):
    return (subset.dataset[int(i)] for i in subset.indices)


def make_splits(dataset, split: str) -> dict:
    """``split`` is ``"random"`` (70/20/10, seed 42) or a JSON file with train/val/test indices."""
    if split != "random":
        path = Path(split) if Path(split).is_absolute() else ROOT / split
        indices = json.loads(path.read_text())
        return {name: Subset(dataset, indices[name]) for name in ("train", "val", "test")}

    # Largest-remainder rounding, so the three parts add up to len(dataset).
    n = len(dataset)
    total = sum(RANDOM_SPLIT.values())
    exact = [n * frac / total for frac in RANDOM_SPLIT.values()]
    lengths = [int(x) for x in exact]
    by_remainder = sorted(range(3), key=lambda i: exact[i] - lengths[i], reverse=True)
    for i in by_remainder[: n - sum(lengths)]:
        lengths[i] += 1
    generator = torch.Generator().manual_seed(RANDOM_SPLIT_SEED)
    return dict(zip(RANDOM_SPLIT, random_split(dataset, lengths, generator=generator)))


def _std_or_one(values: torch.Tensor) -> float:
    if values.numel() < 2:
        return 1.0
    std = float(values.std(unbiased=False))
    return std if math.isfinite(std) and std > 1e-8 else 1.0


def target_statistics(train_split, properties) -> dict:
    """Per-property mean/std of the raw targets on the training split.

    For conductivities we also store the mean/std of log1p(sigma), the space
    in which the conductivity head blends its two predictions.
    """
    values = {name: [] for name in properties}
    for sample in _iter_samples(train_split):
        values[sample["property_name"]].append(float(sample["y_raw"]))

    mean, std = [], []
    for name in properties:
        v = torch.tensor(values[name], dtype=torch.float32)
        mean.append(float(v.mean()) if v.numel() else 0.0)
        std.append(_std_or_one(v) if v.numel() else 1.0)

    log_mean, log_std = [0.0] * len(properties), [1.0] * len(properties)
    for i in conductivity_ids(properties):
        logs = torch.tensor([math.log1p(max(x, 0.0)) for x in values[properties[i]]], dtype=torch.float32)
        if logs.numel():
            log_mean[i], log_std[i] = float(logs.mean()), _std_or_one(logs)

    return {
        "property_names": list(properties),
        "prop_mean": mean,
        "prop_std": std,
        "cond_log_mean": log_mean,
        "cond_log_std": log_std,
    }


def normalize_targets(dataset, stats) -> None:
    """Set ``y`` = z-scored ``y_raw`` for every sample, using training-split statistics."""
    for sample in dataset.samples:
        i = sample["property_id"]
        sample["y"] = (float(sample["y_raw"]) - float(stats["prop_mean"][i])) / max(float(stats["prop_std"][i]), 1e-8)


def _condition(values, default):
    out = [default if v is None or not math.isfinite(float(v)) else float(v) for v in values]
    return torch.tensor(out, dtype=torch.float32)


def collate(samples) -> dict:
    """Batch samples. Node/graph/sample bookkeeping:

    n2g: node -> component graph, n2b: node -> sample, g2b: graph -> sample.
    Solvent graphs come first, then salt graphs.
    """
    from torch_geometric.data import Batch

    solvents, salts, g2b_solv, g2b_salt = [], [], [], []
    for i, sample in enumerate(samples):
        solvents += sample["graphs"]
        salts += sample["salt_graphs"]
        g2b_solv += [i] * len(sample["graphs"])
        g2b_salt += [i] * len(sample["salt_graphs"])

    graphs = Batch.from_data_list(solvents)
    salt_graphs = Batch.from_data_list(salts)
    n2g_solv = torch.arange(graphs.num_graphs).repeat_interleave(graphs.ptr[1:] - graphs.ptr[:-1])
    n2g_salt = torch.arange(salt_graphs.num_graphs).repeat_interleave(salt_graphs.ptr[1:] - salt_graphs.ptr[:-1])
    g2b_solv = torch.tensor(g2b_solv, dtype=torch.long)
    g2b_salt = torch.tensor(g2b_salt, dtype=torch.long)
    salt_offset = int(n2g_solv.max()) + 1 if n2g_solv.numel() else 0

    return {
        "graphs": graphs,
        "salt_graphs": salt_graphs,
        "n2g": torch.cat([n2g_solv, n2g_salt + salt_offset]),
        "n2b": torch.cat([g2b_solv[n2g_solv], g2b_salt[n2g_salt]]),
        "g2b": torch.cat([g2b_solv, g2b_salt]),
        "T": _condition([s.get("T") for s in samples], DEFAULT_T),
        "c": _condition([s.get("c") for s in samples], DEFAULT_C),
        "property_id": torch.tensor([s["property_id"] for s in samples], dtype=torch.long),
        "y": torch.tensor([float(s["y"]) for s in samples], dtype=torch.float32),
        "y_raw": torch.tensor([float(s["y_raw"]) for s in samples], dtype=torch.float32),
    }


def balanced_sampler(subset) -> WeightedRandomSampler:
    """Draw each property equally often (inverse-frequency weights, with replacement)."""
    props = [s["property_id"] for s in _iter_samples(subset)]
    counts = Counter(props)
    return WeightedRandomSampler([1.0 / counts[p] for p in props], num_samples=len(props), replacement=True)


def make_loader(subset, *, batch_size, train=False, balanced=False, num_workers=8) -> DataLoader:
    sampler = balanced_sampler(subset) if train and balanced else None
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=collate,
    )
