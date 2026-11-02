"""Experiment configs.

A config is a short YAML file. It may start with ``base: <path>``: the base
config is loaded first and the keys given here override it (nested mappings
are merged, everything else is replaced). ``configs/mixuni.yaml`` sets every
option, and all other configs derive from it.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

KEYS = {
    "data": {"split", "batch_size", "balanced_sampling"},
    "model": {
        "hidden_dim", "egnn_layers", "mixture_layers", "mixture_heads", "max_components",
        "use_gil", "gil_experts", "denoise_sigma", "denoise_weight",
        "shared_head", "num_experts", "top_k", "property_dim", "physics_head",
    },
    "train": {
        "seed", "epochs", "lr", "weight_decay", "scheduler", "grad_clip",
        "loss_weighting", "pcgrad", "aux_weight", "amp",
    },
}
CHOICES = {
    ("model", "shared_head"): {"moe", "per_property"},
    ("model", "physics_head"): {"stretched_arrhenius", "arrhenius", "vtf", "free_mlp", "none"},
    ("train", "scheduler"): {"cosine", "none"},
    ("train", "loss_weighting"): {"uncertainty", "plain", "gradnorm"},
}


def _merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    return _merge(_read((path.parent / base).resolve()), cfg)


def run_name(path) -> str:
    """``configs/ablations/no_moe.yaml`` -> ``ablations/no_moe``."""
    path = Path(path).resolve()
    try:
        return path.relative_to(CONFIG_DIR).with_suffix("").as_posix()
    except ValueError:
        return path.stem


def validate(cfg: dict) -> dict:
    if not cfg.get("properties"):
        raise ValueError("config must list at least one property")
    for section, keys in KEYS.items():
        given = set(cfg.get(section, {}))
        if given - keys:
            raise ValueError(f"unknown {section} keys: {sorted(given - keys)}")
        if keys - given:
            raise ValueError(f"missing {section} keys: {sorted(keys - given)}")
    for (section, key), allowed in CHOICES.items():
        if cfg[section][key] not in allowed:
            raise ValueError(f"{section}.{key} must be one of {sorted(allowed)}, got {cfg[section][key]!r}")
    return cfg


def load_config(path) -> dict:
    cfg = _read(Path(path).resolve())
    cfg["name"] = run_name(path)
    return validate(cfg)


def save_config(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def conductivity_ids(properties) -> list[int]:
    return [i for i, name in enumerate(properties) if name.endswith("_conductivity")]
