#!/usr/bin/env python
"""Build the processed datasets from the CheMixHub CSVs and the coordinate caches.

    python scripts/prepare_data.py                    # all six properties
    python scripts/prepare_data.py IL_conductivity    # one property

Reads data/raw/chemixhub/ (layout in data/README.md) and writes
data/processed/<property>.pt. Check the result with scripts/check_data.py.
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]

# dataset folder, row filter on the "property" column, coordinate cache, temperature column,
# whether the mixtures contain salts, target transform, and the kept range of the raw value.
PROPERTIES = {
    "IL_conductivity": dict(folder="ionic-liquids", filter="Electrical conductivity", cache="ionic-liquids",
                            temp_col="Temperature, K", salts=True, transform="il_cond", range=(-6.0, 3.0)),
    "IL_viscosity": dict(folder="ionic-liquids", filter="Viscosity", cache="ionic-liquids",
                         temp_col="Temperature, K", salts=True, transform=None, range=(-12.0, 12.0)),
    "PE_conductivity": dict(folder="polymer-electrolyte", filter="", cache="polymer-electrolyte",
                            temp_col="Temperature, K", salts=True, transform="pe_cond", range=(-10.0, 3.0)),
    "NIST_viscosity": dict(folder="nist-logV", filter="", cache="nist-logV",
                           temp_col="T", salts=False, transform=None, range=(-8.0, 10.0)),
    # logV was built with an earlier NIST-logV cache, looked up by compound id (see data/README.md).
    "logV_viscosity": dict(folder="logV", filter="", cache="nist-logV", cache_dir="coord_data_logv",
                           temp_col="T", salts=False, transform=None, range=(-8.0, 10.0)),
    "MiscSolv_density": dict(folder="miscible-solvent", filter="Density", cache="miscible-solvent",
                             temp_col="", salts=False, transform=None, range=(0.5, 2.5)),
}

# atomic number -> one-hot slot
ELEMENTS = {1: 0, 3: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 14: 7, 15: 8, 16: 9, 17: 10, 33: 11, 35: 12, 53: 13}
CARBON = 6


def compound_id(value) -> str:
    """CSV ids may come as 12, '12' or 12.0; use '12' for all of them."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if math.isfinite(number) and number.is_integer() else text


def atom_features(atomic_numbers):
    """[Z / 10, mole fraction (filled in per mixture), one-hot element]."""
    feats = []
    for z in atomic_numbers:
        one_hot = [0.0] * len(ELEMENTS)
        if int(z) in ELEMENTS:
            one_hot[ELEMENTS[int(z)]] = 1.0
        feats.append([int(z) / 10.0, 0.0] + one_hot)
    return torch.tensor(feats, dtype=torch.float32)


def load_structures(npz_path) -> dict:
    """compound id -> graph template (one conformer, 6 Å radius graph, distances as edge features)."""
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for name, coords, numbers, edge_index, edge_attr in zip(
        data["name_list"], data["coordinates"], data["atomic_numbers"], data["edge_index"], data["edge_attr"]
    ):
        edge_index = np.array([np.asarray(edge_index[0], dtype=np.int64), np.asarray(edge_index[1], dtype=np.int64)])
        edge_attr = np.asarray(list(edge_attr), dtype=np.float64)
        if edge_index.size == 0 or edge_attr.size == 0:
            continue  # single atoms have no edges; such compounds are dropped
        out[compound_id(name)] = {
            "x": atom_features(np.asarray(numbers, dtype=np.int64)),
            "pos": torch.tensor(np.asarray(coords, dtype=np.float64), dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        }
    return out


def component_graph(template, name, mole_fraction):
    x = template["x"].clone()
    x[:, 1] = float(mole_fraction)
    return Data(x=x, pos=template["pos"], edge_index=template["edge_index"], edge_attr=template["edge_attr"], name=name)


def dummy_salt():
    """Salt-free mixtures get a single carbon-like placeholder atom as their salt graph."""
    one_hot = [0.0] * len(ELEMENTS)
    one_hot[ELEMENTS[CARBON]] = 1.0
    return Data(x=torch.tensor([[0.6, 0.0] + one_hot]), pos=torch.zeros(1, 3),
                edge_index=torch.tensor([[0], [0]]), edge_attr=torch.tensor([0.0]), name="dummy_salt")


def transform(value, kind):
    """CheMixHub stores log10 conductivity; the model predicts sigma in mS/cm."""
    if kind == "il_cond":
        return 10 ** value * 10.0
    if kind == "pe_cond":
        return 10 ** value * 1000.0
    return float(value)


def build(name, raw_dir: Path):
    spec = PROPERTIES[name]
    csv_dir = raw_dir / "datasets" / spec["folder"] / "processed_data"
    df = pd.read_csv(sorted(csv_dir.glob("processed_*.csv"))[0])
    if spec["filter"]:
        df = df[df["property"].str.contains(spec["filter"], case=False, na=False)]
    df = df.reset_index(drop=True)

    compounds = pd.read_csv(csv_dir / "compounds.csv")
    salt_ids = set()
    if spec["salts"] and "salt" in compounds.columns:
        salt_ids = set(compounds[compounds["salt"] == 1]["compound_id"].map(compound_id))
    structures = load_structures(raw_dir / spec.get("cache_dir", "coord_data") / f"coords-chemixhub-{spec['cache']}.npz")
    temp_col = next((c for c in df.columns if spec["temp_col"] and
                     (c == spec["temp_col"] or spec["temp_col"].lower() in c.lower())), None)
    lo, hi = spec["range"]
    placeholder = dummy_salt()

    samples, skipped = [], 0
    for row_idx, row in df.iterrows():
        value = row.get("value")
        if pd.isna(value) or not lo <= float(value) <= hi:
            skipped += 1
            continue
        ids = [compound_id(i) for i in ast.literal_eval(row["cmp_ids"])]
        fracs = [float(f) for f in ast.literal_eval(row["cmp_mole_fractions"])]
        if len(ids) != len(fracs):
            skipped += 1
            continue

        solvents, salts, ratios = [], [], []
        c = 0.0 if spec["salts"] else None   # salt mole fraction
        for cid, frac in zip(ids, fracs):
            if cid not in structures:
                continue
            graph = component_graph(structures[cid], cid, frac)
            if spec["salts"] and cid in salt_ids:
                salts.append(graph)
                c = frac
            else:
                solvents.append(graph)
                ratios.append(frac)
        if not solvents:
            skipped += 1
            continue

        y = transform(float(value), spec["transform"])
        if spec["transform"] and y <= 0:
            skipped += 1
            continue
        if temp_col is not None and pd.notna(row.get(temp_col)):
            T = float(row[temp_col])
        else:
            T = None if spec["temp_col"] else 298.15

        samples.append({
            "sample_id": f"chemixhub_{name}_{int(row_idx):08d}",
            "property_name": name,
            "y_raw": float(y),
            "T": T,
            "c": c,
            "graphs": solvents,
            "salt_graphs": salts or [placeholder.clone()],
            "ratios": torch.tensor(ratios, dtype=torch.float32),
        })
    return samples, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("properties", nargs="*", help=f"any of {', '.join(PROPERTIES)} (default: all)")
    parser.add_argument("--raw-dir", default=str(ROOT / "data/raw/chemixhub"))
    parser.add_argument("--out-dir", default=str(ROOT / "data/processed"))
    args = parser.parse_args()
    unknown = set(args.properties) - set(PROPERTIES)
    if unknown:
        parser.error(f"unknown properties: {sorted(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.properties or PROPERTIES:
        samples, skipped = build(name, Path(args.raw_dir))
        path = out_dir / f"{name.lower()}.pt"
        torch.save({"property": name, "samples": samples, "skipped_rows": skipped}, path)
        print(f"{name}: {len(samples)} samples ({skipped} rows skipped) -> {path}")


if __name__ == "__main__":
    main()
