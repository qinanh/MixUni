#!/usr/bin/env python
"""Check rebuilt processed files against data/fingerprints.json.

    python scripts/check_data.py                  # all files
    python scripts/check_data.py --identity-only  # sample identity and order only

The split files index samples by position, so a rebuild is only usable if it
has the same samples in the same order. For every file this compares
  - identity: the ordered (sample id, property, component ids, ratios, T, c, y_raw), and
  - graphs:   node features, coordinates and edges of every component graph,
with floats rounded (settings in the JSON). Exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _round(value, digits):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    text = format(value, f".{digits}g")
    return "0" if float(text) == 0.0 else text


def identity_record(sample, digits) -> bytes:
    ratios = sample.get("ratios")
    ratios = None if ratios is None else [_round(v, digits) for v in torch.as_tensor(ratios).reshape(-1).tolist()]
    record = [
        str(sample["sample_id"]),
        str(sample["property_name"]),
        0,  # property id within a single-property file
        [str(getattr(g, "name", "")) for g in sample["graphs"]],
        [str(getattr(g, "name", "")) for g in sample["salt_graphs"]],
        ratios,
        _round(sample.get("T"), digits),
        _round(sample.get("c"), digits),
        _round(sample.get("y_raw"), digits),
    ]
    return json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def graph_record(sample, decimals) -> bytes:
    parts = []
    for group in ("graphs", "salt_graphs"):
        parts.append(f"{group}:{len(sample[group])};".encode("ascii"))
        for graph in sample[group]:
            for key in ("x", "pos", "edge_index", "edge_attr"):
                arr = np.asarray(getattr(graph, key).numpy())
                if arr.dtype.kind == "f":
                    arr = np.rint(arr.astype(np.float64) * 10.0 ** decimals)
                    arr[arr == 0] = 0.0  # fold -0.0
                arr = arr.astype(np.int64)
                parts.append(f"{key}:{'x'.join(map(str, arr.shape))};".encode("ascii"))
                parts.append(np.ascontiguousarray(arr).astype("<i8").tobytes())
    return b"".join(parts)


def check_file(path, expected, settings, with_graphs) -> bool:
    samples = torch.load(path, map_location="cpu", weights_only=False)["samples"]
    width = settings["sample_digest_bytes"]
    want_digests = base64.b64decode(expected["sample_digests"])
    identity, graphs = hashlib.sha256(), hashlib.sha256()
    first_bad = None
    for i, sample in enumerate(samples):
        record = identity_record(sample, settings["sig_digits"])
        identity.update(record + b"\n")
        if first_bad is None and hashlib.sha256(record).digest()[:width] != want_digests[i * width:(i + 1) * width]:
            first_bad = (i, record.decode())
        if with_graphs:
            g = graph_record(sample, settings["graph_decimals"])
            graphs.update(len(g).to_bytes(8, "little") + g)

    ok = len(samples) == expected["count"] and identity.hexdigest() == expected["identity_sha256"]
    if ok:
        print(f"[ok]   {path.name}: {len(samples)} samples, identity matches")
    else:
        print(f"[FAIL] {path.name}: identity mismatch ({len(samples)} samples, expected {expected['count']})")
        if first_bad:
            print(f"       first differing sample {first_bad[0]}: {first_bad[1]}")
    if with_graphs:
        if graphs.hexdigest() == expected["graph_sha256"]:
            print(f"[ok]   {path.name}: graphs match")
        else:
            print(f"[FAIL] {path.name}: graph features/coordinates differ")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "data/processed"))
    parser.add_argument("--identity-only", action="store_true",
                        help="skip the graph hash (e.g. for coordinate caches you generated yourself)")
    args = parser.parse_args()

    stored = json.loads((ROOT / "data/fingerprints.json").read_text())
    failed = 0
    for name, expected in stored["files"].items():
        path = Path(args.data_dir) / name
        if not path.exists():
            print(f"[FAIL] {name}: not found in {args.data_dir}")
            failed += 1
            continue
        failed += not check_file(path, expected, stored["settings"], with_graphs=not args.identity_only)
    print("all files match" if not failed else f"{failed} file(s) differ")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
