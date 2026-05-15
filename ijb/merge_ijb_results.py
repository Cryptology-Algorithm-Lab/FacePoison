#!/usr/bin/env python3
"""Merge per-shard IJB-C summaries into a single markdown table + merged npz.

Reads:  {SAVE_DIR}/summary_ijbc_shard*.npz   (one per shard)
Writes: {SAVE_DIR}/summary_ijbc_merged.npz   (all models)
Prints: markdown table on stdout, in EXPS-declaration order
"""

import glob
import os
import sys

import numpy as np

# Pull EXPS and SAVE_DIR from the runner so ordering matches the source of truth.
from run_ijb import EXPS, SAVE_DIR

METRICS = ["TAR@FAR=1e-3", "TPIR@FPIR=1e-2", "TPIR@FPIR=1e-3", "TPIR@FPIR=1e-4"]


def _load_shards(save_dir):
    paths = sorted(glob.glob(os.path.join(save_dir, "summary_ijbc_shard*.npz")))
    if not paths:
        raise SystemExit(f"No shard summaries found in {save_dir}/")
    merged = {}
    seen = {}
    for p in paths:
        d = np.load(p, allow_pickle=False)
        for k in d.files:
            if k in seen:
                print(f"[warn] duplicate model '{k}' in {p} (already from "
                      f"{seen[k]}); keeping first.", file=sys.stderr)
                continue
            merged[k] = {m: float(v) for m, v in zip(METRICS, d[k])}
            seen[k] = p
    return merged, paths


def main():
    merged, paths = _load_shards(SAVE_DIR)
    print(f"[merge] loaded {len(merged)} models from {len(paths)} shards",
          file=sys.stderr)

    out_npz = os.path.join(SAVE_DIR, "summary_ijbc_merged.npz")
    np.savez(out_npz, **{
        k: np.array([v[m] for m in METRICS]) for k, v in merged.items()
    })
    print(f"[merge] wrote merged npz -> {out_npz}", file=sys.stderr)

    print("| name | TAR@1e-3 | TPIR@1e-2 | TPIR@1e-3 | TPIR@1e-4 |")
    print("|---|---|---|---|---|")
    expected = set()
    for exp in EXPS:
        name = exp["name"]
        expected.add(name)
        v = merged.get(name)
        if v is None:
            print(f"| {name} | — | — | — | — |")
            continue
        print(f"| {name} | "
              f"{v['TAR@FAR=1e-3']:.4f} | "
              f"{v['TPIR@FPIR=1e-2']:.4f} | "
              f"{v['TPIR@FPIR=1e-3']:.4f} | "
              f"{v['TPIR@FPIR=1e-4']:.4f} |")

    # any extras that weren't in EXPS (e.g. you ran a subset) — append below
    extras = [k for k in merged if k not in expected]
    if extras:
        print("\n<!-- not in current EXPS list: -->")
        for name in extras:
            v = merged[name]
            print(f"| {name} | "
                  f"{v['TAR@FAR=1e-3']:.4f} | "
                  f"{v['TPIR@FPIR=1e-2']:.4f} | "
                  f"{v['TPIR@FPIR=1e-3']:.4f} | "
                  f"{v['TPIR@FPIR=1e-4']:.4f} |")


if __name__ == "__main__":
    main()
