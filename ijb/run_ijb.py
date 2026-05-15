#!/usr/bin/env python3
"""
IJB-C runner for state_dict / ONNX checkpoints (no torch.jit conversion needed).

Usage:
    # Place this next to ijb_evals.py in your FacePoison repo, then:
    python run_ijb.py
    # Edit DATA_PATH and EXPS below to point to your IJB-C dir and model paths.

What it does:
    For each entry in EXPS, runs:
      - 1:1 verification on IJB-C  -> TAR@FAR = 1e-3 (and other FAR levels)
      - 1:N identification on IJB-C -> TPIR@FPIR = 1e-2 / 1e-3 / 1e-4

    Supports two checkpoint kinds:
      * state_dict (.pt/.pth): built via ARCH_BUILDERS[arch]() + load_state_dict
      * ONNX (.onnx):         loaded via `onnx2torch.convert`
                              (pip install onnx2torch onnx)

Sharding:
    Uses LPT (Longest-Processing-Time) bin-packing on per-EXP cost so each shard
    gets roughly equal wall-time, not just an equal count of models.

Output: prints a markdown table per model and saves per-shard npz at end.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force a non-interactive matplotlib backend before ijb_evals imports pyplot,
# otherwise plot_roc_and_calculate_tpr's plt.show() blocks between models on
# a headless server.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

import gc

import numpy as np
import torch

from backbones import get_model
from backbones.iresnet import iresnet18, iresnet50, iresnet100  # noqa: F401

from ijb_evals import IJB_test, extract_IJB_data_11, get_embeddings, \
                      extract_gallery_prob_data, \
                      plot_roc_and_calculate_tpr  # noqa: E402


ARCH_BUILDERS = {
    "mbf":   lambda: get_model("mbf"),
    "r18":   iresnet18,
    "r50":   iresnet50,
    "r100":  iresnet100,
    "vit_b": lambda: get_model("vit_b"),
}


# ====== EDIT BELOW ======================================================
# Path to the IJB-C release that contains an `IJBC/` subfolder with
# `IJBC/meta/...` and `IJBC/loose_crop/...`.
DATA_PATH = ''   # TODO: path to IJB-C release directory (contains meta/ and loose_crop/)

SUBSET = "IJBC"          # "IJBB" also supported by the parent script
SAVE_DIR = "./IJB_result"

# Per-arch relative wall-time cost on IJB-C (~470K image forwards).
# Used by the LPT scheduler — tune if your hardware behaves differently.
# Roughly inverse to BATCH_SIZES, biased by FLOPs/sample.
ARCH_COST = {
    "mbf":   1.0,
    "r18":   1.5,
    "r50":   2.5,
    "r100":  4.0,
    "vit_b": 6.0,
}

# Per-arch batch size for IJB-C feature extraction, tuned for A100 (40GB+).
m = 2

BATCH_SIZES = {
    "mbf":   1024 * m,
    "r18":   768 * m,
    "r50":   512 * m,
    "r100":  384 * m,
    "vit_b": 256 * m,
}
DEFAULT_BATCH_SIZE = 256
DEFAULT_ONNX_COST = 3.0   # fallback if param-count estimation fails

# (param_count, cost) anchors used to estimate ONNX cost via log-space interp.
# Derived from the known archs in ARCH_COST.
_PARAM_COST_ANCHORS = [
    (  1_200_000, 1.0),   # MobileFaceNet
    ( 24_000_000, 1.5),   # iResNet-18
    ( 43_500_000, 2.5),   # iResNet-50
    ( 65_000_000, 4.0),   # iResNet-100
    ( 86_000_000, 6.0),   # ViT-Base
]

# Each entry is a dict.
#   Required: name (str), arch (one of ARCH_BUILDERS keys, or "onnx"),
#             ckpt (path to .pt / .pth / .onnx).
#   Optional: cost (float, for LPT scheduling), batch_size (per-model override).
EXPS = [
    # ---- state_dict checkpoint examples (PyTorch .pt / .pth) ----
    # {"name": "R50_clean",  "arch": "r50",
    #  "ckpt": "./workspace/R50_clean/model.pt"},
    # {"name": "R50_poison", "arch": "r50",
    #  "ckpt": "./workspace/R50_poison/model.pt"},

    # ---- ONNX checkpoint example (arch="onnx", optional cost / batch_size) ----
    # {"name": "R50_onnx", "arch": "onnx",
    #  "ckpt": "./workspace/R50.onnx",
    #  "cost": 2.5, "batch_size": 512},
]  # TODO: populate with your own (name, arch, ckpt) entries before running.
# ========================================================================


def _build_model(arch, ckpt_path, device):
    """Load either a state_dict checkpoint or an ONNX file as an nn.Module."""
    if arch == "onnx" or ckpt_path.endswith(".onnx"):
        try:
            from onnx2torch import convert
        except ImportError as e:
            raise ImportError(
                "ONNX checkpoint requested but onnx2torch not installed. "
                "Run: pip install onnx2torch onnx"
            ) from e
        model = convert(ckpt_path)
    else:
        if arch not in ARCH_BUILDERS:
            raise KeyError(
                f"Unknown arch '{arch}' (not in ARCH_BUILDERS, and ckpt "
                f"is not .onnx): {ckpt_path}"
            )
        model = ARCH_BUILDERS[arch]()
        sd = torch.load(ckpt_path, weights_only=True, map_location=device)
        model.load_state_dict(sd)
    return model.eval().to(device)


class TorchInterf:
    """Drop-in for Torch_model_interf — accepts state_dict or ONNX ckpts."""

    def __init__(self, arch, ckpt_path, device="cuda:0"):
        self.device = torch.device(device)
        self.model = _build_model(arch, ckpt_path, self.device)

    @torch.no_grad()
    def __call__(self, imgs):
        imgs = imgs.transpose(0, 3, 1, 2).copy().astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        x = torch.from_numpy(imgs).to(self.device).float()
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.detach().cpu().numpy()


# Backwards-compat alias for any external imports.
StateDictTorchInterf = TorchInterf


def build_ijb_test(arch, ckpt_path, data_path, subset="IJBC",
                   batch_size=128, force_reload=False):
    """Build an IJB_test instance using a state-dict or ONNX checkpoint."""
    interf = TorchInterf(arch, ckpt_path)

    # Bypass IJB_test.__init__'s model_file dispatch and recreate its state.
    tt = object.__new__(IJB_test)
    (templates, medias, p1, p2, label, img_names, landmarks, face_scores) = \
        extract_IJB_data_11(data_path, subset, force_reload=force_reload)
    tt.embs, tt.embs_f = get_embeddings(interf, img_names, landmarks,
                                        batch_size=batch_size)
    tt.data_path, tt.subset, tt.force_reload = data_path, subset, force_reload
    tt.templates, tt.medias = templates, medias
    tt.p1, tt.p2, tt.label = p1, p2, label
    tt.face_scores = face_scores.astype(tt.embs.dtype)
    return tt


# ----------------------- scheduling -----------------------

def _count_onnx_params(ckpt_path):
    """Sum the numel of every initializer in an ONNX graph (no weight load)."""
    import onnx
    model = onnx.load(ckpt_path, load_external_data=False)
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= int(d)
        total += n
    return total


def _estimate_onnx_cost(ckpt_path):
    """Estimate IJB-C wall-time cost for an ONNX model from its parameter count.

    Returns a tuple (cost, source) where source is 'estimated' on success or
    'default' if the ONNX file couldn't be parsed.
    """
    try:
        n_params = _count_onnx_params(ckpt_path)
    except Exception as e:
        print(f"[cost] could not parse {ckpt_path} ({e!r}); using default")
        return DEFAULT_ONNX_COST, "default"
    xs = np.log([p for p, _ in _PARAM_COST_ANCHORS])
    ys = np.array([c for _, c in _PARAM_COST_ANCHORS], dtype=float)
    cost = float(np.interp(np.log(max(n_params, 1)), xs, ys))
    return cost, f"~{n_params/1e6:.1f}M params"


# memoize so we only parse each ONNX file once across calls
_ONNX_COST_CACHE = {}


def _cost_of(exp):
    if "cost" in exp:
        return float(exp["cost"])
    arch = exp.get("arch", "")
    if arch == "onnx" or exp.get("ckpt", "").endswith(".onnx"):
        ckpt = exp["ckpt"]
        if ckpt not in _ONNX_COST_CACHE:
            if os.path.isfile(ckpt):
                _ONNX_COST_CACHE[ckpt] = _estimate_onnx_cost(ckpt)
            else:
                _ONNX_COST_CACHE[ckpt] = (DEFAULT_ONNX_COST, "missing-file")
        return _ONNX_COST_CACHE[ckpt][0]
    if arch not in ARCH_COST:
        raise KeyError(
            f"No cost for arch '{arch}' on exp '{exp['name']}'. "
            f"Add it to ARCH_COST or set exp['cost']."
        )
    return ARCH_COST[arch]


def _cost_source(exp):
    """Human-readable explanation for what cost we picked."""
    if "cost" in exp:
        return "explicit"
    arch = exp.get("arch", "")
    if arch == "onnx" or exp.get("ckpt", "").endswith(".onnx"):
        _cost_of(exp)  # populate cache
        return _ONNX_COST_CACHE[exp["ckpt"]][1]
    return f"arch={arch}"


def _batch_size_of(exp):
    if "batch_size" in exp:
        return int(exp["batch_size"])
    return BATCH_SIZES.get(exp.get("arch", ""), DEFAULT_BATCH_SIZE)


def lpt_schedule(exps, num_shards):
    """LPT (longest-processing-time-first) greedy bin packing.

    Sort jobs by cost desc, then assign each to the currently emptiest shard.
    For 10 jobs / 4 shards this is within 4/3 of optimal (Graham 1969) and
    in practice gives near-perfect balance for our cost distribution.
    """
    indexed = sorted(enumerate(exps), key=lambda ie: -_cost_of(ie[1]))
    bins = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for orig_i, exp in indexed:
        s = min(range(num_shards), key=lambda j: (loads[j], j))
        bins[s].append((orig_i, exp))
        loads[s] += _cost_of(exp)
    # restore original EXPS order within each shard (nicer logs)
    for b in bins:
        b.sort(key=lambda ie: ie[0])
    return bins, loads


def _print_schedule(bins, loads):
    print("[schedule] LPT partition across shards (cost-weighted):")
    for s, (b, load) in enumerate(zip(bins, loads)):
        items = ", ".join(
            f"{e['name']}({_cost_of(e):.1f}|{_cost_source(e)})"
            for _, e in b
        )
        print(f"  shard {s}: load={load:5.1f}  [{items}]")
    if loads:
        maxL, minL = max(loads), min(loads)
        imbalance = (maxL - minL) / maxL if maxL > 0 else 0.0
        print(f"  imbalance = {imbalance*100:.1f}%  "
              f"(max {maxL:.1f} / min {minL:.1f})")


# ----------------------- main -----------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0,
                   help="this process's shard id (0-indexed)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="total number of shards (= number of GPUs you use)")
    p.add_argument("--prewarm", action="store_true",
                   help="Pre-extract IJB metadata caches and exit (no GPU).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the LPT schedule and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.prewarm:
        print("[prewarm] extracting IJB-C 1:1 metadata...")
        extract_IJB_data_11(DATA_PATH, SUBSET, force_reload=False)
        print("[prewarm] extracting IJB-C 1:N gallery/probe metadata...")
        extract_gallery_prob_data(DATA_PATH, SUBSET, force_reload=False)
        print("[prewarm] done.")
        return

    bins, loads = lpt_schedule(EXPS, args.num_shards)
    if args.shard == 0 or args.dry_run:
        _print_schedule(bins, loads)
    if args.dry_run:
        return

    my_exps = [e for _, e in bins[args.shard]]
    print(f"\n[shard {args.shard}/{args.num_shards}] "
          f"load={loads[args.shard]:.1f}, {len(my_exps)} models:")
    for e in my_exps:
        print(f"  - {e['name']}  (cost={_cost_of(e):.1f}, bs={_batch_size_of(e)})")

    summary = {}
    for exp in my_exps:
        name = exp["name"]
        arch = exp.get("arch", "onnx")
        ckpt = exp["ckpt"]
        bs = _batch_size_of(exp)

        if not os.path.isfile(ckpt):
            print(f"[skip] {name}: ckpt not found -> {ckpt}")
            continue

        print(f"\n========== {name} ({arch}, BS={bs}) ==========")
        tt = build_ijb_test(arch, ckpt, DATA_PATH, subset=SUBSET, batch_size=bs)

        # ---- 1:1 verification -> TAR @ FAR levels ----
        score = tt.run_model_test_single()
        tpr_df, _ = plot_roc_and_calculate_tpr(
            [score], names=[name], label=tt.label
        )
        tar_at_1e3 = float(tpr_df.loc[name, 0.001]) if 0.001 in tpr_df.columns \
                     else None

        # ---- 1:N identification -> TPIR @ FPIR ----
        fars, tpirs, _, _ = tt.run_model_test_1N()
        fars = np.array(fars)
        tpirs = np.array(tpirs)
        def tpir_at(target):
            idx = int(np.argmin(np.abs(fars - target)))
            return float(tpirs[idx])

        summary[name] = {
            "TAR@FAR=1e-3":   tar_at_1e3,
            "TPIR@FPIR=1e-2": tpir_at(1e-2),
            "TPIR@FPIR=1e-3": tpir_at(1e-3),
            "TPIR@FPIR=1e-4": tpir_at(1e-4),
        }
        print(f"[summary] {name}: {summary[name]}")

        # Free GPU + figure memory before the next model on this shard.
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        del tt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = os.path.join(SAVE_DIR, f"summary_ijbc_shard{args.shard}.npz")
    np.savez(out, **{k: np.array(list(v.values())) for k, v in summary.items()})
    print(f"\nSaved summary -> {out}")

    print("\n=== FINAL (this shard) ===")
    print("| name | TAR@1e-3 | TPIR@1e-2 | TPIR@1e-3 | TPIR@1e-4 |")
    print("|---|---|---|---|---|")
    for name, v in summary.items():
        print(f"| {name} | "
              f"{v['TAR@FAR=1e-3']:.4f} | "
              f"{v['TPIR@FPIR=1e-2']:.4f} | "
              f"{v['TPIR@FPIR=1e-3']:.4f} | "
              f"{v['TPIR@FPIR=1e-4']:.4f} |")


if __name__ == "__main__":
    main()
