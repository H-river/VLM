#!/usr/bin/env python
"""
07_visual_compare.py
Side-by-side beam intensity comparison: LLM prediction vs ground truth.

Usage:
    python -m lang2setup.scripts.07_visual_compare [--n 20] [--predictions FILE]

Opens a matplotlib window with N pairs arranged in a grid.
Left = LLM prediction, Right = ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# ── project imports ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # optical_sim/
sys.path.insert(0, str(ROOT))

from src.optical_elements import setup_from_dict
from src.simulator import run_simulation


# ── helpers ──────────────────────────────────────────────────

def _load_base_config() -> dict:
    with open(ROOT / "configs" / "base_config.yaml") as f:
        return yaml.safe_load(f)


def _load_bins() -> dict:
    with open(ROOT / "lang2setup" / "configs" / "bins.yaml") as f:
        return yaml.safe_load(f)["bins"]


def _bin_to_physical(bin_val: int, axis_cfg: dict) -> float:
    """Convert a discrete bin index → physical value (center of bin)."""
    lo, hi, n = axis_cfg["min"], axis_cfg["max"], axis_cfg["num"]
    edges = np.linspace(lo, hi, n + 1)
    bin_val = max(0, min(n - 1, bin_val))
    return (edges[bin_val] + edges[bin_val + 1]) / 2.0


def _make_setup(base_cfg: dict, bins_cfg: dict,
                x_bin: int, y_bin: int, angle_bin: int) -> "OpticalSetup":
    """Build an OpticalSetup with alignment from bin values."""
    cfg = json.loads(json.dumps(base_cfg))          # deep copy
    cfg["alignment"]["x_offset"] = _bin_to_physical(x_bin, bins_cfg["x"])
    cfg["alignment"]["y_offset"] = _bin_to_physical(y_bin, bins_cfg["y"])
    cfg["alignment"]["tilt_x"]   = _bin_to_physical(angle_bin, bins_cfg["angle"])
    # Use a smaller grid for speed (256 is enough for visualisation)
    cfg.setdefault("simulation", {})["grid_size"] = 256
    return setup_from_dict(cfg)


def _simulate_intensity(base_cfg, bins_cfg, x, y, a) -> np.ndarray:
    setup = _make_setup(base_cfg, bins_cfg, x, y, a)
    result = run_simulation(setup)
    img = result["intensity"]
    return img


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20,
                        help="Number of comparison pairs")
    parser.add_argument("--predictions", type=str,
                        default=str(ROOT / "lang2setup" / "data" / "llm_predictions.jsonl"))
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of displaying")
    args = parser.parse_args()

    # Load configs
    base_cfg = _load_base_config()
    bins_cfg = _load_bins()

    # Load predictions
    records = []
    with open(args.predictions) as f:
        for line in f:
            records.append(json.loads(line))

    # Pick N diverse samples (spread evenly across the file)
    n = min(args.n, len(records))
    indices = np.linspace(0, len(records) - 1, n, dtype=int)
    samples = [records[i] for i in indices]

    print(f"▶ Simulating {n} pairs (prediction vs ground truth)...")

    # Layout: N rows × 2 cols (pred | gt)
    cols = 4                    # 2 pairs per row → 4 image cols
    pair_cols = cols // 2       # 2 pairs per row
    rows = int(np.ceil(n / pair_cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx, sample in enumerate(samples):
        pred = sample["prediction"]
        tgt  = sample["target"]
        text = sample["text"][:60]

        px, py, pa = pred["x_bin"], pred["y_bin"], pred["angle_bin"]
        tx, ty, ta = tgt["x_bin"], tgt["y_bin"], tgt["angle_bin"]

        # Simulate both
        img_pred = _simulate_intensity(base_cfg, bins_cfg, px, py, pa)
        img_gt   = _simulate_intensity(base_cfg, bins_cfg, tx, ty, ta)

        row = idx // pair_cols
        pair_in_row = idx % pair_cols
        c_pred = pair_in_row * 2
        c_gt   = pair_in_row * 2 + 1

        # Shared colour scale
        vmax = max(img_pred.max(), img_gt.max(), 1e-30)

        ax_p = axes[row, c_pred]
        ax_g = axes[row, c_gt]

        ax_p.imshow(img_pred, cmap="inferno", vmin=0, vmax=vmax, origin="lower")
        ax_g.imshow(img_gt,   cmap="inferno", vmin=0, vmax=vmax, origin="lower")

        match = "✓" if (px == tx and py == ty and pa == ta) else "✗"
        ax_p.set_title(f"#{idx+1} Pred ({px},{py},{pa})", fontsize=8)
        ax_g.set_title(f"GT ({tx},{ty},{ta}) {match}", fontsize=8)

        # Only show text on left side
        ax_p.set_ylabel(text, fontsize=6, rotation=0, labelpad=120,
                        va="center", ha="left")

        for ax in (ax_p, ax_g):
            ax.set_xticks([])
            ax.set_yticks([])

        print(f"  [{idx+1}/{n}] done  pred=({px},{py},{pa}) gt=({tx},{ty},{ta}) {match}")

    # Hide unused axes
    for idx in range(n, rows * pair_cols):
        row = idx // pair_cols
        pair_in_row = idx % pair_cols
        axes[row, pair_in_row * 2].axis("off")
        axes[row, pair_in_row * 2 + 1].axis("off")

    fig.suptitle("LLM Prediction vs Ground Truth – Beam Intensity", fontsize=13, y=1.0)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"  Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
