"""
param_metrics.py
Parameter-level evaluation with tiered metric hierarchy.

Tier 1 (Physical Usefulness): within-2-bin accuracy, MAE, physical error
Tier 2 (Fine Control):        within-1-bin accuracy
Tier 3 (Classification):      exact-bin accuracy

The tiered view emphasizes that physical usefulness matters more than
exact bin classification. A prediction off by 1 bin (≈0.29 mm or ≈1.9 mrad)
is often physically acceptable.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


# Physical constants for converting bin errors to real units
# With 21 bins over ±3mm: bin_width = 6mm/21 ≈ 0.286 mm
# With 21 bins over ±20mrad: bin_width = 40mrad/21 ≈ 1.905 mrad
_BIN_WIDTH_MM = 6.0 / 21       # mm per bin (x, y)
_BIN_WIDTH_MRAD = 40.0 / 21    # mrad per bin (angle)


def evaluate_predictions(predictions: List[Dict[str, int]],
                         ground_truths: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Compute parameter-level metrics organized by tier.

    Returns dict with all metrics — callers can choose which to display.
    """
    n = len(predictions)
    assert n == len(ground_truths)

    fields = ["x_bin", "y_bin", "angle_bin"]
    results = {}

    for field in fields:
        preds = np.array([p[field] for p in predictions])
        golds = np.array([g[field] for g in ground_truths])
        errs = np.abs(preds - golds)

        results[f"{field}_accuracy"] = float(np.mean(errs == 0))
        results[f"{field}_mae"] = float(np.mean(errs))
        results[f"{field}_within_1"] = float(np.mean(errs <= 1))
        results[f"{field}_within_2"] = float(np.mean(errs <= 2))
        results[f"{field}_median_err"] = float(np.median(errs))

    # Joint metrics (all 3 fields)
    for label, threshold in [("exact", 0), ("within_1", 1), ("within_2", 2)]:
        joint = sum(
            all(abs(predictions[i][f] - ground_truths[i][f]) <= threshold for f in fields)
            for i in range(n)
        )
        results[f"joint_{label}"] = joint / n

    # v1 joint (x + angle only)
    for label, threshold in [("exact", 0), ("within_1", 1), ("within_2", 2)]:
        v1 = sum(
            abs(predictions[i]["x_bin"] - ground_truths[i]["x_bin"]) <= threshold and
            abs(predictions[i]["angle_bin"] - ground_truths[i]["angle_bin"]) <= threshold
            for i in range(n)
        )
        results[f"v1_{label}_x_angle"] = v1 / n

    # Physical error estimates (approximate, from bin MAE)
    results["est_x_err_mm"] = results["x_bin_mae"] * _BIN_WIDTH_MM
    results["est_y_err_mm"] = results["y_bin_mae"] * _BIN_WIDTH_MM
    results["est_angle_err_mrad"] = results["angle_bin_mae"] * _BIN_WIDTH_MRAD

    results["n"] = n
    return results


def print_eval_report(metrics: Dict[str, float], compact: bool = False) -> None:
    """
    Pretty-print evaluation results with tiered hierarchy.

    Tier 1 (Physical Usefulness) is shown first and most prominently.
    """
    n = int(metrics.get("n", 0))

    if compact:
        _print_compact(metrics, n)
        return

    print("=" * 62)
    print(f"  Evaluation Report ({n} samples)")
    print("=" * 62)

    # ── Tier 1: Physical Usefulness ──
    print("\n  ▸ Tier 1: Physical Usefulness (most important)")
    print("  " + "─" * 56)
    print(f"    {'Metric':<30} {'Value':>10}")
    print(f"    {'─'*28}   {'─'*10}")
    print(f"    {'joint within ±2 bins (3D)':<30} {metrics.get('joint_within_2', 0):>9.1%}")
    print(f"    {'joint within ±1 bin  (3D)':<30} {metrics.get('joint_within_1', 0):>9.1%}")
    print(f"    {'est. x error':<30} {metrics.get('est_x_err_mm', 0):>8.2f} mm")
    print(f"    {'est. y error':<30} {metrics.get('est_y_err_mm', 0):>8.2f} mm")
    print(f"    {'est. angle error':<30} {metrics.get('est_angle_err_mrad', 0):>7.2f} mrad")

    # ── Tier 2: Fine Control ──
    print("\n  ▸ Tier 2: Fine Control")
    print("  " + "─" * 56)
    for f in ["x_bin", "y_bin", "angle_bin"]:
        label = f.replace("_bin", "")
        w1 = metrics.get(f"{f}_within_1", 0)
        w2 = metrics.get(f"{f}_within_2", 0)
        mae = metrics.get(f"{f}_mae", 0)
        med = metrics.get(f"{f}_median_err", 0)
        print(f"    {label:<8}  ≤1: {w1:>5.1%}   ≤2: {w2:>5.1%}   MAE: {mae:>5.2f}   median: {med:>4.1f}")

    # ── Tier 3: Exact Classification ──
    print("\n  ▸ Tier 3: Exact Classification")
    print("  " + "─" * 56)
    for f in ["x_bin", "y_bin", "angle_bin"]:
        label = f.replace("_bin", "")
        acc = metrics.get(f"{f}_accuracy", 0)
        print(f"    {label:<8}  exact: {acc:>5.1%}")
    print(f"    {'joint':<8}  exact: {metrics.get('joint_exact', 0):>5.1%}")
    print(f"    {'v1(x,a)':<8}  exact: {metrics.get('v1_exact_x_angle', 0):>5.1%}")

    print("=" * 62)


def _print_compact(metrics: Dict[str, float], n: int) -> None:
    """One-line-per-tier compact view for comparing multiple methods."""
    jw2 = metrics.get("joint_within_2", 0)
    jw1 = metrics.get("joint_within_1", 0)
    je = metrics.get("joint_exact", 0)
    xe = metrics.get("est_x_err_mm", 0)
    ae = metrics.get("est_angle_err_mrad", 0)
    print(f"    n={n:>5}  | ±2: {jw2:>5.1%}  ±1: {jw1:>5.1%}  exact: {je:>5.1%}  | x̃≈{xe:.2f}mm  ã≈{ae:.1f}mrad")


def print_comparison_table(method_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a side-by-side comparison of multiple methods,
    organized with physical usefulness first.
    """
    names = list(method_metrics.keys())
    header = f"  {'Metric':<28}" + "".join(f" {n:>14}" for n in names)

    rows = [
        # Tier 1
        ("── Physical Usefulness ──", None),
        ("joint ≤2 bins (3D)", "joint_within_2"),
        ("joint ≤1 bin (3D)", "joint_within_1"),
        ("est. x error (mm)", "est_x_err_mm"),
        ("est. y error (mm)", "est_y_err_mm"),
        ("est. angle error (mrad)", "est_angle_err_mrad"),
        # Tier 2
        ("── Fine Control ──", None),
        ("x within ±1", "x_bin_within_1"),
        ("y within ±1", "y_bin_within_1"),
        ("angle within ±1", "angle_bin_within_1"),
        ("x MAE (bins)", "x_bin_mae"),
        ("y MAE (bins)", "y_bin_mae"),
        ("angle MAE (bins)", "angle_bin_mae"),
        # Tier 3
        ("── Exact Classification ──", None),
        ("x exact", "x_bin_accuracy"),
        ("y exact", "y_bin_accuracy"),
        ("angle exact", "angle_bin_accuracy"),
        ("joint exact (3D)", "joint_exact"),
    ]

    print("\n" + "=" * (30 + 15 * len(names)))
    print(header)
    print("=" * (30 + 15 * len(names)))

    for label, key in rows:
        if key is None:
            print(f"\n  {label}")
            continue
        vals = []
        for n in names:
            v = method_metrics[n].get(key, 0)
            if "mae" in key or "err" in key:
                vals.append(f"{v:>13.2f}")
            else:
                vals.append(f"{v:>13.1%}")
        print(f"  {label:<28}" + "".join(vals))

    print("=" * (30 + 15 * len(names)))
