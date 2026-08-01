#!/usr/bin/env python3
"""Compare a boundary-direction correction on protected grid validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import direction_metrics, load_grid_arrays
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    load_boundary_direction_correction_runtime_v9,
    sha256,
)
from physics_structured_rebuild_v9.evaluate_forward_candidate_protected import (
    high_mask,
    read_jsonl,
)

DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    distance: np.ndarray,
    high: np.ndarray,
) -> dict[str, Any]:
    correct = prediction == truth
    exact = np.all(correct, axis=1)
    report = direction_metrics(truth, prediction, distance)
    return {
        "count": int(len(exact)),
        "joint_exact_count": int(exact.sum()),
        "joint_exact": float(exact.mean()),
        "high_complexity_count": int(high.sum()),
        "high_complexity_joint_count": int(exact[high].sum()),
        "high_complexity_joint_exact": float(exact[high].mean()),
        "equal_field_macro_f1": float(report["equal_field_macro_f1"]),
        "mean_field_accuracy": float(report["mean_field_accuracy"]),
        "per_field": report["per_field"],
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    candidate_path = args.candidate.resolve()
    candidate, artifact = load_boundary_direction_correction_runtime_v9(
        candidate_path,
        torch,
        device,
    )
    blocks = {}
    promotion = True
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        _, _, _, baseline = candidate.base_grid(rows)
        selected, applied = candidate.predict_grid(rows)
        baseline = baseline.reshape(-1, 5)
        selected = selected.reshape(-1, 5)
        applied = applied.reshape(-1, 5)
        truth = np.asarray(arrays.labels, dtype=np.int64)
        high = high_mask(arrays.group_count)
        baseline_metrics = metrics(
            truth,
            baseline,
            arrays.distance_bins,
            high,
        )
        candidate_metrics = metrics(
            truth,
            selected,
            arrays.distance_bins,
            high,
        )
        baseline_exact = np.all(baseline == truth, axis=1)
        candidate_exact = np.all(selected == truth, axis=1)
        block_passed = (
            candidate_metrics["joint_exact_count"]
            >= baseline_metrics["joint_exact_count"]
            and candidate_metrics["high_complexity_joint_count"]
            >= baseline_metrics["high_complexity_joint_count"]
            and candidate_metrics["equal_field_macro_f1"]
            >= baseline_metrics["equal_field_macro_f1"]
        )
        promotion = promotion and block_passed
        blocks[name] = {
            "current": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_only_joint_count": int(
                (candidate_exact & ~baseline_exact).sum()
            ),
            "current_only_joint_count": int(
                (baseline_exact & ~candidate_exact).sum()
            ),
            "oracle_union_joint_count": int(
                (candidate_exact | baseline_exact).sum()
            ),
            "correction_request_count": int(np.any(applied, axis=1).sum()),
            "correction_field_count": int(applied.sum()),
            "promotion_passed": bool(block_passed),
        }
        print(
            json.dumps(
                {"block": name, **blocks[name]},
                sort_keys=True,
            ),
            flush=True,
        )
    report = {
        "version": "boundary_direction_protected_v9_one_seed",
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "rules": artifact["rules"],
        },
        "blocks": blocks,
        "promotion_passed": bool(promotion),
        "source_contract": {
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
