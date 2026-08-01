#!/usr/bin/env python3
"""Measure direction classes induced by complete-basis forward predictions."""

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
from direction_rebuild_v4.data import labels_from_normalized_change, load_grid_arrays
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    load_boundary_direction_correction_runtime_v9,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from specialist_rebuild_v2.common import read_jsonl

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-direction",
        type=Path,
        default=DEFAULT_RUN / "boundary_direction_correction_state_v9.pkl",
    )
    parser.add_argument(
        "--current-forward",
        type=Path,
        default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_RUN / "full_basis_forward_surface_v9.pt",
    )
    parser.add_argument(
        "--old-validation",
        type=Path,
        default=REPO_ROOT.parent
        / "VLM_data/specialist_rebuild_v2/grids/val.jsonl",
    )
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=REPO_ROOT.parent
        / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    correct = prediction == target
    exact = np.all(correct, axis=1)
    return {
        "count": int(len(exact)),
        "all_five_count": int(exact.sum()),
        "all_five_exact": float(exact.mean()),
        "per_field_accuracy": correct.mean(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    torch, device = configure(int(args.seed), args.device)
    base_direction, _ = load_boundary_direction_correction_runtime_v9(
        args.base_direction.resolve(), torch, device
    )
    current_forward, _ = load_forward_selector_ensemble_runtime_v9(
        args.current_forward.resolve(), torch, device
    )
    candidate, _ = load_full_basis_forward_surface_runtime_v9(
        args.candidate.resolve(), torch, device
    )
    blocks = {}
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        _, _, _, hybrid_grid = base_direction.base_grid(rows)
        hybrid = hybrid_grid.reshape(-1, 5)
        current_threshold = labels_from_normalized_change(
            current_forward.predict_changes(rows).reshape(-1, 5)
        )
        candidate_threshold = labels_from_normalized_change(
            candidate.predict_changes(rows).reshape(-1, 5)
        )
        target = arrays.labels
        blocks[name] = {
            "accepted_direction": metric(hybrid, target),
            "accepted_forward_threshold": metric(current_threshold, target),
            "full_basis_threshold": metric(candidate_threshold, target),
            "oracle_union_accepted_direction_full_basis_count": int(
                (
                    np.all(hybrid == target, axis=1)
                    | np.all(candidate_threshold == target, axis=1)
                ).sum()
            ),
            "full_basis_only_success_count": int(
                (
                    np.all(candidate_threshold == target, axis=1)
                    & ~np.all(hybrid == target, axis=1)
                ).sum()
            ),
        }
    report = {
        "version": "full_basis_direction_diagnostic_v9_one_seed",
        "blocks": blocks,
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
