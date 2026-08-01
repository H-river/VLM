#!/usr/bin/env python3
"""Compare the clean full-data direction model with the accepted hybrid."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    load_boundary_direction_correction_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from specialist_rebuild_v2.common import read_jsonl

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_RUN / "boundary_direction_correction_state_v9.pkl",
    )
    parser.add_argument(
        "--forward",
        type=Path,
        default=DEFAULT_RUN / "full_basis_forward_surface_v9.pt",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_RUN / "clean_full_direction_v9.pkl",
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
    current, _ = load_boundary_direction_correction_runtime_v9(
        args.current.resolve(), torch, device
    )
    forward, _ = load_full_basis_forward_surface_runtime_v9(
        args.forward.resolve(), torch, device
    )
    with args.candidate.resolve().open("rb") as stream:
        artifact = pickle.load(stream)
    models = list(artifact["models"])
    blocks = {}
    passed = True
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        _, _, _, current_grid = current.base_grid(rows)
        current_labels = current_grid.reshape(-1, 5)
        prior, correction = forward.predict_correction(rows)
        full_prediction = (
            prior
            + correction * forward.field_blend[None, None, :]
        )
        features = np.concatenate(
            [
                arrays.features,
                prior.reshape(-1, 5),
                full_prediction.reshape(-1, 5),
            ],
            axis=1,
        ).astype(np.float32)
        candidate_labels = np.column_stack(
            [model.predict(features) for model in models]
        ).astype(np.int8)
        current_metric = metric(current_labels, arrays.labels)
        candidate_metric = metric(candidate_labels, arrays.labels)
        block_pass = (
            candidate_metric["all_five_count"]
            >= current_metric["all_five_count"]
        )
        passed = passed and block_pass
        current_success = np.all(current_labels == arrays.labels, axis=1)
        candidate_success = np.all(candidate_labels == arrays.labels, axis=1)
        blocks[name] = {
            "current": current_metric,
            "candidate": candidate_metric,
            "candidate_only_success_count": int(
                (candidate_success & ~current_success).sum()
            ),
            "current_only_success_count": int(
                (current_success & ~candidate_success).sum()
            ),
            "oracle_union_count": int(
                (current_success | candidate_success).sum()
            ),
            "promotion_passed": bool(block_pass),
        }
    report = {
        "version": "clean_full_direction_protected_v9_one_seed",
        "blocks": blocks,
        "promotion_passed": bool(passed),
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
