#!/usr/bin/env python3
"""Calibrate complete-basis field strengths on direct natural requests."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import tolerance_from_current
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.calibrate_system_direction import action_index
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    group_partitions,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import sha256
from physics_structured_rebuild_v9.train_full_basis_forward_surface import (
    choose_blend,
)
from specialist_rebuild_v2.common import STATE_FIELDS, read_jsonl

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_INPUT = DEFAULT_RUN / "full_basis_forward_surface_v9.pt"
DEFAULT_OUTPUT = DEFAULT_RUN / "full_basis_forward_direct_calibrated_v9.pt"
DEFAULT_DIRECT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direct-data", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument(
        "--current", type=Path, default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.abs(prediction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return {
        "count": int(len(exact)),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": passed.mean(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    direct_path = args.direct_data.resolve()
    direct = read_jsonl(direct_path)
    ids = np.asarray([str(row["group_id"]) for row in direct], dtype=np.str_)
    rows = [
        {
            "group_id": str(row["group_id"]),
            "setup": row["setup"],
            "current_beam_state": row["current_beam_state"],
        }
        for row in direct
    ]
    actions = np.asarray(
        [action_index(row["action"]) for row in direct], dtype=np.int64
    )
    positions = np.arange(len(rows), dtype=np.int64)
    target = np.asarray(
        [
            np.asarray(
                [float(row["truth_change"][field]) for field in STATE_FIELDS],
                dtype=np.float32,
            )
            / tolerance_from_current(row["current_beam_state"])
            for row in direct
        ],
        dtype=np.float32,
    )
    torch, device = configure(int(args.seed), args.device)
    input_path = args.input.resolve()
    candidate, artifact = load_full_basis_forward_surface_runtime_v9(
        input_path, torch, device
    )
    prior_grid, correction_grid = candidate.predict_correction(rows)
    prior = prior_grid[positions, actions]
    correction = correction_grid[positions, actions]
    current, _ = load_forward_selector_ensemble_runtime_v9(
        args.current.resolve(), torch, device
    )
    current_prediction = current.predict_changes(rows)[positions, actions]
    calibration, selector_calibration, confirmation = group_partitions(
        ids, int(args.seed)
    )
    blend, trace = choose_blend(
        prior[calibration, None, :],
        correction[calibration, None, :],
        target[calibration, None, :],
    )
    candidate_prediction = prior + correction * blend[None, :]
    splits = {}
    for name, indices in (
        ("candidate_calibration", calibration),
        ("selector_calibration", selector_calibration),
        ("internal_confirmation", confirmation),
    ):
        current_success = np.all(
            np.abs(current_prediction[indices] - target[indices]) <= 1.0,
            axis=1,
        )
        candidate_success = np.all(
            np.abs(candidate_prediction[indices] - target[indices]) <= 1.0,
            axis=1,
        )
        splits[name] = {
            "base_v7": metric(prior[indices], target[indices]),
            "current_accepted": metric(
                current_prediction[indices], target[indices]
            ),
            "candidate": metric(
                candidate_prediction[indices], target[indices]
            ),
            "oracle_union_with_current_count": int(
                (current_success | candidate_success).sum()
            ),
            "candidate_only_success_count": int(
                (candidate_success & ~current_success).sum()
            ),
            "current_only_success_count": int(
                (current_success & ~candidate_success).sum()
            ),
        }
    calibrated = copy.deepcopy(artifact)
    calibrated["version"] = "full_basis_forward_direct_calibrated_v9_one_seed"
    calibrated["field_blend"] = blend
    calibrated["direct_calibration"] = {
        "data": str(direct_path),
        "data_sha256": sha256(direct_path),
        "group_count": int(len(calibration)),
    }
    torch.save(calibrated, output)
    report = {
        "version": calibrated["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "selected_blend": blend.tolist(),
        "splits": splits,
        "trace": trace,
        "source_contract": {
            "model_training_overlap_with_direct_groups": 0,
            "candidate_calibration_groups": int(len(calibration)),
            "selector_calibration_groups": int(len(selector_calibration)),
            "internal_confirmation_groups": int(len(confirmation)),
            "protected_validation_used": False,
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
