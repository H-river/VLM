#!/usr/bin/env python3
"""Calibrate a non-overlapping residual tree on direct training requests."""

from __future__ import annotations

import argparse
import json
import pickle
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
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
)
from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    group_partitions,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import (
    sha256,
    tree_predictions,
)
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    forward_feature,
    read_jsonl,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_DIRECT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_TREE = (
    DEFAULT_RUN
    / "forward_novel_natural_hgb_residual"
    / "forward_novel_natural_hgb_residual_v9.pkl"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "forward_novel_direct_calibrated_v9.pkl"
BLENDS = np.asarray(
    [-0.25, 0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.25],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-data", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--tree-artifact", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def selection_key(
    prior: np.ndarray,
    residual: np.ndarray,
    target: np.ndarray,
    blend: np.ndarray,
) -> tuple[int, int, float, float]:
    prediction = prior + residual * blend[None, :]
    error = np.abs(prediction - target)
    passed = error <= 1.0
    return (
        int(np.all(passed, axis=1).sum()),
        int(passed.sum()),
        -float(error.mean()),
        -float(np.abs(blend).sum()),
    )


def choose_blend(
    prior: np.ndarray,
    residual: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    starts = [
        np.full(5, value, dtype=np.float32)
        for value in (0.0, 0.25, 0.50, 1.0)
    ]
    finals = []
    traces = []
    for start_index, initial in enumerate(starts):
        selected = initial.copy()
        passes = []
        for pass_index in range(6):
            changed = False
            for field in range(5):
                candidates = []
                for value in BLENDS:
                    proposal = selected.copy()
                    proposal[field] = float(value)
                    candidates.append(
                        (
                            selection_key(prior, residual, target, proposal),
                            -float(abs(value)),
                            float(value),
                            proposal,
                        )
                    )
                best = max(candidates, key=lambda row: row[:3])
                if not np.array_equal(selected, best[3]):
                    selected = best[3]
                    changed = True
            prediction = prior + residual * selected[None, :]
            passes.append(
                {
                    "pass": pass_index + 1,
                    "blend": selected.tolist(),
                    "metric": metric(prediction, target),
                }
            )
            if not changed:
                break
        finals.append(
            (
                selection_key(prior, residual, target, selected),
                -float(np.abs(selected).sum()),
                selected,
            )
        )
        traces.append(
            {"start": start_index, "initial": initial.tolist(), "passes": passes}
        )
    return max(finals, key=lambda row: row[:2])[2], traces


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    direct_path = args.direct_data.resolve()
    direct = read_jsonl(direct_path)
    group_ids = np.asarray(
        [str(row["group_id"]) for row in direct],
        dtype=np.str_,
    )
    runtime_rows = [
        {
            "group_id": str(row["group_id"]),
            "setup": row["setup"],
            "current_beam_state": row["current_beam_state"],
        }
        for row in direct
    ]
    action_indices = np.asarray(
        [action_index(row["action"]) for row in direct],
        dtype=np.int64,
    )
    positions = np.arange(len(direct), dtype=np.int64)
    torch, device = configure(int(args.seed), args.device)
    forward_path = args.forward_artifact.resolve()
    base, _ = load_forward_direction_runtime_v7(
        forward_path, torch, device
    )
    prior_grid = base.predict_changes(runtime_rows)
    prior = prior_grid[positions, action_indices]
    engineered = np.asarray(
        [
            forward_feature(
                row["setup"],
                row["current_beam_state"],
                row["action"],
            )
            for row in direct
        ],
        dtype=np.float32,
    )
    features = np.concatenate([engineered, prior], axis=1).astype(np.float32)
    tree_path = args.tree_artifact.resolve()
    with tree_path.open("rb") as stream:
        tree = pickle.load(stream)
    models = list(tree["models"])
    residual = tree_predictions(models, features)
    target = np.asarray(
        [
            np.asarray(
                [
                    float(row["truth_change"][field])
                    for field in STATE_FIELDS
                ],
                dtype=np.float32,
            )
            / tolerance_from_current(row["current_beam_state"])
            for row in direct
        ],
        dtype=np.float32,
    )
    calibration, selector_calibration, confirmation = group_partitions(
        group_ids, int(args.seed)
    )
    blend, trace = choose_blend(
        prior[calibration],
        residual[calibration],
        target[calibration],
    )
    split_reports = {}
    for name, indices in (
        ("candidate_calibration", calibration),
        ("selector_calibration", selector_calibration),
        ("internal_confirmation", confirmation),
    ):
        split_reports[name] = {
            "base": metric(prior[indices], target[indices]),
            "candidate": metric(
                prior[indices] + residual[indices] * blend[None, :],
                target[indices],
            ),
        }
    artifact = {
        "version": "forward_novel_direct_calibrated_v9_one_seed",
        "model": "frozen_v7_plus_residual_tree_v9",
        "route_scope": "predict_forward_from_state_v1",
        "forward_artifact": str(forward_path),
        "forward_artifact_sha256": sha256(forward_path),
        "tree_artifact": str(tree_path),
        "tree_artifact_sha256": sha256(tree_path),
        "field_blend": {
            field: float(blend[index])
            for index, field in enumerate(STATE_FIELDS)
        },
        "field_tree_source": {
            field: "primary" for field in STATE_FIELDS
        },
        "secondary_tree_artifact": None,
        "secondary_tree_artifact_sha256": None,
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "tree_artifact": str(tree_path),
        "selected_blend": blend.tolist(),
        "split_reports": split_reports,
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
