#!/usr/bin/env python3
"""Evaluate a protected boundary correction on state-direction system replay."""

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
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)
from direction_rebuild_v4.data import CLASSES, direction_metrics
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    load_boundary_direction_correction_runtime_v9,
    sha256,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
    read_jsonl,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_CANDIDATE = (
    DEFAULT_RUN / "boundary_direction_protected_calibrated_state_v9.pkl"
)
DEFAULT_OUTPUT = (
    DEFAULT_RUN / "boundary_direction_protected_calibrated_system_validation.json"
)
ROUTE = "predict_direction_from_state_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def metric_bundle(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    report = direction_metrics(truth, prediction)
    return {
        "count": int(report["count"]),
        "all_five_exact_count": int(report["joint_exact_count"]),
        "all_five_exact": float(report["joint_exact"]),
        "mean_field_accuracy": float(report["mean_field_accuracy"]),
        "equal_field_macro_f1": float(report["equal_field_macro_f1"]),
        "per_field": report["per_field"],
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    qwen_data = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if row["target_decision"].get("route_name") == ROUTE
    ]
    if len(canonical) != 150:
        raise ValueError("expected 150 state-direction validation requests")
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen_data / "private/source_cases/val.jsonl")
    }
    rows = [
        {
            "group_id": str(row["example_id"]),
            "setup": row["target_decision"]["arguments"]["setup"],
            "current_beam_state": row["target_decision"]["arguments"][
                "current_beam_state"
            ],
        }
        for row in canonical
    ]

    torch, device = configure(int(args.seed), args.device)
    candidate_path = args.candidate.resolve()
    candidate, artifact = load_boundary_direction_correction_runtime_v9(
        candidate_path,
        torch,
        device,
    )
    _, _, _, base_grid = candidate.base_grid(rows)
    base_grid = base_grid.reshape(len(rows), -1, 5)
    selected_grid, applied_grid = candidate.predict_grid(rows)
    class_index = {str(value): index for index, value in enumerate(CLASSES)}
    baseline = np.empty((len(canonical), 5), dtype=np.int64)
    selected = np.empty_like(baseline)
    applied = np.empty_like(baseline, dtype=np.bool_)
    truth = np.empty_like(baseline)
    for index, row in enumerate(canonical):
        arguments = row["target_decision"]["arguments"]
        grid_index = action_index(arguments["action"])
        baseline[index] = base_grid[index, grid_index]
        selected[index] = selected_grid[index, grid_index]
        applied[index] = applied_grid[index, grid_index]
        private = private_by_group[str(row["group_id"])]
        physical_truth = simulator_forward_truth(
            private,
            arguments["action"],
        )
        truth[index] = [
            class_index[str(physical_truth["directions"][field])]
            for field in DIRECTION_FIELDS
        ]

    baseline_exact = np.all(baseline == truth, axis=1)
    selected_exact = np.all(selected == truth, axis=1)
    baseline_metrics = metric_bundle(truth, baseline)
    candidate_metrics = metric_bundle(truth, selected)
    report = {
        "version": "boundary_direction_system_validation_v9_one_seed",
        "route": ROUTE,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidate_only_all_five_count": int(
            (selected_exact & ~baseline_exact).sum()
        ),
        "baseline_only_all_five_count": int(
            (baseline_exact & ~selected_exact).sum()
        ),
        "oracle_union_all_five_count": int(
            (baseline_exact | selected_exact).sum()
        ),
        "changed_request_count": int(np.any(selected != baseline, axis=1).sum()),
        "changed_field_count": int(np.sum(selected != baseline)),
        "applied_request_count": int(np.any(applied, axis=1).sum()),
        "applied_field_count": int(applied.sum()),
        "promotion_passed": bool(
            candidate_metrics["all_five_exact_count"]
            >= baseline_metrics["all_five_exact_count"]
            and candidate_metrics["equal_field_macro_f1"]
            >= baseline_metrics["equal_field_macro_f1"]
        ),
        "candidate_artifact": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "rules": artifact["rules"],
        },
        "simulator_at_inference": False,
        "private_scoring_note": (
            "The simulator is called only after prediction to obtain private "
            "physical ground truth."
        ),
        "source_contract": {
            "system_validation_used_for_training": False,
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
