#!/usr/bin/env python3
"""Evaluate the grouped forward tree on fixed natural state validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import tolerance_from_current
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
    read_jsonl,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
    load_forward_selector_extension_runtime_v9,
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_tree_runtime import (
    load_grouped_forward_tree_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_selector_runtime import (
    load_grouped_forward_selector_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_risk_selector_runtime import (
    load_grouped_forward_risk_selector_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from physics_structured_rebuild_v9.strict_forward_runtime import (
    load_strict_forward_correction_runtime_v9,
)
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_CANDIDATE = DEFAULT_RUN / "grouped_forward_tree_protected_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "grouped_forward_tree_system_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument(
        "--current-forward",
        type=Path,
        default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-kind",
        choices=(
            "residual",
            "selector_extension",
            "grouped_tree",
            "grouped_selector",
            "grouped_risk_selector",
            "strict_correction",
        ),
        default="grouped_tree",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def success(
    prediction: np.ndarray,
    truth: np.ndarray,
    input_tolerance: np.ndarray,
    scoring_tolerance: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    error = np.abs(
        prediction * input_tolerance - truth
    ) / scoring_tolerance
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return exact, {
        "count": int(len(exact)),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_scoring_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": {
            field: float(passed[:, index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    cache_path = output.with_suffix(".npz")
    if output.exists() or cache_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    qwen = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen / "canonical/val.jsonl")
        if row["target_decision"].get("route_name")
        == "predict_forward_from_state_v1"
    ]
    if len(canonical) != 150:
        raise ValueError("expected 150 state forward validation requests")
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen / "private/source_cases/val.jsonl")
    }
    rows = []
    action_indices = []
    truths = []
    input_tolerances = []
    scoring_tolerances = []
    for index, row in enumerate(canonical):
        arguments = row["target_decision"]["arguments"]
        private = private_by_group[str(row["group_id"])]
        action = arguments["action"]
        truth = simulator_forward_truth(private, action)["change"]
        rows.append(
            {
                "group_id": str(row["example_id"]),
                "setup": arguments["setup"],
                "current_beam_state": arguments["current_beam_state"],
            }
        )
        action_indices.append(action_index(action))
        truths.append([float(truth[field]) for field in STATE_FIELDS])
        input_tolerances.append(
            tolerance_from_current(arguments["current_beam_state"])
        )
        scoring_tolerances.append(
            tolerance_from_current(private["current_beam_state"])
        )
        if (index + 1) % 50 == 0:
            print(
                json.dumps(
                    {"prepared": index + 1, "count": len(canonical)},
                    sort_keys=True,
                ),
                flush=True,
            )
    torch, device = configure(int(args.seed), args.device)
    current, _ = load_forward_selector_ensemble_runtime_v9(
        args.current_forward.resolve(),
        torch,
        device,
    )
    if args.candidate_kind == "selector_extension":
        candidate, _ = load_forward_selector_extension_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    elif args.candidate_kind == "residual":
        candidate, _ = load_residual_forward_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    elif args.candidate_kind == "grouped_selector":
        candidate, _ = load_grouped_forward_selector_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    elif args.candidate_kind == "grouped_risk_selector":
        candidate, _ = load_grouped_forward_risk_selector_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    elif args.candidate_kind == "strict_correction":
        candidate, _ = load_strict_forward_correction_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    else:
        candidate, _ = load_grouped_forward_tree_runtime_v9(
            args.candidate.resolve(),
            torch,
            device,
        )
    current_grid = current.predict_changes(rows)
    candidate_grid = candidate.predict_changes(rows)
    positions = np.arange(len(rows), dtype=np.int64)
    action_indices_array = np.asarray(action_indices, dtype=np.int64)
    current_prediction = current_grid[positions, action_indices_array]
    candidate_prediction = candidate_grid[positions, action_indices_array]
    truth = np.asarray(truths, dtype=np.float32)
    input_tolerance = np.asarray(input_tolerances, dtype=np.float32)
    scoring_tolerance = np.asarray(scoring_tolerances, dtype=np.float32)
    current_success, current_metrics = success(
        current_prediction,
        truth,
        input_tolerance,
        scoring_tolerance,
    )
    candidate_success, candidate_metrics = success(
        candidate_prediction,
        truth,
        input_tolerance,
        scoring_tolerance,
    )
    np.savez_compressed(
        cache_path,
        current_prediction=current_prediction,
        candidate_prediction=candidate_prediction,
        truth_change=truth,
        input_tolerance=input_tolerance,
        scoring_tolerance=scoring_tolerance,
        current_success=current_success,
        candidate_success=candidate_success,
        action_indices=action_indices_array,
    )
    report = {
        "version": "grouped_forward_tree_system_validation_v9_one_seed",
        "route": "predict_forward_from_state_v1",
        "candidate_kind": str(args.candidate_kind),
        "current": current_metrics,
        "candidate": candidate_metrics,
        "oracle_union_success_count": int(
            (current_success | candidate_success).sum()
        ),
        "current_only_success_count": int(
            (current_success & ~candidate_success).sum()
        ),
        "candidate_only_success_count": int(
            (candidate_success & ~current_success).sum()
        ),
        "promotion_passed": (
            candidate_metrics["strict_all_five_count"]
            >= current_metrics["strict_all_five_count"]
        ),
        "artifacts": {
            "current": {
                "path": str(args.current_forward.resolve()),
                "sha256": sha256(args.current_forward.resolve()),
            },
            "candidate": {
                "path": str(args.candidate.resolve()),
                "sha256": sha256(args.candidate.resolve()),
            },
        },
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
