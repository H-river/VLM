#!/usr/bin/env python3
"""Train a direct-request selector for two frozen forward experts."""

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
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    group_partitions,
)
from physics_structured_rebuild_v9.train_forward_selector_extension import (
    extension_features,
)
from physics_structured_rebuild_v9.strict_forward_runtime import sha256
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    forward_feature,
    read_jsonl,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CURRENT = DEFAULT_RUN / "forward_selector_ensemble_v9.pkl"
DEFAULT_CANDIDATE = (
    DEFAULT_RUN / "forward_round2_extra_trees_protected_calibrated_state.pkl"
)
DEFAULT_DIRECT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "direct_forward_expert_selector_v9.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-data", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--current-selector", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def metric(
    success: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    error = np.abs(prediction - target)
    return {
        "count": int(len(success)),
        "strict_all_five_count": int(success.sum()),
        "strict_all_five_success": float(success.mean()),
        "mae_in_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": (error <= 1.0).mean(axis=0).tolist(),
    }


def select_threshold(
    probability: np.ndarray,
    current_success: np.ndarray,
    candidate_success: np.ndarray,
) -> dict[str, Any]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([np.inf], dtype=np.float64),
                np.quantile(
                    probability.astype(np.float64),
                    np.linspace(0.0, 1.0, 401),
                ),
            ]
        )
    )
    baseline = int(current_success.sum())
    choices = []
    for threshold in thresholds:
        choose = probability >= threshold
        selected = np.where(choose, candidate_success, current_success)
        count = int(selected.sum())
        choices.append(
            (
                int(count >= baseline),
                count if count >= baseline else -10**9,
                int((selected & ~current_success).sum()),
                -int((~selected & current_success).sum()),
                -int(choose.sum()),
                float(threshold),
            )
        )
    best = max(choices)
    threshold = float(best[-1])
    choose = probability >= threshold
    selected = np.where(choose, candidate_success, current_success)
    return {
        "threshold": threshold,
        "current_success_count": baseline,
        "candidate_success_count": int(candidate_success.sum()),
        "selected_success_count": int(selected.sum()),
        "oracle_union_success_count": int(
            (current_success | candidate_success).sum()
        ),
        "candidate_count": int(choose.sum()),
        "candidate_only_captured": int(
            (choose & candidate_success & ~current_success).sum()
        ),
        "current_only_sacrificed": int(
            (choose & current_success & ~candidate_success).sum()
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMClassifier

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
    current_path = args.current_selector.resolve()
    candidate_path = args.candidate.resolve()
    current_runtime, _ = load_forward_selector_ensemble_runtime_v9(
        current_path, torch, device
    )
    candidate_runtime, _ = load_residual_forward_runtime_v9(
        candidate_path, torch, device
    )
    current_grid = current_runtime.predict_changes(runtime_rows)
    candidate_grid = candidate_runtime.predict_changes(runtime_rows)
    prior_grid = current_runtime.base.predict_changes(runtime_rows)
    current = current_grid[positions, action_indices]
    candidate = candidate_grid[positions, action_indices]
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
    physical = np.concatenate([engineered, prior], axis=1).astype(np.float32)
    features = extension_features(
        physical, prior, current, candidate
    )
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
    current_success = np.all(np.abs(current - target) <= 1.0, axis=1)
    candidate_success = np.all(np.abs(candidate - target) <= 1.0, axis=1)
    train, calibration, confirmation = group_partitions(
        group_ids, int(args.seed)
    )
    exclusive = current_success ^ candidate_success
    selected_train = train[exclusive[train]]
    labels = candidate_success[selected_train].astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("direct selector lacks both exclusive classes")
    weights = (len(labels) / (2.0 * counts))[labels]
    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=12,
        reg_lambda=3.0,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        random_state=int(args.seed),
    )
    classifier.fit(
        features[selected_train],
        labels,
        sample_weight=weights,
    )
    calibration_probability = classifier.predict_proba(
        features[calibration]
    )[:, 1]
    calibration_selection = select_threshold(
        calibration_probability,
        current_success[calibration],
        candidate_success[calibration],
    )
    threshold = float(calibration_selection["threshold"])

    split_reports = {}
    for name, indices in (
        ("calibration", calibration),
        ("internal_confirmation", confirmation),
    ):
        probability = classifier.predict_proba(features[indices])[:, 1]
        choose = probability >= threshold
        selected_prediction = np.where(
            choose[:, None], candidate[indices], current[indices]
        )
        selected_success = np.where(
            choose,
            candidate_success[indices],
            current_success[indices],
        )
        split_reports[name] = {
            "current": metric(
                current_success[indices], current[indices], target[indices]
            ),
            "candidate": metric(
                candidate_success[indices],
                candidate[indices],
                target[indices],
            ),
            "selected": metric(
                selected_success, selected_prediction, target[indices]
            ),
            "oracle_union_count": int(
                (current_success[indices] | candidate_success[indices]).sum()
            ),
            "candidate_count": int(choose.sum()),
            "candidate_only_captured": int(
                (
                    choose
                    & candidate_success[indices]
                    & ~current_success[indices]
                ).sum()
            ),
            "current_only_sacrificed": int(
                (
                    choose
                    & current_success[indices]
                    & ~candidate_success[indices]
                ).sum()
            ),
        }
    confirmation_report = split_reports["internal_confirmation"]
    internal_passed = bool(
        confirmation_report["selected"]["strict_all_five_count"]
        > confirmation_report["current"]["strict_all_five_count"]
    )
    artifact = {
        "version": "direct_forward_expert_selector_v9_one_seed",
        "model": "hgb_forward_selector_extension_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": threshold,
        "current_selector": str(current_path),
        "current_selector_sha256": sha256(current_path),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": sha256(candidate_path),
        "physical_feature_count": int(physical.shape[1]),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "threshold": threshold,
        "training": {
            "direct_data": str(direct_path),
            "direct_data_sha256": sha256(direct_path),
            "group_count": int(len(group_ids)),
            "train_groups": int(len(train)),
            "calibration_groups": int(len(calibration)),
            "internal_confirmation_groups": int(len(confirmation)),
            "exclusive_train_count": int(len(selected_train)),
            "current_only_train_count": int(counts[0]),
            "candidate_only_train_count": int(counts[1]),
        },
        "calibration_selection": calibration_selection,
        "splits": split_reports,
        "internal_confirmation_passed": internal_passed,
        "source_contract": {
            "generated_setups": 0,
            "generated_images": 0,
            "protected_validation_used_for_training": False,
            "system_validation_used_for_training": False,
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
