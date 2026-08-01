#!/usr/bin/env python3
"""Extend the protected forward selector with one complementary expert."""

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
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.calibrate_forward_expert_selector_restricted import (
    best_protected_threshold,
)
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    DEFAULT_DIFFICULT,
    DEFAULT_FORWARD,
    DEFAULT_OLD,
    DEFAULT_TRAINING,
    action_high_mask,
    calibrated_prediction,
    normalized_success,
    protected_counts,
    raw_success,
    row_indices,
    sha256,
    split_groups,
    system_predictions,
)
from physics_structured_rebuild_v9.train_forward_selector_ensemble import (
    ensemble_features,
    expert_prediction,
    load_selector,
    system_expert_prediction,
)
from specialist_rebuild_v2.common import forward_feature, read_jsonl

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CURRENT = DEFAULT_RUN / "forward_selector_ensemble_v9.pkl"
DEFAULT_CANDIDATE = DEFAULT_RUN / "forward_natural_grid_lightgbm_calibrated"
DEFAULT_OUTPUT = DEFAULT_RUN / "forward_selector_extension_v9.pkl"
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
STATE_ROUTE = "predict_forward_from_state_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--current-selector", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        help="Optional direct state artifact; overrides candidate-dir/forward_state.pkl.",
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--skip-system-evaluation",
        action="store_true",
        help="Train and enforce protected gates without opening system validation.",
    )
    return parser.parse_args()


def load_ensemble(path: Path) -> dict[str, Any]:
    with path.resolve().open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("model") != "hgb_forward_selector_ensemble_v9":
        raise ValueError("unexpected current forward selector ensemble")
    return artifact


def ensemble_prediction(
    artifact: dict[str, Any],
    features: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    primary_artifact = load_selector(
        Path(str(artifact["primary_selector"])).resolve()
    )
    secondary_artifact = load_selector(
        Path(str(artifact["secondary_selector"])).resolve()
    )
    primary, _, _ = expert_prediction(primary_artifact, features, prior)
    secondary, _, _ = expert_prediction(secondary_artifact, features, prior)
    probability = artifact["classifier"].predict_proba(
        ensemble_features(prior, primary, secondary)
    )[:, 1]
    return np.where(
        (probability >= float(artifact["threshold"]))[:, None],
        secondary,
        primary,
    ).astype(np.float32)


def extension_features(
    physical_features: np.ndarray,
    prior: np.ndarray,
    current: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            physical_features,
            prior,
            current,
            candidate,
            np.abs(candidate - current),
        ],
        axis=1,
    ).astype(np.float32)


def current_system_prediction(
    artifact: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    primary_artifact = load_selector(
        Path(str(artifact["primary_selector"])).resolve()
    )
    secondary_artifact = load_selector(
        Path(str(artifact["secondary_selector"])).resolve()
    )
    primary, primary_arrays = system_expert_prediction(primary_artifact)
    secondary, secondary_arrays = system_expert_prediction(secondary_artifact)
    for key in ("prior", "truth_change", "routes"):
        if not np.array_equal(primary_arrays[key], secondary_arrays[key]):
            raise ValueError(f"current selector system caches differ for {key}")
    mask = primary_arrays["routes"] == STATE_ROUTE
    probability = artifact["classifier"].predict_proba(
        ensemble_features(
            primary_arrays["prior"][mask],
            primary,
            secondary,
        )
    )[:, 1]
    selected = np.where(
        (probability >= float(artifact["threshold"]))[:, None],
        secondary,
        primary,
    )
    return selected.astype(np.float32), primary_arrays


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    current_path = args.current_selector.resolve()
    candidate_dir = args.candidate_dir.resolve()
    candidate_path = (
        args.candidate_artifact.resolve()
        if args.candidate_artifact is not None
        else candidate_dir / "forward_state.pkl"
    )
    current_artifact = load_ensemble(current_path)
    with np.load(args.training_cache.resolve(), allow_pickle=False) as cache:
        features = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        target = np.asarray(cache["grid_target_normalized"], dtype=np.float32)
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    training_groups, calibration_groups = split_groups(
        group_ids,
        int(args.seed),
    )
    training_indices = row_indices(training_groups)
    calibration_indices = row_indices(calibration_groups)
    current = ensemble_prediction(current_artifact, features, prior)
    candidate = calibrated_prediction(candidate_path, features, prior)
    current_success = normalized_success(current, target)
    candidate_success = normalized_success(candidate, target)
    exclusive = current_success ^ candidate_success
    selected_train = training_indices[exclusive[training_indices]]
    labels = candidate_success[selected_train].astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("selector extension lacks both exclusive classes")
    sample_weight = (len(labels) / (2.0 * counts))[labels]
    all_features = extension_features(features, prior, current, candidate)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=int(args.seed),
    )
    classifier.fit(
        all_features[selected_train],
        labels,
        sample_weight=sample_weight,
    )

    torch, device = configure(int(args.seed), args.device)
    base, _ = load_forward_direction_runtime_v7(
        args.forward_artifact.resolve(),
        torch,
        device,
    )
    protected_constraints = []
    protected_values = {}
    for block, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_prior = stream_forward_changes(path, base)
        block_features = np.concatenate(
            [arrays.features, block_prior],
            axis=1,
        ).astype(np.float32)
        block_current = ensemble_prediction(
            current_artifact,
            block_features,
            block_prior,
        )
        block_candidate = calibrated_prediction(
            candidate_path,
            block_features,
            block_prior,
        )
        probability = classifier.predict_proba(
            extension_features(
                block_features,
                block_prior,
                block_current,
                block_candidate,
            )
        )[:, 1]
        block_current_success = normalized_success(
            block_current,
            arrays.normalized_changes,
        )
        block_candidate_success = normalized_success(
            block_candidate,
            arrays.normalized_changes,
        )
        high = action_high_mask(arrays.group_count)
        protected_constraints.append(
            (
                probability,
                block_current_success,
                block_candidate_success,
                high,
            )
        )
        protected_values[block] = (
            probability,
            block_current_success,
            block_candidate_success,
            high,
        )
    calibration_probability = classifier.predict_proba(
        all_features[calibration_indices]
    )[:, 1]
    threshold = best_protected_threshold(
        calibration_probability,
        current_success[calibration_indices],
        candidate_success[calibration_indices],
        protected_constraints,
    )
    protected_report = {}
    for block, (
        probability,
        block_current_success,
        block_candidate_success,
        high,
    ) in protected_values.items():
        choose_candidate = probability >= float(threshold["threshold"])
        selected = np.where(
            choose_candidate,
            block_candidate_success,
            block_current_success,
        )
        protected_report[block] = {
            "current": protected_counts(block_current_success, high),
            "selected": protected_counts(selected, high),
            "candidate_count": int(choose_candidate.sum()),
        }

    system_report = None
    if not args.skip_system_evaluation:
        system_current, current_arrays = current_system_prediction(
            current_artifact
        )
        system_candidate_by_route, candidate_arrays = system_predictions(
            candidate_dir
        )
        for key in ("prior", "truth_change", "routes"):
            if not np.array_equal(current_arrays[key], candidate_arrays[key]):
                raise ValueError(f"extension system caches differ for {key}")
        mask = current_arrays["routes"] == STATE_ROUTE
        system_candidate = system_candidate_by_route["state"][mask]
        canonical = [
            row
            for row in read_jsonl(
                args.qwen_data.resolve() / "canonical/val.jsonl"
            )
            if row["target_decision"].get("route_name") == STATE_ROUTE
        ]
        if len(canonical) != int(mask.sum()):
            raise ValueError("system forward state row count differs")
        engineered = np.asarray(
            [
                forward_feature(
                    row["target_decision"]["arguments"]["setup"],
                    row["target_decision"]["arguments"]["current_beam_state"],
                    row["target_decision"]["arguments"]["action"],
                )
                for row in canonical
            ],
            dtype=np.float32,
        )
        if engineered.shape[1] + prior.shape[1] != features.shape[1]:
            raise ValueError("system engineered feature width differs")
        system_physical_features = np.concatenate(
            [
                engineered,
                current_arrays["prior"][mask],
            ],
            axis=1,
        )
        system_probability = classifier.predict_proba(
            extension_features(
                system_physical_features,
                current_arrays["prior"][mask],
                system_current,
                system_candidate,
            )
        )[:, 1]
        choose_candidate = (
            system_probability >= float(threshold["threshold"])
        )
        system_selected = np.where(
            choose_candidate[:, None],
            system_candidate,
            system_current,
        )
        success_arguments = (
            current_arrays["input_tolerance"][mask],
            current_arrays["truth_change"][mask],
            current_arrays["scoring_tolerance"][mask],
        )
        current_system_success = raw_success(
            system_current,
            *success_arguments,
        )
        candidate_system_success = raw_success(
            system_candidate,
            *success_arguments,
        )
        selected_system_success = raw_success(
            system_selected,
            *success_arguments,
        )
        system_report = {
            "current_success_count": int(current_system_success.sum()),
            "candidate_success_count": int(candidate_system_success.sum()),
            "selected_success_count": int(selected_system_success.sum()),
            "oracle_union_success_count": int(
                (current_system_success | candidate_system_success).sum()
            ),
            "candidate_count": int(choose_candidate.sum()),
        }

    artifact = {
        "version": "forward_selector_extension_v9_one_seed",
        "model": "hgb_forward_selector_extension_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": float(threshold["threshold"]),
        "current_selector": str(current_path),
        "current_selector_sha256": sha256(current_path),
        "candidate_dir": str(candidate_dir),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": sha256(candidate_path),
        "physical_feature_count": int(features.shape[1]),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "training": {
            "group_count": int(len(training_groups)),
            "exclusive_count": int(len(selected_train)),
            "current_only_count": int(counts[0]),
            "candidate_only_count": int(counts[1]),
        },
        "calibration": threshold,
        "protected": protected_report,
        "protected_non_regression": all(
            selected >= current_count
            for block in protected_report.values()
            for selected, current_count in zip(
                block["selected"],
                block["current"],
                strict=True,
            )
        ),
        "system_validation": system_report,
        "source_contract": {
            "training_cache": str(args.training_cache.resolve()),
            "system_validation_used_for_training": False,
            "system_validation_opened": not args.skip_system_evaluation,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
