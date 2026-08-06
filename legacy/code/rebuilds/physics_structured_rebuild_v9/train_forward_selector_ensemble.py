#!/usr/bin/env python3
"""Train a protected selector over two complementary forward selectors."""

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
    selector_features,
    sha256,
    split_groups,
    system_predictions,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_PRIMARY = DEFAULT_RUN / "forward_natural_expert_selector_restricted_v9.pkl"
DEFAULT_SECONDARY = (
    DEFAULT_RUN / "forward_natural_lightgbm_field_selector_restricted_v9.pkl"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "forward_selector_ensemble_v9.pkl"
STATE_ROUTE = "predict_forward_from_state_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--primary-selector", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument(
        "--secondary-selector",
        type=Path,
        default=DEFAULT_SECONDARY,
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def load_selector(path: Path) -> dict[str, Any]:
    with path.resolve().open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("model") != "route_restricted_hgb_forward_expert_selector_v9":
        raise ValueError("unexpected restricted forward selector")
    if not bool(artifact["routes"]["state"]["enabled"]):
        raise ValueError("restricted selector state route is disabled")
    return artifact


def expert_prediction(
    artifact: dict[str, Any],
    features: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_path = (
        Path(str(artifact["current_dir"])).resolve() / "forward_state.pkl"
    )
    alternative_path = (
        Path(str(artifact["alternative_dir"])).resolve() / "forward_state.pkl"
    )
    current = calibrated_prediction(current_path, features, prior)
    alternative = calibrated_prediction(alternative_path, features, prior)
    route = artifact["routes"]["state"]
    probability = route["classifier"].predict_proba(
        selector_features(prior, current, alternative)
    )[:, 1]
    choose_alternative = probability >= float(route["threshold"])
    selected = np.where(
        choose_alternative[:, None],
        alternative,
        current,
    )
    return selected.astype(np.float32), current, alternative


def ensemble_features(
    prior: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            prior,
            primary,
            secondary,
            np.abs(primary - secondary),
        ],
        axis=1,
    ).astype(np.float32)


def system_expert_prediction(
    artifact: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    current, current_arrays = system_predictions(
        Path(str(artifact["current_dir"]))
    )
    alternative, alternative_arrays = system_predictions(
        Path(str(artifact["alternative_dir"]))
    )
    for key in ("prior", "truth_change", "routes"):
        if not np.array_equal(current_arrays[key], alternative_arrays[key]):
            raise ValueError(f"selector system caches differ for {key}")
    mask = current_arrays["routes"] == STATE_ROUTE
    current_prediction = current["state"][mask]
    alternative_prediction = alternative["state"][mask]
    route = artifact["routes"]["state"]
    probability = route["classifier"].predict_proba(
        selector_features(
            current_arrays["prior"][mask],
            current_prediction,
            alternative_prediction,
        )
    )[:, 1]
    selected = np.where(
        (probability >= float(route["threshold"]))[:, None],
        alternative_prediction,
        current_prediction,
    )
    return selected.astype(np.float32), current_arrays


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    primary_path = args.primary_selector.resolve()
    secondary_path = args.secondary_selector.resolve()
    primary_artifact = load_selector(primary_path)
    secondary_artifact = load_selector(secondary_path)
    with np.load(args.training_cache.resolve(), allow_pickle=False) as cache:
        features = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        target = np.asarray(
            cache["grid_target_normalized"],
            dtype=np.float32,
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    training_groups, calibration_groups = split_groups(
        group_ids,
        int(args.seed),
    )
    training_indices = row_indices(training_groups)
    calibration_indices = row_indices(calibration_groups)
    primary, _, _ = expert_prediction(primary_artifact, features, prior)
    secondary, _, _ = expert_prediction(
        secondary_artifact,
        features,
        prior,
    )
    primary_success = normalized_success(primary, target)
    secondary_success = normalized_success(secondary, target)
    exclusive = primary_success ^ secondary_success
    selected_train = training_indices[exclusive[training_indices]]
    labels = secondary_success[selected_train].astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("selector ensemble lacks both exclusive classes")
    sample_weight = (len(labels) / (2.0 * counts))[labels]
    all_features = ensemble_features(prior, primary, secondary)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
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
        block_primary, _, _ = expert_prediction(
            primary_artifact,
            block_features,
            block_prior,
        )
        block_secondary, _, _ = expert_prediction(
            secondary_artifact,
            block_features,
            block_prior,
        )
        probability = classifier.predict_proba(
            ensemble_features(
                block_prior,
                block_primary,
                block_secondary,
            )
        )[:, 1]
        current_success = normalized_success(
            block_primary,
            arrays.normalized_changes,
        )
        alternative_success = normalized_success(
            block_secondary,
            arrays.normalized_changes,
        )
        high = action_high_mask(arrays.group_count)
        protected_constraints.append(
            (
                probability,
                current_success,
                alternative_success,
                high,
            )
        )
        protected_values[block] = (
            probability,
            current_success,
            alternative_success,
            high,
        )
    calibration_probability = classifier.predict_proba(
        all_features[calibration_indices]
    )[:, 1]
    threshold = best_protected_threshold(
        calibration_probability,
        primary_success[calibration_indices],
        secondary_success[calibration_indices],
        protected_constraints,
    )
    protected_report = {}
    for block, (
        probability,
        current_success,
        alternative_success,
        high,
    ) in protected_values.items():
        choose_secondary = probability >= threshold["threshold"]
        selected = np.where(
            choose_secondary,
            alternative_success,
            current_success,
        )
        protected_report[block] = {
            "primary": protected_counts(current_success, high),
            "selected": protected_counts(selected, high),
            "secondary_count": int(choose_secondary.sum()),
        }

    system_primary, primary_arrays = system_expert_prediction(
        primary_artifact
    )
    system_secondary, secondary_arrays = system_expert_prediction(
        secondary_artifact
    )
    for key in ("prior", "truth_change", "routes"):
        if not np.array_equal(primary_arrays[key], secondary_arrays[key]):
            raise ValueError(f"ensemble system caches differ for {key}")
    mask = primary_arrays["routes"] == STATE_ROUTE
    system_probability = classifier.predict_proba(
        ensemble_features(
            primary_arrays["prior"][mask],
            system_primary,
            system_secondary,
        )
    )[:, 1]
    choose_secondary = system_probability >= threshold["threshold"]
    system_selected = np.where(
        choose_secondary[:, None],
        system_secondary,
        system_primary,
    )
    success_arguments = (
        primary_arrays["input_tolerance"][mask],
        primary_arrays["truth_change"][mask],
        primary_arrays["scoring_tolerance"][mask],
    )
    primary_system_success = raw_success(
        system_primary,
        *success_arguments,
    )
    secondary_system_success = raw_success(
        system_secondary,
        *success_arguments,
    )
    selected_system_success = raw_success(
        system_selected,
        *success_arguments,
    )
    artifact = {
        "version": "forward_selector_ensemble_v9_one_seed",
        "model": "hgb_forward_selector_ensemble_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": float(threshold["threshold"]),
        "primary_selector": str(primary_path),
        "primary_selector_sha256": sha256(primary_path),
        "secondary_selector": str(secondary_path),
        "secondary_selector_sha256": sha256(secondary_path),
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
            "primary_only_count": int(counts[0]),
            "secondary_only_count": int(counts[1]),
        },
        "calibration": threshold,
        "protected": protected_report,
        "protected_non_regression": all(
            selected >= primary
            for block in protected_report.values()
            for selected, primary in zip(
                block["selected"],
                block["primary"],
                strict=True,
            )
        ),
        "system_validation": {
            "primary_success_count": int(primary_system_success.sum()),
            "secondary_success_count": int(secondary_system_success.sum()),
            "selected_success_count": int(selected_system_success.sum()),
            "oracle_union_success_count": int(
                (primary_system_success | secondary_system_success).sum()
            ),
            "secondary_count": int(choose_secondary.sum()),
        },
        "source_contract": {
            "system_validation_used_for_training": False,
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
