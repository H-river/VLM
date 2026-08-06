#!/usr/bin/env python3
"""Train boundary-aware corrections using the frozen existing cache."""

from __future__ import annotations

import argparse
import hashlib
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

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import (
    direction_metrics,
    distance_bins,
    labels_from_normalized_change,
)
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    correction_features,
    sha256,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
    DIRECTION_FIELDS,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    calibrated_prediction,
)
from physics_structured_rebuild_v9.train_forward_selector_extension import (
    ensemble_prediction,
    load_ensemble,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "qwen_combined_round2_forward_training_features.npz"
DEFAULT_CURRENT_FORWARD = DEFAULT_RUN / "forward_selector_ensemble_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "boundary_direction_correction_state_v9.pkl"
BOUNDARY_LIMITS = (0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 4.0)
CONFIDENCE_LIMITS = (0.34, 0.45, 0.55, 0.65, 0.75)
MARGIN_LIMITS = (0.0, 0.05, 0.10, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--base-direction",
        type=Path,
        default=DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
    )
    parser.add_argument(
        "--current-forward",
        type=Path,
        default=DEFAULT_CURRENT_FORWARD,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument(
        "--joint-repair-weight",
        type=float,
        default=0.0,
        help=(
            "Extra weight when this field is the only incorrect base field, "
            "so correcting it repairs the complete five-output request."
        ),
    )
    parser.add_argument(
        "--near-complete-protection-weight",
        type=float,
        default=0.0,
        help=(
            "Extra weight for retaining a correct field when at most two "
            "base fields are wrong."
        ),
    )
    return parser.parse_args()


def group_partitions(
    group_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bucket = np.asarray(
        [
            int(
                hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16],
                16,
            )
            % 10
            for group_id in group_ids
        ],
        dtype=np.int64,
    )
    return (
        np.flatnonzero(bucket >= 2),
        np.flatnonzero(bucket == 1),
        np.flatnonzero(bucket == 0),
    )


def rows(groups: np.ndarray) -> np.ndarray:
    return (
        groups[:, None] * 81 + np.arange(81, dtype=np.int64)[None, :]
    ).reshape(-1)


def model_probabilities(
    models: list[Any],
    values: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(values), 5, 3), dtype=np.float32)
    for field, model in enumerate(models):
        predicted = model.predict_proba(values)
        output[
            :,
            field,
            np.asarray(model.classes_, dtype=np.int64),
        ] = predicted
    return output


def base_components(
    artifact_path: Path,
    physical_plus_prior: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with artifact_path.resolve().open("rb") as stream:
        artifact = pickle.load(stream)
    threshold_changes = calibrated_prediction(
        Path(str(artifact["threshold_forward_artifact"])).resolve(),
        physical_plus_prior,
        prior,
    )
    with Path(str(artifact["tree_artifact"])).resolve().open("rb") as stream:
        primary = pickle.load(stream)
    models = list(primary["models"])
    secondary_path = artifact.get("secondary_tree_artifact")
    if secondary_path is not None:
        with Path(str(secondary_path)).resolve().open("rb") as stream:
            secondary = pickle.load(stream)
        secondary_models = list(secondary["models"])
        for field, name in enumerate(DIRECTION_FIELDS):
            if artifact["field_tree_source"][name] == "secondary":
                models[field] = secondary_models[field]
    probabilities = model_probabilities(models, physical_plus_prior)
    threshold = labels_from_normalized_change(threshold_changes)
    ordered = np.sort(probabilities, axis=2)
    tree_class = probabilities.argmax(axis=2)
    distance = np.abs(np.abs(threshold_changes) - 1.0)
    selected = threshold.copy()
    for field, name in enumerate(DIRECTION_FIELDS):
        calibration = artifact["field_calibration"][name]
        apply = (
            (distance[:, field] <= float(calibration["boundary_limit"]))
            & (
                ordered[:, field, -1]
                >= float(calibration["confidence_limit"])
            )
            & (
                ordered[:, field, -1] - ordered[:, field, -2]
                >= float(calibration["margin_limit"])
            )
            & (tree_class[:, field] != threshold[:, field])
        )
        selected[apply, field] = tree_class[apply, field]
    return threshold_changes.astype(np.float32), selected.astype(np.int64)


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    normalized: np.ndarray,
) -> dict[str, Any]:
    return direction_metrics(
        truth,
        prediction,
        distance_bins(normalized),
    )


def key(truth: np.ndarray, prediction: np.ndarray) -> tuple[int, int]:
    correct = prediction == truth
    return int(np.all(correct, axis=1).sum()), int(correct.sum())


def calibration_rules(
    truth: np.ndarray,
    base: np.ndarray,
    threshold_changes: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[dict[str, float] | None], list[dict[str, Any]]]:
    ordered = np.sort(probabilities, axis=2)
    model_class = probabilities.argmax(axis=2)
    distance = np.abs(np.abs(threshold_changes) - 1.0)
    choices: list[list[tuple[dict[str, float] | None, np.ndarray]]] = []
    for field in range(5):
        field_choices = [(None, base[:, field].copy())]
        for boundary in BOUNDARY_LIMITS:
            for confidence in CONFIDENCE_LIMITS:
                for margin in MARGIN_LIMITS:
                    rule = {
                        "boundary_limit": float(boundary),
                        "confidence_limit": float(confidence),
                        "margin_limit": float(margin),
                    }
                    apply = (
                        (distance[:, field] <= boundary)
                        & (ordered[:, field, -1] >= confidence)
                        & (
                            ordered[:, field, -1]
                            - ordered[:, field, -2]
                            >= margin
                        )
                        & (model_class[:, field] != base[:, field])
                    )
                    prediction = base[:, field].copy()
                    prediction[apply] = model_class[apply, field]
                    field_choices.append((rule, prediction))
        choices.append(field_choices)

    starts = [
        [0] * 5,
        [
            min(
                range(len(choices[field])),
                key=lambda index: (
                    -int(
                        (
                            choices[field][index][1]
                            == truth[:, field]
                        ).sum()
                    ),
                    index,
                ),
            )
            for field in range(5)
        ],
    ]
    finals = []
    trace = []
    for start_index, initial in enumerate(starts):
        selected = list(initial)
        passes = []
        for pass_index in range(4):
            changed = False
            for field in range(5):
                candidates = []
                for choice_index, (_, field_prediction) in enumerate(
                    choices[field]
                ):
                    proposal = base.copy()
                    for other in range(5):
                        index = (
                            choice_index
                            if other == field
                            else selected[other]
                        )
                        proposal[:, other] = choices[other][index][1]
                    candidates.append(
                        (
                            key(truth, proposal),
                            choice_index == 0,
                            -choice_index,
                            choice_index,
                        )
                    )
                best = max(candidates)
                if selected[field] != best[3]:
                    selected[field] = best[3]
                    changed = True
            prediction = base.copy()
            for field in range(5):
                prediction[:, field] = choices[field][selected[field]][1]
            passes.append(
                {
                    "pass": pass_index + 1,
                    "key": list(key(truth, prediction)),
                    "choice_indices": list(selected),
                }
            )
            if not changed:
                break
        prediction = base.copy()
        for field in range(5):
            prediction[:, field] = choices[field][selected[field]][1]
        finals.append((key(truth, prediction), selected, prediction))
        trace.append(
            {
                "start": start_index,
                "initial": initial,
                "passes": passes,
            }
        )
    best = max(finals, key=lambda value: value[0])
    rules = [
        choices[field][best[1][field]][0] for field in range(5)
    ]
    return rules, trace


def apply_rules(
    base: np.ndarray,
    threshold_changes: np.ndarray,
    probabilities: np.ndarray,
    rules: list[dict[str, float] | None],
) -> np.ndarray:
    ordered = np.sort(probabilities, axis=2)
    model_class = probabilities.argmax(axis=2)
    distance = np.abs(np.abs(threshold_changes) - 1.0)
    selected = base.copy()
    for field, rule in enumerate(rules):
        if rule is None:
            continue
        apply = (
            (distance[:, field] <= rule["boundary_limit"])
            & (ordered[:, field, -1] >= rule["confidence_limit"])
            & (
                ordered[:, field, -1] - ordered[:, field, -2]
                >= rule["margin_limit"]
            )
            & (model_class[:, field] != selected[:, field])
        )
        selected[apply, field] = model_class[apply, field]
    return selected


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMClassifier

    started = time.perf_counter()
    configure(int(args.seed), args.device)
    cache_path = args.training_cache.resolve()
    with np.load(cache_path, allow_pickle=False) as cache:
        physical_plus_prior = np.asarray(
            cache["grid_features"],
            dtype=np.float32,
        )
        prior = np.asarray(
            cache["grid_base_prediction"],
            dtype=np.float32,
        )
        normalized = np.asarray(
            cache["grid_target_normalized"],
            dtype=np.float32,
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    truth = labels_from_normalized_change(normalized)
    base_path = args.base_direction.resolve()
    threshold_changes, base = base_components(
        base_path,
        physical_plus_prior,
        prior,
    )
    current_path = args.current_forward.resolve()
    current_artifact = load_ensemble(current_path)
    current_forward = ensemble_prediction(
        current_artifact,
        physical_plus_prior,
        prior,
    )
    values = correction_features(
        physical_plus_prior[:, :46],
        prior,
        current_forward,
        threshold_changes,
        base,
    )
    train_groups, calibration_groups, test_groups = group_partitions(
        group_ids,
        int(args.seed),
    )
    train_indices = rows(train_groups)
    calibration_indices = rows(calibration_groups)
    test_indices = rows(test_groups)
    models = []
    training_trace = []
    base_wrong_matrix = base[train_indices] != truth[train_indices]
    base_wrong_count = base_wrong_matrix.sum(axis=1)
    for field, name in enumerate(DIRECTION_FIELDS):
        labels = truth[train_indices, field]
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        class_weight = (len(labels) / (3.0 * counts))[labels]
        base_wrong_bool = base_wrong_matrix[:, field]
        base_wrong = base_wrong_bool.astype(np.float64)
        single_field_repair = (
            base_wrong_bool & (base_wrong_count == 1)
        ).astype(np.float64)
        protect_near_complete = (
            (~base_wrong_bool) & (base_wrong_count <= 2)
        ).astype(np.float64)
        near = (
            np.abs(
                np.abs(threshold_changes[train_indices, field]) - 1.0
            )
            <= 0.75
        ).astype(np.float64)
        sample_weight = class_weight * (
            1.0
            + 3.0 * base_wrong
            + 1.5 * near
            + float(args.joint_repair_weight) * single_field_repair
            + float(args.near_complete_protection_weight)
            * protect_near_complete
        )
        model = LGBMClassifier(
            objective="multiclass",
            n_estimators=int(args.estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            min_child_samples=int(args.min_child_samples),
            reg_lambda=1.5,
            feature_fraction=0.90,
            bagging_fraction=0.90,
            bagging_freq=1,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=int(args.seed) + 31 * field,
        )
        field_started = time.perf_counter()
        model.fit(
            values[train_indices],
            labels,
            sample_weight=sample_weight,
        )
        record = {
            "field": name,
            "seconds": time.perf_counter() - field_started,
            "train_accuracy": float(
                np.mean(model.predict(values[train_indices]) == labels)
            ),
            "class_counts": counts.astype(np.int64).tolist(),
            "single_field_repair_count": int(single_field_repair.sum()),
            "near_complete_protection_count": int(
                protect_near_complete.sum()
            ),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        training_trace.append(record)
        models.append(model)

    calibration_probability = model_probabilities(
        models,
        values[calibration_indices],
    )
    rules, calibration_trace = calibration_rules(
        truth[calibration_indices],
        base[calibration_indices],
        threshold_changes[calibration_indices],
        calibration_probability,
    )
    selected_calibration = apply_rules(
        base[calibration_indices],
        threshold_changes[calibration_indices],
        calibration_probability,
        rules,
    )
    test_probability = model_probabilities(models, values[test_indices])
    selected_test = apply_rules(
        base[test_indices],
        threshold_changes[test_indices],
        test_probability,
        rules,
    )
    artifact = {
        "version": "boundary_direction_correction_state_v9_one_seed",
        "model": "boundary_direction_correction_v9",
        "base_direction": str(base_path),
        "base_direction_sha256": sha256(base_path),
        "current_forward": str(current_path),
        "current_forward_sha256": sha256(current_path),
        "models": models,
        "rules": rules,
        "direction_fields": list(DIRECTION_FIELDS),
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
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "group_count": int(len(group_ids)),
            "transition_count": int(len(values)),
            "train_groups": int(len(train_groups)),
            "calibration_groups": int(len(calibration_groups)),
            "internal_test_groups": int(len(test_groups)),
            "trace": training_trace,
            "joint_repair_weight": float(args.joint_repair_weight),
            "near_complete_protection_weight": float(
                args.near_complete_protection_weight
            ),
        },
        "rules": rules,
        "calibration_trace": calibration_trace,
        "baseline": {
            "calibration": metrics(
                truth[calibration_indices],
                base[calibration_indices],
                normalized[calibration_indices],
            ),
            "internal_test": metrics(
                truth[test_indices],
                base[test_indices],
                normalized[test_indices],
            ),
        },
        "selected": {
            "calibration": metrics(
                truth[calibration_indices],
                selected_calibration,
                normalized[calibration_indices],
            ),
            "internal_test": metrics(
                truth[test_indices],
                selected_test,
                normalized[test_indices],
            ),
        },
        "source_contract": {
            "generated_setups": 0,
            "generated_images": 0,
            "system_validation_used_for_training": False,
            "protected_validation_used_for_training": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "rules": rules,
                "baseline_calibration_joint": report["baseline"][
                    "calibration"
                ]["joint_exact"],
                "selected_calibration_joint": report["selected"][
                    "calibration"
                ]["joint_exact"],
                "baseline_internal_test_joint": report["baseline"][
                    "internal_test"
                ]["joint_exact"],
                "selected_internal_test_joint": report["selected"][
                    "internal_test"
                ]["joint_exact"],
                "seconds": report["seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
