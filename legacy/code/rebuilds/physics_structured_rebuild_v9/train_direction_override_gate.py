#!/usr/bin/env python3
"""Train an out-of-sample gate for candidate direction overrides."""

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
    labels_from_normalized_change,
)
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    correction_features,
    override_gate_features,
    positive_probability,
    sha256,
)
from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    base_components,
    group_partitions,
    model_probabilities,
    rows,
)
from physics_structured_rebuild_v9.train_forward_selector_extension import (
    ensemble_prediction,
    load_ensemble,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "qwen_combined_round2_forward_training_features.npz"
DEFAULT_CANDIDATE = DEFAULT_RUN / "boundary_direction_joint_repair_state_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "direction_override_gate_state_v9.pkl"
THRESHOLDS: tuple[float | None, ...] = (
    None,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--estimators", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=30)
    return parser.parse_args()


def gate_bucket(group_ids: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(
        [
            int(
                hashlib.sha256(
                    f"override-gate:{seed}:{group_id}".encode()
                ).hexdigest()[:16],
                16,
            )
            % 5
            for group_id in group_ids
        ],
        dtype=np.int64,
    )


def gate_probability_matrix(
    models: list[Any],
    values: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [positive_probability(model, values) for model in models]
    ).astype(np.float32)


def apply_thresholds(
    base: np.ndarray,
    candidate: np.ndarray,
    probability: np.ndarray,
    thresholds: list[float | None],
) -> tuple[np.ndarray, np.ndarray]:
    selected = base.copy()
    applied = np.zeros_like(base, dtype=np.bool_)
    for field, threshold in enumerate(thresholds):
        if threshold is None:
            continue
        mask = (
            (candidate[:, field] != base[:, field])
            & (probability[:, field] >= float(threshold))
        )
        selected[mask, field] = candidate[mask, field]
        applied[mask, field] = True
    return selected, applied


def metric_bundle(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    report = direction_metrics(truth, prediction)
    return {
        "count": int(report["count"]),
        "joint_exact_count": int(report["joint_exact_count"]),
        "joint_exact": float(report["joint_exact"]),
        "mean_field_accuracy": float(report["mean_field_accuracy"]),
        "equal_field_macro_f1": float(report["equal_field_macro_f1"]),
        "per_field": report["per_field"],
    }


def choose_thresholds(
    truth: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    probability: np.ndarray,
) -> tuple[list[float | None], list[dict[str, Any]]]:
    choices: list[list[tuple[float | None, np.ndarray, np.ndarray]]] = []
    for field in range(5):
        field_choices = []
        for threshold in THRESHOLDS:
            selected, applied = apply_thresholds(
                base[:, [field]],
                candidate[:, [field]],
                probability[:, [field]],
                [threshold],
            )
            field_choices.append(
                (threshold, selected[:, 0], applied[:, 0])
            )
        choices.append(field_choices)

    def evaluate(indices: list[int]) -> tuple[tuple[int, ...], dict[str, Any]]:
        prediction = np.column_stack(
            [choices[field][indices[field]][1] for field in range(5)]
        )
        applied = np.column_stack(
            [choices[field][indices[field]][2] for field in range(5)]
        )
        correct = prediction == truth
        key = (
            int(np.all(correct, axis=1).sum()),
            int(correct.sum()),
            -int(applied.sum()),
        )
        return key, {
            "choice_indices": list(indices),
            "thresholds": [
                choices[field][indices[field]][0] for field in range(5)
            ],
            "joint_exact_count": key[0],
            "field_correct_count": key[1],
            "applied_field_count": int(applied.sum()),
        }

    starts = [[0] * 5]
    starts.append(
        [
            max(
                range(len(choices[field])),
                key=lambda index: (
                    int(
                        (
                            choices[field][index][1] == truth[:, field]
                        ).sum()
                    ),
                    -int(choices[field][index][2].sum()),
                ),
            )
            for field in range(5)
        ]
    )
    finals = []
    trace = []
    for start_index, initial in enumerate(starts):
        selected = list(initial)
        passes = []
        for pass_index in range(6):
            changed = False
            for field in range(5):
                candidates = []
                for choice_index in range(len(choices[field])):
                    proposal = list(selected)
                    proposal[field] = choice_index
                    key, details = evaluate(proposal)
                    candidates.append((key, -choice_index, choice_index, details))
                best = max(candidates, key=lambda value: value[:2])
                if selected[field] != best[2]:
                    selected[field] = best[2]
                    changed = True
            _, details = evaluate(selected)
            passes.append({"pass": pass_index + 1, **details})
            if not changed:
                break
        key, details = evaluate(selected)
        finals.append((key, selected, details))
        trace.append(
            {"start": start_index, "initial": initial, "passes": passes}
        )
    best = max(finals, key=lambda value: value[0])
    return list(best[2]["thresholds"]), trace


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMClassifier

    started = time.perf_counter()
    configure(int(args.seed), args.device)
    candidate_path = args.candidate.resolve()
    with candidate_path.open("rb") as stream:
        candidate_artifact = pickle.load(stream)
    if candidate_artifact.get("model") != "boundary_direction_correction_v9":
        raise ValueError("unexpected direction correction candidate")

    cache_path = args.training_cache.resolve()
    with np.load(cache_path, allow_pickle=False) as cache:
        physical_plus_prior = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        normalized = np.asarray(
            cache["grid_target_normalized"], dtype=np.float32
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    truth = labels_from_normalized_change(normalized)
    base_path = Path(str(candidate_artifact["base_direction"])).resolve()
    current_path = Path(str(candidate_artifact["current_forward"])).resolve()
    threshold_changes, base = base_components(
        base_path, physical_plus_prior, prior
    )
    current_artifact = load_ensemble(current_path)
    current_forward = ensemble_prediction(
        current_artifact, physical_plus_prior, prior
    )
    correction_values = correction_features(
        physical_plus_prior[:, :46],
        prior,
        current_forward,
        threshold_changes,
        base,
    )
    candidate_probability = model_probabilities(
        list(candidate_artifact["models"]), correction_values
    )
    candidate = candidate_probability.argmax(axis=2)
    gate_values = override_gate_features(
        correction_values,
        threshold_changes,
        base,
        candidate_probability,
    )

    _, calibration_groups, test_groups = group_partitions(
        group_ids, int(args.seed)
    )
    subdivisions = gate_bucket(group_ids[calibration_groups], int(args.seed))
    gate_train_groups = calibration_groups[subdivisions <= 2]
    gate_selection_groups = calibration_groups[subdivisions >= 3]
    gate_train_indices = rows(gate_train_groups)
    gate_selection_indices = rows(gate_selection_groups)
    test_indices = rows(test_groups)
    if min(
        len(gate_train_groups), len(gate_selection_groups), len(test_groups)
    ) == 0:
        raise ValueError("empty direction override split")

    gate_models = []
    training_trace = []
    base_wrong = base[gate_train_indices] != truth[gate_train_indices]
    base_wrong_count = base_wrong.sum(axis=1)
    for field in range(5):
        disagreement = (
            candidate[gate_train_indices, field]
            != base[gate_train_indices, field]
        )
        indices = gate_train_indices[disagreement]
        labels = (
            candidate[indices, field] == truth[indices, field]
        ).astype(np.int64)
        if len(np.unique(labels)) != 2:
            raise ValueError(f"gate field {field} has only one target class")
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        class_weight = (len(labels) / (2.0 * counts))[labels]
        local_wrong_count = base_wrong_count[disagreement]
        candidate_repairs_complete = (
            (labels == 1) & (local_wrong_count == 1)
        )
        base_is_complete = local_wrong_count == 0
        sample_weight = class_weight * (
            1.0
            + 5.0 * candidate_repairs_complete.astype(np.float64)
            + 5.0 * base_is_complete.astype(np.float64)
        )
        model = LGBMClassifier(
            objective="binary",
            n_estimators=int(args.estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            min_child_samples=int(args.min_child_samples),
            reg_lambda=2.0,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=int(args.seed) + 101 * field,
        )
        field_started = time.perf_counter()
        model.fit(
            gate_values[indices],
            labels,
            sample_weight=sample_weight,
        )
        record = {
            "field": field,
            "disagreement_count": int(len(indices)),
            "candidate_correct_count": int(labels.sum()),
            "base_complete_negative_count": int(base_is_complete.sum()),
            "complete_repair_positive_count": int(
                candidate_repairs_complete.sum()
            ),
            "seconds": time.perf_counter() - field_started,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        training_trace.append(record)
        gate_models.append(model)

    selection_probability = gate_probability_matrix(
        gate_models, gate_values[gate_selection_indices]
    )
    thresholds, threshold_trace = choose_thresholds(
        truth[gate_selection_indices],
        base[gate_selection_indices],
        candidate[gate_selection_indices],
        selection_probability,
    )

    splits = {}
    for name, indices in (
        ("gate_selection", gate_selection_indices),
        ("internal_confirmation", test_indices),
    ):
        probability = gate_probability_matrix(gate_models, gate_values[indices])
        selected, applied = apply_thresholds(
            base[indices], candidate[indices], probability, thresholds
        )
        splits[name] = {
            "group_count": int(len(indices) // 81),
            "baseline": metric_bundle(truth[indices], base[indices]),
            "candidate": metric_bundle(truth[indices], selected),
            "applied_request_count": int(np.any(applied, axis=1).sum()),
            "applied_field_count": int(applied.sum()),
        }

    confirmation = splits["internal_confirmation"]
    internal_passed = bool(
        confirmation["candidate"]["joint_exact_count"]
        > confirmation["baseline"]["joint_exact_count"]
        and confirmation["candidate"]["equal_field_macro_f1"]
        >= confirmation["baseline"]["equal_field_macro_f1"]
    )
    artifact = dict(candidate_artifact)
    artifact.update(
        {
            "version": "direction_override_gate_state_v9_one_seed",
            "rules": [None] * 5,
            "gate_models": gate_models,
            "gate_thresholds": thresholds,
            "gate_source_candidate": str(candidate_path),
            "gate_source_candidate_sha256": sha256(candidate_path),
            "held_out_test_used": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "source_candidate": str(candidate_path),
        "source_candidate_sha256": sha256(candidate_path),
        "training": {
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "gate_train_groups": int(len(gate_train_groups)),
            "gate_selection_groups": int(len(gate_selection_groups)),
            "internal_confirmation_groups": int(len(test_groups)),
            "trace": training_trace,
        },
        "thresholds": thresholds,
        "threshold_trace": threshold_trace,
        "splits": splits,
        "internal_confirmation_passed": internal_passed,
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
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
