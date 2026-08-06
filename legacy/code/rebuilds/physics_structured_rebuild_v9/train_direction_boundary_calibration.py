#!/usr/bin/env python3
"""Calibrate direction boundaries on rounded Qwen state requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)
from direction_rebuild_v4.data import CLASSES, load_grid_arrays
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
    read_jsonl,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
)
from physics_structured_rebuild_v9.train_direction_expert_selector import (
    BatchHybridDirection,
)
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    DEFAULT_DIFFICULT,
    DEFAULT_FORWARD,
    DEFAULT_OLD,
    action_high_mask,
    protected_counts,
    sha256,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS, forward_feature

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_DIRECT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_LABEL_CACHE = DEFAULT_RUN / "qwen_direction_adapter_features.npz"
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_CURRENT = DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID
DEFAULT_OUTPUT = DEFAULT_RUN / "direction_boundary_calibration_v9.pkl"
LOW = np.linspace(-1.4, -0.6, 9, dtype=np.float32)
HIGH = np.linspace(0.6, 1.4, 9, dtype=np.float32)
ROUTE = "predict_direction_from_state_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-data", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--label-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--current-artifact", type=Path, default=DEFAULT_CURRENT)
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


def split(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.asarray(
        [
            int(
                hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16],
                16,
            )
            % 5
            == 0
            for group_id in group_ids
        ],
        dtype=np.bool_,
    )
    return np.flatnonzero(~calibration), np.flatnonzero(calibration)


def predict_with_boundaries(
    changes: np.ndarray,
    probabilities: np.ndarray,
    field_calibration: list[dict[str, float]],
    boundaries: np.ndarray,
) -> np.ndarray:
    prepared = prepare_direction(changes, probabilities, field_calibration)
    return predict_prepared(prepared, boundaries)


def prepare_direction(
    changes: np.ndarray,
    probabilities: np.ndarray,
    field_calibration: list[dict[str, float]],
) -> dict[str, np.ndarray]:
    """Precompute tree quantities that do not depend on new boundaries."""

    ordered = np.sort(probabilities, axis=2)
    tree_class = probabilities.argmax(axis=2)
    distance = np.abs(np.abs(changes) - 1.0)
    boundary_limit = np.asarray(
        [value["boundary_limit"] for value in field_calibration],
        dtype=np.float32,
    )
    confidence_limit = np.asarray(
        [value["confidence_limit"] for value in field_calibration],
        dtype=np.float32,
    )
    margin_limit = np.asarray(
        [value["margin_limit"] for value in field_calibration],
        dtype=np.float32,
    )
    eligible = (
        (distance <= boundary_limit[None, :])
        & (ordered[:, :, -1] >= confidence_limit[None, :])
        & (
            ordered[:, :, -1] - ordered[:, :, -2]
            >= margin_limit[None, :]
        )
    )
    return {
        "changes": np.asarray(changes, dtype=np.float32),
        "tree_class": np.asarray(tree_class, dtype=np.int64),
        "eligible": np.asarray(eligible, dtype=np.bool_),
    }


def predict_prepared(
    prepared: dict[str, np.ndarray],
    boundaries: np.ndarray,
) -> np.ndarray:
    changes = prepared["changes"]
    tree_class = prepared["tree_class"]
    low = boundaries[:, 0]
    high = boundaries[:, 1]
    threshold = np.where(
        changes < low[None, :],
        0,
        np.where(changes > high[None, :], 2, 1),
    ).astype(np.int64)
    apply = (
        prepared["eligible"]
        & (tree_class != threshold)
    )
    return np.where(apply, tree_class, threshold).astype(np.int64)


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    correct = predicted == target
    exact = np.all(correct, axis=1)
    return {
        "count": int(len(exact)),
        "all_five_exact_count": int(exact.sum()),
        "all_five_exact": float(exact.mean()),
        "per_field_correct": correct.sum(axis=0).astype(int).tolist(),
    }


def search(
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray],
    protected: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    field_calibration: list[dict[str, float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    changes, probabilities, target = calibration
    boundaries = np.asarray([[-1.0, 1.0]] * 5, dtype=np.float32)
    calibration_prepared = prepare_direction(
        changes,
        probabilities,
        field_calibration,
    )
    protected_prepared = [
        prepare_direction(
            block_changes,
            block_probabilities,
            field_calibration,
        )
        for block_changes, block_probabilities, _, _ in protected
    ]
    protected_baseline = []
    for prepared, (_, _, block_target, high) in zip(
        protected_prepared,
        protected,
        strict=True,
    ):
        baseline = predict_prepared(prepared, boundaries)
        success = np.all(baseline == block_target, axis=1)
        protected_baseline.append(protected_counts(success, high))

    def key(candidate: np.ndarray) -> tuple[Any, ...]:
        for prepared, (
            _,
            _,
            block_target,
            high,
        ), baseline_counts in zip(
            protected_prepared,
            protected,
            protected_baseline,
            strict=True,
        ):
            block_prediction = predict_prepared(prepared, candidate)
            success = np.all(block_prediction == block_target, axis=1)
            observed = protected_counts(success, high)
            if any(
                value < baseline
                for value, baseline in zip(
                    observed,
                    baseline_counts,
                    strict=True,
                )
            ):
                return (False, 0, 0, float("-inf"))
        prediction = predict_prepared(calibration_prepared, candidate)
        correct = prediction == target
        exact = np.all(correct, axis=1)
        return (
            True,
            int(exact.sum()),
            int(correct.sum()),
            -float(np.abs(candidate - np.asarray([-1.0, 1.0])).sum()),
        )

    trace = []
    for pass_index in range(4):
        changed = False
        for field in range(5):
            best = None
            for low in LOW:
                for high in HIGH:
                    proposal = boundaries.copy()
                    proposal[field] = (float(low), float(high))
                    proposal_key = key(proposal)
                    if not bool(proposal_key[0]):
                        continue
                    candidate = (
                        proposal_key,
                        -abs(float(low) + 1.0) - abs(float(high) - 1.0),
                        float(low),
                        float(high),
                    )
                    if best is None or candidate > best:
                        best = candidate
            if best is None:
                continue
            selected = np.asarray([best[2], best[3]], dtype=np.float32)
            if not np.array_equal(boundaries[field], selected):
                boundaries[field] = selected
                changed = True
        trace.append(
            {
                "pass": pass_index + 1,
                "boundaries": boundaries.tolist(),
                "calibration": metrics(
                    predict_with_boundaries(
                        changes,
                        probabilities,
                        field_calibration,
                        boundaries,
                    ),
                    target,
                ),
            }
        )
        if not changed:
            break
    return boundaries, trace


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    source = read_jsonl(args.direct_data.resolve())
    with np.load(args.label_cache.resolve(), allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        target = np.asarray(cache["labels"], dtype=np.int64)
        cached_base = np.asarray(cache["state_base_labels"], dtype=np.int64)
    if group_ids.tolist() != [str(row["group_id"]) for row in source]:
        raise ValueError("direction cache and direct rows differ")
    torch, device = configure(int(args.seed), args.device)
    base_path = args.forward_artifact.resolve()
    base, _ = load_forward_direction_runtime_v7(base_path, torch, device)
    rows = [
        {
            "group_id": str(row["group_id"]),
            "setup": row["setup"],
            "current_beam_state": row["current_beam_state"],
        }
        for row in source
    ]
    grid_prior = base.predict_changes(rows)
    prior = np.stack(
        [
            grid_prior[index, action_index(row["action"])]
            for index, row in enumerate(source)
        ]
    ).astype(np.float32)
    features51 = np.concatenate(
        [
            np.asarray(
                [
                    forward_feature(
                        row["setup"],
                        row["current_beam_state"],
                        row["action"],
                    )
                    for row in source
                ],
                dtype=np.float32,
            ),
            prior,
        ],
        axis=1,
    )
    current_path = args.current_artifact.resolve()
    current = BatchHybridDirection(current_path)
    base_prediction, probabilities, changes = current.predict(features51, prior)
    if not np.array_equal(base_prediction, cached_base):
        raise ValueError("rebuilt rounded state direction baseline differs")
    _, calibration_indices = split(group_ids, int(args.seed))

    protected = []
    protected_names = []
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_prior = stream_forward_changes(path, base)
        block_features51 = np.concatenate(
            [arrays.features, block_prior],
            axis=1,
        ).astype(np.float32)
        _, block_probabilities, block_changes = current.predict(
            block_features51,
            block_prior,
        )
        protected.append(
            (
                block_changes,
                block_probabilities,
                arrays.labels,
                action_high_mask(arrays.group_count),
            )
        )
        protected_names.append(name)
    boundaries, trace = search(
        (
            changes[calibration_indices],
            probabilities[calibration_indices],
            target[calibration_indices],
        ),
        protected,
        current.calibrations,
    )
    protected_report = {}
    default_boundaries = np.asarray([[-1.0, 1.0]] * 5, dtype=np.float32)
    for name, (
        block_changes,
        block_probabilities,
        block_target,
        high,
    ) in zip(protected_names, protected, strict=True):
        baseline = predict_with_boundaries(
            block_changes,
            block_probabilities,
            current.calibrations,
            default_boundaries,
        )
        selected = predict_with_boundaries(
            block_changes,
            block_probabilities,
            current.calibrations,
            boundaries,
        )
        protected_report[name] = {
            "current": protected_counts(
                np.all(baseline == block_target, axis=1),
                high,
            ),
            "selected": protected_counts(
                np.all(selected == block_target, axis=1),
                high,
            ),
        }

    qwen_data = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if row["target_decision"].get("route_name") == ROUTE
    ]
    system_rows = [
        {
            "group_id": str(row["example_id"]),
            "setup": row["target_decision"]["arguments"]["setup"],
            "current_beam_state": row["target_decision"]["arguments"][
                "current_beam_state"
            ],
        }
        for row in canonical
    ]
    system_grid_prior = base.predict_changes(system_rows)
    system_prior = np.stack(
        [
            system_grid_prior[index, action_index(
                row["target_decision"]["arguments"]["action"]
            )]
            for index, row in enumerate(canonical)
        ]
    ).astype(np.float32)
    system_features51 = np.concatenate(
        [
            np.asarray(
                [
                    forward_feature(
                        row["target_decision"]["arguments"]["setup"],
                        row["target_decision"]["arguments"][
                            "current_beam_state"
                        ],
                        row["target_decision"]["arguments"]["action"],
                    )
                    for row in canonical
                ],
                dtype=np.float32,
            ),
            system_prior,
        ],
        axis=1,
    )
    system_base, system_probabilities, system_changes = current.predict(
        system_features51,
        system_prior,
    )
    system_selected = predict_with_boundaries(
        system_changes,
        system_probabilities,
        current.calibrations,
        boundaries,
    )
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen_data / "private/source_cases/val.jsonl")
    }
    class_index = {str(value): index for index, value in enumerate(CLASSES)}
    system_truth = [
        simulator_forward_truth(
            private_by_group[str(row["group_id"])],
            row["target_decision"]["arguments"]["action"],
        )
        for row in canonical
    ]
    system_target = np.asarray(
        [
            [
                class_index[
                    str(system_truth[index]["directions"][field])
                ]
                for field in DIRECTION_FIELDS
            ]
            for index in range(len(canonical))
        ],
        dtype=np.int64,
    )

    artifact = {
        "version": "direction_boundary_calibration_v9_one_seed",
        "model": "protected_direction_boundary_calibration_v9",
        "seed": int(args.seed),
        "boundaries": {
            field: {
                "decrease_below": float(boundaries[index, 0]),
                "increase_above": float(boundaries[index, 1]),
            }
            for index, field in enumerate(DIRECTION_FIELDS)
        },
        "base_direction_artifact": str(current_path),
        "base_direction_artifact_sha256": sha256(current_path),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    calibration_baseline = predict_with_boundaries(
        changes[calibration_indices],
        probabilities[calibration_indices],
        current.calibrations,
        default_boundaries,
    )
    calibration_selected = predict_with_boundaries(
        changes[calibration_indices],
        probabilities[calibration_indices],
        current.calibrations,
        boundaries,
    )
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "boundaries": artifact["boundaries"],
        "calibration": {
            "current": metrics(
                calibration_baseline,
                target[calibration_indices],
            ),
            "selected": metrics(
                calibration_selected,
                target[calibration_indices],
            ),
        },
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
        "system_validation": {
            "current": metrics(system_base, system_target),
            "selected": metrics(system_selected, system_target),
            "current_only_count": int(
                np.sum(
                    np.all(system_base == system_target, axis=1)
                    & ~np.all(system_selected == system_target, axis=1)
                )
            ),
            "selected_only_count": int(
                np.sum(
                    ~np.all(system_base == system_target, axis=1)
                    & np.all(system_selected == system_target, axis=1)
                )
            ),
        },
        "trace": trace,
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
