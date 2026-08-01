#!/usr/bin/env python3
"""Calibrate route-specific direction gates on frozen validation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from direction_rebuild_v4.data import (
    CLASSES,
    CLASS_TO_INDEX,
    labels_from_normalized_change,
    load_grid_arrays,
)
from joint_forward_direction_v6.train import geometric_score
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from joint_forward_direction_v7.train import direction_metric_bundle
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    DIRECTION_FIELDS,
    forward_feature,
)

DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_TREE = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed/direction_tree_v4.pkl"
)
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
DEFAULT_QWEN_DATA = (
    REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
)
DEFAULT_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_system_aware"
)

ROUTES = (
    "predict_direction_from_state_v1",
    "predict_direction_from_image_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--tree-artifact", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN_DATA)
    parser.add_argument("--v4-overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def action_index(action: Mapping[str, Any]) -> int:
    for index, candidate in enumerate(ACTION_GRID):
        if all(
            float(action[field]) == float(candidate[field])
            for field in ACTION_FIELDS
        ):
            return index
    raise ValueError("direction request action is outside the 81-action grid")


def tree_probabilities(
    models: list[Any],
    features: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(features), 5, 3), dtype=np.float32)
    for field, model in enumerate(models):
        probability = model.predict_proba(features)
        output[:, field, np.asarray(model.classes_, dtype=np.int64)] = (
            probability
        )
    return output


def tree_features(
    features: np.ndarray,
    changes: np.ndarray,
    feature_mode: str,
    residual_forward: np.ndarray | None = None,
) -> np.ndarray:
    if feature_mode == "engineered_46":
        return features
    base = np.concatenate(
        [
            np.asarray(features, dtype=np.float32),
            np.asarray(changes, dtype=np.float32),
        ],
        axis=1,
    )
    if feature_mode == "engineered_46_plus_forward_v7_change_5":
        return base
    if (
        feature_mode
        == (
            "engineered_46_plus_forward_v7_change_5"
            "_plus_residual_forward_5"
        )
    ):
        if residual_forward is None:
            raise ValueError(
                "residual-forward direction features were not supplied"
            )
        return np.concatenate(
            [
                base,
                np.asarray(residual_forward, dtype=np.float32),
            ],
            axis=1,
        )
    raise ValueError(f"unsupported direction tree feature mode: {feature_mode}")


def gated_predictions(
    changes: np.ndarray,
    probabilities: np.ndarray,
    calibration: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    threshold = labels_from_normalized_change(changes)
    ordered = np.sort(probabilities, axis=-1)
    tree_class = probabilities.argmax(axis=-1)
    distance = np.abs(np.abs(changes) - 1.0)
    apply = (
        (distance <= float(calibration["boundary_limit"]))
        & (
            ordered[..., -1]
            >= float(calibration["confidence_limit"])
        )
        & (
            ordered[..., -1] - ordered[..., -2]
            >= float(calibration["margin_limit"])
        )
        & (tree_class != threshold)
    )
    return np.where(apply, tree_class, threshold).astype(np.int64), apply


def system_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    if target.shape != predicted.shape or target.ndim != 2:
        raise ValueError("system direction arrays must have matching N x 5 shape")
    joint = np.all(target == predicted, axis=1)
    field_metrics = {}
    macro_values = []
    for field_index, field in enumerate(DIRECTION_FIELDS):
        target_field = target[:, field_index]
        predicted_field = predicted[:, field_index]
        class_scores = []
        for class_index, class_name in enumerate(CLASSES):
            true_positive = int(
                np.sum(
                    (target_field == class_index)
                    & (predicted_field == class_index)
                )
            )
            false_positive = int(
                np.sum(
                    (target_field != class_index)
                    & (predicted_field == class_index)
                )
            )
            false_negative = int(
                np.sum(
                    (target_field == class_index)
                    & (predicted_field != class_index)
                )
            )
            denominator = (
                2 * true_positive + false_positive + false_negative
            )
            class_scores.append(
                2.0 * true_positive / denominator if denominator else 0.0
            )
        macro = float(np.mean(class_scores))
        macro_values.append(macro)
        field_metrics[field] = {
            "accuracy": float(np.mean(target_field == predicted_field)),
            "macro_f1": macro,
        }
    return {
        "count": int(len(target)),
        "all_five_exact_count": int(joint.sum()),
        "all_five_exact": float(joint.mean()),
        "equal_field_macro_f1": float(np.mean(macro_values)),
        "per_field": field_metrics,
    }


def specialist_values(
    old_metrics: Mapping[str, Any],
    difficult_metrics: Mapping[str, Any],
) -> tuple[float, ...]:
    return (
        float(old_metrics["overall"]["joint_exact"]),
        float(difficult_metrics["overall"]["joint_exact"]),
        float(
            old_metrics["by_action_complexity"]["three_or_four"][
                "joint_exact"
            ]
        ),
        float(
            difficult_metrics["by_action_complexity"]["three_or_four"][
                "joint_exact"
            ]
        ),
    )


def is_non_regression(
    values: tuple[float, ...],
    baseline: tuple[float, ...],
) -> bool:
    return all(
        candidate + 1e-12 >= reference
        for candidate, reference in zip(values, baseline, strict=True)
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "system_aware_summary.json"
    cache_path = output_dir / "direction_validation_cache.npz"
    artifact_paths = {
        ROUTES[0]: output_dir / "hybrid_direction_state.pkl",
        ROUTES[1]: output_dir / "hybrid_direction_image.pkl",
    }
    if (
        summary_path.exists()
        or cache_path.exists()
        or any(path.exists() for path in artifact_paths.values())
    ):
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, device = configure(int(args.seed), args.device)
    forward_path = args.forward_artifact.resolve()
    tree_path = args.tree_artifact.resolve()
    shared, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )
    with tree_path.open("rb") as stream:
        tree_artifact = pickle.load(stream)
    if tree_artifact.get("model") not in {
        "balanced_five_head_hist_gradient_boosting_direction_v4",
        "balanced_five_head_extra_trees_direction_v9",
        "balanced_five_head_lightgbm_direction_v9",
        "balanced_field_selected_direction_tree_v9",
    }:
        raise ValueError("unexpected direction-tree artifact")
    models = list(tree_artifact["models"])
    tree_feature_mode = str(
        tree_artifact.get("feature_mode", "engineered_46")
    )
    residual_forward_models = None
    residual_forward_path = tree_artifact.get(
        "residual_forward_tree_artifact"
    )
    if residual_forward_path is not None:
        with Path(str(residual_forward_path)).resolve().open("rb") as stream:
            residual_forward_artifact = pickle.load(stream)
        residual_forward_models = list(
            residual_forward_artifact["models"]
        )

    old_path = args.old_validation.resolve()
    difficult_path = args.difficult_validation.resolve()
    old_rows = read_jsonl(old_path)
    difficult_rows = read_jsonl(difficult_path)
    old_arrays = load_grid_arrays(
        old_path,
        include_legacy_features=False,
    )
    difficult_arrays = load_grid_arrays(
        difficult_path,
        include_legacy_features=False,
    )
    old_changes = shared.predict_changes(old_rows).reshape(-1, 5)
    difficult_changes = shared.predict_changes(difficult_rows).reshape(-1, 5)
    old_tree = tree_probabilities(
        models,
        tree_features(
            old_arrays.features,
            old_changes,
            tree_feature_mode,
            (
                None
                if residual_forward_models is None
                else np.stack(
                    [
                        model.predict(
                            tree_features(
                                old_arrays.features,
                                old_changes,
                                (
                                    "engineered_46_plus_"
                                    "forward_v7_change_5"
                                ),
                            )
                        )
                        for model in residual_forward_models
                    ],
                    axis=1,
                )
            ),
        ),
    )
    difficult_tree = tree_probabilities(
        models,
        tree_features(
            difficult_arrays.features,
            difficult_changes,
            tree_feature_mode,
            (
                None
                if residual_forward_models is None
                else np.stack(
                    [
                        model.predict(
                            tree_features(
                                difficult_arrays.features,
                                difficult_changes,
                                (
                                    "engineered_46_plus_"
                                    "forward_v7_change_5"
                                ),
                            )
                        )
                        for model in residual_forward_models
                    ],
                    axis=1,
                )
            ),
        ),
    )

    qwen_data = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if str(row["category"]) in ROUTES
    ]
    if len(canonical) != 300:
        raise ValueError(
            f"expected 300 direction validation requests, found {len(canonical)}"
        )
    private_by_group = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen_data / "private/source_cases/val.jsonl")
    }
    measurement_backend = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.v4_overlay.resolve(),
        device,
    )
    system_rows = []
    truth_cache: dict[tuple[str, tuple[float, ...]], dict[str, Any]] = {}
    for index, row in enumerate(canonical):
        target = row["target_decision"]
        arguments = target["arguments"]
        route = str(target["route_name"])
        if route == ROUTES[0]:
            current = arguments["current_beam_state"]
            measurement_source = "provided_numerical_state"
        else:
            image_path = qwen_data / str(row["images"][0])
            measured = measurement_backend.visual.measure_image(
                image_path,
                arguments["image_calibration"],
            )
            legacy = sensor_to_base_legacy(
                measured["beam_state"],
                [arguments["setup"]],
            )[0]
            current = state_mapping(legacy)
            measurement_source = str(measured["measurement_source"])
        action = arguments["action"]
        private = private_by_group[str(row["group_id"])]
        truth_key = (
            str(row["group_id"]),
            tuple(float(action[field]) for field in ACTION_FIELDS),
        )
        if truth_key not in truth_cache:
            truth_cache[truth_key] = simulator_forward_truth(private, action)
        truth = truth_cache[truth_key]["directions"]
        system_rows.append(
            {
                "example_id": str(row["example_id"]),
                "route": route,
                "setup": arguments["setup"],
                "current": current,
                "action": action,
                "action_index": action_index(action),
                "measurement_source": measurement_source,
                "truth": [
                    CLASS_TO_INDEX[str(truth[field])]
                    for field in DIRECTION_FIELDS
                ],
            }
        )
        if (index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "prepared_direction_requests": index + 1,
                        "total": len(canonical),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    forward_rows = [
        {
            "group_id": row["example_id"],
            "setup": row["setup"],
            "current_beam_state": row["current"],
        }
        for row in system_rows
    ]
    system_grid_changes = shared.predict_changes(forward_rows)
    system_changes = np.stack(
        [
            system_grid_changes[index, int(row["action_index"])]
            for index, row in enumerate(system_rows)
        ]
    )
    system_features = np.stack(
        [
            forward_feature(row["setup"], row["current"], row["action"])
            for row in system_rows
        ]
    )
    system_tree = tree_probabilities(
        models,
        tree_features(
            system_features,
            system_changes,
            tree_feature_mode,
            (
                None
                if residual_forward_models is None
                else np.stack(
                    [
                        model.predict(
                            tree_features(
                                system_features,
                                system_changes,
                                (
                                    "engineered_46_plus_"
                                    "forward_v7_change_5"
                                ),
                            )
                        )
                        for model in residual_forward_models
                    ],
                    axis=1,
                )
            ),
        ),
    )
    system_truth = np.asarray(
        [row["truth"] for row in system_rows],
        dtype=np.int64,
    )
    route_values = np.asarray(
        [row["route"] for row in system_rows],
        dtype=np.str_,
    )
    np.savez_compressed(
        cache_path,
        changes=system_changes,
        tree_probabilities=system_tree,
        truth=system_truth,
        routes=route_values,
        example_ids=np.asarray(
            [row["example_id"] for row in system_rows],
            dtype=np.str_,
        ),
    )

    calibration_grid = [
        {
            "boundary_limit": float(boundary),
            "confidence_limit": float(confidence),
            "margin_limit": float(margin),
        }
        for boundary in (
            -1.0,
            0.10,
            0.20,
            0.25,
            0.35,
            0.50,
            0.75,
            1.0,
            2.0,
            5.0,
            1.0e9,
        )
        for confidence in (0.50, 0.60, 0.70, 0.80, 0.90)
        for margin in (0.0, 0.10, 0.20, 0.30, 0.40)
    ]
    baseline_calibration = {
        "boundary_limit": -1.0,
        "confidence_limit": 0.50,
        "margin_limit": 0.0,
    }
    old_baseline_prediction, _ = gated_predictions(
        old_changes,
        old_tree,
        baseline_calibration,
    )
    difficult_baseline_prediction, _ = gated_predictions(
        difficult_changes,
        difficult_tree,
        baseline_calibration,
    )
    old_baseline = direction_metric_bundle(
        old_arrays,
        old_baseline_prediction,
    )
    difficult_baseline = direction_metric_bundle(
        difficult_arrays,
        difficult_baseline_prediction,
    )
    baseline_values = specialist_values(old_baseline, difficult_baseline)

    candidate_rows = []
    for calibration in calibration_grid:
        old_prediction, old_apply = gated_predictions(
            old_changes,
            old_tree,
            calibration,
        )
        difficult_prediction, difficult_apply = gated_predictions(
            difficult_changes,
            difficult_tree,
            calibration,
        )
        old_metrics = direction_metric_bundle(old_arrays, old_prediction)
        difficult_metrics = direction_metric_bundle(
            difficult_arrays,
            difficult_prediction,
        )
        values = specialist_values(old_metrics, difficult_metrics)
        system_prediction, system_apply = gated_predictions(
            system_changes,
            system_tree,
            calibration,
        )
        per_route = {}
        for route in ROUTES:
            mask = route_values == route
            per_route[route] = {
                "metrics": system_metrics(
                    system_truth[mask],
                    system_prediction[mask],
                ),
                "correction_count": int(system_apply[mask].sum()),
            }
        candidate_rows.append(
            {
                "calibration": calibration,
                "specialist_values": values,
                "specialist_score": geometric_score(list(values)),
                "specialist_non_regression": is_non_regression(
                    values,
                    baseline_values,
                ),
                "specialist": {
                    "old_iid": old_metrics,
                    "difficult": difficult_metrics,
                },
                "specialist_correction_counts": {
                    "old_iid": int(old_apply.sum()),
                    "difficult": int(difficult_apply.sum()),
                },
                "system": per_route,
            }
        )

    selected_by_route = {}
    for route in ROUTES:
        eligible = [
            row for row in candidate_rows if row["specialist_non_regression"]
        ]
        if not eligible:
            raise RuntimeError("no non-regressing direction gate was found")
        selected_by_route[route] = max(
            eligible,
            key=lambda row: (
                row["system"][route]["metrics"]["all_five_exact"],
                row["system"][route]["metrics"]["equal_field_macro_f1"],
                row["specialist_score"],
                -row["system"][route]["correction_count"],
            ),
        )

    baseline_row = next(
        row
        for row in candidate_rows
        if row["calibration"] == baseline_calibration
    )
    combined_baseline, _ = gated_predictions(
        system_changes,
        system_tree,
        baseline_calibration,
    )
    combined_selected = combined_baseline.copy()
    for route in ROUTES:
        mask = route_values == route
        route_prediction, _ = gated_predictions(
            system_changes[mask],
            system_tree[mask],
            selected_by_route[route]["calibration"],
        )
        combined_selected[mask] = route_prediction

    artifacts = {}
    for route, artifact_path in artifact_paths.items():
        selected = selected_by_route[route]
        artifact = {
            "version": (
                "hybrid_system_aware_state_direction_v9_one_seed"
                if route == ROUTES[0]
                else "hybrid_system_aware_image_direction_v9_one_seed"
            ),
            "model": "frozen_v7_forward_plus_boundary_tree_v4",
            "route_scope": route,
            "forward_artifact": str(forward_path),
            "forward_artifact_sha256": sha256(forward_path),
            "tree_artifact": str(tree_path),
            "tree_artifact_sha256": sha256(tree_path),
            "calibration": selected["calibration"],
            "held_out_test_used": False,
        }
        with artifact_path.open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        artifacts[route] = {
            "path": str(artifact_path),
            "sha256": sha256(artifact_path),
            "calibration": artifact["calibration"],
        }

    summary = {
        "version": "system_aware_hybrid_direction_v9_one_seed",
        "artifacts": artifacts,
        "candidate_count": len(candidate_rows),
        "eligible_non_regression_candidate_count": int(
            sum(row["specialist_non_regression"] for row in candidate_rows)
        ),
        "selection_rule": (
            "For each route, maximize all-five exact then equal-field "
            "macro-F1 among gates that do not reduce old-IID, difficult, "
            "old high-complexity, or difficult high-complexity joint exact."
        ),
        "specialist_validation_baseline": {
            "old_iid": old_baseline,
            "difficult": difficult_baseline,
            "selection_values": baseline_values,
        },
        "route_validation": {
            route: {
                "baseline": baseline_row["system"][route],
                "selected": selected_by_route[route]["system"][route],
                "selected_specialist": selected_by_route[route]["specialist"],
                "selected_specialist_values": selected_by_route[route][
                    "specialist_values"
                ],
                "selected_specialist_correction_counts": selected_by_route[
                    route
                ]["specialist_correction_counts"],
            }
            for route in ROUTES
        },
        "combined_system_validation": {
            "baseline": system_metrics(system_truth, combined_baseline),
            "selected": system_metrics(system_truth, combined_selected),
        },
        "data_contract": {
            "system_direction_request_count": len(system_rows),
            "unique_physical_truth_calls": len(truth_cache),
            "state_request_count": int(np.sum(route_values == ROUTES[0])),
            "image_request_count": int(np.sum(route_values == ROUTES[1])),
            "old_validation": str(old_path),
            "difficult_validation": str(difficult_path),
            "qwen_validation": str(
                (qwen_data / "canonical/val.jsonl").resolve()
            ),
            "cache": str(cache_path),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "artifacts": artifacts,
                "candidate_count": len(candidate_rows),
                "eligible_non_regression_candidate_count": summary[
                    "eligible_non_regression_candidate_count"
                ],
                "combined_system_validation": summary[
                    "combined_system_validation"
                ],
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
