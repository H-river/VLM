#!/usr/bin/env python3
"""Calibrate one protected hybrid gate per direction field and route."""

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

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from joint_forward_direction_v7.train import direction_metric_bundle
from physics_structured_rebuild_v9.calibrate_system_direction import (
    ROUTES,
    gated_predictions,
    read_jsonl,
    sha256,
    system_metrics,
    tree_features,
    tree_probabilities,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    DIRECTION_FIELDS,
)

DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_TREE = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "direction_tree_targeted/direction_tree_targeted_v9.pkl"
)
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
DEFAULT_SYSTEM_CACHE = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_targeted_tree/direction_validation_cache.npz"
)
DEFAULT_GLOBAL_DIR = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_targeted_tree"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_fieldwise"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--tree-artifact", type=Path, default=DEFAULT_TREE)
    parser.add_argument(
        "--secondary-tree-artifact",
        type=Path,
        help="Optional second tree whose fields can be selected independently.",
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--system-cache", type=Path, default=DEFAULT_SYSTEM_CACHE)
    parser.add_argument(
        "--secondary-system-cache",
        type=Path,
        help="System cache corresponding to --secondary-tree-artifact.",
    )
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def calibration_grid() -> list[dict[str, float]]:
    return [
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


def calibration_index(
    grid: list[dict[str, float]],
    value: dict[str, float],
) -> int:
    return next(
        index for index, candidate in enumerate(grid) if candidate == value
    )


def action_complexity_mask(group_count: int) -> np.ndarray:
    per_action = np.asarray(
        [
            sum(
                abs(float(action[field])) > 0.0
                for field in ACTION_FIELDS
            )
            >= 3
            for action in ACTION_GRID
        ],
        dtype=np.bool_,
    )
    return np.tile(per_action, group_count)


def combined_prediction(
    predictions: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            predictions[int(selected[field]), :, field]
            for field in range(len(DIRECTION_FIELDS))
        ],
        axis=1,
    )


def protected_counts(
    old_prediction: np.ndarray,
    difficult_prediction: np.ndarray,
    old_labels: np.ndarray,
    difficult_labels: np.ndarray,
    old_high: np.ndarray,
    difficult_high: np.ndarray,
) -> tuple[int, int, int, int]:
    old_exact = np.all(old_prediction == old_labels, axis=1)
    difficult_exact = np.all(
        difficult_prediction == difficult_labels,
        axis=1,
    )
    return (
        int(old_exact.sum()),
        int(difficult_exact.sum()),
        int(old_exact[old_high].sum()),
        int(difficult_exact[difficult_high].sum()),
    )


def coordinate_search(
    *,
    route_mask: np.ndarray,
    system_truth: np.ndarray,
    system_predictions: np.ndarray,
    old_predictions: np.ndarray,
    difficult_predictions: np.ndarray,
    old_labels: np.ndarray,
    difficult_labels: np.ndarray,
    old_high: np.ndarray,
    difficult_high: np.ndarray,
    baseline_protected: tuple[int, int, int, int],
    correction_counts: np.ndarray,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    traces = []
    final_candidates = []
    candidate_count = system_predictions.shape[0]

    for start_index, start in enumerate(starts):
        selected = start.copy()
        start_trace = []
        for pass_index in range(5):
            pass_changed = False
            for field in range(len(DIRECTION_FIELDS)):
                system_other = np.ones(int(route_mask.sum()), dtype=np.bool_)
                old_other = np.ones(len(old_labels), dtype=np.bool_)
                difficult_other = np.ones(
                    len(difficult_labels),
                    dtype=np.bool_,
                )
                system_field_total = 0
                for other in range(len(DIRECTION_FIELDS)):
                    if other == field:
                        continue
                    system_other &= (
                        system_predictions[
                            int(selected[other]),
                            route_mask,
                            other,
                        ]
                        == system_truth[route_mask, other]
                    )
                    old_other &= (
                        old_predictions[
                            int(selected[other]),
                            :,
                            other,
                        ]
                        == old_labels[:, other]
                    )
                    difficult_other &= (
                        difficult_predictions[
                            int(selected[other]),
                            :,
                            other,
                        ]
                        == difficult_labels[:, other]
                    )
                    system_field_total += int(
                        np.sum(
                            system_predictions[
                                int(selected[other]),
                                route_mask,
                                other,
                            ]
                            == system_truth[route_mask, other]
                        )
                    )

                best_index = int(selected[field])
                best_key = None
                for candidate in range(candidate_count):
                    old_field = (
                        old_predictions[candidate, :, field]
                        == old_labels[:, field]
                    )
                    difficult_field = (
                        difficult_predictions[candidate, :, field]
                        == difficult_labels[:, field]
                    )
                    old_joint = old_other & old_field
                    difficult_joint = (
                        difficult_other & difficult_field
                    )
                    protected = (
                        int(old_joint.sum()),
                        int(difficult_joint.sum()),
                        int(old_joint[old_high].sum()),
                        int(difficult_joint[difficult_high].sum()),
                    )
                    if any(
                        observed < minimum
                        for observed, minimum in zip(
                            protected,
                            baseline_protected,
                            strict=True,
                        )
                    ):
                        continue
                    system_field = (
                        system_predictions[
                            candidate,
                            route_mask,
                            field,
                        ]
                        == system_truth[route_mask, field]
                    )
                    system_joint = int(
                        np.sum(system_other & system_field)
                    )
                    field_total = (
                        system_field_total + int(system_field.sum())
                    )
                    key = (
                        system_joint,
                        field_total,
                        sum(protected),
                        -int(correction_counts[candidate, field]),
                        -candidate,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_index = candidate
                if best_index != int(selected[field]):
                    selected[field] = best_index
                    pass_changed = True
            prediction = combined_prediction(
                system_predictions[:, route_mask, :],
                selected,
            )
            start_trace.append(
                {
                    "pass": pass_index + 1,
                    "selected_indices": selected.tolist(),
                    "system": system_metrics(
                        system_truth[route_mask],
                        prediction,
                    ),
                }
            )
            if not pass_changed:
                break
        traces.append(
            {
                "start_index": start_index,
                "passes": start_trace,
            }
        )
        final_candidates.append(selected.copy())

    def final_key(selected: np.ndarray) -> tuple[Any, ...]:
        system_prediction = combined_prediction(
            system_predictions[:, route_mask, :],
            selected,
        )
        metrics = system_metrics(
            system_truth[route_mask],
            system_prediction,
        )
        old_prediction = combined_prediction(old_predictions, selected)
        difficult_prediction = combined_prediction(
            difficult_predictions,
            selected,
        )
        protected = protected_counts(
            old_prediction,
            difficult_prediction,
            old_labels,
            difficult_labels,
            old_high,
            difficult_high,
        )
        return (
            metrics["all_five_exact_count"],
            metrics["equal_field_macro_f1"],
            sum(protected),
            -sum(
                int(correction_counts[int(selected[field]), field])
                for field in range(len(DIRECTION_FIELDS))
            ),
        )

    return max(final_candidates, key=final_key), traces


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "fieldwise_summary.json"
    artifact_paths = {
        ROUTES[0]: output_dir / "hybrid_direction_state.pkl",
        ROUTES[1]: output_dir / "hybrid_direction_image.pkl",
    }
    if summary_path.exists() or any(
        path.exists() for path in artifact_paths.values()
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
        tree = pickle.load(stream)
    if tree.get("model") not in {
        "balanced_five_head_hist_gradient_boosting_direction_v4",
        "balanced_five_head_extra_trees_direction_v9",
        "balanced_five_head_lightgbm_direction_v9",
        "balanced_field_selected_direction_tree_v9",
    }:
        raise ValueError("unexpected direction-tree artifact")
    models = list(tree["models"])
    feature_mode = str(tree["feature_mode"])
    secondary_tree_path = (
        None
        if args.secondary_tree_artifact is None
        else args.secondary_tree_artifact.resolve()
    )
    secondary_models = None
    if secondary_tree_path is not None:
        if args.secondary_system_cache is None:
            raise ValueError(
                "--secondary-system-cache is required with a secondary tree"
            )
        with secondary_tree_path.open("rb") as stream:
            secondary_tree = pickle.load(stream)
        if secondary_tree.get("model") not in {
            "balanced_five_head_hist_gradient_boosting_direction_v4",
            "balanced_five_head_extra_trees_direction_v9",
            "balanced_five_head_lightgbm_direction_v9",
        }:
            raise ValueError("unexpected secondary direction-tree artifact")
        if str(secondary_tree["feature_mode"]) != feature_mode:
            raise ValueError("direction-tree feature modes differ")
        secondary_models = list(secondary_tree["models"])

    old_path = args.old_validation.resolve()
    difficult_path = args.difficult_validation.resolve()
    old_arrays = load_grid_arrays(
        old_path,
        include_legacy_features=False,
    )
    difficult_arrays = load_grid_arrays(
        difficult_path,
        include_legacy_features=False,
    )
    old_changes = shared.predict_changes(
        read_jsonl(old_path)
    ).reshape(-1, 5)
    difficult_changes = shared.predict_changes(
        read_jsonl(difficult_path)
    ).reshape(-1, 5)
    old_probability = tree_probabilities(
        models,
        tree_features(
            old_arrays.features,
            old_changes,
            feature_mode,
        ),
    )
    difficult_probability = tree_probabilities(
        models,
        tree_features(
            difficult_arrays.features,
            difficult_changes,
            feature_mode,
        ),
    )
    old_secondary_probability = None
    difficult_secondary_probability = None
    if secondary_models is not None:
        old_secondary_probability = tree_probabilities(
            secondary_models,
            tree_features(
                old_arrays.features,
                old_changes,
                feature_mode,
            ),
        )
        difficult_secondary_probability = tree_probabilities(
            secondary_models,
            tree_features(
                difficult_arrays.features,
                difficult_changes,
                feature_mode,
            ),
        )

    cache = np.load(args.system_cache.resolve(), allow_pickle=False)
    system_changes = np.asarray(cache["changes"], dtype=np.float32)
    system_probability = np.asarray(
        cache["tree_probabilities"],
        dtype=np.float32,
    )
    system_truth = np.asarray(cache["truth"], dtype=np.int64)
    routes = np.asarray(cache["routes"], dtype=np.str_)
    if (
        system_changes.shape != (300, 5)
        or system_probability.shape != (300, 5, 3)
        or system_truth.shape != (300, 5)
    ):
        raise ValueError("system direction cache shapes differ")
    system_secondary_probability = None
    secondary_cache_path = None
    if secondary_models is not None:
        secondary_cache_path = args.secondary_system_cache.resolve()
        secondary_cache = np.load(
            secondary_cache_path,
            allow_pickle=False,
        )
        for key in ("changes", "truth", "routes", "example_ids"):
            if not np.array_equal(cache[key], secondary_cache[key]):
                raise ValueError(
                    f"primary and secondary system caches differ for {key}"
                )
        system_secondary_probability = np.asarray(
            secondary_cache["tree_probabilities"],
            dtype=np.float32,
        )

    grid = calibration_grid()
    sources = (
        ("primary",)
        if secondary_models is None
        else ("primary", "secondary")
    )
    candidate_specs = [
        (source, calibration)
        for source in sources
        for calibration in grid
    ]
    old_predictions = np.empty(
        (len(candidate_specs), len(old_arrays.labels), 5),
        dtype=np.int8,
    )
    difficult_predictions = np.empty(
        (len(candidate_specs), len(difficult_arrays.labels), 5),
        dtype=np.int8,
    )
    system_predictions = np.empty(
        (len(candidate_specs), len(system_truth), 5),
        dtype=np.int8,
    )
    correction_counts = np.empty(
        (len(candidate_specs), 5),
        dtype=np.int64,
    )
    for index, (source, calibration) in enumerate(candidate_specs):
        if source == "primary":
            selected_old_probability = old_probability
            selected_difficult_probability = difficult_probability
            selected_system_probability = system_probability
        else:
            selected_old_probability = old_secondary_probability
            selected_difficult_probability = difficult_secondary_probability
            selected_system_probability = system_secondary_probability
        if (
            selected_old_probability is None
            or selected_difficult_probability is None
            or selected_system_probability is None
        ):
            raise ValueError("secondary direction probabilities are unavailable")
        old_prediction, _ = gated_predictions(
            old_changes,
            selected_old_probability,
            calibration,
        )
        difficult_prediction, _ = gated_predictions(
            difficult_changes,
            selected_difficult_probability,
            calibration,
        )
        system_prediction, system_apply = gated_predictions(
            system_changes,
            selected_system_probability,
            calibration,
        )
        old_predictions[index] = old_prediction
        difficult_predictions[index] = difficult_prediction
        system_predictions[index] = system_prediction
        correction_counts[index] = system_apply.sum(axis=0)

    baseline_calibration = {
        "boundary_limit": -1.0,
        "confidence_limit": 0.50,
        "margin_limit": 0.0,
    }
    def candidate_index(
        source: str,
        calibration: dict[str, float],
    ) -> int:
        return next(
            index
            for index, candidate in enumerate(candidate_specs)
            if candidate[0] == source and candidate[1] == calibration
        )

    baseline_index = candidate_index("primary", baseline_calibration)
    baseline_selected = np.full(5, baseline_index, dtype=np.int64)
    old_high = action_complexity_mask(old_arrays.group_count)
    difficult_high = action_complexity_mask(
        difficult_arrays.group_count
    )
    baseline_protected = protected_counts(
        old_predictions[baseline_index],
        difficult_predictions[baseline_index],
        old_arrays.labels,
        difficult_arrays.labels,
        old_high,
        difficult_high,
    )

    selected_by_route = {}
    traces = {}
    for route in ROUTES:
        global_path = (
            args.global_dir.resolve()
            / (
                "hybrid_direction_state.pkl"
                if route == ROUTES[0]
                else "hybrid_direction_image.pkl"
            )
        )
        with global_path.open("rb") as stream:
            global_artifact = pickle.load(stream)
        global_index = candidate_index(
            "primary",
            dict(global_artifact["calibration"]),
        )
        full_tree_index = candidate_index(
            "primary",
            {
                "boundary_limit": 1.0e9,
                "confidence_limit": 0.50,
                "margin_limit": 0.0,
            },
        )
        starts = [
            baseline_selected,
            np.full(5, global_index, dtype=np.int64),
            np.full(5, full_tree_index, dtype=np.int64),
        ]
        if secondary_models is not None:
            secondary_full_tree_index = candidate_index(
                "secondary",
                {
                    "boundary_limit": 1.0e9,
                    "confidence_limit": 0.50,
                    "margin_limit": 0.0,
                },
            )
            starts.append(
                np.full(
                    5,
                    secondary_full_tree_index,
                    dtype=np.int64,
                )
            )
        selected, route_trace = coordinate_search(
            route_mask=routes == route,
            system_truth=system_truth,
            system_predictions=system_predictions,
            old_predictions=old_predictions,
            difficult_predictions=difficult_predictions,
            old_labels=old_arrays.labels,
            difficult_labels=difficult_arrays.labels,
            old_high=old_high,
            difficult_high=difficult_high,
            baseline_protected=baseline_protected,
            correction_counts=correction_counts,
            starts=starts,
        )
        selected_by_route[route] = selected
        traces[route] = route_trace

    artifacts = {}
    route_validation = {}
    combined_system = np.empty_like(system_truth)
    for route, artifact_path in artifact_paths.items():
        selected = selected_by_route[route]
        route_mask = routes == route
        system_prediction = combined_prediction(
            system_predictions[:, route_mask, :],
            selected,
        )
        combined_system[route_mask] = system_prediction
        old_prediction = combined_prediction(old_predictions, selected)
        difficult_prediction = combined_prediction(
            difficult_predictions,
            selected,
        )
        field_calibration = {
            field: candidate_specs[int(selected[index])][1]
            for index, field in enumerate(DIRECTION_FIELDS)
        }
        field_tree_source = {
            field: candidate_specs[int(selected[index])][0]
            for index, field in enumerate(DIRECTION_FIELDS)
        }
        artifact = {
            "version": (
                "fieldwise_targeted_tree_state_direction_v9_one_seed"
                if route == ROUTES[0]
                else "fieldwise_targeted_tree_image_direction_v9_one_seed"
            ),
            "model": "frozen_v7_forward_plus_boundary_tree_v4",
            "route_scope": route,
            "forward_artifact": str(forward_path),
            "forward_artifact_sha256": sha256(forward_path),
            "tree_artifact": str(tree_path),
            "tree_artifact_sha256": sha256(tree_path),
            "calibration": baseline_calibration,
            "field_calibration": field_calibration,
            "field_tree_source": field_tree_source,
            "secondary_tree_artifact": (
                None
                if secondary_tree_path is None
                else str(secondary_tree_path)
            ),
            "secondary_tree_artifact_sha256": (
                None
                if secondary_tree_path is None
                else sha256(secondary_tree_path)
            ),
            "held_out_test_used": False,
        }
        with artifact_path.open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        artifacts[route] = {
            "path": str(artifact_path),
            "sha256": sha256(artifact_path),
            "field_calibration": field_calibration,
            "field_tree_source": field_tree_source,
        }
        route_validation[route] = {
            "system": system_metrics(
                system_truth[route_mask],
                system_prediction,
            ),
            "specialist": {
                "old_iid": direction_metric_bundle(
                    old_arrays,
                    old_prediction,
                ),
                "difficult": direction_metric_bundle(
                    difficult_arrays,
                    difficult_prediction,
                ),
                "protected_counts": protected_counts(
                    old_prediction,
                    difficult_prediction,
                    old_arrays.labels,
                    difficult_arrays.labels,
                    old_high,
                    difficult_high,
                ),
            },
            "selected_indices": selected.tolist(),
            "system_field_correction_counts": {
                field: int(correction_counts[int(selected[index]), index])
                for index, field in enumerate(DIRECTION_FIELDS)
            },
        }

    baseline_system = system_predictions[baseline_index]
    summary = {
        "version": "fieldwise_targeted_tree_direction_v9_one_seed",
        "artifacts": artifacts,
        "candidate_count_per_field": len(candidate_specs),
        "selection_rule": (
            "Coordinate search maximizes route all-five exact, then total "
            "field accuracy, while every intermediate combination must retain "
            "the four frozen specialist joint-exact baseline counts."
        ),
        "baseline_protected_counts": baseline_protected,
        "route_validation": route_validation,
        "combined_system_validation": {
            "baseline": system_metrics(system_truth, baseline_system),
            "selected": system_metrics(system_truth, combined_system),
        },
        "search_traces": traces,
        "source_contract": {
            "old_validation": str(old_path),
            "difficult_validation": str(difficult_path),
            "system_cache": str(args.system_cache.resolve()),
            "forward_artifact": str(forward_path),
            "tree_artifact": str(tree_path),
            "secondary_tree_artifact": (
                None
                if secondary_tree_path is None
                else str(secondary_tree_path)
            ),
            "secondary_system_cache": (
                None
                if secondary_cache_path is None
                else str(secondary_cache_path)
            ),
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
                "combined_system_validation": summary[
                    "combined_system_validation"
                ],
                "route_validation": {
                    route: {
                        "system": value["system"],
                        "protected_counts": value["specialist"][
                            "protected_counts"
                        ],
                    }
                    for route, value in route_validation.items()
                },
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
