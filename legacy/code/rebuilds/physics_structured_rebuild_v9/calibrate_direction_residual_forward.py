#!/usr/bin/env python3
"""Calibrate direction gates over route-specific residual-forward changes."""

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

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import (
    labels_from_normalized_change,
    load_grid_arrays,
)
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from joint_forward_direction_v7.train import direction_metric_bundle
from physics_structured_rebuild_v9.calibrate_fieldwise_direction import (
    action_complexity_mask,
    calibration_grid,
    calibration_index,
    combined_prediction,
    coordinate_search,
    protected_counts,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    ROUTES,
    gated_predictions,
    read_jsonl,
    sha256,
    system_metrics,
    tree_features,
    tree_probabilities,
)
from physics_structured_rebuild_v9.calibrate_forward_system import (
    tree_residuals,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS, STATE_FIELDS

DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_DIRECTION_TREE = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "direction_tree_targeted/direction_tree_targeted_v9.pkl"
)
DEFAULT_FORWARD_TREE = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_tree_residual/forward_tree_residual_v9.pkl"
)
DEFAULT_FORWARD_CALIBRATION = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_residual_calibrated"
)
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
DEFAULT_DIRECTION_CACHE = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_targeted_tree/direction_validation_cache.npz"
)
DEFAULT_FORWARD_CACHE = (
    DEFAULT_FORWARD_CALIBRATION / "forward_system_cache.npz"
)
DEFAULT_FIELDWISE_DIR = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_fieldwise"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "hybrid_direction_residual_forward"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument(
        "--direction-tree",
        type=Path,
        default=DEFAULT_DIRECTION_TREE,
    )
    parser.add_argument(
        "--secondary-direction-tree",
        type=Path,
        help="Optional second direction tree selected independently per field.",
    )
    parser.add_argument(
        "--forward-tree",
        type=Path,
        default=DEFAULT_FORWARD_TREE,
    )
    parser.add_argument(
        "--forward-calibration-dir",
        type=Path,
        default=DEFAULT_FORWARD_CALIBRATION,
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument(
        "--direction-cache",
        type=Path,
        default=DEFAULT_DIRECTION_CACHE,
    )
    parser.add_argument(
        "--secondary-direction-cache",
        type=Path,
        help="System cache corresponding to --secondary-direction-tree.",
    )
    parser.add_argument(
        "--forward-cache",
        type=Path,
        default=DEFAULT_FORWARD_CACHE,
    )
    parser.add_argument(
        "--fieldwise-direction-dir",
        type=Path,
        default=DEFAULT_FIELDWISE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def route_filename(route: str, prefix: str) -> str:
    return (
        f"{prefix}_state.pkl"
        if route.endswith("_from_state_v1")
        else f"{prefix}_image.pkl"
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "residual_forward_direction_summary.json"
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
    direction_tree_path = args.direction_tree.resolve()
    forward_tree_path = args.forward_tree.resolve()
    base, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )
    with direction_tree_path.open("rb") as stream:
        direction_tree = pickle.load(stream)
    with forward_tree_path.open("rb") as stream:
        forward_tree = pickle.load(stream)
    direction_models = list(direction_tree["models"])
    direction_feature_mode = str(direction_tree["feature_mode"])
    secondary_direction_tree_path = (
        None
        if args.secondary_direction_tree is None
        else args.secondary_direction_tree.resolve()
    )
    secondary_direction_models = None
    if secondary_direction_tree_path is not None:
        if args.secondary_direction_cache is None:
            raise ValueError(
                "--secondary-direction-cache is required with a second tree"
            )
        with secondary_direction_tree_path.open("rb") as stream:
            secondary_direction_tree = pickle.load(stream)
        if (
            str(secondary_direction_tree["feature_mode"])
            != direction_feature_mode
        ):
            raise ValueError("direction-tree feature modes differ")
        secondary_direction_models = list(
            secondary_direction_tree["models"]
        )
    forward_models = list(forward_tree["models"])

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
    old_prior = base.predict_changes(read_jsonl(old_path)).reshape(-1, 5)
    difficult_prior = base.predict_changes(
        read_jsonl(difficult_path)
    ).reshape(-1, 5)
    old_forward_residual = tree_residuals(
        forward_models,
        old_arrays.features,
        old_prior,
    )
    difficult_forward_residual = tree_residuals(
        forward_models,
        difficult_arrays.features,
        difficult_prior,
    )
    old_tree_probability = tree_probabilities(
        direction_models,
        tree_features(
            old_arrays.features,
            old_prior,
            direction_feature_mode,
        ),
    )
    difficult_tree_probability = tree_probabilities(
        direction_models,
        tree_features(
            difficult_arrays.features,
            difficult_prior,
            direction_feature_mode,
        ),
    )
    old_secondary_tree_probability = None
    difficult_secondary_tree_probability = None
    if secondary_direction_models is not None:
        old_secondary_tree_probability = tree_probabilities(
            secondary_direction_models,
            tree_features(
                old_arrays.features,
                old_prior,
                direction_feature_mode,
            ),
        )
        difficult_secondary_tree_probability = tree_probabilities(
            secondary_direction_models,
            tree_features(
                difficult_arrays.features,
                difficult_prior,
                direction_feature_mode,
            ),
        )

    direction_cache = np.load(
        args.direction_cache.resolve(),
        allow_pickle=False,
    )
    forward_cache = np.load(
        args.forward_cache.resolve(),
        allow_pickle=False,
    )
    system_prior = np.asarray(direction_cache["changes"], dtype=np.float32)
    system_tree_probability = np.asarray(
        direction_cache["tree_probabilities"],
        dtype=np.float32,
    )
    system_truth = np.asarray(direction_cache["truth"], dtype=np.int64)
    system_routes = np.asarray(direction_cache["routes"], dtype=np.str_)
    system_secondary_tree_probability = None
    secondary_direction_cache_path = None
    if secondary_direction_models is not None:
        secondary_direction_cache_path = (
            args.secondary_direction_cache.resolve()
        )
        secondary_direction_cache = np.load(
            secondary_direction_cache_path,
            allow_pickle=False,
        )
        for key in ("changes", "truth", "routes", "example_ids"):
            if not np.array_equal(
                direction_cache[key],
                secondary_direction_cache[key],
            ):
                raise ValueError(
                    f"primary and secondary direction caches differ for {key}"
                )
        system_secondary_tree_probability = np.asarray(
            secondary_direction_cache["tree_probabilities"],
            dtype=np.float32,
        )
    if not np.array_equal(
        system_prior,
        np.asarray(forward_cache["prior"], dtype=np.float32),
    ):
        raise ValueError("direction and forward system-cache priors differ")
    system_forward_residual = np.asarray(
        forward_cache["residual"],
        dtype=np.float32,
    )
    derived_truth = labels_from_normalized_change(
        np.asarray(forward_cache["truth_change"], dtype=np.float32)
        / np.asarray(
            forward_cache["scoring_tolerance"],
            dtype=np.float32,
        )
    )
    if not np.array_equal(system_truth, derived_truth):
        raise ValueError("direction and forward system-cache truths differ")

    original_old = labels_from_normalized_change(old_prior)
    original_difficult = labels_from_normalized_change(difficult_prior)
    old_high = action_complexity_mask(old_arrays.group_count)
    difficult_high = action_complexity_mask(
        difficult_arrays.group_count
    )
    original_protected = protected_counts(
        original_old,
        original_difficult,
        old_arrays.labels,
        difficult_arrays.labels,
        old_high,
        difficult_high,
    )
    grid = calibration_grid()
    sources = (
        ("primary",)
        if secondary_direction_models is None
        else ("primary", "secondary")
    )
    candidate_specs = [
        (source, calibration)
        for source in sources
        for calibration in grid
    ]

    def candidate_index(
        source: str,
        calibration: dict[str, float],
    ) -> int:
        return next(
            index
            for index, candidate in enumerate(candidate_specs)
            if candidate[0] == source and candidate[1] == calibration
        )

    baseline_gate = {
        "boundary_limit": -1.0,
        "confidence_limit": 0.50,
        "margin_limit": 0.0,
    }
    baseline_index = candidate_index("primary", baseline_gate)
    full_tree_index = candidate_index(
        "primary",
        {
            "boundary_limit": 1.0e9,
            "confidence_limit": 0.50,
            "margin_limit": 0.0,
        },
    )

    selected_by_route = {}
    route_data = {}
    traces = {}
    for route in ROUTES:
        forward_calibration_path = (
            args.forward_calibration_dir.resolve()
            / route_filename(route, "forward")
        )
        with forward_calibration_path.open("rb") as stream:
            forward_calibration = pickle.load(stream)
        blends = np.asarray(
            [
                float(forward_calibration["field_blend"][field])
                for field in STATE_FIELDS
            ],
            dtype=np.float32,
        )
        old_changes = (
            old_prior + old_forward_residual * blends[None, :]
        )
        difficult_changes = (
            difficult_prior
            + difficult_forward_residual * blends[None, :]
        )
        system_changes = (
            system_prior + system_forward_residual * blends[None, :]
        )
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
                selected_old_probability = old_tree_probability
                selected_difficult_probability = difficult_tree_probability
                selected_system_probability = system_tree_probability
            else:
                selected_old_probability = old_secondary_tree_probability
                selected_difficult_probability = (
                    difficult_secondary_tree_probability
                )
                selected_system_probability = (
                    system_secondary_tree_probability
                )
            if (
                selected_old_probability is None
                or selected_difficult_probability is None
                or selected_system_probability is None
            ):
                raise ValueError(
                    "secondary direction probabilities are unavailable"
                )
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
            correction_counts[index] = system_apply[
                system_routes == route
            ].sum(axis=0)

        current_fieldwise_path = (
            args.fieldwise_direction_dir.resolve()
            / route_filename(route, "hybrid_direction")
        )
        with current_fieldwise_path.open("rb") as stream:
            current_fieldwise = pickle.load(stream)
        current_sources = current_fieldwise.get("field_tree_source")
        current_indices = np.asarray(
            [
                candidate_index(
                    (
                        str(current_sources[field])
                        if current_sources is not None
                        else "primary"
                    ),
                    dict(current_fieldwise["field_calibration"][field]),
                )
                for field in DIRECTION_FIELDS
            ],
            dtype=np.int64,
        )
        residual_baseline = np.full(
            5,
            baseline_index,
            dtype=np.int64,
        )
        starts = [
            residual_baseline,
            current_indices,
            np.full(5, full_tree_index, dtype=np.int64),
        ]
        if secondary_direction_models is not None:
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
            route_mask=system_routes == route,
            system_truth=system_truth,
            system_predictions=system_predictions,
            old_predictions=old_predictions,
            difficult_predictions=difficult_predictions,
            old_labels=old_arrays.labels,
            difficult_labels=difficult_arrays.labels,
            old_high=old_high,
            difficult_high=difficult_high,
            baseline_protected=original_protected,
            correction_counts=correction_counts,
            starts=starts,
        )
        selected_by_route[route] = selected
        traces[route] = route_trace
        route_data[route] = {
            "forward_calibration_path": forward_calibration_path,
            "old_changes": old_changes,
            "difficult_changes": difficult_changes,
            "system_changes": system_changes,
            "old_predictions": old_predictions,
            "difficult_predictions": difficult_predictions,
            "system_predictions": system_predictions,
            "residual_baseline": residual_baseline,
        }

    artifacts = {}
    route_validation = {}
    combined_selected = np.empty_like(system_truth)
    combined_residual_baseline = np.empty_like(system_truth)
    for route, artifact_path in artifact_paths.items():
        selected = selected_by_route[route]
        data = route_data[route]
        route_mask = system_routes == route
        selected_system = combined_prediction(
            data["system_predictions"][:, route_mask, :],
            selected,
        )
        residual_baseline_system = combined_prediction(
            data["system_predictions"][:, route_mask, :],
            data["residual_baseline"],
        )
        combined_selected[route_mask] = selected_system
        combined_residual_baseline[route_mask] = (
            residual_baseline_system
        )
        selected_old = combined_prediction(
            data["old_predictions"],
            selected,
        )
        selected_difficult = combined_prediction(
            data["difficult_predictions"],
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
        threshold_forward_path = Path(
            data["forward_calibration_path"]
        ).resolve()
        artifact = {
            "version": (
                "residual_forward_tree_state_direction_v9_one_seed"
                if route == ROUTES[0]
                else "residual_forward_tree_image_direction_v9_one_seed"
            ),
            "model": "frozen_v7_forward_plus_boundary_tree_v4",
            "route_scope": route,
            "forward_artifact": str(forward_path),
            "forward_artifact_sha256": sha256(forward_path),
            "threshold_forward_artifact": str(
                threshold_forward_path
            ),
            "threshold_forward_artifact_sha256": sha256(
                threshold_forward_path
            ),
            "tree_artifact": str(direction_tree_path),
            "tree_artifact_sha256": sha256(direction_tree_path),
            "calibration": baseline_gate,
            "field_calibration": field_calibration,
            "field_tree_source": field_tree_source,
            "secondary_tree_artifact": (
                None
                if secondary_direction_tree_path is None
                else str(secondary_direction_tree_path)
            ),
            "secondary_tree_artifact_sha256": (
                None
                if secondary_direction_tree_path is None
                else sha256(secondary_direction_tree_path)
            ),
            "held_out_test_used": False,
        }
        with artifact_path.open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        artifacts[route] = {
            "path": str(artifact_path),
            "sha256": sha256(artifact_path),
            "threshold_forward_artifact": str(
                threshold_forward_path
            ),
            "field_calibration": field_calibration,
            "field_tree_source": field_tree_source,
        }
        route_validation[route] = {
            "system": system_metrics(
                system_truth[route_mask],
                selected_system,
            ),
            "residual_forward_threshold_only": system_metrics(
                system_truth[route_mask],
                residual_baseline_system,
            ),
            "specialist": {
                "old_iid": direction_metric_bundle(
                    old_arrays,
                    selected_old,
                ),
                "difficult": direction_metric_bundle(
                    difficult_arrays,
                    selected_difficult,
                ),
                "protected_counts": protected_counts(
                    selected_old,
                    selected_difficult,
                    old_arrays.labels,
                    difficult_arrays.labels,
                    old_high,
                    difficult_high,
                ),
            },
        }

    original_system = labels_from_normalized_change(system_prior)
    summary = {
        "version": "residual_forward_tree_direction_v9_one_seed",
        "artifacts": artifacts,
        "candidate_count_per_field": len(candidate_specs),
        "original_v7_protected_counts": original_protected,
        "route_validation": route_validation,
        "combined_system_validation": {
            "original_v7_threshold": system_metrics(
                system_truth,
                original_system,
            ),
            "residual_forward_threshold_only": system_metrics(
                system_truth,
                combined_residual_baseline,
            ),
            "selected": system_metrics(
                system_truth,
                combined_selected,
            ),
        },
        "search_traces": traces,
        "source_contract": {
            "old_validation": str(old_path),
            "difficult_validation": str(difficult_path),
            "direction_cache": str(args.direction_cache.resolve()),
            "forward_cache": str(args.forward_cache.resolve()),
            "forward_artifact": str(forward_path),
            "direction_tree": str(direction_tree_path),
            "secondary_direction_tree": (
                None
                if secondary_direction_tree_path is None
                else str(secondary_direction_tree_path)
            ),
            "secondary_direction_cache": (
                None
                if secondary_direction_cache_path is None
                else str(secondary_direction_cache_path)
            ),
            "forward_tree": str(forward_tree_path),
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
                        "threshold_only": value[
                            "residual_forward_threshold_only"
                        ],
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
