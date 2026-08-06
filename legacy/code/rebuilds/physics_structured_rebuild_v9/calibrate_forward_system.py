#!/usr/bin/env python3
"""Calibrate per-field residual-forward blends on specialist and system validation."""

from __future__ import annotations

import argparse
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

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from direction_rebuild_v4.data import load_grid_arrays
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from joint_forward_direction_v7.train import forward_metric_bundle
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
    read_jsonl,
    sha256,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    forward_feature,
)

ROUTES = (
    "predict_forward_from_state_v1",
    "predict_forward_from_image_v1",
)
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_TREE = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_tree_residual/forward_tree_residual_v9.pkl"
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
    / "forward_residual_calibrated"
)
BLENDS = (-0.25, 0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--tree-artifact", type=Path, default=DEFAULT_TREE)
    parser.add_argument(
        "--secondary-tree-artifact",
        type=Path,
        help="Optional second residual tree selected independently per field.",
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN_DATA)
    parser.add_argument("--v4-overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--secondary-system-cache",
        type=Path,
        help="System cache corresponding to --secondary-tree-artifact.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def tree_residuals(
    models: list[Any],
    features: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    combined = np.concatenate(
        [
            np.asarray(features, dtype=np.float32),
            np.asarray(prior, dtype=np.float32),
        ],
        axis=1,
    )
    return np.stack(
        [model.predict(combined) for model in models],
        axis=1,
    ).astype(np.float32)


def action_high_mask(group_count: int) -> np.ndarray:
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


def normalized_prediction(
    prior: np.ndarray,
    residual: np.ndarray,
    selected: np.ndarray,
    secondary_residual: np.ndarray | None = None,
) -> np.ndarray:
    candidate_count = len(BLENDS)
    return np.stack(
        [
            prior[:, field]
            + float(BLENDS[int(selected[field]) % candidate_count])
            * (
                secondary_residual[:, field]
                if int(selected[field]) >= candidate_count
                and secondary_residual is not None
                else residual[:, field]
            )
            for field in range(len(STATE_FIELDS))
        ],
        axis=1,
    )


def specialist_counts(
    old_prediction: np.ndarray,
    difficult_prediction: np.ndarray,
    old_target: np.ndarray,
    difficult_target: np.ndarray,
    old_high: np.ndarray,
    difficult_high: np.ndarray,
) -> tuple[int, int, int, int]:
    old_exact = np.all(
        np.abs(old_prediction - old_target) <= 1.0,
        axis=1,
    )
    difficult_exact = np.all(
        np.abs(difficult_prediction - difficult_target) <= 1.0,
        axis=1,
    )
    return (
        int(old_exact.sum()),
        int(difficult_exact.sum()),
        int(old_exact[old_high].sum()),
        int(difficult_exact[difficult_high].sum()),
    )


def system_metrics(
    truth_change: np.ndarray,
    prediction_normalized: np.ndarray,
    input_tolerance: np.ndarray,
    scoring_tolerance: np.ndarray,
) -> dict[str, Any]:
    prediction_change = prediction_normalized * input_tolerance
    error_in_scoring_tolerance = np.abs(
        prediction_change - truth_change
    ) / scoring_tolerance
    passed = error_in_scoring_tolerance <= 1.0
    exact = np.all(passed, axis=1)
    return {
        "count": int(len(exact)),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_scoring_tolerance_units": float(
            error_in_scoring_tolerance.mean()
        ),
        "per_field_tolerance_pass": {
            field: float(passed[:, index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def coordinate_search(
    *,
    route_mask: np.ndarray,
    old_prior: np.ndarray,
    old_residual: np.ndarray,
    old_secondary_residual: np.ndarray | None,
    difficult_prior: np.ndarray,
    difficult_residual: np.ndarray,
    difficult_secondary_residual: np.ndarray | None,
    system_prior: np.ndarray,
    system_residual: np.ndarray,
    system_secondary_residual: np.ndarray | None,
    old_target: np.ndarray,
    difficult_target: np.ndarray,
    truth_change: np.ndarray,
    input_tolerance: np.ndarray,
    scoring_tolerance: np.ndarray,
    old_high: np.ndarray,
    difficult_high: np.ndarray,
    baseline_counts: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    source_count = 1 if old_secondary_residual is None else 2
    candidate_count = len(BLENDS) * source_count
    starts = [
        np.full(5, BLENDS.index(value), dtype=np.int64)
        for value in (0.0, 0.35, 0.50, 1.0)
    ]
    if source_count == 2:
        starts.extend(
            [
                np.full(
                    5,
                    len(BLENDS) + BLENDS.index(value),
                    dtype=np.int64,
                )
                for value in (0.35, 0.50, 1.0)
            ]
        )
    finals = []
    traces = []
    for start_index, initial in enumerate(starts):
        selected = initial.copy()
        route_trace = []
        for pass_index in range(5):
            changed = False
            for field in range(len(STATE_FIELDS)):
                best_index = int(selected[field])
                best_key = None
                for candidate in range(candidate_count):
                    proposal = selected.copy()
                    proposal[field] = candidate
                    old_prediction = normalized_prediction(
                        old_prior,
                        old_residual,
                        proposal,
                        old_secondary_residual,
                    )
                    difficult_prediction = normalized_prediction(
                        difficult_prior,
                        difficult_residual,
                        proposal,
                        difficult_secondary_residual,
                    )
                    protected = specialist_counts(
                        old_prediction,
                        difficult_prediction,
                        old_target,
                        difficult_target,
                        old_high,
                        difficult_high,
                    )
                    if any(
                        observed < minimum
                        for observed, minimum in zip(
                            protected,
                            baseline_counts,
                            strict=True,
                        )
                    ):
                        continue
                    system_prediction = normalized_prediction(
                        system_prior[route_mask],
                        system_residual[route_mask],
                        proposal,
                        (
                            None
                            if system_secondary_residual is None
                            else system_secondary_residual[route_mask]
                        ),
                    )
                    metrics = system_metrics(
                        truth_change[route_mask],
                        system_prediction,
                        input_tolerance[route_mask],
                        scoring_tolerance[route_mask],
                    )
                    field_total = sum(
                        metrics["per_field_tolerance_pass"].values()
                    )
                    key = (
                        metrics["strict_all_five_count"],
                        field_total,
                        sum(protected),
                        -sum(
                            abs(float(BLENDS[int(value) % len(BLENDS)]))
                            for value in proposal
                        ),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_index = candidate
                if best_index != int(selected[field]):
                    selected[field] = best_index
                    changed = True
            metrics = system_metrics(
                truth_change[route_mask],
                normalized_prediction(
                    system_prior[route_mask],
                    system_residual[route_mask],
                    selected,
                    (
                        None
                        if system_secondary_residual is None
                        else system_secondary_residual[route_mask]
                    ),
                ),
                input_tolerance[route_mask],
                scoring_tolerance[route_mask],
            )
            route_trace.append(
                {
                    "pass": pass_index + 1,
                    "selected_blends": [
                        float(BLENDS[int(value) % len(BLENDS)])
                        for value in selected
                    ],
                    "selected_sources": [
                        (
                            "secondary"
                            if int(value) >= len(BLENDS)
                            else "primary"
                        )
                        for value in selected
                    ],
                    "system": metrics,
                }
            )
            if not changed:
                break
        finals.append(selected.copy())
        traces.append(
            {
                "start_index": start_index,
                "passes": route_trace,
            }
        )

    def final_key(selected: np.ndarray) -> tuple[Any, ...]:
        metrics = system_metrics(
            truth_change[route_mask],
            normalized_prediction(
                system_prior[route_mask],
                system_residual[route_mask],
                selected,
                (
                    None
                    if system_secondary_residual is None
                    else system_secondary_residual[route_mask]
                ),
            ),
            input_tolerance[route_mask],
            scoring_tolerance[route_mask],
        )
        protected = specialist_counts(
            normalized_prediction(
                old_prior,
                old_residual,
                selected,
                old_secondary_residual,
            ),
            normalized_prediction(
                difficult_prior,
                difficult_residual,
                selected,
                difficult_secondary_residual,
            ),
            old_target,
            difficult_target,
            old_high,
            difficult_high,
        )
        return (
            metrics["strict_all_five_count"],
            -metrics["mae_in_scoring_tolerance_units"],
            sum(protected),
            -sum(
                abs(float(BLENDS[int(value) % len(BLENDS)]))
                for value in selected
            ),
        )

    return max(finals, key=final_key), traces


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "forward_calibration_summary.json"
    artifact_paths = {
        ROUTES[0]: output_dir / "forward_state.pkl",
        ROUTES[1]: output_dir / "forward_image.pkl",
    }
    cache_path = output_dir / "forward_system_cache.npz"
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
    base, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )
    with tree_path.open("rb") as stream:
        tree = pickle.load(stream)
    if tree.get("model") not in {
        "v7_stacked_five_head_hist_gradient_boosting_forward_v9",
        "v7_stacked_five_head_extra_trees_forward_v9",
        "v7_stacked_five_head_lightgbm_forward_v9",
    }:
        raise ValueError("unexpected residual-forward tree artifact")
    models = list(tree["models"])
    secondary_tree_path = (
        None
        if args.secondary_tree_artifact is None
        else args.secondary_tree_artifact.resolve()
    )
    secondary_models = None
    if secondary_tree_path is not None:
        if args.secondary_system_cache is None:
            raise ValueError(
                "--secondary-system-cache is required with a second tree"
            )
        with secondary_tree_path.open("rb") as stream:
            secondary_tree = pickle.load(stream)
        if secondary_tree.get("model") not in {
            "v7_stacked_five_head_hist_gradient_boosting_forward_v9",
            "v7_stacked_five_head_extra_trees_forward_v9",
            "v7_stacked_five_head_lightgbm_forward_v9",
        }:
            raise ValueError("unexpected secondary residual-forward artifact")
        if secondary_tree.get("feature_mode") != tree.get("feature_mode"):
            raise ValueError("residual-forward feature modes differ")
        secondary_models = list(secondary_tree["models"])

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
    old_prior = base.predict_changes(old_rows).reshape(-1, 5)
    difficult_prior = base.predict_changes(
        difficult_rows
    ).reshape(-1, 5)
    old_residual = tree_residuals(
        models,
        old_arrays.features,
        old_prior,
    )
    difficult_residual = tree_residuals(
        models,
        difficult_arrays.features,
        difficult_prior,
    )
    old_secondary_residual = None
    difficult_secondary_residual = None
    if secondary_models is not None:
        old_secondary_residual = tree_residuals(
            secondary_models,
            old_arrays.features,
            old_prior,
        )
        difficult_secondary_residual = tree_residuals(
            secondary_models,
            difficult_arrays.features,
            difficult_prior,
        )
    old_target = old_arrays.normalized_changes
    difficult_target = difficult_arrays.normalized_changes
    old_high = action_high_mask(old_arrays.group_count)
    difficult_high = action_high_mask(difficult_arrays.group_count)
    baseline_counts = specialist_counts(
        old_prior,
        difficult_prior,
        old_target,
        difficult_target,
        old_high,
        difficult_high,
    )

    qwen_data = args.qwen_data.resolve()
    canonical = [
        row
        for row in read_jsonl(qwen_data / "canonical/val.jsonl")
        if str(row["category"]) in ROUTES
    ]
    if len(canonical) != 300:
        raise ValueError("expected 300 forward validation requests")
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
        else:
            measured = measurement_backend.visual.measure_image(
                qwen_data / str(row["images"][0]),
                arguments["image_calibration"],
            )
            current = state_mapping(
                sensor_to_base_legacy(
                    measured["beam_state"],
                    [arguments["setup"]],
                )[0]
            )
        action = arguments["action"]
        private = private_by_group[str(row["group_id"])]
        truth_key = (
            str(row["group_id"]),
            tuple(float(action[field]) for field in ACTION_FIELDS),
        )
        if truth_key not in truth_cache:
            truth_cache[truth_key] = simulator_forward_truth(private, action)
        truth = truth_cache[truth_key]["change"]
        system_rows.append(
            {
                "example_id": str(row["example_id"]),
                "route": route,
                "setup": arguments["setup"],
                "current": current,
                "action": action,
                "action_index": action_index(action),
                "truth_change": [
                    float(truth[field]) for field in STATE_FIELDS
                ],
                "input_tolerance": tolerance_from_current(current),
                "scoring_tolerance": tolerance_from_current(
                    private["current_beam_state"]
                ),
            }
        )
        if (index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "prepared_forward_requests": index + 1,
                        "total": len(canonical),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    prediction_rows = [
        {
            "group_id": row["example_id"],
            "setup": row["setup"],
            "current_beam_state": row["current"],
        }
        for row in system_rows
    ]
    system_grid_prior = base.predict_changes(prediction_rows)
    system_prior = np.stack(
        [
            system_grid_prior[index, int(row["action_index"])]
            for index, row in enumerate(system_rows)
        ]
    )
    system_features = np.stack(
        [
            forward_feature(row["setup"], row["current"], row["action"])
            for row in system_rows
        ]
    )
    system_residual = tree_residuals(
        models,
        system_features,
        system_prior,
    )
    truth_change = np.asarray(
        [row["truth_change"] for row in system_rows],
        dtype=np.float32,
    )
    input_tolerance = np.asarray(
        [row["input_tolerance"] for row in system_rows],
        dtype=np.float32,
    )
    scoring_tolerance = np.asarray(
        [row["scoring_tolerance"] for row in system_rows],
        dtype=np.float32,
    )
    routes = np.asarray(
        [row["route"] for row in system_rows],
        dtype=np.str_,
    )
    system_secondary_residual = None
    secondary_cache_path = None
    if secondary_models is not None:
        secondary_cache_path = args.secondary_system_cache.resolve()
        secondary_cache = np.load(
            secondary_cache_path,
            allow_pickle=False,
        )
        for key, expected in (
            ("prior", system_prior),
            ("truth_change", truth_change),
            ("input_tolerance", input_tolerance),
            ("scoring_tolerance", scoring_tolerance),
            ("routes", routes),
        ):
            if not np.array_equal(secondary_cache[key], expected):
                raise ValueError(
                    f"primary and secondary forward caches differ for {key}"
                )
        system_secondary_residual = np.asarray(
            secondary_cache["residual"],
            dtype=np.float32,
        )
    np.savez_compressed(
        cache_path,
        prior=system_prior,
        residual=system_residual,
        truth_change=truth_change,
        input_tolerance=input_tolerance,
        scoring_tolerance=scoring_tolerance,
        routes=routes,
        secondary_residual=(
            np.empty((0, 5), dtype=np.float32)
            if system_secondary_residual is None
            else system_secondary_residual
        ),
    )

    selected_by_route = {}
    traces = {}
    for route in ROUTES:
        selected, trace = coordinate_search(
            route_mask=routes == route,
            old_prior=old_prior,
            old_residual=old_residual,
            old_secondary_residual=old_secondary_residual,
            difficult_prior=difficult_prior,
            difficult_residual=difficult_residual,
            difficult_secondary_residual=difficult_secondary_residual,
            system_prior=system_prior,
            system_residual=system_residual,
            system_secondary_residual=system_secondary_residual,
            old_target=old_target,
            difficult_target=difficult_target,
            truth_change=truth_change,
            input_tolerance=input_tolerance,
            scoring_tolerance=scoring_tolerance,
            old_high=old_high,
            difficult_high=difficult_high,
            baseline_counts=baseline_counts,
        )
        selected_by_route[route] = selected
        traces[route] = trace

    artifacts = {}
    route_validation = {}
    combined_system = np.empty_like(system_prior)
    for route, artifact_path in artifact_paths.items():
        selected = selected_by_route[route]
        route_mask = routes == route
        field_blend = {
            field: float(
                BLENDS[int(selected[index]) % len(BLENDS)]
            )
            for index, field in enumerate(STATE_FIELDS)
        }
        field_tree_source = {
            field: (
                "secondary"
                if int(selected[index]) >= len(BLENDS)
                else "primary"
            )
            for index, field in enumerate(STATE_FIELDS)
        }
        artifact = {
            "version": (
                "calibrated_residual_forward_state_v9_one_seed"
                if route == ROUTES[0]
                else "calibrated_residual_forward_image_v9_one_seed"
            ),
            "model": "frozen_v7_plus_residual_tree_v9",
            "route_scope": route,
            "forward_artifact": str(forward_path),
            "forward_artifact_sha256": sha256(forward_path),
            "tree_artifact": str(tree_path),
            "tree_artifact_sha256": sha256(tree_path),
            "field_blend": field_blend,
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
            "field_blend": field_blend,
            "field_tree_source": field_tree_source,
        }
        old_prediction = normalized_prediction(
            old_prior,
            old_residual,
            selected,
            old_secondary_residual,
        )
        difficult_prediction = normalized_prediction(
            difficult_prior,
            difficult_residual,
            selected,
            difficult_secondary_residual,
        )
        route_prediction = normalized_prediction(
            system_prior[route_mask],
            system_residual[route_mask],
            selected,
            (
                None
                if system_secondary_residual is None
                else system_secondary_residual[route_mask]
            ),
        )
        combined_system[route_mask] = route_prediction
        route_validation[route] = {
            "system": system_metrics(
                truth_change[route_mask],
                route_prediction,
                input_tolerance[route_mask],
                scoring_tolerance[route_mask],
            ),
            "specialist": {
                "old_iid": forward_metric_bundle(
                    old_arrays,
                    old_prediction,
                ),
                "difficult": forward_metric_bundle(
                    difficult_arrays,
                    difficult_prediction,
                ),
                "protected_counts": specialist_counts(
                    old_prediction,
                    difficult_prediction,
                    old_target,
                    difficult_target,
                    old_high,
                    difficult_high,
                ),
            },
        }

    summary = {
        "version": "calibrated_residual_forward_v9_one_seed",
        "artifacts": artifacts,
        "blend_candidates": list(BLENDS),
        "baseline_protected_counts": baseline_counts,
        "route_validation": route_validation,
        "combined_system_validation": {
            "baseline": system_metrics(
                truth_change,
                system_prior,
                input_tolerance,
                scoring_tolerance,
            ),
            "selected": system_metrics(
                truth_change,
                combined_system,
                input_tolerance,
                scoring_tolerance,
            ),
        },
        "search_traces": traces,
        "source_contract": {
            "old_validation": str(old_path),
            "difficult_validation": str(difficult_path),
            "qwen_validation": str(
                (qwen_data / "canonical/val.jsonl").resolve()
            ),
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
            "system_cache": str(cache_path),
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
