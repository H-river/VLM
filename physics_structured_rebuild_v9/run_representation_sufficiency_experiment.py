#!/usr/bin/env python3
"""Test whether five current beam measurements identify future responses."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID, tolerance_from_current
from physics_structured_rebuild_v9.simulator_state_experiment_common import (
    REGIMES,
    bootstrap_mean_ci,
    custom_hidden_action_grid,
    custom_hidden_current_state,
    geometric_focus_residual,
    image_difference,
    json_safe,
    lens_transmission,
    matching_precision,
    phase_descriptors,
    sample_visible_setup,
    setup_from_visible,
    stable_seed,
)
from specialist_rebuild_v2.common import STATE_FIELDS

DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_state_experiments"
    / "representation_sufficiency"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matched-context-count", type=int, default=100)
    parser.add_argument("--maximum-attempts", type=int, default=400)
    parser.add_argument("--maximum-function-evaluations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def parameters_to_hidden(
    parameters: np.ndarray,
    curvature_phase_rad: float,
) -> dict[str, float]:
    return {
        "source_x_offset_mm": float(parameters[0]),
        "source_y_offset_mm": float(parameters[1]),
        "waist_x_scale": float(math.exp(parameters[2])),
        "waist_y_scale": float(math.exp(parameters[3])),
        "amplitude_scale": float(math.exp(parameters[4])),
        "curvature_phase_rad": float(curvature_phase_rad),
    }


def match_hidden_state(
    setup: Any,
    target: np.ndarray,
    phase_sign: float,
    maximum_function_evaluations: int,
) -> dict[str, Any] | None:
    precision = matching_precision(target)
    initial = np.zeros(5, dtype=np.float64)
    lower = np.asarray([-0.9, -0.9, math.log(0.45), math.log(0.45), math.log(0.25)])
    upper = np.asarray([0.9, 0.9, math.log(1.9), math.log(1.9), math.log(4.0)])
    traces = []
    for magnitude in (2.0, 1.0, 0.5):
        phase = phase_sign * magnitude
        evaluations = 0

        def residual(parameters: np.ndarray) -> np.ndarray:
            nonlocal evaluations
            evaluations += 1
            state = custom_hidden_current_state(
                setup,
                parameters_to_hidden(parameters, phase),
            )
            return (state - target) / precision

        started = time.perf_counter()
        optimized = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            method="trf",
            x_scale=np.asarray([0.20, 0.20, 0.15, 0.15, 0.15]),
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
            max_nfev=maximum_function_evaluations,
        )
        hidden = parameters_to_hidden(optimized.x, phase)
        matched = custom_hidden_current_state(setup, hidden)
        normalized = np.abs(matched - target) / precision
        trace = {
            "curvature_phase_rad": phase,
            "optimizer_success": bool(optimized.success),
            "optimizer_status": int(optimized.status),
            "function_evaluations": int(evaluations),
            "cost": float(optimized.cost),
            "maximum_matching_error_precision_units": float(normalized.max()),
            "seconds": float(time.perf_counter() - started),
        }
        traces.append(trace)
        if float(normalized.max()) <= 1.0:
            return {
                "hidden": hidden,
                "current_state": matched,
                "matching_error_precision_units": normalized,
                "trace": traces,
            }
        initial = np.clip(optimized.x, lower, upper)
    return None


def action_complexity(index: int) -> int:
    return int(
        sum(abs(float(value)) > 0.0 for value in ACTION_GRID[index].values())
    )


def phase_comparison(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    normalized = np.abs(left - right) / scale
    return {
        "left": left,
        "right": right,
        "relative_difference_by_descriptor": normalized,
        "maximum_relative_difference": float(normalized.max()),
        "matches_at_one_percent": bool(np.all(normalized <= 0.01)),
    }


def build_matched_context(
    regime: str,
    regime_index: int,
    seed: int,
    maximum_function_evaluations: int,
) -> dict[str, Any] | None:
    started = time.perf_counter()
    visible = sample_visible_setup(regime, regime_index, seed)
    setup = setup_from_visible(visible)
    baseline_hidden: dict[str, float] = {}
    baseline_current = custom_hidden_current_state(setup, baseline_hidden)
    sign = -1.0 if stable_seed(seed, regime, regime_index, "phase") % 2 else 1.0
    match = match_hidden_state(
        setup,
        baseline_current,
        sign,
        maximum_function_evaluations,
    )
    if match is None:
        return None
    candidate_hidden = match["hidden"]
    baseline_states, baseline_arrays = custom_hidden_action_grid(
        setup,
        baseline_hidden,
        return_current_field=True,
    )
    repeat_states, repeat_arrays = custom_hidden_action_grid(
        setup,
        baseline_hidden,
        return_current_field=True,
    )
    candidate_states, candidate_arrays = custom_hidden_action_grid(
        setup,
        candidate_hidden,
        return_current_field=True,
    )
    if baseline_arrays is None or repeat_arrays is None or candidate_arrays is None:
        raise AssertionError("current field arrays were not returned")
    precision = matching_precision(baseline_states[40])
    matching_error = np.abs(
        candidate_states[40] - baseline_states[40]
    ) / precision
    if float(matching_error.max()) > 1.0:
        return None
    pair_tolerance = np.maximum(
        tolerance_from_current(baseline_states[40]),
        tolerance_from_current(candidate_states[40]),
    ).astype(np.float64)
    output_difference = np.abs(
        candidate_states - baseline_states
    ) / pair_tolerance[None, :]
    baseline_response = baseline_states - baseline_states[40][None, :]
    candidate_response = candidate_states - candidate_states[40][None, :]
    response_difference = np.abs(
        candidate_response - baseline_response
    ) / pair_tolerance[None, :]
    repeat_difference = np.abs(
        repeat_states - baseline_states
    ) / pair_tolerance[None, :]
    output_crossing = np.any(output_difference > 1.0, axis=1)
    response_crossing = np.any(response_difference > 1.0, axis=1)
    by_complexity = {}
    for complexity in range(5):
        selected = np.asarray(
            [
                action_complexity(index) == complexity
                for index in range(len(ACTION_GRID))
            ],
            dtype=np.bool_,
        )
        by_complexity[str(complexity)] = {
            "count": int(selected.sum()),
            "output_crossing_rate": float(output_crossing[selected].mean()),
            "response_crossing_rate": float(response_crossing[selected].mean()),
        }
    baseline_phase = phase_descriptors(baseline_arrays)
    repeat_phase = phase_descriptors(repeat_arrays)
    candidate_phase = phase_descriptors(candidate_arrays)
    return {
        "version": "representation_sufficiency_context_v9",
        "context_id": f"{regime}_{regime_index:04d}",
        "regime": regime,
        "regime_index": regime_index,
        "visible_setup": visible,
        "baseline_hidden_state": baseline_hidden,
        "candidate_hidden_state": candidate_hidden,
        "baseline_current_state": baseline_states[40],
        "candidate_current_state": candidate_states[40],
        "current_matching_error_precision_units": matching_error,
        "current_match_maximum_precision_units": float(matching_error.max()),
        "measurement_precision": precision,
        "optimizer_trace": match["trace"],
        "regime_diagnostics": {
            "lens_transmission": lens_transmission(setup),
            "geometric_focus_residual": geometric_focus_residual(setup),
        },
        "future": {
            "action_count": len(ACTION_GRID),
            "output_crossing_action_count": int(output_crossing.sum()),
            "output_crossing_action_rate": float(output_crossing.mean()),
            "response_crossing_action_count": int(response_crossing.sum()),
            "response_crossing_action_rate": float(response_crossing.mean()),
            "any_output_crossing": bool(output_crossing.any()),
            "any_response_crossing": bool(response_crossing.any()),
            "maximum_output_difference_tolerance_units": float(
                output_difference.max()
            ),
            "maximum_response_difference_tolerance_units": float(
                response_difference.max()
            ),
            "output_crossing_count_by_field": {
                field: int((output_difference[:, index] > 1.0).sum())
                for index, field in enumerate(STATE_FIELDS)
            },
            "by_action_complexity": by_complexity,
        },
        "richer_representations": {
            "full_current_intensity_image": image_difference(
                baseline_arrays["intensity"],
                candidate_arrays["intensity"],
            ),
            "same_state_repeat_image": image_difference(
                baseline_arrays["intensity"],
                repeat_arrays["intensity"],
            ),
            "sensor_phase_descriptors": phase_comparison(
                baseline_phase,
                candidate_phase,
            ),
            "same_state_repeat_phase_descriptors": phase_comparison(
                baseline_phase,
                repeat_phase,
            ),
            "source_curvature_descriptor_matches": bool(
                abs(float(candidate_hidden["curvature_phase_rad"])) < 1e-12
            ),
        },
        "same_full_state_repeat_control": {
            "maximum_output_difference_tolerance_units": float(
                repeat_difference.max()
            ),
            "nonzero_difference_count": int(
                np.count_nonzero(repeat_states - baseline_states)
            ),
        },
        "seconds": float(time.perf_counter() - started),
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize(
    contexts: list[dict[str, Any]],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    def ci(key: str, values: list[float], *parts: object) -> dict[str, Any]:
        return bootstrap_mean_ci(
            np.asarray(values, dtype=np.float64),
            stable_seed(seed, key, *parts),
            draws,
        )

    metrics: dict[str, Any] = {
        "five_measurement_representation": {
            "matched_pair_count": len(contexts),
            "future_output_crossing_action_rate": ci(
                "output_cross",
                [
                    context["future"]["output_crossing_action_rate"]
                    for context in contexts
                ],
            ),
            "future_response_crossing_action_rate": ci(
                "response_cross",
                [
                    context["future"]["response_crossing_action_rate"]
                    for context in contexts
                ],
            ),
            "context_with_any_future_output_crossing_rate": ci(
                "output_any",
                [
                    float(context["future"]["any_output_crossing"])
                    for context in contexts
                ],
            ),
            "maximum_current_match_precision_units": float(
                max(
                    context["current_match_maximum_precision_units"]
                    for context in contexts
                )
            ),
        },
        "full_current_intensity_image": {},
        "sensor_phase_descriptors": {},
        "known_source_phase_descriptor": {
            "collision_count": int(
                sum(
                    context["richer_representations"][
                        "source_curvature_descriptor_matches"
                    ]
                    for context in contexts
                )
            )
        },
        "same_full_state_repeat_control": {
            "context_count": len(contexts),
            "context_with_nonzero_output_count": int(
                sum(
                    context["same_full_state_repeat_control"][
                        "nonzero_difference_count"
                    ]
                    > 0
                    for context in contexts
                )
            ),
            "maximum_output_difference_tolerance_units": float(
                max(
                    context["same_full_state_repeat_control"][
                        "maximum_output_difference_tolerance_units"
                    ]
                    for context in contexts
                )
            ),
        },
        "by_regime": {},
    }
    image_matches = [
        bool(
            context["richer_representations"]["full_current_intensity_image"][
                "matches_1pct_rmse"
            ]
            and context["richer_representations"][
                "full_current_intensity_image"
            ]["matches_5pct_maximum"]
        )
        for context in contexts
    ]
    image_collision_contexts = [
        context
        for context, matches in zip(contexts, image_matches, strict=True)
        if matches
    ]
    metrics["full_current_intensity_image"] = {
        "match_definition": (
            "normalized RMSE <= 1% of peak and maximum pixel difference "
            "<= 5% of peak"
        ),
        "collision_count": len(image_collision_contexts),
        "collision_rate": ci(
            "image_collision",
            [float(value) for value in image_matches],
        ),
        "future_crossing_rate_within_collisions": (
            ci(
                "image_collision_future",
                [
                    context["future"]["output_crossing_action_rate"]
                    for context in image_collision_contexts
                ],
            )
            if image_collision_contexts
            else {
                "count": 0,
                "mean": None,
                "ci95_low": None,
                "ci95_high": None,
            }
        ),
    }
    phase_matches = [
        bool(
            context["richer_representations"]["sensor_phase_descriptors"][
                "matches_at_one_percent"
            ]
        )
        for context in contexts
    ]
    phase_collision_contexts = [
        context
        for context, matches in zip(contexts, phase_matches, strict=True)
        if matches
    ]
    metrics["sensor_phase_descriptors"] = {
        "match_definition": (
            "all five phase tilt/curvature descriptors within 1% relative"
        ),
        "collision_count": len(phase_collision_contexts),
        "collision_rate": ci(
            "phase_collision",
            [float(value) for value in phase_matches],
        ),
        "future_crossing_rate_within_collisions": (
            ci(
                "phase_collision_future",
                [
                    context["future"]["output_crossing_action_rate"]
                    for context in phase_collision_contexts
                ],
            )
            if phase_collision_contexts
            else {
                "count": 0,
                "mean": None,
                "ci95_low": None,
                "ci95_high": None,
            }
        ),
    }
    for regime in REGIMES:
        selected = [
            context for context in contexts if context["regime"] == regime
        ]
        metrics["by_regime"][regime] = {
            "count": len(selected),
            "future_output_crossing_action_rate": ci(
                "regime_output_cross",
                [
                    context["future"]["output_crossing_action_rate"]
                    for context in selected
                ],
                regime,
            ),
            "context_with_any_future_output_crossing_rate": ci(
                "regime_output_any",
                [
                    float(context["future"]["any_output_crossing"])
                    for context in selected
                ],
                regime,
            ),
            "image_collision_rate": ci(
                "regime_image_collision",
                [
                    float(
                        context["richer_representations"][
                            "full_current_intensity_image"
                        ]["matches_1pct_rmse"]
                        and context["richer_representations"][
                            "full_current_intensity_image"
                        ]["matches_5pct_maximum"]
                    )
                    for context in selected
                ],
                regime,
            ),
        }
    return metrics


def main() -> None:
    args = parse_args()
    target = int(args.matched_context_count)
    if target < 1:
        raise ValueError("matched-context-count must be positive")
    if int(args.maximum_attempts) < target:
        raise ValueError("maximum-attempts must be at least the target count")
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    failure_dir = output_dir / "failures"
    shard_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"), end="")
        return
    started = time.perf_counter()
    existing = sorted(shard_dir.glob("*.json"))
    contexts = [
        json.loads(path.read_text(encoding="utf-8")) for path in existing
    ]
    recorded_paths = existing + sorted(failure_dir.glob("*.json"))
    attempts = (
        max(int(path.name.split("_", 1)[0]) for path in recorded_paths) + 1
        if recorded_paths
        else 0
    )
    base_target, remainder = divmod(target, len(REGIMES))
    target_by_regime = {
        regime: base_target + int(index < remainder)
        for index, regime in enumerate(REGIMES)
    }

    def regime_count(name: str) -> int:
        return sum(context["regime"] == name for context in contexts)

    while (
        any(
            regime_count(regime) < target_by_regime[regime]
            for regime in REGIMES
        )
        and attempts < int(args.maximum_attempts)
    ):
        regime = REGIMES[attempts % len(REGIMES)]
        regime_index = attempts // len(REGIMES)
        if regime_count(regime) >= target_by_regime[regime]:
            attempts += 1
            continue
        context = build_matched_context(
            regime,
            regime_index,
            int(args.seed),
            int(args.maximum_function_evaluations),
        )
        token = f"{attempts:04d}_{regime}_{regime_index:04d}.json"
        if context is None:
            write_atomic(
                failure_dir / token,
                {
                    "version": "representation_sufficiency_failed_match_v9",
                    "regime": regime,
                    "regime_index": regime_index,
                },
            )
        else:
            write_atomic(shard_dir / token, context)
            contexts.append(context)
        attempts += 1
        print(
            json.dumps(
                {
                    "attempts": attempts,
                    "matched": len(contexts),
                    "target": target,
                    "regime_matched": regime_count(regime),
                    "regime_target": target_by_regime[regime],
                    "regime": regime,
                    "success": context is not None,
                    "context_seconds": (
                        None if context is None else context["seconds"]
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if any(
        regime_count(regime) < target_by_regime[regime]
        for regime in REGIMES
    ):
        raise RuntimeError(
            f"only matched {len(contexts)} balanced contexts in "
            f"{attempts} attempts"
        )
    failed_match_count = len(list(failure_dir.glob("*.json")))
    optimizer_attempt_count = len(contexts) + failed_match_count
    report = {
        "version": "representation_sufficiency_experiment_v9",
        "matched_context_count": len(contexts),
        "optimizer_attempt_count": optimizer_attempt_count,
        "failed_match_count": failed_match_count,
        "match_success_rate": float(len(contexts) / optimizer_attempt_count),
        "schedule_step_count": attempts,
        "independent_unit": "one visible setup with two matched hidden states",
        "regime_counts": {
            regime: regime_count(regime)
            for regime in REGIMES
        },
        "regime_targets": target_by_regime,
        "current_measurement_match": {
            "precision": (
                "0.1 px centroid, 0.2 px sigma, and 0.5% of peak intensity"
            ),
            "all_five_required": True,
        },
        "hidden_state_family": {
            "forced_variable": "source quadratic wavefront phase",
            "optimized_variables": [
                "source_x_offset_mm",
                "source_y_offset_mm",
                "waist_x_scale",
                "waist_y_scale",
                "amplitude_scale",
            ],
            "visible_setup_held_fixed": True,
        },
        "future_tolerance": {
            "centroid_each_px": 1.0,
            "sigma_each_px": 2.0,
            "peak_relative": 0.05,
        },
        "metrics": summarize(
            contexts,
            int(args.seed),
            int(args.bootstrap_draws),
        ),
        "source_contract": {
            "generated_matched_contexts": target - len(existing),
            "resumed_matched_contexts": len(existing),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
            "simulator_grid_size": 1024,
            "simulator_backend": "fresnel_numpy",
        },
        "seconds": float(time.perf_counter() - started),
    }
    write_atomic(summary_path, report)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
