#!/usr/bin/env python3
"""Serialize the exact dev boundary cases unrecovered even with oracle gain."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    Bounds,
    metrics_vector,
    position_vector,
    tolerance_vector,
)

VERSION = "active_diagnosis_v13_boundary_oracle_failure_forensics_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.resolve().read_text().splitlines() if line]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _key(row: Mapping[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _episode_id(key: tuple[str, float]) -> str:
    return f"{key[0]}__g{key[1]:g}"


def _components(metrics: Mapping[str, Any], target: Mapping[str, Any], reference: Mapping[str, Any]) -> np.ndarray:
    return np.abs(metrics_vector(metrics) - metrics_vector(target)) / tolerance_vector(
        metrics_vector(reference)
    )


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _count(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _group_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: {
            "episodes": len(values),
            "initial_distance": _summarize([float(row["initial_distance"]) for row in values]),
            "oracle_final_distance": _summarize([float(row["oracle_final_distance"]) for row in values]),
            "oracle_distance_reduction_fraction": _summarize(
                [float(row["oracle_distance_reduction_fraction"]) for row in values]
            ),
            "mean_model_optimism": _summarize([float(row["mean_model_optimism"]) for row in values]),
            "minimum_bound_clearance_fraction": _summarize(
                [float(row["minimum_bound_clearance_fraction"]) for row in values]
            ),
        }
        for name, values in sorted(grouped.items())
    }


def analyze(
    *,
    suite: Mapping[str, Any],
    v12_config: Mapping[str, Any],
    direct_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bounds = Bounds.from_config(v12_config)
    cases = {str(row["case_id"]): row for row in suite["cases"]}
    direct = {_key(row): row for row in direct_rows}
    oracle = {_key(row): row for row in oracle_rows}
    frozen = {_key(row): row for row in frozen_rows}
    if set(direct) != set(oracle) or set(direct) != set(frozen):
        raise ValueError("control arms are not exactly matched")
    selected = [
        key
        for key, row in oracle.items()
        if str(row["stratum"]) == "reachable_boundary_or_clipping"
        and float(row["evaluator_only_true_gain"]) != 1.0
        and not bool(direct[key]["strict_success"])
        and not bool(row["strict_success"])
    ]
    if len(selected) != 20:
        raise ValueError(f"expected 20 exact boundary oracle failures, found {len(selected)}")
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for key in sorted(selected):
        case = cases[key[0]]
        direct_row, oracle_row, frozen_row = direct[key], oracle[key], frozen[key]
        initial = case["initial_metrics"]
        target = case["target_metrics"]
        initial_components = _components(initial, target, initial)
        final_components = _components(oracle_row["final_metrics"], target, initial)
        initial_position = position_vector(case["initial_positions_mm"])
        all_positions = [initial_position]
        actual_costs = [float(oracle_row["initial_normalized_distance"])]
        optimism = []
        normalized_prediction_mae = []
        normalized_command_max = []
        last_improvement = 0.0
        for step in oracle_row["trace"]:
            true_position = position_vector(
                step["actual_step_audit"]["evaluator_only_next_true_position_mm"]
            )
            all_positions.append(true_position)
            predicted_components = (
                np.abs(
                    metrics_vector(step["predicted_next_metrics"])
                    - metrics_vector(step["observed_next_metrics"])
                )
                / tolerance_vector(metrics_vector(initial))
            )
            command = np.asarray(
                [float(step["command_mm"][field]) for field in ACTION_FIELDS],
                dtype=np.float64,
            )
            command_norm = np.abs(command) / bounds.action_high
            model_optimism = float(step["actual_target_cost"]) - float(
                step["predicted_target_cost"]
            )
            optimism.append(model_optimism)
            normalized_prediction_mae.append(float(np.mean(predicted_components)))
            normalized_command_max.append(float(np.max(command_norm)))
            last_improvement = float(step["before_target_cost"]) - float(
                step["actual_target_cost"]
            )
            actual_costs.append(float(step["actual_target_cost"]))
            step_rows.append(
                {
                    "episode_id": _episode_id(key),
                    "case_id": key[0],
                    "true_gain_evaluator_only": key[1],
                    "control_step": int(step["control_step"]),
                    "before_distance": float(step["before_target_cost"]),
                    "predicted_distance": float(step["predicted_target_cost"]),
                    "actual_distance": float(step["actual_target_cost"]),
                    "actual_improvement": float(step["before_target_cost"])
                    - float(step["actual_target_cost"]),
                    "model_optimism": model_optimism,
                    "prediction_error_components_normalized": {
                        field: float(predicted_components[index])
                        for index, field in enumerate(OUTPUT_FIELDS)
                    },
                    "prediction_mae_normalized": float(np.mean(predicted_components)),
                    "command_fraction_of_step_bound": {
                        field: float(command_norm[index])
                        for index, field in enumerate(ACTION_FIELDS)
                    },
                    "maximum_command_fraction_of_step_bound": float(np.max(command_norm)),
                    "step_saturated": bool(step["actual_step_audit"]["step_saturated"]),
                    "absolute_position_saturated": bool(
                        step["actual_step_audit"]["absolute_position_saturated"]
                    ),
                }
            )
        positions = np.stack(all_positions)
        lower_clearance = (positions - bounds.position_low[None, :]) / (
            bounds.position_high - bounds.position_low
        )[None, :]
        upper_clearance = (bounds.position_high[None, :] - positions) / (
            bounds.position_high - bounds.position_low
        )[None, :]
        clearance = np.minimum(lower_clearance, upper_clearance)
        closest_flat = int(np.argmin(clearance))
        closest_time, closest_axis = np.unravel_index(closest_flat, clearance.shape)
        closest_side = (
            "lower"
            if lower_clearance[closest_time, closest_axis]
            <= upper_clearance[closest_time, closest_axis]
            else "upper"
        )
        monotonic = bool(np.all(np.diff(actual_costs) < 0.0))
        reduction_fraction = 1.0 - float(oracle_row["final_normalized_distance"]) / float(
            oracle_row["initial_normalized_distance"]
        )
        if int(oracle_row["saturation_count"]) > 0:
            mechanism = "saturation_or_limit_contact"
        elif monotonic and last_improvement > 0.25:
            mechanism = "control_budget_exhausted_while_progressing"
        elif float(np.mean(optimism)) > 1.0:
            mechanism = "forward_model_optimism"
        else:
            mechanism = "weak_or_nonmonotonic_progress"
        episode_rows.append(
            {
                "episode_id": _episode_id(key),
                "case_id": key[0],
                "group_id": str(case["group_id"]),
                "regime": str(case["regime"]),
                "true_gain_evaluator_only": key[1],
                "initial_distance": float(oracle_row["initial_normalized_distance"]),
                "direct_final_distance": float(direct_row["final_normalized_distance"]),
                "oracle_final_distance": float(oracle_row["final_normalized_distance"]),
                "frozen_probe_final_distance": float(frozen_row["final_normalized_distance"]),
                "oracle_distance_reduction_fraction": reduction_fraction,
                "dominant_initial_error_metric": OUTPUT_FIELDS[int(np.argmax(initial_components))],
                "dominant_initial_error_normalized": float(np.max(initial_components)),
                "dominant_final_error_metric": OUTPUT_FIELDS[int(np.argmax(final_components))],
                "dominant_final_error_normalized": float(np.max(final_components)),
                "oracle_control_steps": int(oracle_row["control_steps"]),
                "oracle_saturation_count": int(oracle_row["saturation_count"]),
                "monotonic_actual_improvement": monotonic,
                "last_step_actual_improvement": last_improvement,
                "mean_model_optimism": float(np.mean(optimism)),
                "mean_prediction_mae_normalized": float(np.mean(normalized_prediction_mae)),
                "mean_maximum_command_fraction_of_step_bound": float(
                    np.mean(normalized_command_max)
                ),
                "minimum_bound_clearance_fraction": float(np.min(clearance)),
                "closest_bound_axis": POSITION_FIELDS[closest_axis],
                "closest_bound_side": closest_side,
                "closest_bound_step": int(closest_time),
                "assigned_mechanism": mechanism,
            }
        )
    actual_improvement = [float(row["actual_improvement"]) for row in step_rows]
    prediction_mae = [float(row["prediction_mae_normalized"]) for row in step_rows]
    command_fraction = [
        float(row["maximum_command_fraction_of_step_bound"]) for row in step_rows
    ]
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "selection_rule": "boundary stratum, nonnominal gain, direct and oracle-known controller strict failure",
        "episodes": episode_rows,
        "steps": step_rows,
        "summary": {
            "episodes": len(episode_rows),
            "steps": len(step_rows),
            "exact_episode_ids": [row["episode_id"] for row in episode_rows],
            "mechanism_counts": _count(episode_rows, "assigned_mechanism"),
            "regime_counts": _count(episode_rows, "regime"),
            "closest_bound_axis_counts": _count(episode_rows, "closest_bound_axis"),
            "dominant_initial_error_counts": _count(
                episode_rows, "dominant_initial_error_metric"
            ),
            "dominant_final_error_counts": _count(
                episode_rows, "dominant_final_error_metric"
            ),
            "initial_distance": _summarize(
                [float(row["initial_distance"]) for row in episode_rows]
            ),
            "oracle_final_distance": _summarize(
                [float(row["oracle_final_distance"]) for row in episode_rows]
            ),
            "oracle_distance_reduction_fraction": _summarize(
                [float(row["oracle_distance_reduction_fraction"]) for row in episode_rows]
            ),
            "minimum_bound_clearance_fraction": _summarize(
                [float(row["minimum_bound_clearance_fraction"]) for row in episode_rows]
            ),
            "step_actual_improvement": _summarize(actual_improvement),
            "step_prediction_mae_normalized": _summarize(prediction_mae),
            "step_maximum_command_fraction": _summarize(command_fraction),
            "prediction_mae_vs_actual_improvement_correlation": float(
                np.corrcoef(prediction_mae, actual_improvement)[0, 1]
            ),
            "command_fraction_vs_actual_improvement_correlation": float(
                np.corrcoef(command_fraction, actual_improvement)[0, 1]
            ),
        },
        "by_true_gain": _group_numeric(episode_rows, "true_gain_evaluator_only"),
        "by_case": _group_numeric(episode_rows, "case_id"),
        "interpretation_guard": (
            "This is descriptive failure mining on the exact development oracle failures. "
            "It does not select a protected policy or establish causal benefit."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--v12-config", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        suite=_read_json(args.suite),
        v12_config=_read_json(args.v12_config),
        direct_rows=_read_jsonl(args.direct),
        oracle_rows=_read_jsonl(args.oracle),
        frozen_rows=_read_jsonl(args.frozen),
    )
    _atomic_json(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
