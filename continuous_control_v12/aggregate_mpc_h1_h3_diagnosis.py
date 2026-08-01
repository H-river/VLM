#!/usr/bin/env python3
"""Aggregate matched v12 Oracle/Learned H1/H3 diagnostic episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import ACTION_FIELDS, OUTPUT_FIELDS
from continuous_control_v12.run_mpc_h1_h3_diagnosis import EPISODE_VERSION

CONTROLLERS = ("oracle_h1", "oracle_h3", "learned_h1", "learned_h3")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_correlation(
    x: Sequence[float],
    y: Sequence[float],
    kind: str,
) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if (
        len(left) < 3
        or np.allclose(left, left[0])
        or np.allclose(right, right[0])
    ):
        return None
    result = spearmanr(left, right) if kind == "spearman" else pearsonr(left, right)
    return float(result.statistic)


def _bootstrap(
    values: Sequence[float],
    *,
    seed: int,
    samples: int,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"estimate": math.nan, "low": math.nan, "high": math.nan}
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            statistic(array[rng.integers(0, len(array), size=len(array))])
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(statistic(array)),
        "low": float(low),
        "high": float(high),
    }


def _episode_row(episode: Mapping[str, Any]) -> dict[str, Any]:
    trace = list(episode["trace"])
    predicted_improvement_steps = int(episode["predicted_improvement_steps"])
    false_steps = int(episode["false_improvement_steps"])
    row: dict[str, Any] = {
        "case_id": episode["case_id"],
        "group_id": episode["group_id"],
        "controller": episode["controller"],
        "method": episode["method"],
        "horizon": int(episode["horizon"]),
        "stratum": episode["stratum"],
        "regime": episode["regime"],
        "initial_distance_band": episode["initial_distance_band"],
        "initial_normalized_distance": float(
            episode["initial_normalized_distance"]
        ),
        "final_normalized_distance": float(
            episode["final_normalized_distance"]
        ),
        "normalized_distance_reduction": float(
            episode["normalized_distance_reduction"]
        ),
        "final_to_initial_distance_ratio": float(
            episode["final_to_initial_distance_ratio"]
        ),
        "strict_success": int(bool(episode["success"])),
        "any_improvement": int(bool(episode["any_improvement"])),
        "maximum_temporary_worsening": float(
            episode["maximum_temporary_worsening"]
        ),
        "executed_steps": int(episode["executed_steps"]),
        "cumulative_motion_l1_mm": float(
            episode["cumulative_motion_l1_mm"]
        ),
        "termination_reason": episode["termination_reason"],
        "planner_runtime_seconds": float(
            episode["planner_runtime_seconds"]
        ),
        "wall_runtime_seconds": float(episode["wall_runtime_seconds"]),
        "planner_rollout_backend_calls": int(
            episode["planner_rollout_backend_calls"]
        ),
        "model_rollout_calls": int(episode["model_rollout_calls"]),
        "simulator_calls": int(episode["simulator_calls"]),
        "predicted_improvement_steps": predicted_improvement_steps,
        "false_improvement_steps": false_steps,
        "false_improvement_rate": (
            math.nan
            if predicted_improvement_steps == 0
            else float(false_steps / predicted_improvement_steps)
        ),
        "selected_actions_outside_central_rate": (
            math.nan
            if not trace
            else float(
                np.mean(
                    [
                        bool(
                            step["action_distribution"][
                                "outside_central_training_range"
                            ]
                        )
                        for step in trace
                    ]
                )
            )
        ),
        "selected_actions_outside_tail_rate": (
            math.nan
            if not trace
            else float(
                np.mean(
                    [
                        bool(
                            step["action_distribution"][
                                "outside_tail_training_range"
                            ]
                        )
                        for step in trace
                    ]
                )
            )
        ),
        "selected_action_h1_prediction_mae": (
            math.nan
            if not trace
            else float(
                np.mean(
                    [
                        step["selected_action_h1_prediction_mae"]
                        for step in trace
                    ]
                )
            )
        ),
        "clipping_or_boundary_failure": int(
            bool(episode["clipping_or_boundary_failure"])
        ),
        "q_goal_used_by_controller": int(
            bool(episode["q_goal_used_by_controller"])
        ),
    }
    for field in OUTPUT_FIELDS:
        row[f"final_failure_{field}"] = int(
            bool(episode["final_output_tolerance_failure"][field])
        )
    for field in ACTION_FIELDS:
        row[f"cumulative_motion_{field}"] = float(
            episode["cumulative_motion_mm"][field]
        )
    return row


def _step_rows(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for episode in episodes:
        for step in episode["trace"]:
            row: dict[str, Any] = {
                "case_id": episode["case_id"],
                "group_id": episode["group_id"],
                "controller": episode["controller"],
                "method": episode["method"],
                "horizon": int(episode["horizon"]),
                "stratum": episode["stratum"],
                "regime": episode["regime"],
                "initial_distance_band": episode["initial_distance_band"],
                "episode_step": int(step["episode_step"]),
                "before_normalized_distance": float(
                    step["before_normalized_distance"]
                ),
                "predicted_next_target_cost": float(
                    step["predicted_next_target_cost"]
                ),
                "actual_next_target_cost": float(
                    step["actual_next_target_cost"]
                ),
                "predicted_first_step_improvement": float(
                    step["predicted_first_step_improvement"]
                ),
                "actual_first_step_improvement": float(
                    step["actual_first_step_improvement"]
                ),
                "false_improvement": int(
                    bool(
                        step[
                            "predicted_improvement_actual_worsening"
                        ]
                    )
                ),
                "selected_action_h1_prediction_mae": float(
                    step["selected_action_h1_prediction_mae"]
                ),
                "selected_action_h1_prediction_linf": float(
                    step["selected_action_h1_prediction_linf"]
                ),
                "predicted_terminal_target_cost": float(
                    step["predicted_terminal_target_cost"]
                ),
                "actual_selected_sequence_terminal_cost": (
                    math.nan
                    if step["actual_selected_sequence_terminal_cost"] is None
                    else float(
                        step["actual_selected_sequence_terminal_cost"]
                    )
                ),
                "action_normalized_linf": float(
                    step["action_distribution"]["normalized_linf"]
                ),
                "action_normalized_l1": float(
                    step["action_distribution"]["normalized_l1"]
                ),
                "outside_central_training_range": int(
                    bool(
                        step["action_distribution"][
                            "outside_central_training_range"
                        ]
                    )
                ),
                "outside_tail_training_range": int(
                    bool(
                        step["action_distribution"][
                            "outside_tail_training_range"
                        ]
                    )
                ),
                "at_per_step_action_bound": int(
                    bool(
                        step["action_distribution"][
                            "at_per_step_action_bound"
                        ]
                    )
                ),
                "minimum_position_bound_distance_before_mm": float(
                    step["position_bound_distance_before"]["minimum_mm"]
                ),
                "minimum_position_bound_distance_after_mm": float(
                    step["position_bound_distance_after"]["minimum_mm"]
                ),
                "action_bound_adjacent": int(
                    bool(
                        step["action_distribution"][
                            "at_per_step_action_bound"
                        ]
                    )
                    or float(
                        step["position_bound_distance_after"]["minimum_mm"]
                    )
                    <= 0.05
                ),
                "optical_clipping_active_before": int(
                    float(
                        step["current_auxiliary"]["clipping_fraction"]
                    )
                    > 0.01
                ),
                "optical_clipping_active_after": int(
                    float(
                        step["observed_auxiliary"]["clipping_fraction"]
                    )
                    > 0.01
                ),
                "camera_boundary_active_before": int(
                    bool(
                        step["current_auxiliary"][
                            "camera_boundary_indicator"
                        ]
                    )
                ),
                "camera_boundary_active_after": int(
                    bool(
                        step["observed_auxiliary"][
                            "camera_boundary_indicator"
                        ]
                    )
                ),
                "planner_runtime_seconds": float(
                    step["planner_runtime_seconds"]
                ),
                "planner_rollout_backend_calls": int(
                    step["planner_rollout_backend_calls"]
                ),
            }
            for field in ACTION_FIELDS:
                row[f"requested_{field}"] = float(
                    step["requested_action_mm"][field]
                )
                row[f"effective_{field}"] = float(
                    step["effective_action_mm"][field]
                )
            for field in OUTPUT_FIELDS:
                row[f"predicted_{field}"] = float(
                    step["predicted_next_metrics"][field]
                )
                row[f"actual_{field}"] = float(
                    step["actual_next_metrics"][field]
                )
                row[f"normalized_residual_{field}"] = float(
                    step["tolerance_normalized_prediction_residual"][field]
                )
            output.append(row)
    return output


def _counterfactual_rows(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for episode in episodes:
        for rollout in episode["h3_selected_sequence_counterfactuals"]:
            for depth in rollout["depths"]:
                row: dict[str, Any] = {
                    "case_id": episode["case_id"],
                    "group_id": episode["group_id"],
                    "controller": episode["controller"],
                    "stratum": episode["stratum"],
                    "regime": episode["regime"],
                    "episode_step": int(rollout["episode_step"]),
                    "depth": int(depth["depth"]),
                    "normalized_prediction_mae": float(
                        depth["tolerance_normalized_mae"]
                    ),
                    "normalized_prediction_linf": float(
                        depth["tolerance_normalized_linf"]
                    ),
                    "predicted_target_cost": float(
                        depth["predicted_target_cost"]
                    ),
                    "actual_target_cost": float(
                        depth["actual_target_cost"]
                    ),
                    "predicted_terminal_improvement": float(
                        rollout["predicted_terminal_improvement"]
                    ),
                    "actual_terminal_improvement": float(
                        rollout["actual_terminal_improvement"]
                    ),
                    "optimism_gap": float(rollout["optimism_gap"]),
                }
                for field in OUTPUT_FIELDS:
                    row[f"normalized_residual_{field}"] = float(
                        depth["tolerance_normalized_prediction_residual"][
                            field
                        ]
                    )
                output.append(row)
    return output


def _candidate_rows(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for episode in episodes:
        for candidate in episode["candidate_ranking_audits"]:
            first_action = candidate["effective_sequence"][0]
            output.append(
                {
                    "case_id": episode["case_id"],
                    "group_id": episode["group_id"],
                    "controller": episode["controller"],
                    "horizon": int(episode["horizon"]),
                    "stratum": episode["stratum"],
                    "regime": episode["regime"],
                    "episode_step": int(candidate["episode_step"]),
                    "selection": candidate["selection"],
                    "selection_rank": int(candidate["selection_rank"]),
                    "candidate_index": int(candidate["candidate_index"]),
                    "predicted_score": float(candidate["predicted_score"]),
                    "predicted_terminal_cost": float(
                        candidate["predicted_terminal_cost"]
                    ),
                    "actual_terminal_cost": float(
                        candidate["actual_terminal_cost"]
                    ),
                    "predicted_terminal_improvement": float(
                        candidate["predicted_terminal_improvement"]
                    ),
                    "actual_terminal_improvement": float(
                        candidate["actual_terminal_improvement"]
                    ),
                    "optimism_gap": float(candidate["optimism_gap"]),
                    "false_improvement": int(
                        bool(
                            candidate[
                                "predicted_improvement_actual_worsening"
                            ]
                        )
                    ),
                    **{
                        f"first_{field}": float(first_action[field])
                        for field in ACTION_FIELDS
                    },
                }
            )
    return output


def _summarize_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    predicted = sum(int(row["predicted_improvement_steps"]) for row in rows)
    false = sum(int(row["false_improvement_steps"]) for row in rows)
    result = {
        "groups": len(rows),
        "strict_success_rate": float(
            np.mean([row["strict_success"] for row in rows])
        ),
        "strict_success_group_bootstrap_95": _bootstrap(
            [float(row["strict_success"]) for row in rows],
            seed=seed,
            samples=samples,
        ),
        "any_improvement_rate": float(
            np.mean([row["any_improvement"] for row in rows])
        ),
        "predicted_improvement_actual_worsening_rate": (
            None if predicted == 0 else float(false / predicted)
        ),
        "mean_final_normalized_distance": float(
            np.mean([row["final_normalized_distance"] for row in rows])
        ),
        "median_final_normalized_distance": float(
            np.median([row["final_normalized_distance"] for row in rows])
        ),
        "mean_normalized_distance_reduction": float(
            np.mean([row["normalized_distance_reduction"] for row in rows])
        ),
        "median_normalized_distance_reduction": float(
            np.median([row["normalized_distance_reduction"] for row in rows])
        ),
        "mean_final_to_initial_distance_ratio": float(
            np.mean([row["final_to_initial_distance_ratio"] for row in rows])
        ),
        "median_final_to_initial_distance_ratio": float(
            np.median(
                [row["final_to_initial_distance_ratio"] for row in rows]
            )
        ),
        "median_steps": float(
            np.median([row["executed_steps"] for row in rows])
        ),
        "median_cumulative_motion_l1_mm": float(
            np.median([row["cumulative_motion_l1_mm"] for row in rows])
        ),
        "boundary_clipping_failure_rate": float(
            np.mean([row["clipping_or_boundary_failure"] for row in rows])
        ),
        "mean_planner_runtime_seconds": float(
            np.mean([row["planner_runtime_seconds"] for row in rows])
        ),
        "mean_simulator_calls": float(
            np.mean([row["simulator_calls"] for row in rows])
        ),
        "mean_model_rollout_calls": float(
            np.mean([row["model_rollout_calls"] for row in rows])
        ),
        "selected_actions_outside_central_rate": float(
            np.nanmean(
                [row["selected_actions_outside_central_rate"] for row in rows]
            )
        ),
        "selected_actions_outside_tail_rate": float(
            np.nanmean(
                [row["selected_actions_outside_tail_rate"] for row in rows]
            )
        ),
        "per_output_final_tolerance_failure_rate": {
            field: float(
                np.mean([row[f"final_failure_{field}"] for row in rows])
            )
            for field in OUTPUT_FIELDS
        },
    }
    return result


def _paired_comparisons(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    by_case = {
        (str(row["case_id"]), str(row["controller"])): row
        for row in episode_rows
    }
    pairs = (
        ("oracle_h1", "oracle_h3"),
        ("learned_h1", "learned_h3"),
        ("oracle_h1", "learned_h1"),
        ("oracle_h3", "learned_h3"),
    )
    output = {}
    case_ids = sorted({str(row["case_id"]) for row in episode_rows})
    for pair_index, (left_name, right_name) in enumerate(pairs):
        rows = [
            (by_case[(case_id, left_name)], by_case[(case_id, right_name)])
            for case_id in case_ids
            if (case_id, left_name) in by_case
            and (case_id, right_name) in by_case
        ]
        success_delta = [
            float(right["strict_success"] - left["strict_success"])
            for left, right in rows
        ]
        final_delta = [
            float(
                right["final_normalized_distance"]
                - left["final_normalized_distance"]
            )
            for left, right in rows
        ]
        reduction_delta = [
            float(
                right["normalized_distance_reduction"]
                - left["normalized_distance_reduction"]
            )
            for left, right in rows
        ]
        output[f"{right_name}_minus_{left_name}"] = {
            "groups": len(rows),
            "strict_success_rate_difference": _bootstrap(
                success_delta,
                seed=seed + pair_index * 101,
                samples=samples,
            ),
            "mean_final_distance_difference": _bootstrap(
                final_delta,
                seed=seed + pair_index * 101 + 1,
                samples=samples,
            ),
            "mean_distance_reduction_difference": _bootstrap(
                reduction_delta,
                seed=seed + pair_index * 101 + 2,
                samples=samples,
            ),
            "right_better_final_distance_rate": (
                None
                if not rows
                else float(np.mean(np.asarray(final_delta) < 0.0))
            ),
            "left_better_final_distance_rate": (
                None
                if not rows
                else float(np.mean(np.asarray(final_delta) > 0.0))
            ),
            "ties_final_distance_rate": (
                None
                if not rows
                else float(np.mean(np.isclose(final_delta, 0.0)))
            ),
        }
    return output


def _candidate_decisions(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                str(row["case_id"]),
                str(row["controller"]),
                int(row["episode_step"]),
            )
        ].append(row)
    decisions = []
    for (case_id, controller, episode_step), candidates in sorted(
        grouped.items()
    ):
        predicted = [float(row["predicted_terminal_cost"]) for row in candidates]
        actual = [float(row["actual_terminal_cost"]) for row in candidates]
        top = [
            row
            for row in candidates
            if row["selection"] == "predicted_top"
            and int(row["selection_rank"]) == 1
        ][0]
        improving = [
            row
            for row in candidates
            if float(row["predicted_terminal_improvement"]) > 0.0
        ]
        decisions.append(
            {
                "case_id": case_id,
                "controller": controller,
                "episode_step": episode_step,
                "audited_candidates": len(candidates),
                "spearman_predicted_actual_cost": _safe_correlation(
                    predicted, actual, "spearman"
                ),
                "pearson_predicted_actual_cost": _safe_correlation(
                    predicted, actual, "pearson"
                ),
                "top_candidate_regret": float(
                    top["actual_terminal_cost"] - min(actual)
                ),
                "predicted_improving_candidates": len(improving),
                "predicted_improving_actual_worsening_fraction": (
                    None
                    if not improving
                    else float(
                        np.mean([row["false_improvement"] for row in improving])
                    )
                ),
                "mean_optimism_gap": float(
                    np.mean([row["optimism_gap"] for row in candidates])
                ),
                "top_candidate_optimism_gap": float(top["optimism_gap"]),
            }
        )
    by_controller = {}
    for controller in ("learned_h1", "learned_h3"):
        selected = [
            row for row in decisions if row["controller"] == controller
        ]
        by_controller[controller] = {
            "decisions": len(selected),
            "mean_spearman_rank_correlation": float(
                np.nanmean(
                    [
                        math.nan
                        if row["spearman_predicted_actual_cost"] is None
                        else row["spearman_predicted_actual_cost"]
                        for row in selected
                    ]
                )
            ),
            "median_spearman_rank_correlation": float(
                np.nanmedian(
                    [
                        math.nan
                        if row["spearman_predicted_actual_cost"] is None
                        else row["spearman_predicted_actual_cost"]
                        for row in selected
                    ]
                )
            ),
            "mean_top_candidate_regret": float(
                np.mean([row["top_candidate_regret"] for row in selected])
            ),
            "mean_top_candidate_optimism_gap": float(
                np.mean([row["top_candidate_optimism_gap"] for row in selected])
            ),
            "mean_predicted_improving_actual_worsening_fraction": float(
                np.nanmean(
                    [
                        math.nan
                        if row[
                            "predicted_improving_actual_worsening_fraction"
                        ]
                        is None
                        else row[
                            "predicted_improving_actual_worsening_fraction"
                        ]
                        for row in selected
                    ]
                )
            ),
        }
    return decisions, by_controller


def _h3_rollout_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_depth = {}
    for depth in (1, 2, 3):
        selected = [row for row in rows if int(row["depth"]) == depth]
        by_depth[str(depth)] = {
            "records": len(selected),
            "mean_normalized_prediction_mae": float(
                np.mean(
                    [row["normalized_prediction_mae"] for row in selected]
                )
            ),
            "median_normalized_prediction_mae": float(
                np.median(
                    [row["normalized_prediction_mae"] for row in selected]
                )
            ),
            "mean_normalized_prediction_linf": float(
                np.mean(
                    [row["normalized_prediction_linf"] for row in selected]
                )
            ),
            "per_output_mean_normalized_residual": {
                field: float(
                    np.mean(
                        [row[f"normalized_residual_{field}"] for row in selected]
                    )
                )
                for field in OUTPUT_FIELDS
            },
        }
    grouped: defaultdict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["episode_step"]))][
            int(row["depth"])
        ] = float(row["normalized_prediction_mae"])
    classifications = []
    for key, values in grouped.items():
        if set(values) != {1, 2, 3}:
            continue
        curvature = values[3] - 2.0 * values[2] + values[1]
        scale = max(values[3], 0.1)
        classification = (
            "approximately_linear"
            if abs(curvature) <= 0.15 * scale
            else "superlinear"
            if curvature > 0.0
            else "sublinear"
        )
        classifications.append(
            {
                "case_id": key[0],
                "episode_step": key[1],
                "depth_1_mae": values[1],
                "depth_2_mae": values[2],
                "depth_3_mae": values[3],
                "second_difference": curvature,
                "classification": classification,
            }
        )
    counts = {
        name: sum(row["classification"] == name for row in classifications)
        for name in ("sublinear", "approximately_linear", "superlinear")
    }
    return {
        "by_depth": by_depth,
        "growth_classification_definition": (
            "second difference within 15% of max(depth3_mae,0.1) is "
            "approximately linear; positive is superlinear; negative sublinear"
        ),
        "growth_classification_counts": counts,
        "growth_classifications": classifications,
    }


def _step_action_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = {}
    for controller in CONTROLLERS:
        controller_rows = [
            row for row in rows if row["controller"] == controller
        ]
        output[controller] = {}
        for label, predicate in (
            ("ordinary_action_step", lambda row: not row["action_bound_adjacent"]),
            ("action_bound_adjacent_step", lambda row: row["action_bound_adjacent"]),
        ):
            selected = [row for row in controller_rows if predicate(row)]
            output[controller][label] = {
                "steps": len(selected),
                "mean_actual_improvement": (
                    None
                    if not selected
                    else float(
                        np.mean(
                            [row["actual_first_step_improvement"] for row in selected]
                        )
                    )
                ),
                "false_improvement_rate": (
                    None
                    if not selected
                    else float(np.mean([row["false_improvement"] for row in selected]))
                ),
                "mean_selected_action_h1_prediction_mae": (
                    None
                    if not selected
                    else float(
                        np.mean(
                            [
                                row["selected_action_h1_prediction_mae"]
                                for row in selected
                            ]
                        )
                    )
                ),
            }
    return output


def _plots(
    output_dir: Path,
    episodes: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "oracle_h1": "#1f77b4",
        "oracle_h3": "#17becf",
        "learned_h1": "#d62728",
        "learned_h3": "#ff7f0e",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    success = [
        summary["overall"][controller]["strict_success_rate"]
        for controller in CONTROLLERS
    ]
    low = [
        summary["overall"][controller]["strict_success_group_bootstrap_95"][
            "low"
        ]
        for controller in CONTROLLERS
    ]
    high = [
        summary["overall"][controller]["strict_success_group_bootstrap_95"][
            "high"
        ]
        for controller in CONTROLLERS
    ]
    axes[0].bar(
        CONTROLLERS,
        success,
        color=[colors[name] for name in CONTROLLERS],
        yerr=[
            np.asarray(success) - np.asarray(low),
            np.asarray(high) - np.asarray(success),
        ],
        capsize=4,
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("strict all-five success")
    axes[0].tick_params(axis="x", rotation=25)
    final = [
        [
            row["final_normalized_distance"]
            for row in episodes
            if row["controller"] == controller
        ]
        for controller in CONTROLLERS
    ]
    axes[1].boxplot(final, tick_labels=CONTROLLERS, showfliers=False)
    axes[1].set_ylabel("final normalized distance")
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "oracle_learned_h1_h3_comparison.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, controller in zip(axes, ("learned_h1", "learned_h3"), strict=True):
        selected = [row for row in steps if row["controller"] == controller]
        x = np.asarray(
            [row["predicted_first_step_improvement"] for row in selected]
        )
        y = np.asarray(
            [row["actual_first_step_improvement"] for row in selected]
        )
        axis.scatter(x, y, s=14, alpha=0.55, color=colors[controller])
        if len(x):
            limit = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0))
            axis.plot([-limit, limit], [-limit, limit], color="black", lw=1)
            axis.axhline(0.0, color="gray", lw=0.8)
            axis.axvline(0.0, color="gray", lw=0.8)
        axis.set_title(controller)
        axis.set_xlabel("predicted first-step improvement")
        axis.set_ylabel("actual simulator improvement")
    figure.tight_layout()
    figure.savefig(output_dir / "predicted_vs_actual_improvement.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for field in OUTPUT_FIELDS:
        values = [
            np.mean(
                [
                    row[f"normalized_residual_{field}"]
                    for row in counterfactuals
                    if int(row["depth"]) == depth
                ]
            )
            for depth in (1, 2, 3)
        ]
        axis.plot((1, 2, 3), values, marker="o", label=field)
    axis.set_xlabel("counterfactual rollout depth")
    axis.set_ylabel("mean tolerance-normalized residual")
    axis.set_xticks((1, 2, 3))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_error_by_depth_and_output.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for controller in ("learned_h1", "learned_h3"):
        selected = [
            row for row in candidates if row["controller"] == controller
        ]
        axes[0].scatter(
            [row["predicted_terminal_cost"] for row in selected],
            [row["actual_terminal_cost"] for row in selected],
            s=10,
            alpha=0.35,
            label=controller,
            color=colors[controller],
        )
        axes[1].hist(
            [row["optimism_gap"] for row in selected],
            bins=30,
            alpha=0.5,
            label=controller,
            color=colors[controller],
        )
    axes[0].set_xlabel("predicted candidate terminal cost")
    axes[0].set_ylabel("actual simulator terminal cost")
    axes[0].legend()
    axes[1].set_xlabel("optimism gap")
    axes[1].set_ylabel("audited candidates")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "candidate_optimism_rank_calibration.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for controller in CONTROLLERS:
        selected = [row for row in steps if row["controller"] == controller]
        axes[0].hist(
            [row["action_normalized_linf"] for row in selected],
            bins=np.linspace(0.0, 1.05, 22),
            histtype="step",
            linewidth=1.5,
            label=controller,
            color=colors[controller],
        )
    axes[0].set_xlabel("selected action normalized L-infinity")
    axes[0].set_ylabel("steps")
    axes[0].legend()
    learned = [
        row for row in steps if str(row["method"]) == "learned"
    ]
    axes[1].scatter(
        [row["minimum_position_bound_distance_after_mm"] for row in learned],
        [row["selected_action_h1_prediction_mae"] for row in learned],
        c=[row["false_improvement"] for row in learned],
        cmap="coolwarm",
        s=15,
        alpha=0.6,
    )
    axes[1].set_xlabel("distance to nearest absolute actuator bound (mm)")
    axes[1].set_ylabel("selected-action H1 prediction MAE")
    figure.tight_layout()
    figure.savefig(output_dir / "action_distribution_bound_proximity.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    locked = json.loads(
        args.config.resolve().read_text(encoding="utf-8")
    )
    suite = json.loads(
        args.suite.resolve().read_text(encoding="utf-8")
    )
    episodes = []
    for path in sorted(args.episodes_dir.resolve().glob("*.json")):
        if "oracle_retry" in path.name:
            continue
        episode = json.loads(path.read_text(encoding="utf-8"))
        if episode.get("version") != EPISODE_VERSION:
            raise ValueError(f"unexpected episode version: {path}")
        episodes.append(episode)
    expected = len(suite["cases"]) * len(CONTROLLERS)
    if not args.allow_incomplete and len(episodes) != expected:
        raise RuntimeError(
            f"expected {expected} primary episodes, found {len(episodes)}"
        )
    keys = [
        (str(row["case_id"]), str(row["controller"])) for row in episodes
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate case/controller episode")
    if any(
        row["simulator_semantics_version"]
        != suite["simulator_semantics_version"]
        for row in episodes
    ):
        raise RuntimeError("mixed simulator semantics in episodes")
    if any(bool(row["q_goal_used_by_controller"]) for row in episodes):
        raise RuntimeError("q_goal leaked into controller")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite aggregate: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = [_episode_row(row) for row in episodes]
    step_rows = _step_rows(episodes)
    counterfactual_rows = _counterfactual_rows(episodes)
    candidate_rows = _candidate_rows(episodes)
    samples = int(locked["bootstrap"]["samples"])
    seed = int(locked["bootstrap"]["seed"])
    summary: dict[str, Any] = {
        "version": "v12_mpc_h1_h3_diagnostic_aggregate_v1",
        "episodes": len(episode_rows),
        "groups": len({row["group_id"] for row in episode_rows}),
        "expected_episodes": expected,
        "complete": len(episode_rows) == expected,
        "overall": {},
        "by_stratum": {},
        "by_initial_distance_band": {},
    }
    for controller_index, controller in enumerate(CONTROLLERS):
        selected = [
            row for row in episode_rows if row["controller"] == controller
        ]
        summary["overall"][controller] = _summarize_episode_rows(
            selected,
            seed=seed + controller_index,
            samples=samples,
        )
    for stratum in sorted({row["stratum"] for row in episode_rows}):
        summary["by_stratum"][stratum] = {}
        for controller_index, controller in enumerate(CONTROLLERS):
            selected = [
                row
                for row in episode_rows
                if row["controller"] == controller
                and row["stratum"] == stratum
            ]
            summary["by_stratum"][stratum][controller] = (
                _summarize_episode_rows(
                    selected,
                    seed=seed + 1000 + controller_index,
                    samples=samples,
                )
            )
    for band in ("low", "medium", "high"):
        summary["by_initial_distance_band"][band] = {}
        for controller_index, controller in enumerate(CONTROLLERS):
            selected = [
                row
                for row in episode_rows
                if row["controller"] == controller
                and row["initial_distance_band"] == band
            ]
            summary["by_initial_distance_band"][band][controller] = (
                _summarize_episode_rows(
                    selected,
                    seed=seed + 2000 + controller_index,
                    samples=samples,
                )
            )
    summary["paired_comparisons"] = _paired_comparisons(
        episode_rows, seed=seed + 3000, samples=samples
    )
    decision_rows, candidate_summary = _candidate_decisions(candidate_rows)
    summary["learned_candidate_ranking"] = candidate_summary
    summary["h3_selected_sequence_rollout"] = _h3_rollout_summary(
        counterfactual_rows
    )
    summary["ordinary_vs_action_bound_adjacent_steps"] = (
        _step_action_summary(step_rows)
    )
    learned_steps = [
        row for row in step_rows if row["method"] == "learned"
    ]
    summary["learned_selected_step_prediction"] = {
        controller: {
            "steps": len(
                [
                    row
                    for row in learned_steps
                    if row["controller"] == controller
                ]
            ),
            "mean_h1_prediction_mae": float(
                np.mean(
                    [
                        row["selected_action_h1_prediction_mae"]
                        for row in learned_steps
                        if row["controller"] == controller
                    ]
                )
            ),
            "predicted_actual_improvement_spearman": _safe_correlation(
                [
                    row["predicted_first_step_improvement"]
                    for row in learned_steps
                    if row["controller"] == controller
                ],
                [
                    row["actual_first_step_improvement"]
                    for row in learned_steps
                    if row["controller"] == controller
                ],
                "spearman",
            ),
            "predicted_actual_improvement_pearson": _safe_correlation(
                [
                    row["predicted_first_step_improvement"]
                    for row in learned_steps
                    if row["controller"] == controller
                ],
                [
                    row["actual_first_step_improvement"]
                    for row in learned_steps
                    if row["controller"] == controller
                ],
                "pearson",
            ),
        }
        for controller in ("learned_h1", "learned_h3")
    }
    _write_csv(output_dir / "episodes.csv", episode_rows)
    _write_csv(output_dir / "steps.csv", step_rows)
    _write_csv(
        output_dir / "h3_counterfactual_rollouts.csv",
        counterfactual_rows,
    )
    _write_csv(output_dir / "candidate_ranking_audits.csv", candidate_rows)
    _write_csv(
        output_dir / "candidate_decision_summary.csv", decision_rows
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "paired_comparisons.json",
        summary["paired_comparisons"],
    )
    _plots(
        output_dir,
        episode_rows,
        step_rows,
        counterfactual_rows,
        candidate_rows,
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
