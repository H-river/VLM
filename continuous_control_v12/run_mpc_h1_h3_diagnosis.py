#!/usr/bin/env python3
"""Run resumable matched Oracle/Learned H1/H3 episodes on a fixed v12 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    Bounds,
    action_dict,
    assert_no_q_star,
    metrics_dict,
    metrics_vector,
    normalized_distance,
    normalized_error,
    position_dict,
    position_vector,
    project_action,
    stable_seed,
)
from continuous_control_v12.mpc import (
    CEMMPC,
    learned_predictor,
    simulator_predictor,
)
from continuous_control_v12.simulator import (
    CORRECTED_SEMANTICS_VERSION,
    simulate_state,
)
from continuous_control_v12.world_model import (
    ForwardEnsemble,
    load_forward_ensemble,
)

EPISODE_VERSION = "v12_mpc_h1_h3_diagnostic_episode_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_locked_sources(locked: Mapping[str, Any]) -> None:
    checks = (
        ("corrected_v12_config", "corrected_v12_config_sha256"),
        ("base_simulator_config", "base_simulator_config_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    )
    for path_key, hash_key in checks:
        path = Path(str(locked[path_key])).resolve()
        if _sha256(path) != str(locked[hash_key]):
            raise ValueError(f"locked source hash mismatch: {path}")
    manifest = (
        Path(str(locked["training_dataset"])).resolve() / "manifest.json"
    )
    if _sha256(manifest) != str(
        locked["training_dataset_manifest_sha256"]
    ):
        raise ValueError("locked training dataset manifest hash mismatch")


def _action_array(action: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(action, Mapping):
        return np.asarray(
            [float(action[field]) for field in ACTION_FIELDS],
            dtype=np.float64,
        )
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("action must have four values")
    return values


def _serializable_auxiliary(capture: Mapping[str, Any]) -> dict[str, Any]:
    auxiliary = capture["auxiliary"]
    return {
        key: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else None
            if value is None
            else float(value)
            if isinstance(value, (int, float, np.integer, np.floating))
            else value
        )
        for key, value in auxiliary.items()
    }


def _action_distribution_flags(
    action: np.ndarray,
    distribution: Mapping[str, Any],
    bounds: Bounds,
) -> dict[str, Any]:
    central_outside = []
    tail_outside = []
    for index, field in enumerate(ACTION_FIELDS):
        summary = distribution["per_axis"][field]
        central_outside.append(
            bool(
                action[index] < float(summary["q05"]) - 1e-12
                or action[index] > float(summary["q95"]) + 1e-12
            )
        )
        tail_outside.append(
            bool(
                action[index] < float(summary["q01"]) - 1e-12
                or action[index] > float(summary["q99"]) + 1e-12
            )
        )
    normalized = np.abs(action / bounds.action_high)
    return {
        "normalized_linf": float(normalized.max()),
        "normalized_l1": float(normalized.sum()),
        "per_axis_outside_central_training_range": {
            field: central_outside[index]
            for index, field in enumerate(ACTION_FIELDS)
        },
        "per_axis_outside_tail_training_range": {
            field: tail_outside[index]
            for index, field in enumerate(ACTION_FIELDS)
        },
        "outside_central_training_range": bool(any(central_outside)),
        "outside_tail_training_range": bool(any(tail_outside)),
        "at_per_step_action_bound": bool(np.any(normalized >= 1.0 - 1e-10)),
    }


def _position_bound_distances(
    positions: np.ndarray,
    bounds: Bounds,
) -> dict[str, Any]:
    low = positions - bounds.position_low
    high = bounds.position_high - positions
    nearest = np.minimum(low, high)
    return {
        "minimum_mm": float(nearest.min()),
        "per_axis_mm": {
            field: float(nearest[index])
            for index, field in enumerate(POSITION_FIELDS)
        },
        "at_absolute_bound": bool(np.any(nearest <= 1e-10)),
    }


def simulate_counterfactual_sequence(
    *,
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    base_config_path: str,
    bounds: Bounds,
    start_positions: Mapping[str, Any] | Sequence[float],
    effective_sequence: Sequence[Mapping[str, Any] | Sequence[float]],
) -> dict[str, Any]:
    """Execute one sequence on cloned arrays; caller state is never mutated."""

    positions = position_vector(start_positions).copy()
    start_copy = positions.copy()
    depths = []
    for depth, action_value in enumerate(effective_sequence, start=1):
        requested = _action_array(action_value)
        effective = project_action(positions, requested, bounds)
        if not np.allclose(
            requested, effective, atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                "planner effective sequence changed under simulator replay"
            )
        positions = positions + effective
        capture = simulate_state(
            setup_context,
            position_dict(positions),
            simulator_fixed,
            base_config_path,
            bounds,
        )
        depths.append(
            {
                "depth": depth,
                "effective_action_mm": action_dict(effective),
                "positions_mm": position_dict(positions),
                "metrics": capture["metrics"],
                "auxiliary": _serializable_auxiliary(capture),
            }
        )
    if not np.array_equal(start_copy, position_vector(start_positions)):
        raise RuntimeError("counterfactual rollout mutated the start state")
    return {
        "depths": depths,
        "simulator_calls": len(depths),
        "final_positions_mm": position_dict(positions),
    }


def _counterfactual_comparison(
    *,
    predicted_metrics: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    target_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any] | Sequence[float],
) -> dict[str, Any]:
    reference = metrics_vector(reference_metrics)
    target = metrics_vector(target_metrics)
    depth_rows = []
    for predicted, actual in zip(
        predicted_metrics, replay["depths"], strict=True
    ):
        predicted_values = metrics_vector(predicted)
        actual_values = metrics_vector(actual["metrics"])
        residual = np.abs(predicted_values - actual_values)
        normalized_residual = normalized_error(
            predicted_values, actual_values, reference
        )
        depth_rows.append(
            {
                "depth": int(actual["depth"]),
                "predicted_metrics": metrics_dict(predicted_values),
                "actual_metrics": metrics_dict(actual_values),
                "absolute_prediction_residual": {
                    field: float(residual[index])
                    for index, field in enumerate(OUTPUT_FIELDS)
                },
                "tolerance_normalized_prediction_residual": {
                    field: float(normalized_residual[index])
                    for index, field in enumerate(OUTPUT_FIELDS)
                },
                "tolerance_normalized_mae": float(
                    normalized_residual.mean()
                ),
                "tolerance_normalized_linf": float(
                    normalized_residual.max()
                ),
                "predicted_target_cost": normalized_distance(
                    predicted_values, target, reference
                ),
                "actual_target_cost": normalized_distance(
                    actual_values, target, reference
                ),
                "actual_auxiliary": actual["auxiliary"],
            }
        )
    return {
        "depths": depth_rows,
        "predicted_terminal_cost": depth_rows[-1][
            "predicted_target_cost"
        ],
        "actual_terminal_cost": depth_rows[-1]["actual_target_cost"],
    }


def _audit_candidates(
    *,
    audit_candidates: Sequence[Mapping[str, Any]],
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    base_config_path: str,
    bounds: Bounds,
    positions: np.ndarray,
    current_metrics: np.ndarray,
    target_metrics: np.ndarray,
    reference_metrics: np.ndarray,
    episode_step: int,
) -> tuple[list[dict[str, Any]], int]:
    records = []
    simulator_calls = 0
    before = normalized_distance(
        current_metrics, target_metrics, reference_metrics
    )
    for candidate in audit_candidates:
        replay = simulate_counterfactual_sequence(
            setup_context=setup_context,
            simulator_fixed=simulator_fixed,
            base_config_path=base_config_path,
            bounds=bounds,
            start_positions=positions,
            effective_sequence=candidate["effective_sequence"],
        )
        simulator_calls += int(replay["simulator_calls"])
        comparison = _counterfactual_comparison(
            predicted_metrics=candidate["predicted_rollout_metrics"],
            replay=replay,
            target_metrics=metrics_dict(target_metrics),
            reference_metrics=reference_metrics,
        )
        predicted_improvement = float(
            candidate["predicted_terminal_improvement"]
        )
        actual_improvement = float(
            before - comparison["actual_terminal_cost"]
        )
        records.append(
            {
                "episode_step": episode_step,
                "selection": candidate["selection"],
                "selection_rank": int(candidate["selection_rank"]),
                "candidate_index": int(candidate["candidate_index"]),
                "requested_sequence": candidate["requested_sequence"],
                "effective_sequence": candidate["effective_sequence"],
                "predicted_score": float(candidate["predicted_score"]),
                "predicted_terminal_cost": float(
                    candidate["predicted_terminal_target_cost"]
                ),
                "actual_terminal_cost": float(
                    comparison["actual_terminal_cost"]
                ),
                "predicted_terminal_improvement": predicted_improvement,
                "actual_terminal_improvement": actual_improvement,
                "optimism_gap": float(
                    predicted_improvement - actual_improvement
                ),
                "predicted_improvement_actual_worsening": bool(
                    predicted_improvement > 0.0
                    and actual_improvement < 0.0
                ),
                "counterfactual_depths": comparison["depths"],
            }
        )
    return records, simulator_calls


def run_diagnostic_episode(
    *,
    case: Mapping[str, Any],
    method: str,
    horizon: int,
    locked: Mapping[str, Any],
    v12_config: Mapping[str, Any],
    model: ForwardEnsemble | None,
    mode_config: Mapping[str, Any],
) -> dict[str, Any]:
    if method not in {"oracle", "learned"}:
        raise ValueError(f"unknown diagnostic method: {method}")
    if horizon not in {1, 3}:
        raise ValueError(f"diagnostic horizon must be 1 or 3, got {horizon}")
    if case["simulator_fixed"]["simulator_semantics_version"] != (
        CORRECTED_SEMANTICS_VERSION
    ):
        raise ValueError("episode refuses non-corrected simulator semantics")
    deployed_input = {
        "setup_context": case["setup_context"],
        "simulator_fixed": case["simulator_fixed"],
        "initial_positions_mm": case["initial_positions_mm"],
        "initial_metrics": case["initial_metrics"],
        "target_metrics": case["target_metrics"],
    }
    assert_no_q_star(deployed_input)
    bounds = Bounds.from_config(v12_config)
    base_config_path = str(
        Path(str(locked["base_simulator_config"])).resolve()
    )
    if method == "oracle":
        predictor = simulator_predictor(
            setup_context=case["setup_context"],
            simulator_fixed=case["simulator_fixed"],
            base_config_path=base_config_path,
            bounds=bounds,
        )
    else:
        if model is None:
            raise ValueError("learned diagnostic requires the locked model")
        predictor = learned_predictor(model, case["setup_context"])
    planner_config = {
        "horizon": horizon,
        "population": int(mode_config["population"]),
        "elites": int(mode_config["elites"]),
        "cem_iterations": int(mode_config["cem_iterations"]),
        "mean_error_weight": float(
            locked["primary"]["mean_error_weight"]
        ),
        "movement_weight": float(locked["primary"]["movement_weight"]),
        "limit_penalty": float(locked["primary"]["limit_penalty"]),
        "boundary_penalty": float(locked["primary"]["boundary_penalty"]),
        "uncertainty_weight": float(
            locked["primary"]["uncertainty_weight"]
        ),
        "candidate_audit_top_k": (
            int(mode_config["candidate_audit_top_k"])
            if method == "learned"
            else 0
        ),
        "candidate_audit_reference_k": (
            int(mode_config["candidate_audit_reference_k"])
            if method == "learned"
            else 0
        ),
    }
    planner_seed = stable_seed(
        int(locked["root_seed"]),
        str(case["case_id"]),
        "matched_mpc",
    )
    planner = CEMMPC(
        bounds=bounds,
        predictor=predictor,
        config=planner_config,
        seed=planner_seed,
    )
    positions = position_vector(case["initial_positions_mm"]).copy()
    metrics = metrics_vector(case["initial_metrics"]).copy()
    target = metrics_vector(case["target_metrics"])
    reference = metrics.copy()
    current_auxiliary = dict(case["initial_auxiliary"])
    trace = []
    candidate_audits: list[dict[str, Any]] = []
    h3_counterfactuals: list[dict[str, Any]] = []
    cumulative_per_axis = np.zeros(4, dtype=np.float64)
    maximum_temporary_worsening = 0.0
    simulator_calls = 0
    model_calls = 0
    planner_seconds = 0.0
    final_reason = "max_steps"
    started = time.perf_counter()
    for episode_step in range(int(mode_config["max_closed_loop_steps"])):
        before = normalized_distance(metrics, target, reference)
        if before <= 1.0:
            final_reason = "reached"
            break
        before_positions = positions.copy()
        before_metrics = metrics.copy()
        plan_started = time.perf_counter()
        plan = planner.plan(
            positions_mm=positions,
            current_metrics=metrics,
            target_metrics=target,
            allowed_dofs=locked["primary"]["allowed_dofs"],
            tolerance_reference=reference,
        )
        step_planner_seconds = time.perf_counter() - plan_started
        planner_seconds += step_planner_seconds
        if method == "oracle":
            simulator_calls += int(plan["rollout_backend_calls"])
        else:
            model_calls += int(plan["rollout_backend_calls"])
        requested_action = _action_array(
            plan.get(
                "selected_requested_action", plan["selected_action"]
            )
        )
        effective_action = project_action(
            positions, requested_action, bounds
        )
        planner_effective_action = _action_array(
            plan.get(
                "selected_effective_action", plan["selected_action"]
            )
        )
        if not np.allclose(
            effective_action,
            planner_effective_action,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("planner/executor action projection mismatch")
        if method == "learned" and plan["candidate_audit"]:
            audited, calls = _audit_candidates(
                audit_candidates=plan["candidate_audit"],
                setup_context=case["setup_context"],
                simulator_fixed=case["simulator_fixed"],
                base_config_path=base_config_path,
                bounds=bounds,
                positions=positions,
                current_metrics=metrics,
                target_metrics=target,
                reference_metrics=reference,
                episode_step=episode_step,
            )
            candidate_audits.extend(audited)
            simulator_calls += calls
        selected_counterfactual = None
        if method == "learned" and horizon == 3:
            replay = simulate_counterfactual_sequence(
                setup_context=case["setup_context"],
                simulator_fixed=case["simulator_fixed"],
                base_config_path=base_config_path,
                bounds=bounds,
                start_positions=positions,
                effective_sequence=plan["planned_effective_sequence"],
            )
            simulator_calls += int(replay["simulator_calls"])
            selected_counterfactual = _counterfactual_comparison(
                predicted_metrics=plan["predicted_rollout_metrics"],
                replay=replay,
                target_metrics=case["target_metrics"],
                reference_metrics=reference,
            )
            selected_counterfactual.update(
                {
                    "episode_step": episode_step,
                    "predicted_terminal_improvement": float(
                        before
                        - selected_counterfactual[
                            "predicted_terminal_cost"
                        ]
                    ),
                    "actual_terminal_improvement": float(
                        before
                        - selected_counterfactual[
                            "actual_terminal_cost"
                        ]
                    ),
                }
            )
            selected_counterfactual["optimism_gap"] = float(
                selected_counterfactual["predicted_terminal_improvement"]
                - selected_counterfactual["actual_terminal_improvement"]
            )
            h3_counterfactuals.append(selected_counterfactual)
        next_positions = positions + effective_action
        observed = simulate_state(
            case["setup_context"],
            position_dict(next_positions),
            case["simulator_fixed"],
            base_config_path,
            bounds,
        )
        simulator_calls += 1
        next_metrics = metrics_vector(observed["metrics"])
        after = normalized_distance(next_metrics, target, reference)
        predicted_next_metrics = metrics_vector(
            plan["predicted_next_metrics"]
        )
        predicted_after = normalized_distance(
            predicted_next_metrics, target, reference
        )
        predicted_improvement = float(before - predicted_after)
        actual_improvement = float(before - after)
        maximum_temporary_worsening = max(
            maximum_temporary_worsening,
            float(max(0.0, after - before)),
        )
        cumulative_per_axis += np.abs(effective_action)
        residual = normalized_error(
            predicted_next_metrics, next_metrics, reference
        )
        action_flags = _action_distribution_flags(
            effective_action,
            case["_training_action_distribution"],
            bounds,
        )
        trace.append(
            {
                "episode_step": episode_step,
                "current_positions_mm": position_dict(before_positions),
                "current_metrics": metrics_dict(before_metrics),
                "current_auxiliary": current_auxiliary,
                "requested_action_mm": action_dict(requested_action),
                "effective_action_mm": action_dict(effective_action),
                "requested_effective_action_gap_mm": action_dict(
                    requested_action - effective_action
                ),
                "action_distribution": action_flags,
                "position_bound_distance_before": _position_bound_distances(
                    before_positions, bounds
                ),
                "position_bound_distance_after": _position_bound_distances(
                    next_positions, bounds
                ),
                "predicted_next_metrics": metrics_dict(
                    predicted_next_metrics
                ),
                "actual_next_metrics": metrics_dict(next_metrics),
                "tolerance_normalized_prediction_residual": {
                    field: float(residual[index])
                    for index, field in enumerate(OUTPUT_FIELDS)
                },
                "selected_action_h1_prediction_mae": float(residual.mean()),
                "selected_action_h1_prediction_linf": float(residual.max()),
                "before_normalized_distance": float(before),
                "predicted_next_target_cost": float(predicted_after),
                "actual_next_target_cost": float(after),
                "predicted_first_step_improvement": predicted_improvement,
                "actual_first_step_improvement": actual_improvement,
                "predicted_improvement_actual_worsening": bool(
                    predicted_improvement > 0.0
                    and actual_improvement < 0.0
                ),
                "predicted_terminal_target_cost": float(
                    plan["predicted_terminal_distance"]
                ),
                "actual_selected_sequence_terminal_cost": (
                    None
                    if selected_counterfactual is None
                    else float(
                        selected_counterfactual["actual_terminal_cost"]
                    )
                ),
                "planner_runtime_seconds": step_planner_seconds,
                "planner_rollout_backend_calls": int(
                    plan["rollout_backend_calls"]
                ),
                "candidate_sequences_evaluated": int(
                    plan["candidate_sequences_evaluated"]
                ),
                "planner_iteration_history": plan["iteration_history"],
                "planned_requested_sequence": plan[
                    "planned_requested_sequence"
                ],
                "planned_effective_sequence": plan[
                    "planned_effective_sequence"
                ],
                "predicted_rollout_metrics": plan[
                    "predicted_rollout_metrics"
                ],
                "observed_auxiliary": _serializable_auxiliary(observed),
            }
        )
        positions = next_positions
        metrics = next_metrics
        current_auxiliary = _serializable_auxiliary(observed)
        if not bool(observed["auxiliary"]["simulator_valid"]):
            final_reason = "invalid_simulation"
            break
        if after <= 1.0:
            final_reason = "reached"
            break
        final_reason = "max_steps"
    initial_distance = normalized_distance(
        case["initial_metrics"], target, reference
    )
    final_distance = normalized_distance(metrics, target, reference)
    q_goal = position_vector(case["q_goal_mm"])
    success = bool(final_distance <= 1.0)
    result = {
        "version": EPISODE_VERSION,
        "case_id": str(case["case_id"]),
        "group_id": str(case["group_id"]),
        "stratum": str(case["stratum"]),
        "regime": str(case["regime"]),
        "initial_distance_band": str(case["initial_distance_band"]),
        "method": method,
        "horizon": horizon,
        "controller": f"{method}_h{horizon}",
        "planner_seed": int(planner_seed),
        "locked_mpc_config": planner_config,
        "simulator_semantics_version": (
            case["simulator_fixed"]["simulator_semantics_version"]
        ),
        "checkpoint": (
            None if method == "oracle" else str(locked["checkpoint"])
        ),
        "checkpoint_sha256": (
            None if method == "oracle" else str(locked["checkpoint_sha256"])
        ),
        "q_goal_used_by_controller": False,
        "future_state_used_by_controller": False,
        "reachability_label_used_by_controller": False,
        "candidate_simulator_audit_used_by_controller": False,
        "initial_positions_mm": case["initial_positions_mm"],
        "initial_metrics": case["initial_metrics"],
        "target_metrics": case["target_metrics"],
        "final_positions_mm": position_dict(positions),
        "final_metrics": metrics_dict(metrics),
        "initial_normalized_distance": float(initial_distance),
        "final_normalized_distance": float(final_distance),
        "final_to_initial_distance_ratio": float(
            final_distance / max(initial_distance, 1e-30)
        ),
        "normalized_distance_reduction": float(
            initial_distance - final_distance
        ),
        "success": success,
        "any_improvement": bool(final_distance < initial_distance),
        "maximum_temporary_worsening": float(
            maximum_temporary_worsening
        ),
        "executed_steps": len(trace),
        "termination_reason": final_reason,
        "cumulative_motion_mm": {
            field: float(cumulative_per_axis[index])
            for index, field in enumerate(ACTION_FIELDS)
        },
        "cumulative_motion_l1_mm": float(cumulative_per_axis.sum()),
        "final_actuator_linf_distance_to_q_goal_mm": float(
            np.max(np.abs(positions - q_goal))
        ),
        "final_output_tolerance_failure": {
            field: bool(value > 1.0)
            for field, value in zip(
                OUTPUT_FIELDS,
                normalized_error(metrics, target, reference),
                strict=True,
            )
        },
        "clipping_or_boundary_failure": bool(
            not success
            and (
                bool(current_auxiliary["camera_boundary_indicator"])
                or float(current_auxiliary["clipping_fraction"]) > 0.01
            )
        ),
        "planner_runtime_seconds": float(planner_seconds),
        "wall_runtime_seconds": float(time.perf_counter() - started),
        "planner_rollout_backend_calls": int(
            sum(row["planner_rollout_backend_calls"] for row in trace)
        ),
        "model_rollout_calls": int(model_calls),
        "simulator_calls": int(simulator_calls),
        "predicted_improvement_steps": int(
            sum(
                row["predicted_first_step_improvement"] > 0.0
                for row in trace
            )
        ),
        "false_improvement_steps": int(
            sum(
                row["predicted_improvement_actual_worsening"]
                for row in trace
            )
        ),
        "trace": trace,
        "h3_selected_sequence_counterfactuals": h3_counterfactuals,
        "candidate_ranking_audits": candidate_audits,
        "evaluator_q_goal_diagnostic": {
            "minimum_bounded_position_steps": int(
                case["minimum_bounded_position_steps"]
            ),
            "known_reachability": str(case["known_reachability"]),
            "q_goal_mm": case["q_goal_mm"],
            "visibility": "evaluator_only_never_planner_input",
        },
    }
    return result


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("version") != EPISODE_VERSION:
            raise RuntimeError(f"invalid existing episode: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _primary_tasks(
    cases: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str, int]]:
    return [
        (case, method, horizon)
        for case in cases
        for method in ("oracle", "learned")
        for horizon in (1, 3)
    ]


def _smoke_tasks(
    cases: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str, int]]:
    by_stratum: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        by_stratum.setdefault(str(case["stratum"]), case)
    ordered = [
        by_stratum["one_step_reachable_interior"],
        by_stratum["multi_step_reachable_interior"],
        by_stratum["reachable_boundary_or_clipping"],
    ]
    return [
        (ordered[0], "oracle", 1),
        (ordered[0], "oracle", 3),
        (ordered[1], "learned", 1),
        (ordered[1], "learned", 3),
        (ordered[2], "learned", 1),
        (ordered[2], "learned", 3),
    ]


def _retry_tasks(
    cases: Sequence[Mapping[str, Any]],
    failed_case_ids: set[str],
) -> list[tuple[Mapping[str, Any], str, int]]:
    return [
        (case, "oracle", horizon)
        for case in cases
        if str(case["case_id"]) in failed_case_ids
        for horizon in (1, 3)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("primary", "smoke", "oracle_retry"),
        default="primary",
    )
    parser.add_argument("--failed-cases", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    locked = json.loads(
        args.config.resolve().read_text(encoding="utf-8")
    )
    _verify_locked_sources(locked)
    suite = json.loads(
        args.suite.resolve().read_text(encoding="utf-8")
    )
    if suite["simulator_semantics_version"] != CORRECTED_SEMANTICS_VERSION:
        raise ValueError("runner refuses non-corrected diagnostic suite")
    if _sha256(args.config.resolve()) != str(suite["locked_config_sha256"]):
        raise ValueError("suite/runner locked config hash mismatch")
    v12_config = json.loads(
        Path(str(locked["corrected_v12_config"]))
        .resolve()
        .read_text(encoding="utf-8")
    )
    cases = list(suite["cases"])
    selected_ids = set(map(str, args.case_id))
    if selected_ids:
        cases = [
            case for case in cases if str(case["case_id"]) in selected_ids
        ]
        if len(cases) != len(selected_ids):
            raise ValueError("one or more selected diagnostic cases are absent")
    for case in cases:
        case["_training_action_distribution"] = suite[
            "training_action_distribution"
        ]
    if args.mode == "smoke":
        tasks = _smoke_tasks(cases)
        mode_config = {
            **locked["smoke"],
            "max_closed_loop_steps": locked["smoke"][
                "max_closed_loop_steps"
            ],
        }
    elif args.mode == "oracle_retry":
        if args.failed_cases is None:
            raise ValueError("oracle retry requires --failed-cases")
        failed = set(
            json.loads(
                args.failed_cases.resolve().read_text(encoding="utf-8")
            )["case_ids"]
        )
        tasks = _retry_tasks(cases, failed)
        mode_config = {
            **locked["oracle_retry"],
            "candidate_audit_top_k": 0,
            "candidate_audit_reference_k": 0,
            "max_closed_loop_steps": locked["oracle_retry"][
                "max_closed_loop_steps"
            ],
        }
    else:
        tasks = _primary_tasks(cases)
        mode_config = {
            **locked["primary"],
            "max_closed_loop_steps": locked["primary"][
                "max_closed_loop_steps"
            ],
        }
    tasks = [
        task
        for index, task in enumerate(tasks)
        if index % args.shard_count == args.shard_index
    ]
    needs_model = any(method == "learned" for _, method, _ in tasks)
    model = (
        load_forward_ensemble(
            Path(str(locked["checkpoint"])).resolve(),
            device_name="cpu",
        )
        if needs_model
        else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for task_index, (case, method, horizon) in enumerate(tasks, start=1):
        suffix = "oracle_retry" if args.mode == "oracle_retry" else method
        output = output_dir / (
            f"{case['case_id']}__{suffix}_h{horizon}.json"
        )
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing.get("version") != EPISODE_VERSION:
                raise RuntimeError(f"invalid resumable episode: {output}")
            skipped += 1
            print(
                json.dumps(
                    {
                        "event": "skip_existing",
                        "output": str(output),
                        "task": task_index,
                        "tasks": len(tasks),
                    }
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "event": "episode_start",
                    "case_id": case["case_id"],
                    "method": method,
                    "horizon": horizon,
                    "task": task_index,
                    "tasks": len(tasks),
                }
            ),
            flush=True,
        )
        result = run_diagnostic_episode(
            case=case,
            method=method,
            horizon=horizon,
            locked=locked,
            v12_config=v12_config,
            model=model,
            mode_config=mode_config,
        )
        if args.mode == "oracle_retry":
            result["diagnostic_role"] = (
                "post_primary_oracle_budget_diagnostic_not_primary_score"
            )
            result["primary_result_replaced"] = False
        else:
            result["diagnostic_role"] = args.mode
        _atomic_write_json(output, result)
        completed += 1
        print(
            json.dumps(
                {
                    "event": "episode_complete",
                    "case_id": case["case_id"],
                    "method": method,
                    "horizon": horizon,
                    "success": result["success"],
                    "initial_distance": result[
                        "initial_normalized_distance"
                    ],
                    "final_distance": result["final_normalized_distance"],
                    "runtime_seconds": result["wall_runtime_seconds"],
                    "simulator_calls": result["simulator_calls"],
                    "model_calls": result["model_rollout_calls"],
                    "output": str(output),
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "shard_complete",
                "mode": args.mode,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "completed": completed,
                "skipped": skipped,
                "tasks": len(tasks),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
