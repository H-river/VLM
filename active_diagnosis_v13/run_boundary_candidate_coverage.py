#!/usr/bin/env python3
"""Simulator-score nested boundary-conditioned action-sequence proposals on dev."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.faults import (
    command_for_desired_physical_delta,
    effective_planning_bounds,
    realize_hidden_gain_step,
)
from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    Bounds,
    action_dict,
    metrics_vector,
    normalized_distance,
    position_dict,
    position_vector,
    stable_seed,
)
from continuous_control_v12.simulator import simulate_state

VERSION = "active_diagnosis_v13_boundary_candidate_coverage_v2"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.resolve().read_text().splitlines() if line]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _exact_failure_keys(
    direct_rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]]
) -> list[tuple[str, float]]:
    direct = {_key(row): row for row in direct_rows}
    oracle = {_key(row): row for row in oracle_rows}
    selected = [
        key
        for key, row in oracle.items()
        if str(row["stratum"]) == "reachable_boundary_or_clipping"
        and key[1] != 1.0
        and not bool(direct[key]["strict_success"])
        and not bool(row["strict_success"])
    ]
    if len(selected) != 20:
        raise ValueError(f"expected 20 exact boundary failures, found {len(selected)}")
    return sorted(selected)


def _condition_uniform(
    uniform: np.ndarray,
    position: np.ndarray,
    physical_bounds: Bounds,
) -> tuple[np.ndarray, list[str]]:
    transformed = uniform.copy()
    conditioned_axes = []
    lower_steps = (position - physical_bounds.position_low) / physical_bounds.action_high
    upper_steps = (physical_bounds.position_high - position) / physical_bounds.action_high
    for axis in range(4):
        if lower_steps[axis] < 2.0:
            transformed[axis] = np.sqrt(transformed[axis])
            conditioned_axes.append(f"{ACTION_FIELDS[axis]}:away_from_lower")
        elif upper_steps[axis] < 2.0:
            transformed[axis] = 1.0 - np.sqrt(1.0 - transformed[axis])
            conditioned_axes.append(f"{ACTION_FIELDS[axis]}:away_from_upper")
    return transformed, conditioned_axes


def _evaluate_sequence(
    *,
    raw_uniforms: np.ndarray,
    proposal: str,
    case: dict[str, Any],
    true_gain: float,
    start_true_position: np.ndarray,
    start_command_position: np.ndarray,
    start_metrics: np.ndarray,
    initial_metrics: np.ndarray,
    target: np.ndarray,
    bounds: Bounds,
    base_config_path: str,
) -> dict[str, Any]:
    physical_bounds = effective_planning_bounds(bounds, true_gain)
    true_position = start_true_position.copy()
    command_position = start_command_position.copy()
    metrics = start_metrics.copy()
    desired_actions = []
    issued_commands = []
    conditioned_axes = []
    saturation_count = 0
    for step in range(len(raw_uniforms)):
        uniform = raw_uniforms[step]
        if proposal == "boundary_conditioned":
            uniform, step_conditioned = _condition_uniform(
                uniform, true_position, physical_bounds
            )
            conditioned_axes.extend(step_conditioned)
        elif proposal != "uniform_feasible":
            raise ValueError(f"unknown proposal: {proposal}")
        feasible_low = np.maximum(
            physical_bounds.action_low,
            physical_bounds.position_low - true_position,
        )
        feasible_high = np.minimum(
            physical_bounds.action_high,
            physical_bounds.position_high - true_position,
        )
        desired = feasible_low + (feasible_high - feasible_low) * uniform
        command = command_for_desired_physical_delta(
            desired, true_gain, command_position, bounds
        )
        realized = realize_hidden_gain_step(
            true_position=true_position,
            commanded_position_belief=command_position,
            requested_command=command,
            true_gain=true_gain,
            bounds=bounds,
        )
        capture = simulate_state(
            case["setup_context"],
            position_dict(realized.next_true_position),
            case["simulator_fixed"],
            base_config_path,
            bounds,
        )
        metrics = metrics_vector(capture["metrics"])
        true_position = realized.next_true_position
        command_position = realized.next_commanded_position_belief
        saturation_count += int(
            realized.step_saturated or realized.absolute_position_saturated
        )
        desired_actions.append(action_dict(desired))
        issued_commands.append(action_dict(command))
    return {
        "desired_physical_sequence": desired_actions,
        "issued_command_sequence": issued_commands,
        "terminal_normalized_distance": float(
            normalized_distance(metrics, target, initial_metrics)
        ),
        "strict_success": bool(normalized_distance(metrics, target, initial_metrics) <= 1.0),
        "saturation_count": saturation_count,
        "conditioned_axes": sorted(set(conditioned_axes)),
    }


def _worker(
    *,
    key: tuple[str, float],
    case: dict[str, Any],
    oracle_row: dict[str, Any],
    v12_config: dict[str, Any],
    base_config_path: str,
    root_seed: int,
    maximum_budget: int,
    sequence_horizon: int,
) -> dict[str, Any]:
    bounds = Bounds.from_config(v12_config)
    start_true_position = position_vector(oracle_row["evaluator_only_final_true_position_mm"])
    start_command_position = position_vector(oracle_row["final_commanded_position_belief_mm"])
    start_metrics = metrics_vector(oracle_row["final_metrics"])
    initial_metrics = metrics_vector(case["initial_metrics"])
    target = metrics_vector(case["target_metrics"])
    rng = np.random.default_rng(
        stable_seed(root_seed, key[0], f"{key[1]:g}", "boundary_candidate_coverage")
    )
    raw = rng.uniform(size=(maximum_budget, sequence_horizon, 4))
    proposal_rows = {}
    for proposal in ("uniform_feasible", "boundary_conditioned"):
        candidates = [
            _evaluate_sequence(
                raw_uniforms=raw[index],
                proposal=proposal,
                case=case,
                true_gain=key[1],
                start_true_position=start_true_position,
                start_command_position=start_command_position,
                start_metrics=start_metrics,
                initial_metrics=initial_metrics,
                target=target,
                bounds=bounds,
                base_config_path=base_config_path,
            )
            for index in range(maximum_budget)
        ]
        first_actions = np.asarray(
            [
                [candidate["desired_physical_sequence"][0][field] for field in ACTION_FIELDS]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        normalized = first_actions / effective_planning_bounds(
            bounds, key[1]
        ).action_high[None, :]
        distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
        proposal_rows[proposal] = {
            "candidates": candidates,
            "mean_first_action_pairwise_normalized_distance": float(
                distances[np.triu_indices(maximum_budget, k=1)].mean()
            ),
            "conditioning_active_candidates": sum(
                bool(candidate["conditioned_axes"]) for candidate in candidates
            ),
        }
    return {
        "version": VERSION,
        "record_id": f"{key[0]}__g{key[1]:g}__boundary_candidate_coverage",
        "case_id": key[0],
        "group_id": str(case["group_id"]),
        "stratum": str(case["stratum"]),
        "regime": str(case["regime"]),
        "evaluator_only_true_gain": key[1],
        "source_budget4_final_distance": float(oracle_row["final_normalized_distance"]),
        "source_budget4_strict_success": bool(oracle_row["strict_success"]),
        "sequence_horizon": sequence_horizon,
        "maximum_budget": maximum_budget,
        "planner_root_seed": root_seed,
        "proposal_raw_uniforms_matched": True,
        "proposals": proposal_rows,
    }


def _aggregate(
    rows: list[dict[str, Any]], budgets: list[int]
) -> dict[str, Any]:
    proposals = {}
    for proposal in ("uniform_feasible", "boundary_conditioned"):
        curve = []
        for budget in budgets:
            best_distances = [
                min(
                    float(candidate["terminal_normalized_distance"])
                    for candidate in row["proposals"][proposal]["candidates"][:budget]
                )
                for row in rows
            ]
            curve.append(
                {
                    "candidate_budget": budget,
                    "episodes": len(rows),
                    "simulator_best_of_k_strict_success": float(
                        np.mean(np.asarray(best_distances) <= 1.0)
                    ),
                    "simulator_best_of_k_successes": int(
                        np.sum(np.asarray(best_distances) <= 1.0)
                    ),
                    "mean_best_terminal_distance": float(np.mean(best_distances)),
                    "median_best_terminal_distance": float(np.median(best_distances)),
                }
            )
        proposals[proposal] = {
            "coverage_curve": curve,
            "mean_first_action_pairwise_normalized_distance": float(
                np.mean(
                    [
                        row["proposals"][proposal][
                            "mean_first_action_pairwise_normalized_distance"
                        ]
                        for row in rows
                    ]
                )
            ),
            "mean_saturation_events_per_candidate": float(
                np.mean(
                    [
                        candidate["saturation_count"]
                        for row in rows
                        for candidate in row["proposals"][proposal]["candidates"]
                    ]
                )
            ),
        }
    maximum = budgets[-1]
    uniform_best = {
        _key(row): min(
            float(candidate["terminal_normalized_distance"])
            for candidate in row["proposals"]["uniform_feasible"]["candidates"][:maximum]
        )
        for row in rows
    }
    conditioned_best = {
        _key(row): min(
            float(candidate["terminal_normalized_distance"])
            for candidate in row["proposals"]["boundary_conditioned"]["candidates"][:maximum]
        )
        for row in rows
    }
    differences = np.asarray(
        [conditioned_best[key] - uniform_best[key] for key in sorted(uniform_best)],
        dtype=np.float64,
    )
    return {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "selection": "exact 20 nonnominal boundary failures under both direct and budget-four oracle-known control",
        "episodes": len(rows),
        "budgets": budgets,
        "proposals": proposals,
        "boundary_conditioned_minus_uniform_best_distance_at_max_budget": {
            "mean": float(np.mean(differences)),
            "median": float(np.median(differences)),
            "conditioned_better_episodes": int(np.sum(differences < -1e-12)),
            "uniform_better_episodes": int(np.sum(differences > 1e-12)),
            "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        },
        "interpretation_guard": (
            "This is an evaluator-only simulator best-of-budget coverage bound on selected "
            "development failures. It is not a policy result and cannot use protected data."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v13-config", type=Path, required=True)
    parser.add_argument("--v12-config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--oracle-budget4", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 24, 48, 96, 192])
    parser.add_argument("--sequence-horizon", type=int, default=2)
    parser.add_argument("--root-seed-override", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = sorted(set(args.budgets))
    if budgets[0] <= 0 or args.sequence_horizon <= 0:
        raise ValueError("budgets and sequence horizon must be positive")
    v13_config = _read_json(args.v13_config)
    root_seed = (
        int(v13_config["root_seed"])
        if args.root_seed_override is None
        else int(args.root_seed_override)
    )
    v12_config = _read_json(args.v12_config)
    suite = _read_json(args.suite)
    cases = {str(row["case_id"]): row for row in suite["cases"]}
    direct_rows = _read_jsonl(args.direct)
    oracle_rows = _read_jsonl(args.oracle_budget4)
    oracle = {_key(row): row for row in oracle_rows}
    selected = _exact_failure_keys(direct_rows, oracle_rows)
    completed_rows = _read_jsonl(args.records)
    completed = {_key(row) for row in completed_rows}
    pending = [key for key in selected if key not in completed]
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                _worker,
                key=key,
                case=cases[key[0]],
                oracle_row=oracle[key],
                v12_config=v12_config,
                base_config_path=str(
                    Path(str(v13_config["baseline"]["base_simulator_config"])).resolve()
                ),
                root_seed=root_seed,
                maximum_budget=budgets[-1],
                sequence_horizon=int(args.sequence_horizon),
            )
            for key in pending
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            _append_jsonl(args.records, row)
            print(
                json.dumps(
                    {
                        "event": "boundary_candidate_case_complete",
                        "completed_this_run": index,
                        "pending_this_run": len(pending),
                        "record_id": row["record_id"],
                    }
                ),
                flush=True,
            )
    rows = _read_jsonl(args.records)
    if len(rows) != len(selected) or {_key(row) for row in rows} != set(selected):
        raise ValueError("candidate-coverage records are incomplete or duplicated")
    report = _aggregate(rows, budgets)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
