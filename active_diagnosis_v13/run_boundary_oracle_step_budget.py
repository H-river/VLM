#!/usr/bin/env python3
"""Test longer control budgets on exact dev boundary oracle failures only."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.run_gate_a import (
    _atomic_json,
    _execute_control_episode,
    _load_config,
    _load_runtime,
)

VERSION = "active_diagnosis_v13_boundary_oracle_step_budget_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case_id"]), float(row["evaluator_only_true_gain"])


def _exact_failure_keys(
    direct_rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]]
) -> list[tuple[str, float]]:
    direct = {_key(row): row for row in direct_rows}
    oracle = {_key(row): row for row in oracle_rows}
    if set(direct) != set(oracle):
        raise ValueError("baseline direct/oracle arms are not matched")
    selected = [
        key
        for key, row in oracle.items()
        if str(row["stratum"]) == "reachable_boundary_or_clipping"
        and key[1] != 1.0
        and not bool(direct[key]["strict_success"])
        and not bool(row["strict_success"])
    ]
    if len(selected) != 20:
        raise ValueError(f"expected 20 exact failures, found {len(selected)}")
    return sorted(selected)


def _worker(
    *,
    budget: int,
    config_path: str,
    suite_path: str,
    selected_keys: list[tuple[str, float]],
    output_path: str,
) -> dict[str, Any]:
    config = _load_config(Path(config_path))
    config = copy.deepcopy(config)
    config["baseline"]["max_control_steps"] = int(budget)
    _, bounds, model = _load_runtime(config)
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    cases = {str(row["case_id"]): row for row in suite["cases"]}
    output = Path(output_path)
    complete = {_key(row) for row in _read_jsonl(output)}
    for index, key in enumerate(selected_keys, start=1):
        if key in complete:
            continue
        result = _execute_control_episode(
            case=cases[key[0]],
            true_gain=key[1],
            mode="oracle_known",
            config=config,
            bounds=bounds,
            model=model,
            probe_selection=None,
            classifier_bundle=None,
            split="development",
            policy_name=f"oracle_boundary_budget_{budget}",
        )
        _append_jsonl(output, result)
        print(
            json.dumps(
                {
                    "event": "boundary_budget_complete",
                    "budget": budget,
                    "task": index,
                    "tasks": len(selected_keys),
                    "case_id": key[0],
                    "gain": key[1],
                    "success": result["strict_success"],
                    "distance": result["final_normalized_distance"],
                }
            ),
            flush=True,
        )
    rows = _read_jsonl(output)
    if len(rows) != len(selected_keys) or {_key(row) for row in rows} != set(selected_keys):
        raise ValueError(f"budget {budget} output is incomplete")
    return {
        "budget": budget,
        "output": str(output.resolve()),
        "episodes": len(rows),
        "successes": sum(bool(row["strict_success"]) for row in rows),
    }


def _trace_projection(row: dict[str, Any], steps: int = 4) -> list[dict[str, Any]]:
    return [
        {
            "command_mm": step["command_mm"],
            "predicted_next_metrics": step["predicted_next_metrics"],
            "observed_next_metrics": step["observed_next_metrics"],
            "before_target_cost": step["before_target_cost"],
            "predicted_target_cost": step["predicted_target_cost"],
            "actual_target_cost": step["actual_target_cost"],
            "actual_step_audit": step["actual_step_audit"],
        }
        for step in row["trace"][:steps]
    ]


def _paired_bootstrap(
    baseline: dict[tuple[str, float], dict[str, Any]],
    treatment: dict[tuple[str, float], dict[str, Any]],
    seed: int,
    samples: int = 4000,
) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for key in sorted(baseline):
        grouped[str(baseline[key]["group_id"])].append(
            float(treatment[key]["strict_success"])
            - float(baseline[key]["strict_success"])
        )
    group_values = np.asarray(
        [float(np.mean(grouped[group])) for group in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(group_values[rng.integers(0, len(group_values), len(group_values))]))
        for _ in range(samples)
    ]
    low, high = np.quantile(draws, [0.025, 0.975])
    episode_estimate = float(
        np.mean(
            [
                float(treatment[key]["strict_success"])
                - float(baseline[key]["strict_success"])
                for key in baseline
            ]
        )
    )
    return {
        "episode_estimate": episode_estimate,
        "group_mean_estimate": float(np.mean(group_values)),
        "group_bootstrap_low": float(low),
        "group_bootstrap_high": float(high),
        "independent_groups": int(len(group_values)),
    }


def _summarize_arm(
    baseline: dict[tuple[str, float], dict[str, Any]],
    treatment: dict[tuple[str, float], dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    rows = [treatment[key] for key in sorted(treatment)]
    first_four_exact = [
        _trace_projection(treatment[key]) == _trace_projection(baseline[key])
        for key in baseline
    ]
    by_gain: dict[str, Any] = {}
    for gain in sorted({key[1] for key in treatment}):
        subset = [treatment[key] for key in treatment if key[1] == gain]
        by_gain[f"{gain:g}"] = {
            "episodes": len(subset),
            "successes": sum(bool(row["strict_success"]) for row in subset),
            "success_rate": float(np.mean([bool(row["strict_success"]) for row in subset])),
        }
    return {
        "episodes": len(rows),
        "successes": sum(bool(row["strict_success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["strict_success"]) for row in rows])),
        "success_difference_vs_budget4": _paired_bootstrap(baseline, treatment, seed),
        "first_four_trace_exact_replays": sum(first_four_exact),
        "first_four_trace_all_exact": all(first_four_exact),
        "mean_final_distance": float(np.mean([row["final_normalized_distance"] for row in rows])),
        "median_final_distance": float(np.median([row["final_normalized_distance"] for row in rows])),
        "saturation_episodes": sum(int(row["saturation_count"]) > 0 for row in rows),
        "mean_saturation_count": float(np.mean([row["saturation_count"] for row in rows])),
        "mean_control_steps": float(np.mean([row["control_steps"] for row in rows])),
        "by_true_gain": by_gain,
        "recovered_episode_ids": [
            f"{key[0]}__g{key[1]:g}"
            for key in sorted(treatment)
            if not bool(baseline[key]["strict_success"])
            and bool(treatment[key]["strict_success"])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--oracle-budget4", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] <= 4:
        raise ValueError("all treatment budgets must exceed four")
    direct_rows = _read_jsonl(args.direct.resolve())
    baseline_rows = _read_jsonl(args.oracle_budget4.resolve())
    selected_keys = _exact_failure_keys(direct_rows, baseline_rows)
    output_paths = {
        budget: args.output_dir / f"oracle_boundary_budget_{budget}.jsonl"
        for budget in budgets
    }
    worker_results = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(budgets))) as pool:
        futures = [
            pool.submit(
                _worker,
                budget=budget,
                config_path=str(args.config.resolve()),
                suite_path=str(args.suite.resolve()),
                selected_keys=selected_keys,
                output_path=str(output_paths[budget].resolve()),
            )
            for budget in budgets
        ]
        for future in as_completed(futures):
            worker_results.append(future.result())
    baseline_all = {_key(row): row for row in baseline_rows}
    baseline = {key: baseline_all[key] for key in selected_keys}
    arms = {}
    for budget in budgets:
        rows = {_key(row): row for row in _read_jsonl(output_paths[budget])}
        if set(rows) != set(selected_keys):
            raise ValueError(f"budget {budget} keys are incomplete")
        arms[str(budget)] = _summarize_arm(
            baseline, rows, seed=2026080100 + budget
        )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "selection": "exact 20 nonnominal boundary episodes failed by both direct and oracle-known budget-4 control",
        "baseline_budget": 4,
        "treatment_budgets": budgets,
        "exact_episode_ids": [f"{key[0]}__g{key[1]:g}" for key in selected_keys],
        "worker_results": sorted(worker_results, key=lambda row: row["budget"]),
        "arms": arms,
        "interpretation_guard": (
            "Post-freeze development-only causal ablation. It diagnoses the control horizon "
            "and cannot replace the frozen protected policy without a new preregistered study."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report["arms"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
