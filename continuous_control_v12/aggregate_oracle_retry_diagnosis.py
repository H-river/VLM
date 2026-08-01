"""Aggregate the labelled, non-primary oracle CEM-budget retry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _load_directory(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(path.glob("*.json"))
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _episode_row(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": episode["case_id"],
        "stratum": episode["stratum"],
        "initial_distance_band": episode["initial_distance_band"],
        "horizon": episode["horizon"],
        "success": episode["success"],
        "initial_normalized_distance": episode[
            "initial_normalized_distance"
        ],
        "final_normalized_distance": episode["final_normalized_distance"],
        "normalized_distance_reduction": episode[
            "normalized_distance_reduction"
        ],
        "executed_steps": episode["executed_steps"],
        "planner_runtime_seconds": episode["planner_runtime_seconds"],
        "simulator_calls": episode["simulator_calls"],
        "termination_reason": episode["termination_reason"],
        "diagnostic_role": episode["diagnostic_role"],
        "primary_result_replaced": episode["primary_result_replaced"],
    }


def _summarize(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    distances = np.asarray(
        [row["final_normalized_distance"] for row in episodes],
        dtype=np.float64,
    )
    return {
        "episodes": len(episodes),
        "strict_successes": sum(bool(row["success"]) for row in episodes),
        "strict_success_rate": float(
            np.mean([bool(row["success"]) for row in episodes])
        ),
        "mean_final_normalized_distance": float(distances.mean()),
        "median_final_normalized_distance": float(np.median(distances)),
        "mean_simulator_calls": float(
            np.mean([row["simulator_calls"] for row in episodes])
        ),
        "mean_planner_runtime_seconds": float(
            np.mean([row["planner_runtime_seconds"] for row in episodes])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--retry-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    selected = list(map(str, selection["case_ids"]))
    expected = {(case_id, horizon) for case_id in selected for horizon in (1, 3)}

    retry = _load_directory(args.retry_dir)
    retry_keys = {
        (str(row["case_id"]), int(row["horizon"])) for row in retry
    }
    if retry_keys != expected or len(retry) != len(expected):
        raise ValueError(
            f"incomplete retry: got {len(retry)} unique={len(retry_keys)}, "
            f"expected={len(expected)}"
        )
    for row in retry:
        if row["method"] != "oracle":
            raise ValueError("oracle retry contains a learned-model episode")
        if row["diagnostic_role"] != (
            "post_primary_oracle_budget_diagnostic_not_primary_score"
        ):
            raise ValueError("oracle retry is missing its non-primary label")
        if row["primary_result_replaced"] is not False:
            raise ValueError("oracle retry claims to replace a primary result")

    primary_all = _load_directory(args.primary_dir)
    primary = {
        (str(row["case_id"]), int(row["horizon"])): row
        for row in primary_all
        if row["method"] == "oracle"
        and str(row["case_id"]) in set(selected)
    }
    if set(primary) != expected:
        raise ValueError("matching primary oracle episodes are incomplete")

    comparison_rows: list[dict[str, Any]] = []
    for current in sorted(
        retry, key=lambda row: (row["case_id"], row["horizon"])
    ):
        key = (str(current["case_id"]), int(current["horizon"]))
        before = primary[key]
        comparison_rows.append(
            {
                "case_id": key[0],
                "stratum": current["stratum"],
                "horizon": key[1],
                "primary_success": before["success"],
                "retry_success": current["success"],
                "rescued_strict_success": (
                    not bool(before["success"]) and bool(current["success"])
                ),
                "primary_final_normalized_distance": before[
                    "final_normalized_distance"
                ],
                "retry_final_normalized_distance": current[
                    "final_normalized_distance"
                ],
                "retry_minus_primary_final_distance": (
                    current["final_normalized_distance"]
                    - before["final_normalized_distance"]
                ),
                "primary_simulator_calls": before["simulator_calls"],
                "retry_simulator_calls": current["simulator_calls"],
            }
        )

    iteration: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for episode in retry:
        for step in episode["trace"]:
            for item in step["planner_iteration_history"]:
                iteration[(int(episode["horizon"]), int(item["iteration"]))].append(
                    item
                )
    convergence_rows = []
    for (horizon, iteration_index), records in sorted(iteration.items()):
        convergence_rows.append(
            {
                "horizon": horizon,
                "iteration": iteration_index,
                "records": len(records),
                "mean_best_score": float(
                    np.mean([record["best_score"] for record in records])
                ),
                "mean_best_terminal_distance": float(
                    np.mean(
                        [
                            record["best_terminal_distance"]
                            for record in records
                        ]
                    )
                ),
                "mean_best_first_step_distance": float(
                    np.mean(
                        [
                            record["best_first_step_distance"]
                            for record in records
                        ]
                    )
                ),
                "mean_elite_score": float(
                    np.mean(
                        [record["elite_score_mean"] for record in records]
                    )
                ),
            }
        )

    by_horizon: dict[str, Any] = {}
    for horizon in (1, 3):
        retry_h = [row for row in retry if int(row["horizon"]) == horizon]
        primary_h = [primary[(case_id, horizon)] for case_id in selected]
        comparisons_h = [
            row for row in comparison_rows if int(row["horizon"]) == horizon
        ]
        by_horizon[f"h{horizon}"] = {
            "primary_selected_cases": _summarize(primary_h),
            "retry": _summarize(retry_h),
            "strict_failures_rescued": sum(
                bool(row["rescued_strict_success"])
                for row in comparisons_h
            ),
            "mean_retry_minus_primary_final_distance": float(
                np.mean(
                    [
                        row["retry_minus_primary_final_distance"]
                        for row in comparisons_h
                    ]
                )
            ),
            "cases_with_lower_retry_final_distance": sum(
                row["retry_minus_primary_final_distance"] < 0
                for row in comparisons_h
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(
        args.output_dir / "episodes.csv",
        [_episode_row(row) for row in retry],
    )
    _write_csv(
        args.output_dir / "paired_primary_vs_retry.csv", comparison_rows
    )
    _write_csv(
        args.output_dir / "cem_convergence_by_iteration.csv",
        convergence_rows,
    )
    _write_json(
        args.output_dir / "summary.json",
        {
            "version": "v12_mpc_h1_h3_oracle_retry_aggregate_v1",
            "diagnostic_only_not_primary": True,
            "primary_result_replaced": False,
            "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "selected_cases": selected,
            "by_horizon": by_horizon,
        },
    )
    print(
        json.dumps(
            {
                "event": "oracle_retry_aggregate_complete",
                "episodes": len(retry),
                "output_dir": str(args.output_dir.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
