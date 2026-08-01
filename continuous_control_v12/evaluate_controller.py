#!/usr/bin/env python3
"""Aggregate saved v12 controller episodes with group-level bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from continuous_control_v12.evaluation import (
    controller_metrics,
    group_bootstrap_ci,
)
from continuous_control_v12.schema import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026072903)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    loaded = read_jsonl(args.episodes.resolve())
    episodes = []
    matched: dict[str, dict[str, dict]] = {}
    for row in loaded:
        episode = dict(row.get("episode", row))
        episode.setdefault("group_id", row.get("group_id"))
        episode.setdefault("target_category", row.get("target_category"))
        episode.setdefault("regime", row.get("regime", "unspecified"))
        episode["mode"] = row.get("mode", episode.get("mode", "unspecified"))
        episode["target_id"] = row.get(
            "target_id", episode.get("target_id", "unspecified")
        )
        episodes.append(episode)
        matched.setdefault(str(episode["target_id"]), {})[
            str(episode["mode"])
        ] = episode
    for pair in matched.values():
        if "oracle" in pair and "learned" in pair:
            oracle_distance = float(
                pair["oracle"]["final_normalized_distance"]
            )
            pair["oracle"]["oracle_regret"] = 0.0
            pair["learned"]["oracle_regret"] = float(
                pair["learned"]["final_normalized_distance"]
                - oracle_distance
            )
    result = controller_metrics(episodes)
    result["strict_success_group_bootstrap_95"] = group_bootstrap_ci(
        [float(bool(row["success"])) for row in episodes],
        [str(row["group_id"]) for row in episodes],
        seed=int(args.seed),
        samples=1000,
    )
    comparison_rows = []
    for target_id, pair in sorted(matched.items()):
        if "oracle" not in pair or "learned" not in pair:
            continue
        oracle = pair["oracle"]
        learned = pair["learned"]
        learned_trace = list(learned.get("trace", []))
        predicted_steps = sum(
            float(step.get("predicted_improvement", 0.0)) > 0.0
            for step in learned_trace
        )
        exploitation_events = sum(
            bool(step.get("planner_exploitation_event", False))
            for step in learned_trace
        )
        comparison_rows.append(
            {
                "target_id": target_id,
                "group_id": learned["group_id"],
                "target_category": learned["target_category"],
                "regime": learned["regime"],
                "oracle_success": bool(oracle["success"]),
                "learned_success": bool(learned["success"]),
                "initial_normalized_distance": float(
                    learned["initial_normalized_distance"]
                ),
                "oracle_final_normalized_distance": float(
                    oracle["final_normalized_distance"]
                ),
                "learned_final_normalized_distance": float(
                    learned["final_normalized_distance"]
                ),
                "oracle_regret": float(learned["oracle_regret"]),
                "oracle_steps": int(oracle["steps"]),
                "learned_steps": int(learned["steps"]),
                "learned_cumulative_motion_mm": float(
                    learned["cumulative_actuator_movement_mm"]
                ),
                "predicted_improvement_steps": int(predicted_steps),
                "planner_exploitation_events": int(exploitation_events),
                "planner_exploitation_rate": (
                    None
                    if not predicted_steps
                    else float(exploitation_events / predicted_steps)
                ),
                "illegal_actions": int(learned.get("illegal_actions", 0)),
                "mean_predicted_vs_actual_distance_gap": (
                    None
                    if not learned_trace
                    else float(
                        np.mean(
                            [
                                float(
                                    step.get(
                                        "predicted_versus_actual_distance_gap",
                                        0.0,
                                    )
                                )
                                for step in learned_trace
                            ]
                        )
                    )
                ),
                "mean_uncertainty": (
                    None
                    if not learned_trace
                    else float(
                        np.mean(
                            [
                                np.mean(step["planner"]["uncertainty"])
                                for step in learned_trace
                            ]
                        )
                    )
                ),
            }
        )
    total_predicted = sum(
        row["predicted_improvement_steps"] for row in comparison_rows
    )
    total_exploitation = sum(
        row["planner_exploitation_events"] for row in comparison_rows
    )
    result["matched_oracle_learned"] = {
        "pairs": len(comparison_rows),
        "planner_predicted_improvement_steps": total_predicted,
        "planner_exploitation_events": total_exploitation,
        "planner_exploitation_rate": (
            None
            if not total_predicted
            else float(total_exploitation / total_predicted)
        ),
        "rows": comparison_rows,
    }
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise RuntimeError(
                f"refusing to overwrite controller evaluation: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        table_path = output.with_suffix(".csv")
        if comparison_rows:
            with table_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(comparison_rows[0])
                )
                writer.writeheader()
                writer.writerows(comparison_rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
