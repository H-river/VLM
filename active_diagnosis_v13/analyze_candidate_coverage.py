#!/usr/bin/env python3
"""Audit existing learned-H1 predicted-top candidate rollouts on dev groups."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import ACTION_FIELDS, OUTPUT_FIELDS

VERSION = "active_diagnosis_v13_existing_candidate_coverage_v1"


def _case_index(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[1])


def _action(candidate: Mapping[str, Any]) -> np.ndarray:
    action = candidate["effective_sequence"][0]
    return np.asarray([float(action[field]) for field in ACTION_FIELDS])


def _mean_pairwise_distance(candidates: Sequence[Mapping[str, Any]]) -> float:
    if len(candidates) < 2:
        return 0.0
    scale = np.asarray([0.05, 0.05, 0.02, 0.02], dtype=np.float64)
    actions = np.stack([_action(row) / scale for row in candidates])
    distances = [
        float(np.linalg.norm(actions[left] - actions[right]))
        for left in range(len(actions))
        for right in range(left + 1, len(actions))
    ]
    return float(np.mean(distances))


def summarize_decisions(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not decisions:
        raise ValueError("no candidate decisions")
    maximum_k = max(len(row["candidates"]) for row in decisions)
    curves = []
    for k in range(1, maximum_k + 1):
        costs = [
            min(float(candidate["actual_terminal_cost"]) for candidate in row["candidates"][:k])
            for row in decisions
            if len(row["candidates"]) >= k
        ]
        curves.append(
            {
                "k": k,
                "decisions": len(costs),
                "strict_success_coverage": float(np.mean(np.asarray(costs) <= 1.0)),
                "mean_best_actual_cost": float(np.mean(costs)),
                "median_best_actual_cost": float(np.median(costs)),
            }
        )
    regrets = []
    hard_negatives = 0
    top1_success = 0
    best_success = 0
    diversity = []
    residual_by_field: defaultdict[str, list[float]] = defaultdict(list)
    for decision in decisions:
        candidates = list(decision["candidates"])
        top1 = float(candidates[0]["actual_terminal_cost"])
        best = min(float(row["actual_terminal_cost"]) for row in candidates)
        regrets.append(top1 - best)
        top1_success += int(top1 <= 1.0)
        best_success += int(best <= 1.0)
        hard_negatives += int(top1 > 1.0 and best <= 1.0)
        diversity.append(_mean_pairwise_distance(candidates))
        if top1 > 1.0 and best < top1:
            for candidate in candidates:
                if float(candidate["actual_terminal_cost"]) != best:
                    continue
                depth = candidate["counterfactual_depths"][0]
                values = depth["tolerance_normalized_prediction_residual"]
                for field in OUTPUT_FIELDS:
                    residual_by_field[field].append(abs(float(values[field])))
                break
    count = len(decisions)
    return {
        "decisions": count,
        "h1_cem_top1_strict_success": top1_success / count,
        "simulator_oracle_best_of_k_strict_success": best_success / count,
        "best_of_k_success_gain": (best_success - top1_success) / count,
        "matched_hard_negative_recoveries": hard_negatives,
        "mean_candidate_ranking_regret": float(np.mean(regrets)),
        "median_candidate_ranking_regret": float(np.median(regrets)),
        "predicted_best_but_true_failed_hard_negative_frequency": hard_negatives / count,
        "mean_normalized_pairwise_action_diversity": float(np.mean(diversity)),
        "ranking_failure_best_candidate_mean_abs_prediction_error": {
            field: (None if not values else float(np.mean(values)))
            for field, values in sorted(residual_by_field.items())
        },
        "top_k_coverage_curve": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "protected"), default="development"
    )
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.split == "protected":
        if args.freeze is None or not args.freeze.exists():
            raise ValueError("protected candidate audit requires frozen decision")
        frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
        if not bool(frozen.get("frozen", False)):
            raise ValueError("protected candidate audit requires frozen=true")
    episodes = []
    for path in sorted(args.episode_dir.resolve().glob("*__learned_h1.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        index = _case_index(str(episode["case_id"]))
        belongs = index <= 9 if args.split == "development" else index >= 10
        if belongs:
            episodes.append(episode)
    decisions = []
    for episode in episodes:
        grouped: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for candidate in episode["candidate_ranking_audits"]:
            if candidate["selection"] == "predicted_top":
                grouped[int(candidate["episode_step"])].append(candidate)
        for episode_step, candidates in sorted(grouped.items()):
            candidates.sort(key=lambda row: int(row["selection_rank"]))
            decisions.append(
                {
                    "case_id": episode["case_id"],
                    "group_id": episode["group_id"],
                    "episode_step": episode_step,
                    "candidates": candidates,
                }
            )
    all_steps = summarize_decisions(decisions)
    first_steps = summarize_decisions(
        [row for row in decisions if int(row["episode_step"]) == 0]
    )
    report = {
        "version": VERSION,
        "split": args.split,
        "protected_set_used_for_selection": args.split == "protected",
        "source_role": "preexisting_exact_simulator_candidate_audits_not_hidden_gain_gate_a",
        "episodes": len(episodes),
        "all_closed_loop_decisions": all_steps,
        "initial_decisions_only": first_steps,
        "limitations": [
            "The audit covers the ten predicted-top candidates retained by the prior H1 run, not every final CEM population member.",
            "Best-of-k is a one-decision simulator upper bound; it is not a reconstructed counterfactual closed loop.",
            "These candidates use nominal gain and are supporting fallback evidence, not a substitute for v13 fault evaluation."
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f"candidate_coverage.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "candidate_coverage.json")
    with (output_dir / "top_k_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "scope",
                "k",
                "decisions",
                "strict_success_coverage",
                "mean_best_actual_cost",
                "median_best_actual_cost",
            ),
        )
        writer.writeheader()
        for scope, summary in (("all_steps", all_steps), ("initial_only", first_steps)):
            for row in summary["top_k_coverage_curve"]:
                writer.writerow({"scope": scope, **row})
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
