#!/usr/bin/env python3
"""Compose direct and deterministic-tool evaluations into one seven-task system gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = (
    "setup_interpretation",
    "information_sufficiency",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "constrained_intervention",
    "counterfactual_reasoning",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-summary", type=Path, required=True)
    parser.add_argument("--compact-summary", type=Path, required=True)
    parser.add_argument("--forward-summary", type=Path, required=True)
    parser.add_argument("--counterfactual-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct = load(args.direct_summary)
    compact = load(args.compact_summary)
    forward = load(args.forward_summary)
    counterfactual = load(args.counterfactual_summary)
    direct_scores = {
        task: float(direct["per_task"][task]["task_score"])
        for task in TASKS
    }
    compact_score = float(compact["end_to_end_exact_match"])
    task_scores = {
        "setup_interpretation": direct_scores["setup_interpretation"],
        "information_sufficiency": compact_score,
        "causal_effects": direct_scores["causal_effects"],
        "forward_prediction": float(forward["end_to_end_exact_match"]),
        "diagnosis": direct_scores["diagnosis"],
        "constrained_intervention": compact_score,
        "counterfactual_reasoning": float(counterfactual["end_to_end_exact_match"]),
    }
    routes = {
        "setup_interpretation": "direct_structured_answer",
        "information_sufficiency": "enumerate_compatible_completions_then_compact_interpretation",
        "causal_effects": "direct_structured_answer",
        "forward_prediction": "registered_state_deterministic_simulator",
        "diagnosis": "direct_structured_answer",
        "constrained_intervention": "exhaustive_grid_tool_then_compact_interpretation",
        "counterfactual_reasoning": "registered_paired_state_deterministic_simulator",
    }
    macro = sum(task_scores.values()) / len(TASKS)
    direct_macro = float(direct["macro_task_score"])
    checks = {
        "all_task_scores_at_least_0_90": min(task_scores.values()) >= 0.90,
        "equal_task_macro_at_least_0_90": macro >= 0.90,
        "forward_exact_at_least_0_98": task_scores["forward_prediction"] >= 0.98,
        "counterfactual_exact_at_least_0_98": task_scores["counterfactual_reasoning"] >= 0.98,
        "compact_tool_tasks_exact_at_least_0_98": min(
            task_scores["information_sufficiency"], task_scores["constrained_intervention"]
        ) >= 0.98,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "equal_task_macro": macro,
        "direct_only_macro": direct_macro,
        "absolute_macro_gain_over_direct_only": macro - direct_macro,
        "task_scores": task_scores,
        "direct_only_task_scores": direct_scores,
        "routes": routes,
        "claim_boundary": (
            "Scores routed through deterministic tools measure the LLM's tool choice, source-path "
            "construction, and result interpretation; they do not claim native pixel-level metrology."
        ),
        "source_summaries": {
            "direct": str(args.direct_summary.resolve()),
            "compact": str(args.compact_summary.resolve()),
            "forward": str(args.forward_summary.resolve()),
            "counterfactual": str(args.counterfactual_summary.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Seven-task optics system gate",
        "",
        f"Passed: **{report['passed']}**",
        f"Equal-task macro: **{macro:.3f}** (direct-only reference: {direct_macro:.3f})",
        "",
        "| Task | Route | Score | Direct-only reference |",
        "|---|---|---:|---:|",
    ]
    for task in TASKS:
        lines.append(f"| {task} | {routes[task]} | {task_scores[task]:.3f} | {direct_scores[task]:.3f} |")
    lines += ["", report["claim_boundary"], ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
