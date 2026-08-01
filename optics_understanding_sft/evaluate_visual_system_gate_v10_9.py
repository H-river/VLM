#!/usr/bin/env python3
"""Strict production gate for expanded full-sensor visual evidence v10.9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-summary", type=Path, required=True)
    parser.add_argument("--withheld-summary", type=Path, required=True)
    parser.add_argument("--state-tool-summary", type=Path, required=True)
    parser.add_argument("--pair-tool-summary", type=Path, required=True)
    parser.add_argument("--expanded-audit", type=Path, required=True)
    parser.add_argument("--seven-task-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expert = load(args.expert_summary)
    withheld = load(args.withheld_summary)
    state = load(args.state_tool_summary)
    pair = load(args.pair_tool_summary)
    audit = load(args.expanded_audit)
    seven = load(args.seven_task_summary)
    pair_prefix = "visual_pair_direction_extraction."
    expert_pair_f1 = mean(
        [value for key, value in expert["field_macro_f1"].items() if key.startswith(pair_prefix)]
    )
    withheld_pair_f1 = mean(
        [value for key, value in withheld["field_macro_f1"].items() if key.startswith(pair_prefix)]
    )
    state_f1 = float(state["equal_field_macro_f1"])
    pair_f1 = float(pair["equal_field_macro_f1"])
    state_macro_low = float(
        state["group_bootstrap"]["equal_field_macro_f1_95ci"]["low"]
    )
    state_joint_low = float(
        state["group_bootstrap"]["joint_exact_match_95ci"]["low"]
    )
    pair_macro_low = float(
        pair["group_bootstrap"]["equal_field_macro_f1_95ci"]["low"]
    )
    pair_joint_low = float(
        pair["group_bootstrap"]["joint_exact_match_95ci"]["low"]
    )
    production_f1 = (4.0 * state_f1 + 5.0 * pair_f1) / 9.0
    metrics = {
        "expanded_source_groups": int(audit["source_groups"]),
        "expanded_generated_records": int(audit["generated_records"]),
        "state_tool_equal_field_macro_f1": state_f1,
        "state_tool_joint_exact": float(state["joint_exact_match"]),
        "pair_tool_equal_field_macro_f1": pair_f1,
        "pair_tool_joint_exact": float(pair["joint_exact_match"]),
        "state_tool_macro_f1_95ci_low": state_macro_low,
        "state_tool_joint_exact_95ci_low": state_joint_low,
        "pair_tool_macro_f1_95ci_low": pair_macro_low,
        "pair_tool_joint_exact_95ci_low": pair_joint_low,
        "production_visual_equal_field_macro_f1": production_f1,
        "visual_expert_pixel_contribution_macro_f1": expert_pair_f1 - withheld_pair_f1,
        "pair_pixel_contribution_macro_f1": expert_pair_f1 - withheld_pair_f1,
        "seven_task_system_macro": float(seven["equal_task_macro"]),
    }
    checks = {
        "expanded_audit_passed": bool(audit["passed"]),
        "expanded_class_coverage_complete": bool(audit["class_coverage_complete"]),
        "expanded_source_groups_at_least_150": metrics["expanded_source_groups"] >= 150,
        "state_macro_f1_at_least_0_95": state_f1 >= 0.95,
        "state_joint_exact_at_least_0_90": metrics["state_tool_joint_exact"] >= 0.90,
        "pair_macro_f1_at_least_0_85": pair_f1 >= 0.85,
        "pair_joint_exact_at_least_0_90": metrics["pair_tool_joint_exact"] >= 0.90,
        "state_macro_f1_95ci_low_at_least_0_95": state_macro_low >= 0.95,
        "state_joint_exact_95ci_low_at_least_0_85": state_joint_low >= 0.85,
        "pair_macro_f1_95ci_low_at_least_0_80": pair_macro_low >= 0.80,
        "pair_joint_exact_95ci_low_at_least_0_85": pair_joint_low >= 0.85,
        "production_visual_macro_f1_at_least_0_90": production_f1 >= 0.90,
        "visual_pixel_contribution_at_least_0_15": metrics[
            "visual_expert_pixel_contribution_macro_f1"
        ]
        >= 0.15,
        "seven_task_macro_at_least_0_90": metrics["seven_task_system_macro"] >= 0.90,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "routes": {
            "single_image_state_evidence": "full_sensor_interpolated_overlay_threshold_tool",
            "paired_change_direction_evidence": "full_sensor_interpolated_overlay_top10_peak_tool",
            "visual_pixel_use_benchmark": "qwen25vl_pair_expert_with_signed_difference",
            "exact_numeric_optics": "registered_deterministic_simulator_tools",
        },
        "claim_boundary": (
            "Production categorical image measurements are deterministic and calibrated on synthetic "
            "train scenarios. The VLM selects tools, maps visible inputs, and interprets results. Exact "
            "numbers remain simulator-computed; no real-laboratory validity is claimed."
        ),
        "sources": {name: str(path.resolve()) for name, path in {
            "expert": args.expert_summary,
            "withheld": args.withheld_summary,
            "state_tool": args.state_tool_summary,
            "pair_tool": args.pair_tool_summary,
            "expanded_audit": args.expanded_audit,
            "seven_task": args.seven_task_summary,
        }.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
