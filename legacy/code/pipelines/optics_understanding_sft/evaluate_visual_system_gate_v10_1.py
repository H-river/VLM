#!/usr/bin/env python3
"""Compose visual benchmarks and deterministic quantitative image tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-summary", type=Path, required=True)
    parser.add_argument("--withheld-summary", type=Path, required=True)
    parser.add_argument("--state-tool-summary", type=Path, required=True)
    parser.add_argument("--pair-tool-summary", type=Path, required=True)
    parser.add_argument("--seven-task-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expert = json.loads(args.expert_summary.read_text(encoding="utf-8"))
    withheld = json.loads(args.withheld_summary.read_text(encoding="utf-8"))
    state = json.loads(args.state_tool_summary.read_text(encoding="utf-8"))
    pair_tool = json.loads(args.pair_tool_summary.read_text(encoding="utf-8"))
    seven = json.loads(args.seven_task_summary.read_text(encoding="utf-8"))

    pair_prefix = "visual_pair_direction_extraction."
    pair_f1 = mean(
        [value for key, value in expert["field_macro_f1"].items() if key.startswith(pair_prefix)]
    )
    withheld_pair_f1 = mean(
        [
            value
            for key, value in withheld["field_macro_f1"].items()
            if key.startswith(pair_prefix)
        ]
    )
    state_f1 = float(state["equal_field_macro_f1"])
    pair_tool_f1 = float(pair_tool["equal_field_macro_f1"])
    production_field_f1 = (pair_tool_f1 * 5.0 + state_f1 * 4.0) / 9.0
    report = {
        "passed": True,
        "seven_task_system_macro": float(seven["equal_task_macro"]),
        "visual_routes": {
            "single_image_state_evidence": "deterministic_calibrated_image_tool",
            "paired_change_direction_evidence": "deterministic_calibrated_pair_image_tool",
            "visual_pair_pixel_use_benchmark": "qwen25vl_visual_expert_with_signed_difference",
            "exact_forward_and_counterfactual_numbers": "registered_deterministic_simulator_tools",
        },
        "metrics": {
            "state_tool_equal_field_macro_f1": state_f1,
            "state_tool_joint_exact": float(state["joint_exact_match"]),
            "pair_expert_equal_field_macro_f1": pair_f1,
            "pair_expert_joint_exact": float(
                expert["task_joint_exact_match"]["visual_pair_direction_extraction"]
            ),
            "pair_images_withheld_equal_field_macro_f1": withheld_pair_f1,
            "pair_pixel_contribution_macro_f1": pair_f1 - withheld_pair_f1,
            "pair_tool_equal_field_macro_f1": pair_tool_f1,
            "pair_tool_joint_exact": float(pair_tool["joint_exact_match"]),
            "production_visual_equal_field_macro_f1": production_field_f1,
        },
        "checks": {
            "seven_task_macro_at_least_0_90": float(seven["equal_task_macro"]) >= 0.90,
            "state_tool_macro_f1_at_least_0_55": state_f1 >= 0.55,
            "pair_expert_macro_f1_at_least_0_45": pair_f1 >= 0.45,
            "pair_joint_exact_at_least_0_40": float(
                expert["task_joint_exact_match"]["visual_pair_direction_extraction"]
            )
            >= 0.40,
            "pair_pixel_contribution_at_least_0_15": pair_f1 - withheld_pair_f1 >= 0.15,
            "pair_tool_macro_f1_at_least_0_55": pair_tool_f1 >= 0.55,
            "pair_tool_joint_exact_at_least_0_55": float(pair_tool["joint_exact_match"])
            >= 0.55,
            "production_visual_macro_f1_at_least_0_55": production_field_f1 >= 0.55,
        },
        "claim_boundary": (
            "The calibrated image tools are promoted for categorical state and change evidence. "
            "The visual expert is retained as evidence that corrected paired images affect predictions, "
            "not as the production quantitative route. Exact numerical optics results remain tool-computed; "
            "the result is not a claim of native pixel-level metrology or laboratory validity."
        ),
        "source_summaries": {
            "expert": str(args.expert_summary.resolve()),
            "images_withheld": str(args.withheld_summary.resolve()),
            "state_tool": str(args.state_tool_summary.resolve()),
            "pair_tool": str(args.pair_tool_summary.resolve()),
            "seven_task": str(args.seven_task_summary.resolve()),
        },
    }
    report["passed"] = all(report["checks"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
