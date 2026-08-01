#!/usr/bin/env python3
"""Compose seven-task, visual measurement, and visual-tool orchestration gates."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-summary", type=Path, required=True)
    parser.add_argument("--direct-tool-summary", type=Path)
    parser.add_argument("--direct-tool-confirmation-summary", type=Path)
    parser.add_argument("--compact-summary", type=Path, required=True)
    parser.add_argument("--forward-summary", type=Path, required=True)
    parser.add_argument("--counterfactual-summary", type=Path, required=True)
    parser.add_argument("--visual-summary", type=Path, required=True)
    parser.add_argument("--orchestration-summary", type=Path, required=True)
    parser.add_argument("--robustness-summary", type=Path)
    parser.add_argument("--robust-orchestration-summary", type=Path)
    parser.add_argument("--saturation-orchestration-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    direct = load(args.direct_summary)
    direct_tool = load(args.direct_tool_summary) if args.direct_tool_summary else None
    direct_tool_confirmation = (
        load(args.direct_tool_confirmation_summary)
        if args.direct_tool_confirmation_summary
        else None
    )
    if direct_tool_confirmation is not None and direct_tool is None:
        parser.error("--direct-tool-confirmation-summary requires --direct-tool-summary")
    compact = load(args.compact_summary)
    forward = load(args.forward_summary)
    counterfactual = load(args.counterfactual_summary)
    visual = load(args.visual_summary)
    orchestration = load(args.orchestration_summary)
    robustness = load(args.robustness_summary) if args.robustness_summary else None
    robust_orchestration = (
        load(args.robust_orchestration_summary)
        if args.robust_orchestration_summary
        else None
    )
    saturation_orchestration = (
        load(args.saturation_orchestration_summary)
        if args.saturation_orchestration_summary
        else None
    )

    task_scores = {
        "setup_interpretation": float(direct["per_task"]["setup_interpretation"]["task_score"]),
        "information_sufficiency": float(compact["end_to_end_exact_match"]),
        "causal_effects": float(direct["per_task"]["causal_effects"]["task_score"]),
        "forward_prediction": float(forward["end_to_end_exact_match"]),
        "diagnosis": float(direct["per_task"]["diagnosis"]["task_score"]),
        "constrained_intervention": float(compact["end_to_end_exact_match"]),
        "counterfactual_reasoning": float(counterfactual["end_to_end_exact_match"]),
    }
    if direct_tool is not None:
        for task in ("setup_interpretation", "causal_effects", "diagnosis"):
            development_score = float(
                direct_tool["by_source_task"][task]["end_to_end_exact_match"]
            )
            confirmation_score = (
                float(
                    direct_tool_confirmation["by_source_task"][task][
                        "end_to_end_exact_match"
                    ]
                )
                if direct_tool_confirmation is not None
                else development_score
            )
            task_scores[task] = min(development_score, confirmation_score)
    seven_macro = sum(task_scores.values()) / len(task_scores)
    metrics = {
        "seven_task_equal_macro": seven_macro,
        "seven_task_scores": task_scores,
        "visual_production_equal_field_macro_f1": float(
            visual["metrics"]["production_visual_equal_field_macro_f1"]
        ),
        "visual_state_tool_joint_exact": float(visual["metrics"]["state_tool_joint_exact"]),
        "visual_pair_tool_joint_exact": float(visual["metrics"]["pair_tool_joint_exact"]),
        "visual_expert_pixel_contribution_macro_f1": float(
            visual["metrics"]["pair_pixel_contribution_macro_f1"]
        ),
        "visual_tool_orchestration_end_to_end_exact": float(
            orchestration["end_to_end_exact_match"]
        ),
    }
    if direct_tool is not None:
        metrics["direct_tool_orchestration_end_to_end_exact"] = float(
            direct_tool["end_to_end_exact_match"]
        )
    if direct_tool_confirmation is not None:
        metrics["direct_tool_confirmation_end_to_end_exact"] = float(
            direct_tool_confirmation["end_to_end_exact_match"]
        )
        metrics["direct_tool_confirmation_group_wilson_95ci_low"] = float(
            direct_tool_confirmation["physical_group_end_to_end_wilson_95ci"]["low"]
        )
    checks = {
        "all_seven_tasks_at_least_0_90": min(task_scores.values()) >= 0.90,
        "seven_task_macro_at_least_0_95": seven_macro >= 0.95,
        "visual_measurement_gate_passed": bool(visual["passed"]),
        "visual_production_macro_f1_at_least_0_55": metrics[
            "visual_production_equal_field_macro_f1"
        ]
        >= 0.55,
        "visual_orchestration_exact_at_least_0_95": metrics[
            "visual_tool_orchestration_end_to_end_exact"
        ]
        >= 0.95,
    }
    if direct_tool is not None:
        checks["direct_tool_orchestration_at_least_0_95"] = metrics[
            "direct_tool_orchestration_end_to_end_exact"
        ] >= 0.95
    if direct_tool_confirmation is not None:
        checks["direct_tool_confirmation_at_least_0_95"] = metrics[
            "direct_tool_confirmation_end_to_end_exact"
        ] >= 0.95
        checks["direct_tool_confirmation_group_wilson_low_at_least_0_95"] = metrics[
            "direct_tool_confirmation_group_wilson_95ci_low"
        ] >= 0.95
    if robustness is not None:
        metrics["visual_robustness_average_macro_f1"] = float(
            robustness["average_production_visual_macro_f1"]
        )
        metrics["visual_robustness_worst_condition_macro_f1"] = float(
            robustness["worst_condition_production_visual_macro_f1"]
        )
        if "worst_condition_conservative_production_macro_f1_95ci_low" in robustness:
            metrics["visual_robustness_worst_condition_conservative_95ci_low"] = float(
                robustness[
                    "worst_condition_conservative_production_macro_f1_95ci_low"
                ]
            )
        checks["visual_robustness_gate_passed"] = bool(robustness["passed"])
        checks["visual_robustness_worst_condition_at_least_0_80"] = metrics[
            "visual_robustness_worst_condition_macro_f1"
        ] >= 0.80
    if robust_orchestration is not None:
        metrics["visual_robust_orchestration_end_to_end_exact"] = float(
            robust_orchestration["end_to_end_exact_match"]
        )
        checks["visual_robust_orchestration_at_least_0_95"] = metrics[
            "visual_robust_orchestration_end_to_end_exact"
        ] >= 0.95
    if saturation_orchestration is not None:
        metrics["saturation_orchestration_raw_source_mapping_exact"] = float(
            saturation_orchestration["source_mapping_exact_match"]
        )
        metrics["saturation_orchestration_validated_end_to_end_exact"] = float(
            saturation_orchestration[
                "validated_production_end_to_end_exact_match"
            ]
        )
        checks["saturation_validated_orchestration_at_least_0_95"] = metrics[
            "saturation_orchestration_validated_end_to_end_exact"
        ] >= 0.95
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "architecture": {
            "direct_language_routes": (
                []
                if direct_tool is not None
                else ["setup_interpretation", "causal_effects", "diagnosis"]
            ),
            "deterministic_direct_reasoning_routes": (
                ["setup_interpretation", "causal_effects", "diagnosis"]
                if direct_tool is not None
                else []
            ),
            "deterministic_simulator_routes": [
                "information_sufficiency",
                "forward_prediction",
                "constrained_intervention",
                "counterfactual_reasoning",
            ],
            "deterministic_image_measurement_routes": [
                "single_image_state_evidence",
                "paired_change_direction_evidence",
            ],
            "llm_responsibilities": [
                "choose_registered_tool",
                "construct_ordered_tool_inputs",
                "interpret_validated_tool_result",
            ],
        },
        "claim_boundary": (
            "Exact optics numbers and thresholded image measurements are tool-computed. The LLM is "
            "promoted for routing, registered-input construction, and interpretation; this is synthetic "
            "simulator evidence and not a real-laboratory metrology claim."
        ),
        "source_summaries": {
            "direct": str(args.direct_summary.resolve()),
            **(
                {"direct_tool": str(args.direct_tool_summary.resolve())}
                if args.direct_tool_summary
                else {}
            ),
            **(
                {
                    "direct_tool_confirmation": str(
                        args.direct_tool_confirmation_summary.resolve()
                    )
                }
                if args.direct_tool_confirmation_summary
                else {}
            ),
            "compact": str(args.compact_summary.resolve()),
            "forward": str(args.forward_summary.resolve()),
            "counterfactual": str(args.counterfactual_summary.resolve()),
            "visual": str(args.visual_summary.resolve()),
            "orchestration": str(args.orchestration_summary.resolve()),
            **(
                {"robustness": str(args.robustness_summary.resolve())}
                if args.robustness_summary
                else {}
            ),
            **(
                {
                    "saturation_orchestration": str(
                        args.saturation_orchestration_summary.resolve()
                    )
                }
                if args.saturation_orchestration_summary
                else {}
            ),
            **(
                {
                    "robust_orchestration": str(
                        args.robust_orchestration_summary.resolve()
                    )
                }
                if args.robust_orchestration_summary
                else {}
            ),
        },
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
