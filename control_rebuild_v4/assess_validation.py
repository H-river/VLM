#!/usr/bin/env python3
"""Apply pre-registered validation gates before any held-out evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
THRESHOLDS = {
    "same_distribution_overall_forward_delta": 0.01,
    "same_distribution_overall_forward_retrieval_delta": 0.01,
    "same_distribution_overall_inverse_delta": 0.01,
    "same_distribution_inverse_status_delta_floor": -0.01,
    "hard_category_delta_floor": 0.0,
    "iid_forward_floor": 0.30835390946502055,
    "iid_inverse_floor": 0.385,
    "measurement_all_condition_floor": 0.65,
    "visual_inverse_floor": 0.44,
    "paired_error_retention_floor": 0.70,
    "three_step_closed_loop_floor": 0.45,
    "direct_measurement_floor": 0.70,
    "ready_execution_floor": 0.95,
    "qwen_physical_loss_ceiling": 0.03,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class Gates:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def minimum(self, label: str, observed: Any, threshold: float) -> None:
        self.rows.append(
            {
                "label": label,
                "comparison": ">=",
                "observed": observed,
                "threshold": threshold,
                "passed": finite(observed) and float(observed) >= threshold,
            }
        )

    def maximum(self, label: str, observed: Any, threshold: float) -> None:
        self.rows.append(
            {
                "label": label,
                "comparison": "<=",
                "observed": observed,
                "threshold": threshold,
                "passed": finite(observed) and float(observed) <= threshold,
            }
        )

    def exact(self, label: str, observed: Any, expected: Any) -> None:
        self.rows.append(
            {
                "label": label,
                "comparison": "==",
                "observed": observed,
                "threshold": expected,
                "passed": observed == expected,
            }
        )


def assess_validation(
    run_dir: Path,
    expected_comparison_groups: int = 450,
) -> dict[str, Any]:
    if expected_comparison_groups < 1:
        raise ValueError("expected comparison group count must be positive")
    run_dir = run_dir.resolve()
    comparison = read_json(run_dir / "v3_v4_selection_validation_comparison.json")
    controlled = read_json(run_dir / "controlled_validation.json")
    inverse_summary = read_json(run_dir / "inverse_control_v4_summary.json")
    closed_loop = read_json(run_dir / "closed_loop_val.json")
    overlay = read_json(run_dir / "orchestrated_runtime_validation.json")
    system = read_json(run_dir / "orchestrated_system_validation.json")

    gates = Gates()
    gates.exact("comparison is complete", comparison.get("complete"), True)
    gates.exact(
        (
            f"comparison uses all {expected_comparison_groups} difficult "
            "validation groups"
        ),
        comparison.get("models", {}).get("v4", {}).get("all", {}).get("group_count"),
        expected_comparison_groups,
    )
    gates.exact(
        "controlled validation is complete",
        controlled.get("complete"),
        True,
    )
    gates.exact("closed-loop validation is complete", closed_loop.get("complete"), True)
    gates.exact(
        "route integration validation is complete", overlay.get("complete"), True
    )
    gates.exact("combined system validation is complete", system.get("complete"), True)

    same = comparison["comparison"]
    overall = same["all"]
    gates.minimum(
        "same-distribution forward all-five improvement",
        overall["forward_strict_all_five"]["absolute_delta"],
        THRESHOLDS["same_distribution_overall_forward_delta"],
    )
    gates.minimum(
        "same-distribution forward-only target-retrieval improvement",
        overall["forward_only_target_success_feasible"]["absolute_delta"],
        THRESHOLDS["same_distribution_overall_forward_retrieval_delta"],
    )
    gates.minimum(
        "same-distribution learned-inverse target improvement",
        overall["inverse_target_success_feasible"]["absolute_delta"],
        THRESHOLDS["same_distribution_overall_inverse_delta"],
    )
    gates.minimum(
        "same-distribution inverse status does not materially regress",
        overall["inverse_status_accuracy"]["absolute_delta"],
        THRESHOLDS["same_distribution_inverse_status_delta_floor"],
    )
    for category in ("ood_boundary", "high_nonlinearity"):
        category_result = same["by_category"][category]
        gates.minimum(
            f"{category} forward all-five does not regress",
            category_result["forward_strict_all_five"]["absolute_delta"],
            THRESHOLDS["hard_category_delta_floor"],
        )
        gates.minimum(
            f"{category} inverse target success does not regress",
            category_result["inverse_target_success_feasible"]["absolute_delta"],
            THRESHOLDS["hard_category_delta_floor"],
        )

    val = controlled["split_results"]["val"]
    gates.minimum(
        "old-IID forward accuracy remains within one percentage point of v3",
        val["numerical"]["forward"]["strict_all_five_success"],
        THRESHOLDS["iid_forward_floor"],
    )
    gates.minimum(
        "old-IID inverse target accuracy remains within one percentage point of v3",
        val["numerical"]["inverse"]["learned_residual_corrected"][
            "target_success_feasible"
        ],
        THRESHOLDS["iid_inverse_floor"],
    )
    gates.minimum(
        "all-condition measurement accuracy",
        val["measurement"]["all_conditions"]["strict_all_five_success"],
        THRESHOLDS["measurement_all_condition_floor"],
    )
    gates.minimum(
        "visual inverse physical target success",
        val["visual_inverse"]["model_measurement"]["physical_target_success_feasible"],
        THRESHOLDS["visual_inverse_floor"],
    )

    expanded = inverse_summary["validation"]["metrics_at_selected_alpha"]
    clean_inverse = float(expanded["expanded_clean"]["target_success_feasible"])
    noisy_inverse = float(
        expanded["expanded_measurement_augmented"]["target_success_feasible"]
    )
    retention = noisy_inverse / clean_inverse if clean_inverse > 0.0 else 0.0
    gates.minimum(
        "paired measurement-error inverse retention",
        retention,
        THRESHOLDS["paired_error_retention_floor"],
    )
    gates.minimum(
        "three-step closed-loop target success",
        closed_loop["metrics"]["reached_by_step"]["3"]["rate"],
        THRESHOLDS["three_step_closed_loop_floor"],
    )

    direct = overlay["direct_measurement_validation"]
    gates.exact("all seven registered routes execute", overlay.get("routes_passed"), 7)
    gates.exact(
        "route integration uses no simulator at inference",
        overlay.get("simulator_inference_calls"),
        0,
    )
    gates.minimum(
        "direct public-contract image measurement",
        direct["strict_all_five_success"],
        THRESHOLDS["direct_measurement_floor"],
    )
    system_metrics = system["metrics"]
    gates.minimum(
        "saved-Qwen valid ready execution",
        system_metrics["successful_valid_execution_rate"],
        THRESHOLDS["ready_execution_floor"],
    )
    gates.maximum(
        "maximum Qwen physical loss versus correctly routed specialist",
        max(
            system_metrics[
                "physical_loss_versus_correctly_routed_specialist_by_task"
            ].values()
        ),
        THRESHOLDS["qwen_physical_loss_ceiling"],
    )
    gates.exact(
        "combined system uses no simulator at inference",
        system_metrics["simulator_calls_during_inference_count"],
        0,
    )

    failed = [row["label"] for row in gates.rows if not row["passed"]]
    result = {
        "decision_rule_version": "control_rebuild_v4_preregistered_validation_gates",
        "scope": (
            "validation-only decision made before the one-time held-out " "evaluation"
        ),
        "thresholds": THRESHOLDS,
        "gates": gates.rows,
        "gate_count": len(gates.rows),
        "passed_gate_count": len(gates.rows) - len(failed),
        "failed_gates": failed,
        "proceed_to_heldout": not failed,
        "held_out_evaluation_opened": False,
        "complete": True,
    }
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "validation_decision.json"
    )
    result = assess_validation(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
