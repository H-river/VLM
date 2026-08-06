#!/usr/bin/env python3
"""Create a compact quantitative report from frozen Qwen and v3/v4 results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_numerical"
QWEN_SELECTION = (
    REPO_ROOT
    / "Qwen_orchestration/results/v1/stage2_safe_runtime_v1/full_selection.json"
)
QWEN_END_TO_END = (
    REPO_ROOT
    / "Qwen_orchestration/results/v1/stage2_safe_runtime_v1"
    / "checkpoint-1000_end_to_end.json"
)
V3_EVALUATION = (
    REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed/controlled_evaluation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--phase", choices=("validation", "final"), required=True)
    parser.add_argument(
        "--decision-file",
        default="validation_decision.json",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float | None) -> str:
    return "not available" if value is None else f"{100.0 * value:.2f}%"


def rate_with_count(value: float | None, denominator: int | None) -> str:
    if value is None or denominator is None:
        return "not available"
    numerator = int(round(value * denominator))
    return f"{percent(value)} ({numerator:,}/{denominator:,})"


def percentage_points(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def qwen_metrics() -> dict[str, Any]:
    selection = read_json(QWEN_SELECTION)
    candidate = next(
        row for row in selection["candidates"] if int(row["checkpoint_step"]) == 1000
    )
    metrics = candidate["metrics"]
    end_to_end = read_json(QWEN_END_TO_END)["metrics"]
    return {
        "checkpoint_step": 1000,
        "formally_promoted": selection["selected_checkpoint"] is not None,
        "schema_valid": metrics["schema_valid_rate"],
        "registry_valid_ready": metrics["registry_valid_ready_call_rate"],
        "correct_ready_route": metrics["ready_route_exact_accuracy"],
        "required_argument_groups": metrics["required_argument_group_exact_accuracy"],
        "numerical_value_and_unit_path": metrics["numeric_value_unit_exact_accuracy"],
        "image_role_binding": metrics["image_role_exact_accuracy"],
        "clarification_status": metrics["clarification_recall"],
        "unsupported_status": metrics["unsupported_recall"],
        "valid_ready_execution": end_to_end["successful_valid_execution_rate"],
        "denominators": {
            "schema_valid": int(metrics["count"]),
            "registry_valid_ready": int(
                metrics["stage2_denominators"]["ready_records"]
            ),
            "correct_ready_route": int(metrics["stage2_denominators"]["ready_records"]),
            "required_argument_groups": int(
                metrics["stage2_denominators"]["required_argument_groups"]
            ),
            "numerical_value_and_unit_path": int(
                metrics["stage2_denominators"]["numeric_values_with_unit_paths"]
            ),
            "image_role_binding": int(
                metrics["stage2_denominators"]["visual_ready_records"]
            ),
            "clarification_status": int(
                metrics["stage2_denominators"]["clarification_records"]
            ),
            "unsupported_status": 200,
        },
    }


def specialist_block(split: dict[str, Any]) -> dict[str, Any]:
    numerical = split["numerical"]
    inverse = numerical.get("inverse")
    learned = None if inverse is None else inverse["learned_residual_corrected"]
    measurement = split["measurement"]["all_conditions"]
    visual = split["visual_inverse"]["model_measurement"]
    visual_oracle = split["visual_inverse"]["oracle_measurement"]
    return {
        "forward_transition_count": numerical["forward"]["count"],
        "forward_strict_all_five": numerical["forward"]["strict_all_five_success"],
        "forward_mae_tolerance_units": numerical["forward"]["mae_in_tolerance_units"],
        "inverse_target_success_feasible": (
            None if learned is None else learned["target_success_feasible"]
        ),
        "inverse_reachable_request_count": (
            None if learned is None else learned["reachable_pair_count"]
        ),
        "inverse_minimum_movement_exact": (
            None if learned is None else learned["minimum_movement_exact_feasible"]
        ),
        "inverse_status_accuracy": (
            None if learned is None else learned["status_accuracy"]
        ),
        "measurement_strict_all_five": measurement["strict_all_five_success"],
        "measurement_view_count": measurement["count"],
        "measurement_mae_tolerance_units": measurement["mae_in_tolerance_units"],
        "visual_target_success_feasible": visual["physical_target_success_feasible"],
        "visual_reachable_request_count": visual["reachable_request_count"],
        "visual_minimum_movement_exact": visual["minimum_movement_exact_feasible"],
        "visual_status_accuracy": visual["status_accuracy"],
        "visual_oracle_measurement_target_success_feasible": visual_oracle[
            "physical_target_success_feasible"
        ],
    }


def closed_loop_block(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    metrics = read_json(path)["metrics"]
    return {
        "request_count": metrics["request_count"],
        "one_step_success": metrics["reached_by_step"]["1"]["rate"],
        "three_step_success": metrics["reached_by_step"]["3"]["rate"],
        "mean_executed_steps": metrics["mean_executed_steps"],
        "zero_action_stalls": metrics["zero_action_stalls"],
    }


def overlay_block(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    direct = value.get("direct_measurement_validation")
    return {
        "complete": bool(value["complete"]),
        "routes_expected": int(value["routes_expected"]),
        "routes_passed": int(value["routes_passed"]),
        "all_routes_passed": bool(value["all_routes_passed"]),
        "simulator_inference_calls": int(value["simulator_inference_calls"]),
        "scope": str(value["scope"]),
        "candidate_manifest": str(value["overlay_manifest"]),
        "direct_measurement": (
            None
            if direct is None
            else {
                "count": int(direct["count"]),
                "strict_all_five_success": float(direct["strict_all_five_success"]),
                "mae_in_tolerance_units": float(direct["mae_in_tolerance_units"]),
                "measurement_source_counts": dict(direct["measurement_source_counts"]),
                "scope": str(direct["scope"]),
            }
        ),
    }


def whole_system_block(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    metrics = value["metrics"]
    return {
        "complete": bool(value["complete"]),
        "record_count": int(value["record_count"]),
        "task_ready_counts": dict(value["task_ready_counts"]),
        "successful_valid_execution": float(metrics["successful_valid_execution_rate"]),
        "direction_macro_f1": float(metrics["end_to_end_direction_macro_f1"]),
        "direction_physical_macro_f1": float(
            metrics["end_to_end_direction_physical_macro_f1"]
        ),
        "direction_physical_all_five_exact": float(
            metrics["end_to_end_direction_physical_all_five_exact"]
        ),
        "correctly_routed_direction_physical_macro_f1": float(
            metrics["correctly_routed_direction_physical_macro_f1"]
        ),
        "forward_consistency_with_v4_oracle": float(
            metrics["end_to_end_forward_strict_all_five_success"]
        ),
        "forward_physical_strict_all_five": float(
            metrics["end_to_end_forward_physical_strict_all_five_success"]
        ),
        "correctly_routed_forward_physical_strict_all_five": float(
            metrics["correctly_routed_forward_physical_strict_all_five_success"]
        ),
        "direct_measurement_strict_all_five": float(
            metrics["end_to_end_visual_measurement_strict_all_five_success"]
        ),
        "correctly_routed_measurement_strict_all_five": float(
            metrics["oracle_visual_measurement_strict_all_five_success"]
        ),
        "inverse_physical_target_reached": float(
            metrics["end_to_end_inverse_target_reached_rate"]
        ),
        "v4_oracle_inverse_physical_target_reached": float(
            metrics["oracle_inverse_target_reached_rate"]
        ),
        "simulator_calls_during_inference": int(
            metrics["simulator_calls_during_inference_count"]
        ),
        "private_simulator_scoring_calls": int(
            metrics.get(
                "private_simulator_scoring_calls_total",
                metrics["private_simulator_scoring_calls"],
            )
        ),
        "scope": str(value["scope"]),
    }


def failure_focused_validation_block(
    run_dir: Path,
    data_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Expose the difficult v4 checkpoint-selection distribution explicitly."""

    forward_path = run_dir / "forward_physics_residual_v4_summary.json"
    inverse_path = run_dir / "inverse_control_v4_summary.json"
    if not forward_path.is_file() or not inverse_path.is_file():
        return None
    forward = read_json(forward_path)
    inverse = read_json(inverse_path)
    forward_expanded = forward["validation"]["validation_expanded"]
    forward_retrieval = forward_expanded.get("forward_cost_only")
    if forward_retrieval is None:
        legacy_retrieval = forward_expanded["inverse_selection"]
        forward_retrieval = {
            "pair_count": legacy_retrieval["request_count"],
            "reachable_pair_count": legacy_retrieval["request_count"],
            "target_success_feasible": legacy_retrieval["target_success"],
            "metric_source": "legacy_three_target_proxy",
        }
    else:
        forward_retrieval = {
            **forward_retrieval,
            "metric_source": "exact_six_request_physical_retrieval",
        }
    inverse_expanded = inverse["validation"]["metrics_at_selected_alpha"]
    group_count = int(forward_expanded["groups"])
    category_counts = None
    dataset_manifest_path = (
        DEFAULT_DATA / "manifest.json"
        if data_dir is None
        else data_dir / "manifest.json"
    )
    if dataset_manifest_path.is_file():
        dataset_manifest = read_json(dataset_manifest_path)
        category_counts = (
            dataset_manifest.get("verification", {})
            .get("splits", {})
            .get("val", {})
            .get("category_counts")
        )
        if category_counts is None:
            category_counts = (
                dataset_manifest.get("split_summary", {})
                .get("val", {})
                .get("category_counts")
            )
    comparison_path = run_dir / "v3_v4_selection_validation_comparison.json"
    same_distribution_comparison = (
        read_json(comparison_path) if comparison_path.is_file() else None
    )
    return {
        "scope": (
            f"{group_count}-group v4 checkpoint-selection validation; "
            "this is not held-out evaluation"
        ),
        "group_count": group_count,
        "category_counts": (
            None
            if category_counts is None
            else {key: int(value) for key, value in category_counts.items()}
        ),
        "forward": {
            "transition_count": int(forward_expanded["forward"]["count"]),
            "strict_all_five_success": float(
                forward_expanded["forward"]["strict_all_five_success"]
            ),
            "mae_in_tolerance_units": float(
                forward_expanded["forward"]["mae_in_tolerance_units"]
            ),
        },
        "forward_only_target_retrieval": {
            "pair_count": int(forward_retrieval["pair_count"]),
            "request_count": int(forward_retrieval["reachable_pair_count"]),
            "target_success": float(forward_retrieval["target_success_feasible"]),
            "metric_source": str(forward_retrieval["metric_source"]),
        },
        "inverse_clean": {
            key: inverse_expanded["expanded_clean"][key]
            for key in (
                "pair_count",
                "reachable_pair_count",
                "target_success_feasible",
                "status_accuracy",
            )
        },
        "inverse_paired_measurement_errors": {
            key: inverse_expanded["expanded_measurement_augmented"][key]
            for key in (
                "pair_count",
                "reachable_pair_count",
                "target_success_feasible",
                "status_accuracy",
            )
        },
        "same_distribution_v3_v4": (
            None
            if same_distribution_comparison is None
            else {
                "scope": same_distribution_comparison["scope"],
                "comparison": same_distribution_comparison["comparison"],
                "examples": same_distribution_comparison.get("examples"),
            }
        ),
        "held_out_used_for_training_or_selection": 0,
    }


def validation_decision_block(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    if "proceed_to_server_training" in value:
        authorization_target = "server_training"
        authorized = bool(value["proceed_to_server_training"])
    else:
        authorization_target = "held_out_evaluation"
        authorized = bool(value["proceed_to_heldout"])
    return {
        "complete": bool(value["complete"]),
        "authorization_target": authorization_target,
        "authorized": authorized,
        "proceed_to_heldout": (
            authorized if authorization_target == "held_out_evaluation" else False
        ),
        "proceed_to_server_training": (
            authorized if authorization_target == "server_training" else False
        ),
        "gate_count": int(value["gate_count"]),
        "passed_gate_count": int(value["passed_gate_count"]),
        "failed_gates": list(value["failed_gates"]),
        "scope": str(value["scope"]),
    }


def table_rows(
    split: str,
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> list[str]:
    definitions = [
        (
            "Forward: all five within tolerance",
            "forward_strict_all_five",
            "forward_transition_count",
        ),
        (
            "Inverse: physical target reached",
            "inverse_target_success_feasible",
            "inverse_reachable_request_count",
        ),
        (
            "Measurement with known synthetic transform: all five within tolerance",
            "measurement_strict_all_five",
            "measurement_view_count",
        ),
        (
            "Visual inverse with known synthetic transform: physical target reached",
            "visual_target_success_feasible",
            "visual_reachable_request_count",
        ),
    ]
    rows = []
    for label, key, denominator_key in definitions:
        old = None if previous is None else previous.get(key)
        old_denominator = None if previous is None else previous.get(denominator_key)
        rows.append(
            f"| {split} | {label} | "
            f"{rate_with_count(old, old_denominator)} | "
            f"{rate_with_count(current.get(key), current.get(denominator_key))} |"
        )
    return rows


def collect_examples(evaluation: dict[str, Any]) -> dict[str, Any]:
    definitions = {
        "forward": (
            "Success means all five predicted numerical changes are within "
            "one declared tolerance of simulator truth."
        ),
        "inverse": (
            "Success means the selected action is one of the "
            "simulator-confirmed matching actions for a reachable request."
        ),
        "measurement": (
            "Success means all five measured beam values are within one "
            "declared tolerance of the image ground truth."
        ),
        "visual_inverse": (
            "Success means the action selected from measured current and "
            "desired images physically reaches the reachable target."
        ),
    }
    tasks = {task: {"success": None, "failure": None} for task in definitions}
    for split, value in evaluation["split_results"].items():
        sources = {
            "forward": value["numerical"].get("examples", {}).get("forward", {}),
            "inverse": value["numerical"].get("examples", {}).get("inverse", {}),
            "measurement": value["measurement"].get("examples", {}),
            "visual_inverse": value["visual_inverse"].get("examples", {}),
        }
        for task, examples in sources.items():
            for outcome in ("success", "failure"):
                if tasks[task][outcome] is None and examples.get(outcome) is not None:
                    tasks[task][outcome] = {
                        "source_split": split,
                        **examples[outcome],
                    }
    return {
        "definitions": definitions,
        "tasks": tasks,
    }


def write_examples(
    run_dir: Path,
    phase: str,
    evaluation: dict[str, Any],
) -> tuple[Path, Path]:
    examples = collect_examples(evaluation)
    output_json = run_dir / f"{phase}_examples.json"
    output_markdown = run_dir / f"{phase.upper()}_EXAMPLES.md"
    result = {
        "report_version": "control_rebuild_v4_real_examples",
        "phase": phase,
        **examples,
    }
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Specialist v4 {phase} success and failure examples",
        "",
        "These are real records from the declared evaluation phase. Numerical "
        "values are copied from the saved evaluator output.",
    ]
    labels = {
        "forward": "Forward prediction",
        "inverse": "Numerical inverse control",
        "measurement": "Beam-image measurement",
        "visual_inverse": "Visual inverse control",
    }
    for task, label in labels.items():
        lines.extend(["", f"## {label}", "", examples["definitions"][task]])
        for outcome in ("success", "failure"):
            lines.extend(["", f"### {outcome.title()}", ""])
            value = examples["tasks"][task][outcome]
            if value is None:
                lines.append(f"No {outcome} example exists in this evaluation.")
            else:
                lines.extend(
                    [
                        "```json",
                        json.dumps(value, indent=2, sort_keys=True),
                        "```",
                    ]
                )
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_json, output_markdown


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    evaluation_path = (
        run_dir / "controlled_validation.json"
        if args.phase == "validation"
        else run_dir / "controlled_evaluation.json"
    )
    evaluation = read_json(evaluation_path)
    v4 = {
        split: specialist_block(value)
        for split, value in evaluation["split_results"].items()
    }
    v3 = {}
    if args.phase == "final" and V3_EVALUATION.is_file():
        previous = read_json(V3_EVALUATION)
        v3 = {
            split: specialist_block(value)
            for split, value in previous["split_results"].items()
            if split in v4
        }
    closed_loop = {}
    if args.phase == "validation":
        closed_loop["val"] = closed_loop_block(run_dir / "closed_loop_val.json")
    else:
        closed_loop["test_iid"] = closed_loop_block(
            run_dir / "closed_loop_test_iid.json"
        )
        closed_loop["test_ood_physics"] = closed_loop_block(
            run_dir / "closed_loop_test_ood_physics.json"
        )

    result = {
        "report_version": "qwen_and_specialists_v4_one_seed",
        "phase": args.phase,
        "qwen_frozen_validation": qwen_metrics(),
        "specialists_v4": v4,
        "specialists_v3_comparison": v3,
        "closed_loop_v4": closed_loop,
        "failure_focused_selection_validation": (
            failure_focused_validation_block(run_dir, data_dir)
        ),
        "validation_decision": validation_decision_block(run_dir / args.decision_file),
        "qwen_to_v4_execution_overlay": overlay_block(
            run_dir / "orchestrated_runtime_validation.json"
        ),
        "qwen_plus_v4_system_validation": whole_system_block(
            run_dir / "orchestrated_system_validation.json"
        ),
        "evidence_scope": {
            "qwen": "frozen 1600-record in-domain validation",
            "specialists": (
                "validation"
                if args.phase == "validation"
                else "one final held-out evaluation"
            ),
            "same_denominator_for_qwen_and_specialists": False,
            "held_out_used_for_training_or_selection": 0,
            "failure_focused_selection_validation": (
                "used to choose forward and inverse checkpoints; reported "
                "separately from held-out results"
            ),
        },
    }
    examples_json, examples_markdown = write_examples(
        run_dir,
        args.phase,
        evaluation,
    )
    result["examples_artifacts"] = {
        "json": str(examples_json),
        "markdown": str(examples_markdown),
    }
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else run_dir / f"{args.phase}_performance_summary.json"
    )
    output_markdown = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else run_dir / f"{args.phase.upper()}_REPORT.md"
    )
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    qwen = result["qwen_frozen_validation"]
    qwen_denominator = qwen["denominators"]
    lines = [
        "# Qwen orchestration and specialist v4 performance",
        "",
        "Qwen numbers below are frozen in-domain validation results. "
        "Specialist numbers use a separate specialist dataset, so the two "
        "sets of percentages must not be multiplied or treated as one "
        "shared denominator.",
        "",
        "## Frozen Qwen orchestration",
        "",
        "| Metric | Rate |",
        "|---|---:|",
        "| JSON-schema-valid decisions | "
        f"{rate_with_count(qwen['schema_valid'], qwen_denominator['schema_valid'])} |",
        "| Registry-valid ready calls | "
        f"{rate_with_count(qwen['registry_valid_ready'], qwen_denominator['registry_valid_ready'])} |",
        "| Correct route on ready requests | "
        f"{rate_with_count(qwen['correct_ready_route'], qwen_denominator['correct_ready_route'])} |",
        "| Exact required argument groups | "
        f"{rate_with_count(qwen['required_argument_groups'], qwen_denominator['required_argument_groups'])} |",
        "| Exact numerical value and unit path | "
        f"{rate_with_count(qwen['numerical_value_and_unit_path'], qwen_denominator['numerical_value_and_unit_path'])} |",
        "| Exact image-role binding | "
        f"{rate_with_count(qwen['image_role_binding'], qwen_denominator['image_role_binding'])} |",
        "| Correct clarification status | "
        f"{rate_with_count(qwen['clarification_status'], qwen_denominator['clarification_status'])} |",
        "| Correct unsupported status | "
        f"{rate_with_count(qwen['unsupported_status'], qwen_denominator['unsupported_status'])} |",
        "",
        "Formal Qwen promotion remains **not passed**, because schema, "
        "registry-valid calls, and image-role binding remain below their "
        "frozen gates.",
        "",
        "## Specialist results",
        "",
        "| Split | Metric | v3 | v4 |",
        "|---|---|---:|---:|",
    ]
    for split, block in v4.items():
        lines.extend(table_rows(split, block, v3.get(split)))
    lines.extend(["", "## Failure-focused selection validation", ""])
    failure_focused = result["failure_focused_selection_validation"]
    if failure_focused is None:
        lines.append("Failure-focused checkpoint-selection metrics are not available.")
    else:
        lines.append(
            "These metrics use the new difficult validation distribution that "
            "selected the forward and inverse checkpoints. They are not "
            "held-out test results."
        )
        categories = failure_focused["category_counts"]
        if categories is not None:
            lines.extend(
                [
                    "",
                    f"The {failure_focused['group_count']} setups contain "
                    f"**{categories.get('iid_expanded', 0)} expanded "
                    "in-distribution**, "
                    f"**{categories.get('ood_boundary', 0)} boundary**, and "
                    f"**{categories.get('high_nonlinearity', 0)} "
                    "high-nonlinearity** setups.",
                ]
            )
        forward_focus = failure_focused["forward"]
        retrieval = failure_focused["forward_only_target_retrieval"]
        inverse_clean = failure_focused["inverse_clean"]
        inverse_noisy = failure_focused["inverse_paired_measurement_errors"]
        lines.extend(
            [
                "",
                "| Intermediate metric | Rate |",
                "|---|---:|",
                "| Forward numerical prediction: all five within tolerance | "
                f"{rate_with_count(forward_focus['strict_all_five_success'], forward_focus['transition_count'])} |",
                "| Forward-only candidate retrieval: physical target found | "
                f"{rate_with_count(retrieval['target_success'], retrieval['request_count'])} |",
                "| Inverse control with clean numerical states: physical target reached | "
                f"{rate_with_count(inverse_clean['target_success_feasible'], inverse_clean['reachable_pair_count'])} |",
                "| Inverse control with paired measurement errors: physical target reached | "
                f"{rate_with_count(inverse_noisy['target_success_feasible'], inverse_noisy['reachable_pair_count'])} |",
                "| Inverse clean three-class status accuracy | "
                f"{rate_with_count(inverse_clean['status_accuracy'], inverse_clean['pair_count'])} |",
                "| Inverse paired-error three-class status accuracy | "
                f"{rate_with_count(inverse_noisy['status_accuracy'], inverse_noisy['pair_count'])} |",
            ]
        )
        same_distribution = failure_focused["same_distribution_v3_v4"]
        if same_distribution is not None:
            metric_labels = {
                "forward_strict_all_five": ("Forward all-five numerical success"),
                "forward_only_target_success_feasible": (
                    "Forward-only physical target retrieval"
                ),
                "inverse_target_success_feasible": (
                    "Learned inverse physical target success"
                ),
                "inverse_status_accuracy": "Inverse three-class status accuracy",
            }
            scopes = {
                "all": f"All {failure_focused['group_count']}",
                "iid_expanded": "Expanded in-distribution",
                "ood_boundary": "Boundary",
                "high_nonlinearity": "High nonlinearity",
            }
            comparison = same_distribution["comparison"]
            lines.extend(
                [
                    "",
                    "### Frozen v3 versus candidate v4 on identical inputs",
                    "",
                    "| Validation subset | Metric | v3 | v4 | v4 minus v3 |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for scope, label in scopes.items():
                values = (
                    comparison["all"]
                    if scope == "all"
                    else comparison["by_category"].get(scope)
                )
                if values is None:
                    continue
                for metric, metric_label in metric_labels.items():
                    row = values[metric]
                    lines.append(
                        f"| {label} | {metric_label} | "
                        f"{percent(row['v3'])} | {percent(row['v4'])} | "
                        f"{percentage_points(row['absolute_delta'])} |"
                    )
            examples = same_distribution.get("examples")
            if examples is not None:
                lines.extend(
                    [
                        "",
                        "### Real side-by-side diagnostic records",
                        "",
                        "| Task | v4 fixed a v3 failure | "
                        "v4 regressed from a v3 success |",
                        "|---|---|---|",
                    ]
                )
                for task, label in (
                    ("forward", "Forward prediction"),
                    ("inverse", "Inverse control"),
                ):
                    improvement = examples["all"][task]["v4_improvement"]
                    regression = examples["all"][task]["v4_regression"]

                    def identifier(value: dict[str, Any] | None) -> str:
                        if value is None:
                            return "none in this validation"
                        return str(value.get("request_id", value.get("group_id")))

                    lines.append(
                        f"| {label} | {identifier(improvement)} | "
                        f"{identifier(regression)} |"
                    )
    lines.extend(["", "## Closed-loop results", ""])
    available_loop = {
        split: block for split, block in closed_loop.items() if block is not None
    }
    if not available_loop:
        lines.append("Closed-loop results are not available.")
    else:
        lines.extend(
            [
                "| Split | Requests | Reached after one step | "
                "Reached after three steps |",
                "|---|---:|---:|---:|",
            ]
        )
        for split, block in available_loop.items():
            lines.append(
                f"| {split} | {block['request_count']} | "
                f"{percent(block['one_step_success'])} | "
                f"{percent(block['three_step_success'])} |"
            )
    decision = result["validation_decision"]
    if decision is None:
        lines.extend(["", "## Validation decision", ""])
        lines.append("The pre-registered validation decision is not available.")
    elif decision["authorization_target"] == "server_training":
        lines.extend(["", "## Server-training decision", ""])
        if decision["authorized"]:
            lines.append(
                f"**Pass:** {decision['passed_gate_count']}/"
                f"{decision['gate_count']} registered performance bars passed. "
                "Full unique-data server training is recommended. No held-out "
                "test was opened."
            )
        else:
            lines.append(
                f"**Stop:** {decision['passed_gate_count']}/"
                f"{decision['gate_count']} registered performance bars passed. "
                "Full server training is not yet recommended. No held-out test "
                "was opened."
            )
            lines.extend(["", "Failed gates:"])
            lines.extend(f"- {label}" for label in decision["failed_gates"])
    elif decision["authorized"]:
        lines.extend(["", "## Held-out evaluation decision", ""])
        lines.append(
            f"**Pass:** {decision['passed_gate_count']}/"
            f"{decision['gate_count']} pre-registered gates passed. "
            "The one-time held-out evaluation is authorized."
        )
    else:
        lines.extend(["", "## Held-out evaluation decision", ""])
        lines.append(
            f"**Stop:** {decision['passed_gate_count']}/"
            f"{decision['gate_count']} pre-registered gates passed. "
            "The held-out evaluation is not authorized."
        )
        lines.extend(["", "Failed gates:"])
        lines.extend(f"- {label}" for label in decision["failed_gates"])
    lines.extend(["", "## Qwen-to-v4 execution overlay", ""])
    overlay = result["qwen_to_v4_execution_overlay"]
    if overlay is None:
        lines.append("The bounded route-integration validation is not available.")
    else:
        lines.append(
            f"Contract-valid execution passed **{overlay['routes_passed']}/"
            f"{overlay['routes_expected']} registered routes**, with "
            f"**{overlay['simulator_inference_calls']} simulator calls at "
            "inference**. This checks route-to-specialist integration, not "
            "task accuracy."
        )
        lines.append("")
        lines.append(
            "The executed specialist generation was loaded from the "
            "hash-pinned `candidate_overlay_manifest.json`."
        )
        direct = overlay["direct_measurement"]
        if direct is not None:
            lines.append("")
            lines.append(
                "Under the public image-plus-calibration contract, direct beam "
                "measurement placed all five outputs within tolerance for "
                f"**{rate_with_count(direct['strict_all_five_success'], direct['count'])}**. "
                "This specialist metric uses ground-truth sensor coordinates "
                "and does not include Qwen routing errors."
            )
    lines.extend(["", "## Saved Qwen decisions plus v4 specialists", ""])
    system = result["qwen_plus_v4_system_validation"]
    if system is None:
        lines.append("The combined in-domain system validation is not available.")
    else:
        counts = system["task_ready_counts"]
        lines.extend(
            [
                "These results execute the saved checkpoint-1000 decisions, so "
                "they include Qwen parsing, routing, argument, and image-binding "
                "errors. Direction and forward physical metrics compare with "
                "simulator ground truth. Consistency with target-decision "
                "execution is retained as a separate diagnostic.",
                "",
                "| Metric | Rate |",
                "|---|---:|",
                "| Valid ready requests executed | "
                f"{rate_with_count(system['successful_valid_execution'], sum(counts.values()))} |",
                "| Direction macro F1 versus target-decision execution | "
                f"{percent(system['direction_macro_f1'])} |",
                "| Direction physical macro F1 | "
                f"{percent(system['direction_physical_macro_f1'])} |",
                "| Direction physical all-five exact | "
                f"{rate_with_count(system['direction_physical_all_five_exact'], counts.get('direction_prediction'))} |",
                "| Correctly routed direction specialist physical macro F1 | "
                f"{percent(system['correctly_routed_direction_physical_macro_f1'])} |",
                "| Forward physical all-five success | "
                f"{rate_with_count(system['forward_physical_strict_all_five'], counts.get('forward_prediction'))} |",
                "| Correctly routed forward specialist physical all-five success | "
                f"{rate_with_count(system['correctly_routed_forward_physical_strict_all_five'], counts.get('forward_prediction'))} |",
                "| Forward consistency versus target-decision execution | "
                f"{rate_with_count(system['forward_consistency_with_v4_oracle'], counts.get('forward_prediction'))} |",
                "| Direct measurement all-five physical success | "
                f"{rate_with_count(system['direct_measurement_strict_all_five'], counts.get('beam_profile_measurement'))} |",
                "| Correctly routed measurement specialist physical all-five success | "
                f"{rate_with_count(system['correctly_routed_measurement_strict_all_five'], counts.get('beam_profile_measurement'))} |",
                "| Inverse physical target reached | "
                f"{rate_with_count(system['inverse_physical_target_reached'], counts.get('inverse_control'))} |",
                "| Correctly routed v4 inverse oracle | "
                f"{rate_with_count(system['v4_oracle_inverse_physical_target_reached'], counts.get('inverse_control'))} |",
            ]
        )
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": args.phase,
                "json": str(output_json),
                "markdown": str(output_markdown),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
