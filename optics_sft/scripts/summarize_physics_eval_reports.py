#!/usr/bin/env python3
"""Summarize physics-aware optics SFT evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_ARGS = {
    "sim_in_loop": "sim_in_loop_report",
    "forward": "forward_report",
    "counterfactual": "counterfactual_report",
    "closed_loop": "closed_loop_report",
    "ood": "ood_report",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize physics-aware optics SFT evaluation reports.")
    parser.add_argument("--sim-in-loop-report", type=Path)
    parser.add_argument("--forward-report", type=Path)
    parser.add_argument("--counterfactual-report", type=Path)
    parser.add_argument("--closed-loop-report", type=Path)
    parser.add_argument("--ood-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON report: {path}")
    return data


def extract_metrics(data: Mapping[str, Any]) -> dict[str, Any]:
    summary = data.get("summary")
    if isinstance(summary, dict):
        metrics = dict(summary)
        if isinstance(data.get("by_changed_parameter"), dict):
            metrics["by_changed_parameter"] = data["by_changed_parameter"]
        return metrics

    # OOD split tooling writes a manifest, not an evaluator summary.
    manifest_keys = (
        "counts",
        "counts_by_sample_type",
        "input_count",
        "id_candidate_count",
        "ood_candidate_count",
        "ignored_count",
        "ignored_by_reason",
        "ood_parameter",
        "ranges",
    )
    manifest_metrics = {key: data[key] for key in manifest_keys if key in data}
    if manifest_metrics:
        return manifest_metrics
    return {}


def load_reports(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for report_name, arg_name in REPORT_ARGS.items():
        path = getattr(args, arg_name)
        if path is None:
            reports[report_name] = {"status": "missing", "path": None, "metrics": {}}
            continue
        if not path.exists():
            reports[report_name] = {
                "status": "missing",
                "path": str(path),
                "metrics": {},
                "error": "file_not_found",
            }
            continue
        data = load_json(path)
        reports[report_name] = {
            "status": "present",
            "path": str(path),
            "metrics": extract_metrics(data),
        }
    return reports


def get_metric(reports: Mapping[str, Any], report_name: str, key: str) -> Any:
    report = reports.get(report_name, {})
    metrics = report.get("metrics", {}) if isinstance(report, Mapping) else {}
    if isinstance(metrics, Mapping):
        return metrics.get(key)
    return None


def present(reports: Mapping[str, Any], report_name: str) -> bool:
    report = reports.get(report_name, {})
    return isinstance(report, Mapping) and report.get("status") == "present"


def fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, sort_keys=True)}`"
    return str(value)


def bullet(label: str, value: Any) -> str:
    return f"- {label}: {fmt(value)}"


def section_missing(report_label: str) -> list[str]:
    return [f"- Missing: no {report_label} report was provided."]


def json_schema_reliability(reports: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    any_present = False
    for report_name, label in (
        ("sim_in_loop", "simulator-in-loop"),
        ("forward", "forward"),
        ("counterfactual", "counterfactual"),
    ):
        if not present(reports, report_name):
            lines.append(f"- {label}: missing")
            continue
        any_present = True
        lines.append(bullet(f"{label} json_valid_rate", get_metric(reports, report_name, "json_valid_rate")))
    if present(reports, "sim_in_loop"):
        lines.append(bullet("simulator-in-loop control_plan_valid_rate", get_metric(reports, "sim_in_loop", "control_plan_valid_rate")))
    if present(reports, "forward"):
        lines.append(bullet("forward predicted_after_state_valid_rate", get_metric(reports, "forward", "predicted_after_state_valid_rate")))
        lines.append(bullet("forward predicted_change_valid_rate", get_metric(reports, "forward", "predicted_change_valid_rate")))
    if present(reports, "counterfactual"):
        lines.append(bullet("counterfactual required_fields_valid_rate", get_metric(reports, "counterfactual", "required_fields_valid_rate")))
    if not any_present:
        return ["- Missing: no prediction reports with JSON/schema reliability metrics were provided."]
    return lines


def action_accuracy(reports: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if present(reports, "sim_in_loop"):
        lines.append(bullet("inverse-control action_mae", get_metric(reports, "sim_in_loop", "action_mae")))
        lines.append(bullet("inverse-control lens_sign_accuracy", get_metric(reports, "sim_in_loop", "lens_sign_accuracy")))
    else:
        lines.extend(section_missing("simulator-in-loop"))
    if present(reports, "counterfactual"):
        lines.append(bullet("scenario_a_action_mae", get_metric(reports, "counterfactual", "scenario_a_action_mae")))
        lines.append(bullet("scenario_b_action_mae", get_metric(reports, "counterfactual", "scenario_b_action_mae")))
        lines.append(bullet("action_difference_mae", get_metric(reports, "counterfactual", "action_difference_mae")))
    else:
        lines.extend(section_missing("counterfactual"))
    return lines


def simulator_improvement(reports: Mapping[str, Any]) -> list[str]:
    if not present(reports, "sim_in_loop"):
        return section_missing("simulator-in-loop")
    return [
        bullet("simulated_count", get_metric(reports, "sim_in_loop", "simulated_count")),
        bullet("mean_initial_error_px", get_metric(reports, "sim_in_loop", "mean_initial_error_px")),
        bullet("mean_post_action_error_px", get_metric(reports, "sim_in_loop", "mean_post_action_error_px")),
        bullet("mean_error_reduction_ratio", get_metric(reports, "sim_in_loop", "mean_error_reduction_ratio")),
        bullet("median_error_reduction_ratio", get_metric(reports, "sim_in_loop", "median_error_reduction_ratio")),
        bullet("success_rate_under_2px", get_metric(reports, "sim_in_loop", "success_rate_under_2px")),
        bullet("divergence_rate", get_metric(reports, "sim_in_loop", "divergence_rate")),
    ]


def forward_prediction(reports: Mapping[str, Any]) -> list[str]:
    if not present(reports, "forward"):
        return section_missing("forward")
    return [
        bullet("count", get_metric(reports, "forward", "count")),
        bullet("centroid_x_mae_px", get_metric(reports, "forward", "centroid_x_mae_px")),
        bullet("centroid_y_mae_px", get_metric(reports, "forward", "centroid_y_mae_px")),
        bullet("centroid_euclidean_mae_px", get_metric(reports, "forward", "centroid_euclidean_mae_px")),
        bullet("sigma_x_mae_px", get_metric(reports, "forward", "sigma_x_mae_px")),
        bullet("sigma_y_mae_px", get_metric(reports, "forward", "sigma_y_mae_px")),
        bullet("peak_intensity_mae", get_metric(reports, "forward", "peak_intensity_mae")),
        bullet("delta_centroid_euclidean_mae_px", get_metric(reports, "forward", "delta_centroid_euclidean_mae_px")),
    ]


def counterfactual_consistency(reports: Mapping[str, Any]) -> list[str]:
    if not present(reports, "counterfactual"):
        return section_missing("counterfactual")
    return [
        bullet("count", get_metric(reports, "counterfactual", "count")),
        bullet("should_action_change_accuracy", get_metric(reports, "counterfactual", "should_action_change_accuracy")),
        bullet("changed_parameter_accuracy", get_metric(reports, "counterfactual", "changed_parameter_accuracy")),
        bullet("action_difference_mae", get_metric(reports, "counterfactual", "action_difference_mae")),
        bullet(
            "identical_action_rate_when_should_change",
            get_metric(reports, "counterfactual", "identical_action_rate_when_should_change"),
        ),
        bullet("by_changed_parameter", get_metric(reports, "counterfactual", "by_changed_parameter")),
    ]


def closed_loop_control(reports: Mapping[str, Any]) -> list[str]:
    if not present(reports, "closed_loop"):
        return section_missing("closed-loop")
    return [
        bullet("count", get_metric(reports, "closed_loop", "count")),
        bullet("valid_case_count", get_metric(reports, "closed_loop", "valid_case_count")),
        bullet("success_rate", get_metric(reports, "closed_loop", "success_rate")),
        bullet("mean_steps_to_success", get_metric(reports, "closed_loop", "mean_steps_to_success")),
        bullet("median_final_residual_px", get_metric(reports, "closed_loop", "median_final_residual_px")),
        bullet("mean_final_residual_px", get_metric(reports, "closed_loop", "mean_final_residual_px")),
        bullet("divergence_rate", get_metric(reports, "closed_loop", "divergence_rate")),
        bullet("oscillation_rate", get_metric(reports, "closed_loop", "oscillation_rate")),
        bullet("invalid_action_rate", get_metric(reports, "closed_loop", "invalid_action_rate")),
    ]


def ood_generalization(reports: Mapping[str, Any]) -> list[str]:
    if not present(reports, "ood"):
        return section_missing("OOD")
    lines = [
        bullet("ood_parameter", get_metric(reports, "ood", "ood_parameter")),
        bullet("ranges", get_metric(reports, "ood", "ranges")),
        bullet("counts", get_metric(reports, "ood", "counts")),
        bullet("counts_by_sample_type", get_metric(reports, "ood", "counts_by_sample_type")),
    ]
    for key in (
        "success_rate",
        "mean_error_reduction_ratio",
        "mean_final_residual_px",
        "centroid_euclidean_mae_px",
        "changed_parameter_accuracy",
    ):
        value = get_metric(reports, "ood", key)
        if value is not None:
            lines.append(bullet(f"OOD metric {key}", value))
    if all(get_metric(reports, "ood", key) is None for key in ("success_rate", "mean_error_reduction_ratio", "centroid_euclidean_mae_px")):
        lines.append("- OOD performance metrics: not available in this report; only split coverage was provided.")
    return lines


def interpretation_lines() -> list[str]:
    return [
        "- Treat simulator post-action improvement and closed-loop success as the main success metrics.",
        "- Use action MAE as a secondary diagnostic because a numerically different action can still improve the beam.",
        "- Forward prediction quality and counterfactual consistency are evidence of physics learning, especially when evaluated on held-out or OOD splits.",
        "- Missing sections mean the corresponding evaluator report was not provided; no metrics are inferred from absent reports.",
    ]


def render_markdown(reports: Mapping[str, Any]) -> str:
    sections = [
        ("JSON/Schema Reliability", json_schema_reliability(reports)),
        ("Action-Level Accuracy", action_accuracy(reports)),
        ("Simulator Post-Action Improvement", simulator_improvement(reports)),
        ("Forward Physics Prediction", forward_prediction(reports)),
        ("Counterfactual Consistency", counterfactual_consistency(reports)),
        ("Closed-Loop Control", closed_loop_control(reports)),
        ("OOD Generalization", ood_generalization(reports)),
        ("Interpretation", interpretation_lines()),
    ]
    lines = ["# Physics-Aware Optics SFT Evaluation Summary", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.extend(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    reports = load_reports(args)
    metrics_summary = {
        "reports": reports,
        "interpretation": {
            "main_success_metrics": [
                "simulator post-action improvement",
                "closed-loop success",
            ],
            "secondary_metrics": ["action MAE"],
            "physics_learning_indicators": [
                "forward prediction",
                "counterfactual consistency",
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics_summary.md").write_text(
        render_markdown(reports),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir / 'metrics_summary.json'}")
    print(f"Wrote {args.output_dir / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()
