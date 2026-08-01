#!/usr/bin/env python3
"""Produce a paired, task-aware error taxonomy for validation runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import read_jsonl


COMPONENT_ALIASES = {
    "gaussian_source": {"gaussian_source", "source", "laser", "gaussian_laser", "gaussian_beam_source"},
    "thin_lens": {"thin_lens", "lens", "thin_lens_element"},
    "camera_sensor": {"camera_sensor", "camera", "sensor", "image_sensor"},
}
ADJUSTABLE_ALIASES = {
    "lens_x": {"lens_x", "lens_x_delta_mm", "lens_horizontal", "lens_x_offset"},
    "lens_y": {"lens_y", "lens_y_delta_mm", "lens_vertical", "lens_y_offset"},
    "camera_x": {"camera_x", "camera_x_delta_mm", "camera_horizontal", "camera_x_offset"},
    "camera_y": {"camera_y", "camera_y_delta_mm", "camera_vertical", "camera_y_offset"},
}
CAUSAL_FIELDS = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=RESULT_DIR",
        help="Named evaluator result directory; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def canonical_alias(value: Any, aliases: Mapping[str, set[str]]) -> str:
    token = normalize(value)
    for canonical, options in aliases.items():
        if token in options:
            return canonical
    return token


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def confusion(rows: Iterable[tuple[Any, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{target} -> {predicted}" for target, predicted in rows).items()))


def prediction_map(result_dir: Path) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(result_dir / "predictions.jsonl")}


def detail_map(result_dir: Path) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(result_dir / "details.jsonl")}


def setup_analysis(records: list[dict[str, Any]], predictions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    semantic_order = 0
    exact_order = 0
    semantic_adjustable = 0
    focal_ratios: list[float] = []
    distance_errors: list[float] = []
    for record in records:
        target = record["target"]["answer"]
        parsed = predictions[record["example_id"]].get("parsed_json") or {}
        answer = parsed.get("answer") if isinstance(parsed.get("answer"), dict) else {}
        predicted_order = answer.get("component_order")
        if isinstance(predicted_order, list):
            exact_order += predicted_order == target["component_order"]
            mapped = [canonical_alias(value, COMPONENT_ALIASES) for value in predicted_order]
            semantic_order += mapped == target["component_order"]
        predicted_adjustable = answer.get("adjustable_parameters")
        if isinstance(predicted_adjustable, list):
            mapped = {canonical_alias(value, ADJUSTABLE_ALIASES) for value in predicted_adjustable}
            semantic_adjustable += mapped == set(target["adjustable_parameters"])
        focal = answer.get("lens_focal_length_m")
        if finite(focal):
            focal_ratios.append(float(focal) / float(target["lens_focal_length_m"]))
        distance = answer.get("total_source_to_sensor_mm")
        if finite(distance):
            distance_errors.append(float(distance) - float(target["total_source_to_sensor_mm"]))
    count = len(records)
    ratio_bins = Counter()
    for ratio in focal_ratios:
        if abs(ratio - 1.0) <= 0.01:
            ratio_bins["correct_scale"] += 1
        elif abs(ratio - 10.0) <= 0.1:
            ratio_bins["ten_times_too_large"] += 1
        elif abs(ratio - 0.1) <= 0.01:
            ratio_bins["ten_times_too_small"] += 1
        else:
            ratio_bins["other_scale"] += 1
    return {
        "count": count,
        "component_order_exact_rate": exact_order / count if count else None,
        "component_order_semantic_alias_rate": semantic_order / count if count else None,
        "adjustable_set_semantic_alias_rate": semantic_adjustable / count if count else None,
        "focal_scale_categories": dict(sorted(ratio_bins.items())),
        "focal_ratio_median": statistics.median(focal_ratios) if focal_ratios else None,
        "distance_signed_error_mm_mean": safe_mean(distance_errors),
        "distance_abs_error_mm_mean": safe_mean(abs(value) for value in distance_errors),
    }


def status_analysis(records: list[dict[str, Any]], predictions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for record in records:
        parsed = predictions[record["example_id"]].get("parsed_json")
        pairs.append((record["target"]["status"], parsed.get("status") if isinstance(parsed, dict) else None))
    return {"count": len(records), "status_confusion": confusion(pairs)}


def causal_analysis(records: list[dict[str, Any]], predictions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(records), "per_field_confusion": {}}
    for field in CAUSAL_FIELDS:
        pairs = []
        for record in records:
            target = record["target"]["answer"]["effects"][field]
            parsed = predictions[record["example_id"]].get("parsed_json") or {}
            answer = parsed.get("answer") if isinstance(parsed.get("answer"), dict) else {}
            effects = answer.get("effects") if isinstance(answer.get("effects"), dict) else {}
            pairs.append((target, effects.get(field)))
        result["per_field_confusion"][field] = confusion(pairs)
    return result


def forward_analysis(
    records: list[dict[str, Any]], predictions: Mapping[str, dict[str, Any]], details: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    consistency_centroid: list[float] = []
    consistency_sigma: list[float] = []
    consistency_peak: list[float] = []
    for record in records:
        parsed = predictions[record["example_id"]].get("parsed_json") or {}
        answer = parsed.get("answer") if isinstance(parsed.get("answer"), dict) else {}
        after = answer.get("after_state") if isinstance(answer.get("after_state"), dict) else {}
        change = answer.get("change") if isinstance(answer.get("change"), dict) else {}
        current = record["prompt_inputs"].get("current_observation")
        if not isinstance(current, dict):
            continue
        if all(finite(after.get(key)) and finite(change.get(key)) for key in ("centroid_x_px", "centroid_y_px")):
            dx = float(after["centroid_x_px"]) - float(current["centroid_x_px"]) - float(change["centroid_x_px"])
            dy = float(after["centroid_y_px"]) - float(current["centroid_y_px"]) - float(change["centroid_y_px"])
            consistency_centroid.append(math.hypot(dx, dy))
        if all(finite(after.get(key)) and finite(change.get(key)) for key in ("sigma_x_px", "sigma_y_px")):
            consistency_sigma.extend(
                abs(float(after[key]) - float(current[key]) - float(change[key]))
                for key in ("sigma_x_px", "sigma_y_px")
            )
        if finite(after.get("peak_intensity")) and finite(change.get("peak_intensity")):
            consistency_peak.append(
                abs(
                    float(after["peak_intensity"])
                    - float(current["peak_intensity"])
                    - float(change["peak_intensity"])
                )
            )
    keys = (
        "after_centroid_within_2px",
        "after_sigma_within_2px",
        "after_peak_within_5pct",
        "change_centroid_within_2px",
        "change_sigma_within_2px",
        "change_peak_within_5pct",
    )
    return {
        "count": len(records),
        "tolerance_pass_rates": {
            key: safe_mean(float(details[row["example_id"]].get(key, 0.0)) for row in records) for key in keys
        },
        "self_consistency": {
            "centroid_error_px_mean": safe_mean(consistency_centroid),
            "centroid_within_0_1px_rate": safe_mean(float(value <= 0.1) for value in consistency_centroid),
            "sigma_abs_error_px_mean": safe_mean(consistency_sigma),
            "peak_abs_error_mean": safe_mean(consistency_peak),
        },
    }


def control_analysis(records: list[dict[str, Any]], predictions: Mapping[str, dict[str, Any]], details: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    base = status_analysis(records, predictions)
    feasible = [row for row in records if row["target"]["status"] == "feasible"]
    infeasible = [row for row in records if row["target"]["status"] == "infeasible_within_limits"]
    base.update(
        {
            "feasible_count": len(feasible),
            "infeasible_count": len(infeasible),
            "feasible_status_recall": safe_mean(
                float((predictions[row["example_id"]].get("parsed_json") or {}).get("status") == "feasible")
                for row in feasible
            ),
            "infeasible_status_recall": safe_mean(
                float(
                    (predictions[row["example_id"]].get("parsed_json") or {}).get("status")
                    == "infeasible_within_limits"
                )
                for row in infeasible
            ),
            "feasible_simulator_success_rate": safe_mean(
                float(details[row["example_id"]].get("simulator_outcome_success", 0.0)) for row in feasible
            ),
            "feasible_minimum_motion_rate": safe_mean(
                float(details[row["example_id"]].get("minimum_motion_optimal", 0.0)) for row in feasible
            ),
        }
    )
    return base


def run_analysis(
    records: list[dict[str, Any]], predictions: Mapping[str, dict[str, Any]], details: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_type"]].append(record)
    common = {
        "record_count": len(records),
        "invalid_json_count": sum(not details[row["example_id"]]["json_valid"] for row in records),
        "invalid_schema_count": sum(not details[row["example_id"]]["schema_valid"] for row in records),
    }
    return {
        "common": common,
        "setup_interpretation": setup_analysis(by_task["setup_interpretation"], predictions),
        "information_sufficiency": status_analysis(by_task["information_sufficiency"], predictions),
        "causal_effects": causal_analysis(by_task["causal_effects"], predictions),
        "forward_prediction": forward_analysis(by_task["forward_prediction"], predictions, details),
        "diagnosis": status_analysis(by_task["diagnosis"], predictions),
        "constrained_intervention": control_analysis(by_task["constrained_intervention"], predictions, details),
        "counterfactual_reasoning": {
            **status_analysis(by_task["counterfactual_reasoning"], predictions),
            "changed_parameter_exact_rate": safe_mean(
                float(details[row["example_id"]].get("changed_parameter_exact", 0.0))
                for row in by_task["counterfactual_reasoning"]
            ),
            "direction_exact_rate": safe_mean(
                float(details[row["example_id"]].get("direction_exact", 0.0))
                for row in by_task["counterfactual_reasoning"]
            ),
        },
    }


def markdown(result: Mapping[str, Any]) -> str:
    lines = ["# Validation error taxonomy", ""]
    for name, run in result["runs"].items():
        common = run["common"]
        setup = run["setup_interpretation"]
        control = run["constrained_intervention"]
        forward = run["forward_prediction"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"Invalid JSON/schema: {common['invalid_json_count']} / {common['invalid_schema_count']} of {common['record_count']}.",
                f"Setup component order: exact {setup['component_order_exact_rate']:.1%}, semantic aliases {setup['component_order_semantic_alias_rate']:.1%}.",
                f"Setup adjustable-set semantic match: {setup['adjustable_set_semantic_alias_rate']:.1%}; median focal scale ratio: {setup['focal_ratio_median']:.3f}.",
                f"Forward centroid self-consistency within 0.1 px: {forward['self_consistency']['centroid_within_0_1px_rate']:.1%}.",
                f"Control feasible status recall/simulator success: {control['feasible_status_recall']:.1%} / {control['feasible_simulator_success_rate']:.1%}.",
                "",
            ]
        )
    lines.extend(
        [
            "The semantic alias metric is diagnostic only and does not alter the frozen v1 checkpoint-selection score.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records_jsonl)
    runs: dict[str, Any] = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run must be NAME=RESULT_DIR: {spec}")
        name, directory = spec.split("=", 1)
        result_dir = Path(directory)
        runs[name] = run_analysis(records, prediction_map(result_dir), detail_map(result_dir))
    result = {"records": str(args.records_jsonl), "runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "error_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "error_analysis.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({name: data["common"] for name, data in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
