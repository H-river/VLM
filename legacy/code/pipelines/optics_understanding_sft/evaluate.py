#!/usr/bin/env python3
"""Score model predictions for the optics-understanding pilot benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import read_jsonl
from .run_inference import extract_json_object


TASKS = (
    "setup_interpretation",
    "information_sufficiency",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "constrained_intervention",
    "counterfactual_reasoning",
)
STATE_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "peak_intensity",
    "sigma_x_px",
    "sigma_y_px",
)
ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
ALLOWED_STATUSES = {
    "setup_interpretation": {"answerable"},
    "information_sufficiency": {"answerable", "insufficient_information"},
    "causal_effects": {"answerable"},
    "forward_prediction": {"answerable"},
    "diagnosis": {"unique", "ambiguous", "unsupported"},
    "constrained_intervention": {"feasible", "infeasible_within_limits"},
    "counterfactual_reasoning": {"answerable"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rubric-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 preserves the published pilot score; v2 fixes rubric artifacts for subsequent experiments.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Evaluate the intersection instead of requiring one prediction per record.",
    )
    return parser.parse_args()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_float(value: Any) -> float | None:
    return float(value) if finite_number(value) else None


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def set_f1(predicted: Any, target: Any) -> float:
    if not isinstance(predicted, list) or not isinstance(target, list):
        return 0.0
    pred_set = {str(value) for value in predicted}
    target_set = {str(value) for value in target}
    if not pred_set and not target_set:
        return 1.0
    if not pred_set or not target_set:
        return 0.0
    overlap = len(pred_set & target_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(target_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def vector_error(predicted: Mapping[str, Any], target: Mapping[str, Any], x: str, y: str) -> float | None:
    px, py = safe_float(predicted.get(x)), safe_float(predicted.get(y))
    tx, ty = safe_float(target.get(x)), safe_float(target.get(y))
    if None in (px, py, tx, ty):
        return None
    return math.hypot(px - tx, py - ty)


def abs_error(predicted: Mapping[str, Any], target: Mapping[str, Any], field: str) -> float | None:
    pv, tv = safe_float(predicted.get(field)), safe_float(target.get(field))
    return abs(pv - tv) if pv is not None and tv is not None else None


def relative_or_absolute_ok(predicted: Any, target: Any, *, relative: float, absolute: float = 1e-4) -> bool:
    pv, tv = safe_float(predicted), safe_float(target)
    if pv is None or tv is None:
        return False
    return abs(pv - tv) <= max(absolute, relative * max(abs(tv), absolute))


def schema_valid(task: str, prediction: Any) -> tuple[bool, str | None]:
    if not isinstance(prediction, dict):
        return False, "prediction is not an object"
    status = prediction.get("status")
    if status not in ALLOWED_STATUSES[task]:
        return False, f"invalid status {status!r}"
    if not isinstance(prediction.get("answer"), dict):
        return False, "answer is not an object"
    answer = prediction["answer"]
    if task == "setup_interpretation":
        valid = (
            isinstance(answer.get("component_order"), list)
            and isinstance(answer.get("adjustable_parameters"), list)
            and finite_number(answer.get("total_source_to_sensor_mm"))
            and finite_number(answer.get("lens_focal_length_m"))
        )
    elif task == "information_sufficiency":
        valid = isinstance(answer.get("missing_fields"), list)
        if status == "answerable":
            valid = valid and answer.get("centroid_x_direction") in {"increase", "decrease", "no_change"}
        else:
            valid = valid and isinstance(answer.get("nonidentifiable_output"), str)
    elif task == "causal_effects":
        effects = answer.get("effects")
        valid = isinstance(effects, dict) and all(
            effects.get(field) in {"increase", "decrease", "no_change"}
            for field in ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")
        )
    elif task == "forward_prediction":
        valid = all(
            isinstance(answer.get(section), dict)
            and all(finite_number(answer[section].get(field)) for field in STATE_FIELDS)
            for section in ("after_state", "change")
        )
    elif task == "diagnosis":
        valid = isinstance(answer.get("plausible_causes"), list)
    elif task == "constrained_intervention":
        plan = answer.get("control_plan")
        if status == "feasible":
            valid = isinstance(plan, dict) and all(finite_number(plan.get(field)) for field in ACTION_FIELDS)
            valid = valid and finite_number(answer.get("expected_residual_px"))
        else:
            valid = plan is None and finite_number(answer.get("best_achievable_residual_px"))
    elif task == "counterfactual_reasoning":
        valid = (
            isinstance(answer.get("changed_parameter"), str)
            and isinstance(answer.get("centroid_direction_preserved"), bool)
            and all(
                isinstance(answer.get(section), dict)
                and all(finite_number(answer[section].get(field)) for field in STATE_FIELDS)
                for section in ("response_a", "response_b", "response_difference")
            )
        )
    else:
        valid = False
    if not valid:
        return False, "answer does not match task schema"
    return True, None


def score_setup(pred: Mapping[str, Any], target: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    pa, ta = pred["answer"], target["answer"]
    total_error = abs_error(pa, ta, "total_source_to_sensor_mm")
    focal_error = abs_error(pa, ta, "lens_focal_length_m")
    components = float(pa.get("component_order") == ta.get("component_order"))
    adjustable = set_f1(pa.get("adjustable_parameters"), ta.get("adjustable_parameters"))
    total_ok = float(total_error is not None and total_error <= 0.1)
    focal_ok = float(focal_error is not None and focal_error <= 1e-6)
    return {
        "task_score": mean([components, adjustable, total_ok, focal_ok]),
        "component_order_exact": components,
        "adjustable_parameters_f1": adjustable,
        "total_distance_abs_error_mm": total_error,
        "focal_length_abs_error_m": focal_error,
    }


def normalized_token(value: Any) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token


def semantic_component_order(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    aliases = {
        "gaussian_source": {"gaussian_source", "source", "laser", "gaussian_laser", "gaussian_beam_source"},
        "thin_lens": {"thin_lens", "lens", "thin_lens_element"},
        "camera_sensor": {"camera_sensor", "camera", "sensor", "image_sensor"},
    }
    result = []
    for item in value:
        token = normalized_token(item)
        result.append(next((canonical for canonical, choices in aliases.items() if token in choices), token))
    return result


def score_setup_v2(pred: Mapping[str, Any], target: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Score prompt-identifiable setup semantics; keep hidden adjustability diagnostic only."""
    original = score_setup(pred, target, context)
    pa, ta = pred["answer"], target["answer"]
    semantic = float(semantic_component_order(pa.get("component_order")) == ta.get("component_order"))
    total_error = original["total_distance_abs_error_mm"]
    focal_error = original["focal_length_abs_error_m"]
    total_ok = float(total_error is not None and total_error <= 0.1)
    focal_ok = float(focal_error is not None and focal_error <= 1e-6)
    return {
        **original,
        "task_score": mean([semantic, total_ok, focal_ok]),
        "component_order_semantic_exact": semantic,
        "adjustable_parameters_excluded_as_prompt_hidden": 1.0,
    }


def score_sufficiency(pred: Mapping[str, Any], target: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    pa, ta = pred["answer"], target["answer"]
    status = float(pred.get("status") == target.get("status"))
    missing_f1 = set_f1(pa.get("missing_fields"), ta.get("missing_fields"))
    if target["status"] == "answerable":
        secondary = float(pa.get("centroid_x_direction") == ta.get("centroid_x_direction"))
    else:
        secondary = float(pa.get("nonidentifiable_output") == ta.get("nonidentifiable_output"))
    answer_score = mean([missing_f1, secondary])
    return {
        "task_score": 0.6 * status + 0.4 * float(answer_score),
        "status_exact": status,
        "missing_fields_f1": missing_f1,
        "secondary_exact": secondary,
    }


def score_causal(pred: Mapping[str, Any], target: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    pe = pred["answer"].get("effects", {})
    te = target["answer"]["effects"]
    fields = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")
    correctness = {field: float(isinstance(pe, dict) and pe.get(field) == te[field]) for field in fields}
    return {"task_score": mean(correctness.values()), "effect_accuracy": mean(correctness.values()), **correctness}


def state_metrics(predicted: Any, target: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    predicted = predicted if isinstance(predicted, dict) else {}
    centroid_error = vector_error(predicted, target, "centroid_x_px", "centroid_y_px")
    sigma_x_error = abs_error(predicted, target, "sigma_x_px")
    sigma_y_error = abs_error(predicted, target, "sigma_y_px")
    intensity_error = abs_error(predicted, target, "peak_intensity")
    return {
        f"{prefix}_centroid_error_px": centroid_error,
        f"{prefix}_sigma_x_abs_error_px": sigma_x_error,
        f"{prefix}_sigma_y_abs_error_px": sigma_y_error,
        f"{prefix}_peak_abs_error": intensity_error,
        f"{prefix}_centroid_within_2px": float(centroid_error is not None and centroid_error <= 2.0),
        f"{prefix}_sigma_within_2px": float(
            sigma_x_error is not None and sigma_y_error is not None and sigma_x_error <= 2.0 and sigma_y_error <= 2.0
        ),
        f"{prefix}_peak_within_5pct": float(
            relative_or_absolute_ok(predicted.get("peak_intensity"), target.get("peak_intensity"), relative=0.05)
        ),
    }


def score_forward(pred: Mapping[str, Any], target: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    pa, ta = pred["answer"], target["answer"]
    after = state_metrics(pa.get("after_state"), ta["after_state"], "after")
    change = state_metrics(pa.get("change"), ta["change"], "change")
    predicted_change = pa.get("change") if isinstance(pa.get("change"), dict) else {}
    change_peak_ok = float(
        abs_error(predicted_change, ta["change"], "peak_intensity") is not None
        and abs_error(predicted_change, ta["change"], "peak_intensity")
        <= max(1e-4, 0.05 * abs(float(ta["after_state"]["peak_intensity"])))
    )
    change["change_peak_within_5pct"] = change_peak_ok
    success = [
        after["after_centroid_within_2px"],
        after["after_sigma_within_2px"],
        after["after_peak_within_5pct"],
        change["change_centroid_within_2px"],
        change["change_sigma_within_2px"],
        change["change_peak_within_5pct"],
    ]
    return {"task_score": mean(success), **after, **change}


def score_diagnosis(pred: Mapping[str, Any], target: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    status = float(pred.get("status") == target.get("status"))
    cause_f1 = set_f1(pred["answer"].get("plausible_causes"), target["answer"].get("plausible_causes"))
    return {"task_score": 0.5 * status + 0.5 * cause_f1, "status_exact": status, "plausible_causes_f1": cause_f1}


def cached_control_state(private: Mapping[str, Any], allowed_index: int) -> Mapping[str, Any] | None:
    name = f"grid_{allowed_index}"
    for spec in private.get("replay_specs", []):
        if spec.get("name") == name:
            return spec.get("expected_state")
    return None


def score_control(pred: Mapping[str, Any], target: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    status_exact = float(pred.get("status") == target.get("status"))
    inputs = context["record"]["prompt_inputs"]
    constraints = inputs["actuator_constraints"]
    active = constraints["active_actuator"]
    allowed = [float(value) for value in constraints["allowed_values_mm"]]
    tolerance = float(constraints["success_tolerance_px"])
    plan = pred["answer"].get("control_plan")
    grid_valid = 0.0
    outcome_success = 0.0
    optimality = 0.0
    residual: float | None = None
    movement_gap: float | None = None
    if pred.get("status") == "infeasible_within_limits":
        grid_valid = float(plan is None)
        outcome_success = float(target.get("status") == "infeasible_within_limits")
        optimality = outcome_success
    elif isinstance(plan, dict) and all(finite_number(plan.get(field)) for field in ACTION_FIELDS):
        inactive_ok = all(abs(float(plan[field])) <= 5e-7 for field in ACTION_FIELDS if field != active)
        value = float(plan[active])
        nearest_index = min(range(len(allowed)), key=lambda idx: abs(value - allowed[idx]))
        active_ok = abs(value - allowed[nearest_index]) <= 1e-6
        grid_valid = float(active_ok and inactive_ok)
        if grid_valid:
            state = cached_control_state(context["private_eval"], nearest_index)
            if state:
                target_obs = inputs["target_observation"]
                residual = vector_error(state, target_obs, "centroid_x_px", "centroid_y_px")
                outcome_success = float(residual is not None and residual <= tolerance)
                true_plan = target["answer"].get("control_plan")
                if isinstance(true_plan, dict):
                    movement_gap = abs(value) - abs(float(true_plan[active]))
                    optimality = float(outcome_success and movement_gap <= 1e-6)
    return {
        "task_score": 0.3 * status_exact + 0.2 * grid_valid + 0.3 * outcome_success + 0.2 * optimality,
        "status_exact": status_exact,
        "action_grid_valid": grid_valid,
        "simulator_outcome_success": outcome_success,
        "minimum_motion_optimal": optimality,
        "simulator_residual_px": residual,
        "movement_gap_mm": movement_gap,
    }


def score_control_v2(pred: Mapping[str, Any], target: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Gate downstream control credit on the correct feasibility branch."""
    original = score_control(pred, target, context)
    status_exact = float(pred.get("status") == target.get("status"))
    if not status_exact:
        return {**original, "task_score": 0.0, "branch_gated": 1.0}
    answer = pred["answer"]
    if target["status"] == "feasible":
        task_score = mean(
            [
                status_exact,
                float(original["action_grid_valid"]),
                float(original["simulator_outcome_success"]),
                float(original["minimum_motion_optimal"]),
            ]
        )
        best_residual_ok = None
    else:
        plan_null = float(answer.get("control_plan") is None)
        predicted_best = safe_float(answer.get("best_achievable_residual_px"))
        target_best = safe_float(target["answer"].get("best_achievable_residual_px"))
        best_residual_ok = float(
            predicted_best is not None and target_best is not None and abs(predicted_best - target_best) <= 2.0
        )
        task_score = 0.4 * status_exact + 0.2 * plan_null + 0.4 * best_residual_ok
    return {
        **original,
        "task_score": task_score,
        "branch_gated": 1.0,
        "best_achievable_within_2px": best_residual_ok,
    }


def score_counterfactual(pred: Mapping[str, Any], target: Mapping[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
    pa, ta = pred["answer"], target["answer"]
    changed = float(pa.get("changed_parameter") == ta.get("changed_parameter"))
    direction = float(pa.get("centroid_direction_preserved") is ta.get("centroid_direction_preserved"))
    diff = state_metrics(pa.get("response_difference"), ta["response_difference"], "difference")
    predicted_difference = pa.get("response_difference") if isinstance(pa.get("response_difference"), dict) else {}
    peak_error = abs_error(predicted_difference, ta["response_difference"], "peak_intensity")
    peak_scale = max(
        abs(float(ta["response_a"]["peak_intensity"])),
        abs(float(ta["response_b"]["peak_intensity"])),
        1e-4,
    )
    diff["difference_peak_within_5pct"] = float(peak_error is not None and peak_error <= 0.05 * peak_scale)
    diff_score = mean(
        [
            diff["difference_centroid_within_2px"],
            diff["difference_sigma_within_2px"],
            diff["difference_peak_within_5pct"],
        ]
    )
    return {"task_score": mean([changed, direction, float(diff_score)]), "changed_parameter_exact": changed, "direction_exact": direction, **diff}


def score_counterfactual_v2(pred: Mapping[str, Any], target: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    original = score_counterfactual(pred, target, context)
    numeric = mean(
        [
            original["difference_centroid_within_2px"],
            original["difference_sigma_within_2px"],
            original["difference_peak_within_5pct"],
        ]
    )
    return {
        **original,
        "task_score": 0.2 * original["changed_parameter_exact"] + 0.2 * original["direction_exact"] + 0.6 * float(numeric),
        "numeric_difference_score": numeric,
    }


SCORERS = {
    "setup_interpretation": score_setup,
    "information_sufficiency": score_sufficiency,
    "causal_effects": score_causal,
    "forward_prediction": score_forward,
    "diagnosis": score_diagnosis,
    "constrained_intervention": score_control,
    "counterfactual_reasoning": score_counterfactual,
}

SCORERS_V2 = {
    **SCORERS,
    "setup_interpretation": score_setup_v2,
    "constrained_intervention": score_control_v2,
    "counterfactual_reasoning": score_counterfactual_v2,
}


def private_index(master_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case in master_rows:
        for item in case["records"]:
            record = item["record"]
            result[record["example_id"]] = {
                "private_eval": item["private_eval"],
                "distribution": case["distribution"],
                "ood_parameter": case.get("ood_parameter"),
            }
    return result


def prediction_json(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    parsed = row.get("parsed_json")
    if isinstance(parsed, dict):
        return parsed, None
    return extract_json_object(str(row.get("raw_prediction_text", "")))


def evaluate_rows(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    master_rows: list[dict[str, Any]],
    *,
    allow_partial: bool = False,
    rubric_version: str = "v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if rubric_version not in {"v1", "v2"}:
        raise ValueError(f"unsupported rubric_version: {rubric_version}")
    prediction_map = {row["example_id"]: row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise ValueError("duplicate prediction example_id")
    record_ids = {row["example_id"] for row in records}
    unknown = sorted(set(prediction_map) - record_ids)
    if unknown:
        raise ValueError(f"predictions contain {len(unknown)} unknown example_id values")
    missing = sorted(record_ids - set(prediction_map))
    if missing and not allow_partial:
        raise ValueError(f"missing predictions for {len(missing)} records")
    private = private_index(master_rows)
    details: list[dict[str, Any]] = []
    for record in records:
        example_id = record["example_id"]
        if example_id not in prediction_map:
            continue
        pred_row = prediction_map[example_id]
        parsed, parse_error = prediction_json(pred_row)
        json_valid = parsed is not None
        valid, schema_error = schema_valid(record["task_type"], parsed) if json_valid else (False, parse_error)
        envelope_valid = (
            isinstance(parsed, dict)
            and parsed.get("status") in ALLOWED_STATUSES[record["task_type"]]
            and isinstance(parsed.get("answer"), dict)
        )
        base = {
            "example_id": example_id,
            "group_id": record["group_id"],
            "task_type": record["task_type"],
            "modality": record["modality"],
            "distribution": private[example_id]["distribution"],
            "json_valid": json_valid,
            "schema_valid": valid,
            "error": schema_error,
            "target_status": record["target"]["status"],
            "predicted_status": parsed.get("status") if isinstance(parsed, dict) else None,
            "latency_seconds": pred_row.get("latency_seconds"),
            "input_tokens": pred_row.get("input_tokens"),
            "output_tokens": pred_row.get("output_tokens"),
            "peak_cuda_memory_bytes": pred_row.get("peak_cuda_memory_bytes"),
        }
        if envelope_valid:
            context = {"record": record, **private[example_id]}
            try:
                scorers = SCORERS_V2 if rubric_version == "v2" else SCORERS
                metrics = scorers[record["task_type"]](parsed, record["target"], context)
            except (KeyError, TypeError, ValueError) as exc:
                base["schema_valid"] = False
                base["error"] = f"malformed task answer: {exc}"
                metrics = {"task_score": 0.0}
        else:
            metrics = {"task_score": 0.0}
        details.append({**base, **metrics})
    summary = summarize(details, expected=len(records), missing=len(missing))
    summary["rubric_version"] = rubric_version
    return details, summary


def summarize(details: list[dict[str, Any]], *, expected: int, missing: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_distribution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_task[row["task_type"]].append(row)
        by_modality[row["modality"]].append(row)
        by_distribution[row["distribution"]].append(row)
    task_summary = {}
    for task in TASKS:
        rows = by_task.get(task, [])
        numeric_metrics: dict[str, Any] = {}
        keys = sorted({key for row in rows for key, value in row.items() if finite_number(value)})
        for key in keys:
            if key in {"input_tokens", "output_tokens", "latency_seconds", "peak_cuda_memory_bytes"}:
                continue
            numeric_metrics[key] = mean(float(row[key]) for row in rows if finite_number(row.get(key)))
        if len(ALLOWED_STATUSES[task]) > 1 and rows:
            f1_values = []
            for label in sorted(ALLOWED_STATUSES[task]):
                tp = sum(row["target_status"] == label and row["predicted_status"] == label for row in rows)
                fp = sum(row["target_status"] != label and row["predicted_status"] == label for row in rows)
                fn = sum(row["target_status"] == label and row["predicted_status"] != label for row in rows)
                denominator = 2 * tp + fp + fn
                f1_values.append((2 * tp / denominator) if denominator else 0.0)
            numeric_metrics["status_macro_f1"] = mean(f1_values)
        task_summary[task] = {
            "count": len(rows),
            "json_valid_rate": mean(float(row["json_valid"]) for row in rows),
            "schema_valid_rate": mean(float(row["schema_valid"]) for row in rows),
            **numeric_metrics,
        }
    task_scores = [task_summary[task].get("task_score") for task in TASKS]
    macro = mean(float(value) for value in task_scores if value is not None)
    def group_scores(groups: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            key: {
                "count": len(rows),
                "task_score": mean(float(row["task_score"]) for row in rows),
                "json_valid_rate": mean(float(row["json_valid"]) for row in rows),
                "schema_valid_rate": mean(float(row["schema_valid"]) for row in rows),
            }
            for key, rows in sorted(groups.items())
        }
    task_modality = {
        task: group_scores(
            {
                modality: [row for row in rows if row["modality"] == modality]
                for modality in sorted({row["modality"] for row in rows})
            }
        )
        for task, rows in sorted(by_task.items())
    }
    latency_values = [float(row["latency_seconds"]) for row in details if finite_number(row.get("latency_seconds"))]
    token_in = [float(row["input_tokens"]) for row in details if finite_number(row.get("input_tokens"))]
    token_out = [float(row["output_tokens"]) for row in details if finite_number(row.get("output_tokens"))]
    peak = [float(row["peak_cuda_memory_bytes"]) for row in details if finite_number(row.get("peak_cuda_memory_bytes"))]
    return {
        "expected_records": expected,
        "evaluated_records": len(details),
        "missing_predictions": missing,
        "json_valid_rate": mean(float(row["json_valid"]) for row in details),
        "schema_valid_rate": mean(float(row["schema_valid"]) for row in details),
        "macro_task_score": macro,
        "per_task": task_summary,
        "by_modality": group_scores(by_modality),
        "by_task_modality": task_modality,
        "by_distribution": group_scores(by_distribution),
        "runtime": {
            "total_latency_seconds": sum(latency_values),
            "mean_latency_seconds": mean(latency_values),
            "mean_input_tokens": mean(token_in),
            "mean_output_tokens": mean(token_out),
            "peak_cuda_memory_bytes": max(peak) if peak else None,
        },
        "target_status_counts": dict(sorted(Counter(row["target_status"] for row in details).items())),
        "predicted_status_counts": dict(sorted(Counter(str(row["predicted_status"]) for row in details).items())),
    }


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Optics understanding evaluation",
        "",
        f"Rubric: **{summary.get('rubric_version', 'v1')}**.",
        "",
        f"Evaluated **{summary['evaluated_records']} / {summary['expected_records']}** records.",
        f"Macro task score: **{summary['macro_task_score']:.3f}**",
        f"Valid JSON: **{summary['json_valid_rate']:.1%}**; valid schema: **{summary['schema_valid_rate']:.1%}**.",
        "",
        "| Task | N | Score | JSON | Schema |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = summary["per_task"][task]
        score = item.get("task_score")
        lines.append(
            f"| {task} | {item['count']} | {score:.3f} | {item['json_valid_rate']:.1%} | {item['schema_valid_rate']:.1%} |"
            if score is not None
            else f"| {task} | 0 | n/a | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "| Modality | N | Score | JSON | Schema |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for modality, item in summary["by_modality"].items():
        lines.append(
            f"| {modality} | {item['count']} | {item['task_score']:.3f} | "
            f"{item['json_valid_rate']:.1%} | {item['schema_valid_rate']:.1%} |"
        )
    runtime = summary["runtime"]
    lines.extend(
        [
            "",
            f"Total generation latency: **{runtime['total_latency_seconds']:.1f} s**; "
            f"mean: **{runtime['mean_latency_seconds']:.2f} s/record**.",
            f"Mean tokens: **{runtime['mean_input_tokens']:.1f} input / {runtime['mean_output_tokens']:.1f} output**. "
            f"Peak allocated CUDA memory: **{runtime['peak_cuda_memory_bytes'] / (1024 ** 3):.2f} GiB**.",
            "",
            "Scores are normalized to 0-1. Control success is checked against simulator-cached exhaustive grid states.",
            "",
        ]
    )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    details, summary = evaluate_rows(
        read_jsonl(args.records_jsonl),
        read_jsonl(args.predictions_jsonl),
        read_jsonl(args.master_jsonl),
        allow_partial=args.allow_partial,
        rubric_version=args.rubric_version,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"evaluated": len(details), "macro_task_score": summary["macro_task_score"]}, indent=2))


if __name__ == "__main__":
    main()
