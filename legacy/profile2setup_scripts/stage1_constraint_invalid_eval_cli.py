"""Evaluate Stage 1 constraint, invalid, and ambiguous-request behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile2setup.evaluation.param_metrics import load_tolerances
from profile2setup.schema import VARIABLE_ORDER, validate_setup_dict
from profile2setup.training.normalization import denormalize_delta_vector, load_variables_config


CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
EVAL_CATEGORIES = {"constraint", "invalid", "ambiguous_multi_intent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 constraint and invalid-request behavior."
    )
    parser.add_argument("--labels", required=True, help="Stage 1 labels JSONL")
    parser.add_argument(
        "--llm-predictions",
        default="profile2setup/results/stage1_understanding/finetuned_llm_predictions.jsonl",
        help="Fine-tuned LLM predictions JSONL, if already generated",
    )
    parser.add_argument(
        "--local-model-eval",
        required=True,
        help="Local model evaluate_cli JSON with detailed examples",
    )
    parser.add_argument("--variables-config", required=True, help="Variables YAML config")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--summary-out", required=True, help="Output Markdown path")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _clean_delta(value: Any) -> dict[str, float] | None:
    if value is None or not isinstance(value, dict):
        return None
    if not validate_setup_dict(value):
        return None
    return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}


def _direction(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "increase"
    if value < -tolerance:
        return "decrease"
    return "unchanged"


def _directions_from_delta(
    delta: dict[str, float] | None,
    tolerances: dict[str, float],
) -> dict[str, str] | None:
    if delta is None:
        return None
    return {
        name: _direction(float(delta[name]), float(tolerances[name]))
        for name in CANONICAL_VARIABLE_ORDER
    }


def _changed_from_directions(directions: dict[str, str] | None) -> list[str] | None:
    if directions is None:
        return None
    return [name for name in CANONICAL_VARIABLE_ORDER if directions.get(name) != "unchanged"]


def _normalize_direction_map(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, str] = {}
    for name in CANONICAL_VARIABLE_ORDER:
        direction = value.get(name)
        if direction not in {"increase", "decrease", "unchanged"}:
            return None
        out[name] = direction
    return out


def _parse_llm_prediction_row(row: dict) -> dict | None:
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        return prediction
    raw = row.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _llm_rows_by_id(path: Path) -> tuple[dict[str, dict], list[str]]:
    if not path.exists():
        return {}, [f"LLM predictions file not found: {path}"]
    rows = _load_jsonl(path)
    by_id: dict[str, dict] = {}
    warnings: list[str] = []
    for idx, row in enumerate(rows, start=1):
        record_id = row.get("record_id") or row.get("id")
        if not isinstance(record_id, str) or not record_id:
            warnings.append(f"LLM prediction row {idx} missing record_id")
            continue
        by_id[record_id] = row
    return by_id, warnings


def _local_examples_by_id(path: Path) -> tuple[dict[str, dict], list[str]]:
    obj = _load_json(path)
    examples = obj.get("examples")
    if not isinstance(examples, list):
        return {}, [f"local model eval missing examples list: {path}"]
    by_id: dict[str, dict] = {}
    warnings: list[str] = []
    for idx, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            warnings.append(f"local model eval example {idx} is not an object")
            continue
        record_id = example.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            warnings.append(f"local model eval example {idx} missing record_id")
            continue
        by_id[record_id] = example
    return by_id, warnings


def _local_predicted_delta(
    example: dict | None,
    variables_config: dict,
) -> tuple[dict[str, float] | None, str | None]:
    if example is None:
        return None, "missing local model eval example"
    predicted_delta_norm = example.get("predicted_delta_norm")
    if not isinstance(predicted_delta_norm, dict):
        return None, "missing predicted_delta_norm"
    try:
        return (
            denormalize_delta_vector(
                [predicted_delta_norm[name] for name in CANONICAL_VARIABLE_ORDER],
                variables_config,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - row-level diagnostic.
        return None, f"could not denormalize predicted_delta_norm: {exc}"


def _llm_prediction_summary(
    row: dict | None,
    tolerances: dict[str, float],
) -> tuple[dict[str, Any], str | None]:
    if row is None:
        return {
            "available": False,
            "valid": None,
            "rejection_reason": None,
            "changed_variables": None,
            "change_direction": None,
            "predicted_delta": None,
        }, "missing LLM prediction row"

    prediction = _parse_llm_prediction_row(row)
    if not isinstance(prediction, dict):
        return {
            "available": False,
            "valid": None,
            "rejection_reason": None,
            "changed_variables": None,
            "change_direction": None,
            "predicted_delta": None,
        }, "LLM prediction row does not contain parseable prediction JSON"

    understanding = prediction.get("setup_understanding")
    understanding = understanding if isinstance(understanding, dict) else {}
    explicit_direction = _normalize_direction_map(understanding.get("change_direction"))
    predicted_delta = _clean_delta(prediction.get("predicted_delta"))
    delta_direction = _directions_from_delta(predicted_delta, tolerances)
    change_direction = explicit_direction or delta_direction

    changed = understanding.get("changed_variables")
    if isinstance(changed, list) and all(isinstance(item, str) for item in changed):
        changed_variables = [name for name in CANONICAL_VARIABLE_ORDER if name in set(changed)]
    else:
        changed_variables = _changed_from_directions(change_direction)

    return {
        "available": True,
        "valid": prediction.get("valid"),
        "rejection_reason": prediction.get("rejection_reason"),
        "changed_variables": changed_variables,
        "change_direction": change_direction,
        "predicted_delta": predicted_delta,
    }, None


def _constraint_match(
    predicted_direction: dict[str, str] | None,
    expected_direction: dict[str, str],
) -> bool | None:
    if predicted_direction is None:
        return None
    return all(predicted_direction.get(name) == expected_direction.get(name) for name in CANONICAL_VARIABLE_ORDER)


def _fixed_violations(
    *,
    expected_fixed: list[str],
    predicted_delta: dict[str, float] | None,
    predicted_direction: dict[str, str] | None,
    tolerances: dict[str, float],
) -> tuple[int, int, dict[str, bool]]:
    results: dict[str, bool] = {}
    violations = 0
    for name in expected_fixed:
        if name not in CANONICAL_VARIABLE_ORDER:
            continue
        if predicted_delta is not None:
            violates = abs(float(predicted_delta[name])) > float(tolerances[name])
        elif predicted_direction is not None:
            violates = predicted_direction.get(name) != "unchanged"
        else:
            violates = False
        results[name] = bool(violates)
        if violates:
            violations += 1
    return violations, len(results), results


def _is_rejection(summary: dict[str, Any]) -> bool:
    valid = summary.get("valid")
    reason = summary.get("rejection_reason")
    return valid is False and isinstance(reason, str) and bool(reason.strip())


def compute_constraint_invalid_eval(
    *,
    labels_path: Path,
    llm_predictions_path: Path,
    local_model_eval_path: Path,
    variables_config_path: Path,
) -> dict:
    labels = [row for row in _load_jsonl(labels_path) if row.get("category") in EVAL_CATEGORIES]
    variables_config = load_variables_config(variables_config_path)
    tolerances = load_tolerances(variables_config)
    llm_by_id, llm_warnings = _llm_rows_by_id(llm_predictions_path)
    local_by_id, local_warnings = _local_examples_by_id(local_model_eval_path)

    llm = {
        "constraint_correct": 0,
        "constraint_total": 0,
        "fixed_violations": 0,
        "fixed_total": 0,
        "invalid_rejections": 0,
        "invalid_total": 0,
        "contradiction_detections": 0,
        "contradiction_total": 0,
    }
    local = {
        "constraint_correct": 0,
        "constraint_total": 0,
        "fixed_violations": 0,
        "fixed_total": 0,
        "invalid_forced_predictions": 0,
        "invalid_total": 0,
        "ambiguous_forced_predictions": 0,
        "ambiguous_total": 0,
    }

    records: list[dict[str, Any]] = []
    for label in labels:
        record_id = label["record_id"]
        category = label["category"]
        expected_direction = label["expected_change_direction"]
        expected_fixed = list(label.get("expected_fixed_variables") or [])

        llm_summary, llm_warning = _llm_prediction_summary(llm_by_id.get(record_id), tolerances)
        local_delta, local_warning = _local_predicted_delta(local_by_id.get(record_id), variables_config)
        local_direction = _directions_from_delta(local_delta, tolerances)
        local_changed = _changed_from_directions(local_direction)

        llm_constraint_match = None
        local_constraint_match = None
        if category == "constraint":
            llm_constraint_match = _constraint_match(llm_summary["change_direction"], expected_direction)
            if llm_constraint_match is not None:
                llm["constraint_total"] += 1
                llm["constraint_correct"] += int(llm_constraint_match)

            local_constraint_match = _constraint_match(local_direction, expected_direction)
            if local_constraint_match is not None:
                local["constraint_total"] += 1
                local["constraint_correct"] += int(local_constraint_match)

            if llm_summary["available"]:
                llm_fixed_violations, llm_fixed_total, llm_fixed_results = _fixed_violations(
                    expected_fixed=expected_fixed,
                    predicted_delta=llm_summary["predicted_delta"],
                    predicted_direction=llm_summary["change_direction"],
                    tolerances=tolerances,
                )
                llm["fixed_violations"] += llm_fixed_violations
                llm["fixed_total"] += llm_fixed_total
            else:
                llm_fixed_results = {}

            local_fixed_violations, local_fixed_total, local_fixed_results = _fixed_violations(
                expected_fixed=expected_fixed,
                predicted_delta=local_delta,
                predicted_direction=local_direction,
                tolerances=tolerances,
            )
            local["fixed_violations"] += local_fixed_violations
            local["fixed_total"] += local_fixed_total
        else:
            llm_fixed_results = {}
            local_fixed_results = {}

        llm_rejection = _is_rejection(llm_summary)
        if category == "invalid":
            llm["invalid_total"] += int(llm_summary["available"])
            llm["invalid_rejections"] += int(llm_summary["available"] and llm_rejection)
            local["invalid_total"] += 1
            local["invalid_forced_predictions"] += int(local_delta is not None)

        if category == "ambiguous_multi_intent":
            llm["contradiction_total"] += int(llm_summary["available"])
            llm["contradiction_detections"] += int(llm_summary["available"] and llm_rejection)
            local["ambiguous_total"] += 1
            local["ambiguous_forced_predictions"] += int(local_delta is not None)

        records.append(
            {
                "record_id": record_id,
                "category": category,
                "prompt": label.get("prompt"),
                "expected_changed_variables": label.get("expected_changed_variables"),
                "expected_change_direction": expected_direction,
                "expected_fixed_variables": expected_fixed,
                "expected_rejection_reason": label.get("expected_rejection_reason"),
                "llm": {
                    **llm_summary,
                    "constraint_following_correct": llm_constraint_match,
                    "fixed_variable_violations": llm_fixed_results,
                    "rejected": llm_rejection if llm_summary["available"] else None,
                    "warning": llm_warning,
                },
                "local": {
                    "native_rejection": "not_applicable_no_rejection_head",
                    "predicted_delta": local_delta,
                    "changed_variables": local_changed,
                    "change_direction": local_direction,
                    "constraint_following_correct": local_constraint_match,
                    "fixed_variable_violations": local_fixed_results,
                    "forced_numerical_prediction": local_delta is not None,
                    "warning": local_warning,
                },
            }
        )

    result = {
        "metadata": {
            "labels_path": str(labels_path),
            "llm_predictions_path": str(llm_predictions_path),
            "local_model_eval_path": str(local_model_eval_path),
            "variables_config_path": str(variables_config_path),
            "evaluated_categories": sorted(EVAL_CATEGORIES),
            "variable_order": CANONICAL_VARIABLE_ORDER,
            "local_rejection_note": (
                "The local PyTorch profile2setup model has no native rejection head by design; "
                "native rejection metrics are not applicable."
            ),
        },
        "counts": {
            "records_total": len(labels),
            "constraint": sum(1 for row in labels if row.get("category") == "constraint"),
            "invalid": sum(1 for row in labels if row.get("category") == "invalid"),
            "ambiguous_multi_intent": sum(1 for row in labels if row.get("category") == "ambiguous_multi_intent"),
            "llm_prediction_rows_loaded": len(llm_by_id),
            "local_examples_loaded": len(local_by_id),
        },
        "tolerances": {name: float(tolerances[name]) for name in CANONICAL_VARIABLE_ORDER},
        "metrics": {
            "llm": {
                "constraint_following_accuracy": _rate(llm["constraint_correct"], llm["constraint_total"]),
                "constraint_correct": llm["constraint_correct"],
                "constraint_total": llm["constraint_total"],
                "fixed_variable_violation_rate": _rate(llm["fixed_violations"], llm["fixed_total"]),
                "fixed_variable_violations": llm["fixed_violations"],
                "fixed_variable_total": llm["fixed_total"],
                "invalid_rejection_accuracy": _rate(llm["invalid_rejections"], llm["invalid_total"]),
                "invalid_rejections": llm["invalid_rejections"],
                "invalid_total": llm["invalid_total"],
                "contradiction_detection_accuracy": _rate(
                    llm["contradiction_detections"],
                    llm["contradiction_total"],
                ),
                "contradiction_detections": llm["contradiction_detections"],
                "contradiction_total": llm["contradiction_total"],
            },
            "local": {
                "constraint_following_accuracy": _rate(local["constraint_correct"], local["constraint_total"]),
                "constraint_correct": local["constraint_correct"],
                "constraint_total": local["constraint_total"],
                "fixed_variable_violation_rate": _rate(local["fixed_violations"], local["fixed_total"]),
                "fixed_variable_violations": local["fixed_violations"],
                "fixed_variable_total": local["fixed_total"],
                "native_rejection": "not_applicable_no_rejection_head",
                "invalid_forced_prediction_rate": _rate(
                    local["invalid_forced_predictions"],
                    local["invalid_total"],
                ),
                "invalid_forced_predictions": local["invalid_forced_predictions"],
                "invalid_total": local["invalid_total"],
                "ambiguous_forced_prediction_rate": _rate(
                    local["ambiguous_forced_predictions"],
                    local["ambiguous_total"],
                ),
                "ambiguous_forced_predictions": local["ambiguous_forced_predictions"],
                "ambiguous_total": local["ambiguous_total"],
            },
        },
        "warnings": llm_warnings + local_warnings,
        "records": records,
    }
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary(result: dict, path: Path) -> None:
    llm = result["metrics"]["llm"]
    local = result["metrics"]["local"]
    lines = [
        "# Stage 1C Constraint and Invalid-Request Evaluation",
        "",
        "## Purpose",
        "",
        (
            "This evaluates constraint following, invalid-request rejection, and ambiguous/contradictory "
            "prompt handling on the Stage-1 understanding benchmark."
        ),
        "",
        "## Inputs",
        "",
        f"- Labels: `{result['metadata']['labels_path']}`",
        f"- LLM predictions: `{result['metadata']['llm_predictions_path']}`",
        f"- Local model eval: `{result['metadata']['local_model_eval_path']}`",
        f"- Variables config: `{result['metadata']['variables_config_path']}`",
        "",
        "## Counts",
        "",
        f"- Constraint records: `{result['counts']['constraint']}`",
        f"- Invalid records: `{result['counts']['invalid']}`",
        f"- Ambiguous/multi-intent records: `{result['counts']['ambiguous_multi_intent']}`",
        f"- LLM prediction rows loaded: `{result['counts']['llm_prediction_rows_loaded']}`",
        f"- Local examples loaded: `{result['counts']['local_examples_loaded']}`",
        "",
        "## Metrics",
        "",
        "| Metric | LLM/API | Local PyTorch |",
        "|---|---:|---:|",
        f"| constraint_following_accuracy | {_fmt(llm['constraint_following_accuracy'])} | {_fmt(local['constraint_following_accuracy'])} |",
        f"| fixed_variable_violation_rate | {_fmt(llm['fixed_variable_violation_rate'])} | {_fmt(local['fixed_variable_violation_rate'])} |",
        f"| invalid_rejection_accuracy | {_fmt(llm['invalid_rejection_accuracy'])} | not applicable |",
        f"| invalid_forced_prediction_rate | not applicable | {_fmt(local['invalid_forced_prediction_rate'])} |",
        f"| contradiction_detection_accuracy | {_fmt(llm['contradiction_detection_accuracy'])} | not applicable |",
        f"| ambiguous_forced_prediction_rate | not applicable | {_fmt(local['ambiguous_forced_prediction_rate'])} |",
        "",
        "## Interpretation",
        "",
        (
            "- The local PyTorch model has no native rejection mechanism by design, so invalid rejection is "
            "reported as not applicable rather than a failure."
        ),
        (
            "- Local forced-prediction rates indicate whether the numerical model produced setup deltas for "
            "requests that are labeled invalid or contradictory."
        ),
        (
            "- LLM rejection metrics are only available when the LLM predictions JSONL exists and contains "
            "rows for these Stage-1 benchmark record IDs."
        ),
        "",
        "## Warnings",
        "",
    ]
    warnings = result.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No loader warnings.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = compute_constraint_invalid_eval(
        labels_path=Path(args.labels),
        llm_predictions_path=Path(args.llm_predictions),
        local_model_eval_path=Path(args.local_model_eval),
        variables_config_path=Path(args.variables_config),
    )
    _save_json(result, Path(args.out))
    write_summary(result, Path(args.summary_out))
    print(json.dumps({"out": args.out, "summary_out": args.summary_out, "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
