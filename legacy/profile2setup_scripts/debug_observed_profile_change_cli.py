"""Debug observed_profile_change metric behavior for LLM/API predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile2setup.evaluation.llm_api_eval import (
    _observed_profile_change,
    _validate_prediction,
    normalize_observed_profile_label,
)


OBSERVED_FIELDS = [
    "centroid_x",
    "centroid_y",
    "beam_width_x",
    "beam_width_y",
    "peak_intensity",
    "total_intensity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug observed_profile_change accuracy.")
    parser.add_argument("--predictions", required=True, help="LLM/API predictions JSONL")
    parser.add_argument("--data", required=True, help="Original data JSONL")
    parser.add_argument("--eval-json", default=None, help="Existing LLM eval JSON, optional")
    parser.add_argument("--out", required=True, help="Output debug JSON")
    parser.add_argument("--summary-out", required=True, help="Output debug Markdown")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(obj)
    return rows


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _parse_prediction(row: dict) -> dict | None:
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


def _classify_mismatch(field: str, predicted: Any, target: Any) -> str:
    if predicted is None:
        return "missing_field"
    if predicted == target:
        return "match"
    normalized_pred = normalize_observed_profile_label(field, predicted)
    normalized_target = normalize_observed_profile_label(field, target)
    if normalized_pred == normalized_target:
        return "vocabulary_mismatch"
    if field in {"centroid_x", "centroid_y"} and str(normalized_pred).startswith("moves_"):
        return "coordinate_or_direction_mismatch"
    return "semantic_or_model_mismatch"


def build_debug(predictions_path: Path, data_path: Path, eval_json_path: Path | None) -> dict:
    predictions = _load_jsonl(predictions_path)
    data_rows = _load_jsonl(data_path)
    data_by_id = {row.get("id"): row for row in data_rows if row.get("id")}
    eval_json = _load_json(eval_json_path) if eval_json_path is not None else None

    rows = []
    raw_correct = 0
    raw_total = 0
    normalized_correct = 0
    normalized_total = 0
    all_valid_json_raw_correct = 0
    all_valid_json_raw_total = 0
    all_valid_json_normalized_correct = 0
    all_valid_json_normalized_total = 0
    cause_counts: dict[str, int] = {}
    all_valid_json_cause_counts: dict[str, int] = {}
    predicted_value_counts: dict[str, dict[str, int]] = {field: {} for field in OBSERVED_FIELDS}
    target_value_counts: dict[str, dict[str, int]] = {field: {} for field in OBSERVED_FIELDS}

    for pred_row in predictions:
        prediction = _parse_prediction(pred_row)
        if prediction is None:
            continue
        _, _, schema_valid, canonical = _validate_prediction(pred_row)
        record_id = pred_row.get("record_id")
        data_record = data_by_id.get(record_id)
        target_change = _observed_profile_change(data_record) if isinstance(data_record, dict) else None
        predicted_change = prediction.get("observed_profile_change")
        evaluator_counted = bool(schema_valid)

        field_rows = {}
        if isinstance(target_change, dict) and isinstance(predicted_change, dict):
            for field in OBSERVED_FIELDS:
                target_value = target_change.get(field)
                predicted_value = predicted_change.get(field)
                normalized_pred = normalize_observed_profile_label(field, predicted_value)
                normalized_target = normalize_observed_profile_label(field, target_value)
                raw_equal = predicted_value == target_value
                normalized_equal = normalized_pred == normalized_target
                cause = _classify_mismatch(field, predicted_value, target_value)
                all_valid_json_raw_total += 1
                all_valid_json_normalized_total += 1
                all_valid_json_raw_correct += int(raw_equal)
                all_valid_json_normalized_correct += int(normalized_equal)
                all_valid_json_cause_counts[cause] = all_valid_json_cause_counts.get(cause, 0) + 1
                if evaluator_counted:
                    raw_total += 1
                    normalized_total += 1
                    raw_correct += int(raw_equal)
                    normalized_correct += int(normalized_equal)
                    cause_counts[cause] = cause_counts.get(cause, 0) + 1
                predicted_value_counts[field][str(predicted_value)] = (
                    predicted_value_counts[field].get(str(predicted_value), 0) + 1
                )
                target_value_counts[field][str(target_value)] = target_value_counts[field].get(str(target_value), 0) + 1
                field_rows[field] = {
                    "predicted": predicted_value,
                    "target": target_value,
                    "raw_equal": raw_equal,
                    "normalized_predicted": normalized_pred,
                    "normalized_target": normalized_target,
                    "normalized_equal": normalized_equal,
                    "mismatch_cause": cause,
                }
        else:
            cause = "missing_target_or_prediction_observed_profile_change"
            all_valid_json_cause_counts[cause] = all_valid_json_cause_counts.get(cause, 0) + 1
            if evaluator_counted:
                cause_counts[cause] = cause_counts.get(cause, 0) + 1

        rows.append(
            {
                "record_id": record_id,
                "task_type": pred_row.get("task_type"),
                "valid_json": pred_row.get("valid_json"),
                "schema_valid": schema_valid,
                "canonical_variables": canonical,
                "counted_by_evaluator": evaluator_counted,
                "predicted_observed_profile_change": predicted_change,
                "target_observed_profile_change": target_change,
                "field_results": field_rows,
            }
        )

    raw_accuracy = None if raw_total == 0 else raw_correct / raw_total
    normalized_accuracy = None if normalized_total == 0 else normalized_correct / normalized_total
    all_valid_json_raw_accuracy = (
        None if all_valid_json_raw_total == 0 else all_valid_json_raw_correct / all_valid_json_raw_total
    )
    all_valid_json_normalized_accuracy = (
        None
        if all_valid_json_normalized_total == 0
        else all_valid_json_normalized_correct / all_valid_json_normalized_total
    )
    headline = (
        (eval_json or {}).get("understanding_metrics", {}).get("observed_profile_change_accuracy")
        if eval_json
        else None
    )
    if cause_counts.get("vocabulary_mismatch", 0) > cause_counts.get("semantic_or_model_mismatch", 0):
        diagnosis = (
            "The zero raw accuracy is primarily caused by vocabulary mismatch and exact-string comparison."
        )
    else:
        diagnosis = (
            "The metric includes non-vocabulary mismatches; observed_profile_change_accuracy should not be used as a headline Stage-1 metric until labels/vocabulary are fixed."
        )

    return {
        "metadata": {
            "predictions_path": str(predictions_path),
            "data_path": str(data_path),
            "eval_json_path": None if eval_json_path is None else str(eval_json_path),
            "observed_fields": OBSERVED_FIELDS,
            "current_evaluator_behavior": "exact string equality unless --normalize-observed-profile-labels is passed",
        },
        "summary": {
            "valid_prediction_rows": len(rows),
            "evaluator_counted_rows": sum(1 for row in rows if row["counted_by_evaluator"]),
            "note": "Headline raw/normalized counts mirror evaluator behavior by requiring schema-valid predictions; all_valid_json_* counts include every parseable JSON row.",
            "raw_correct": raw_correct,
            "raw_total": raw_total,
            "raw_accuracy": raw_accuracy,
            "normalized_correct": normalized_correct,
            "normalized_total": normalized_total,
            "normalized_accuracy": normalized_accuracy,
            "all_valid_json_raw_correct": all_valid_json_raw_correct,
            "all_valid_json_raw_total": all_valid_json_raw_total,
            "all_valid_json_raw_accuracy": all_valid_json_raw_accuracy,
            "all_valid_json_normalized_correct": all_valid_json_normalized_correct,
            "all_valid_json_normalized_total": all_valid_json_normalized_total,
            "all_valid_json_normalized_accuracy": all_valid_json_normalized_accuracy,
            "existing_eval_observed_profile_change_accuracy": headline,
            "cause_counts": cause_counts,
            "all_valid_json_cause_counts": all_valid_json_cause_counts,
            "diagnosis": diagnosis,
            "recommendation": (
                "observed_profile_change_accuracy should not be used as a headline Stage-1 metric until labels/vocabulary are fixed."
            ),
        },
        "predicted_value_counts": predicted_value_counts,
        "target_value_counts": target_value_counts,
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(debug: dict, path: Path) -> None:
    summary = debug["summary"]
    lines = [
        "# Observed Profile Change Debug",
        "",
        "## Purpose",
        "",
        "This report debugs why `observed_profile_change_accuracy` was 0.0 for the Same-50 fine-tuned LLM/API evaluation.",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{debug['metadata']['predictions_path']}`",
        f"- Data: `{debug['metadata']['data_path']}`",
        f"- Existing eval JSON: `{debug['metadata']['eval_json_path']}`",
        "",
        "## Metric Implementation",
        "",
        "The current evaluator derives target observed-profile labels from `current_metrics` and `target_metrics`, then compares each field with exact string equality. The new evaluator flag `--normalize-observed-profile-labels` preserves old behavior by default and only normalizes labels when explicitly requested.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| existing eval observed_profile_change_accuracy | {_fmt(summary['existing_eval_observed_profile_change_accuracy'])} |",
        f"| raw debug accuracy | {_fmt(summary['raw_accuracy'])} |",
        f"| normalized debug accuracy | {_fmt(summary['normalized_accuracy'])} |",
        f"| raw correct / total | {summary['raw_correct']} / {summary['raw_total']} |",
        f"| normalized correct / total | {summary['normalized_correct']} / {summary['normalized_total']} |",
        f"| parseable-row raw correct / total | {summary['all_valid_json_raw_correct']} / {summary['all_valid_json_raw_total']} |",
        f"| parseable-row normalized correct / total | {summary['all_valid_json_normalized_correct']} / {summary['all_valid_json_normalized_total']} |",
        f"| evaluator-counted rows | {summary['evaluator_counted_rows']} |",
        "",
        "The headline debug counts mirror the evaluator by requiring schema-valid predictions. The parseable-row counts are included only to show what changes if the single schema-invalid prediction is inspected instead of excluded.",
        "",
        "## Cause Counts",
        "",
        "| Cause | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(summary["cause_counts"].items()):
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            summary["diagnosis"],
            "",
            summary["recommendation"],
            "",
            "## Value Vocabulary",
            "",
            "### Predicted Values",
            "",
        ]
    )
    for field, values in debug["predicted_value_counts"].items():
        lines.append(f"- `{field}`: `{values}`")
    lines.extend(["", "### Target Values", ""])
    for field, values in debug["target_value_counts"].items():
        lines.append(f"- `{field}`: `{values}`")

    lines.extend(["", "## Per-Row Debug Table", ""])
    for row in debug["rows"]:
        lines.extend(
            [
                f"### {row['record_id']}",
                "",
                f"- task_type: `{row.get('task_type')}`",
                f"- valid_json: `{row.get('valid_json')}`",
                f"- schema_valid: `{row.get('schema_valid')}`",
                f"- counted_by_evaluator: `{row.get('counted_by_evaluator')}`",
                "",
                "| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |",
                "|---|---|---|---:|---|---|---:|---|",
            ]
        )
        if not row["field_results"]:
            lines.append("| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |")
        else:
            for field in OBSERVED_FIELDS:
                result = row["field_results"].get(field) or {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            field,
                            str(result.get("predicted")),
                            str(result.get("target")),
                            str(result.get("raw_equal")),
                            str(result.get("normalized_predicted")),
                            str(result.get("normalized_target")),
                            str(result.get("normalized_equal")),
                            str(result.get("mismatch_cause")),
                        ]
                    )
                    + " |"
                )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    debug = build_debug(
        predictions_path=Path(args.predictions),
        data_path=Path(args.data),
        eval_json_path=Path(args.eval_json) if args.eval_json else None,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2, sort_keys=True)
    write_markdown(debug, Path(args.summary_out))
    print(json.dumps({"out": args.out, "summary_out": args.summary_out, "summary": debug["summary"]}, indent=2))


if __name__ == "__main__":
    main()
