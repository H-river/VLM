"""Evaluation helpers for profile2setup reasoning VLM JSONL outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legacy.reasoning_vlm.validator import validate_reasoning_command
from legacy.reasoning_vlm.vlm_parser import parse_vlm_json


_METRIC_NAMES = (
    "valid_invalid_classification_accuracy",
    "conflict_detection_accuracy",
    "unsupported_request_detection_accuracy",
    "fixed_variable_extraction_accuracy",
    "allowed_variable_extraction_accuracy",
    "observed_profile_change_accuracy",
)


def _load_jsonl(path) -> list[dict]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {jsonl_path}")

    records: list[dict] = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_num}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {jsonl_path}:{line_num}")
            records.append(record)
    return records


def _save_json(obj: Any, path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _assistant_content(record: dict) -> Any | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message.get("content")
    return None


def _text_from_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts) if parts else None
    return None


def _parse_command(value: Any) -> tuple[dict | None, str | None]:
    if value is None:
        return None, "missing"
    try:
        if isinstance(value, dict):
            return validate_reasoning_command(value), None
        text = _text_from_content(value)
        if text is None:
            return None, f"unsupported command value type: {type(value).__name__}"
        return parse_vlm_json(text), None
    except Exception as exc:  # noqa: BLE001 - evaluator reports validation errors as data.
        return None, f"{type(exc).__name__}: {exc}"


def _extract_prediction_and_target(record: dict) -> tuple[Any, Any | None, str]:
    if "predicted_reasoning_command" in record:
        return (
            record.get("predicted_reasoning_command"),
            record.get("target_reasoning_command"),
            "prediction_record",
        )
    if "predicted_reasoning_text" in record:
        return (
            record.get("predicted_reasoning_text"),
            record.get("target_reasoning_command"),
            "prediction_record",
        )

    assistant = _assistant_content(record)
    if assistant is not None:
        return assistant, assistant, "sft_assistant_label"

    return record.get("reasoning_command"), record.get("target_reasoning_command"), "reasoning_command_record"


def _constraints(command: dict) -> dict:
    value = command.get("constraints")
    return value if isinstance(value, dict) else {}


def _variable_set(command: dict, key: str) -> set[str]:
    values = _constraints(command).get(key) or []
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def _has_unsupported(command: dict) -> bool:
    values = command.get("unsupported_requests") or []
    return bool(values) if isinstance(values, list) else False


def _same_observed_profile_change(predicted: dict, target: dict) -> bool:
    return predicted.get("observed_profile_change") == target.get("observed_profile_change")


def _new_counter() -> dict[str, int]:
    return {"correct": 0, "total": 0}


def _add(counter: dict[str, int], correct: bool) -> None:
    counter["total"] += 1
    if correct:
        counter["correct"] += 1


def _finalize_counter(counter: dict[str, int]) -> dict[str, float | int | None]:
    total = int(counter["total"])
    correct = int(counter["correct"])
    return {
        "correct": correct,
        "total": total,
        "accuracy": None if total == 0 else float(correct / total),
    }


def _empty_metrics() -> dict[str, dict[str, int]]:
    return {name: _new_counter() for name in _METRIC_NAMES}


def _score_record(predicted: dict | None, target: dict, metrics: dict[str, dict[str, int]]) -> None:
    if predicted is None:
        for name in _METRIC_NAMES:
            _add(metrics[name], False)
        return

    _add(
        metrics["valid_invalid_classification_accuracy"],
        bool(predicted.get("valid")) == bool(target.get("valid")),
    )
    _add(
        metrics["conflict_detection_accuracy"],
        bool(predicted.get("conflict_detected")) == bool(target.get("conflict_detected")),
    )
    _add(
        metrics["unsupported_request_detection_accuracy"],
        _has_unsupported(predicted) == _has_unsupported(target),
    )
    _add(
        metrics["fixed_variable_extraction_accuracy"],
        _variable_set(predicted, "fixed_variables") == _variable_set(target, "fixed_variables"),
    )
    _add(
        metrics["allowed_variable_extraction_accuracy"],
        _variable_set(predicted, "allowed_variables") == _variable_set(target, "allowed_variables"),
    )
    _add(
        metrics["observed_profile_change_accuracy"],
        _same_observed_profile_change(predicted, target),
    )


def evaluate_reasoning_vlm_jsonl(input_path, out_path=None, max_error_examples: int = 20) -> dict:
    """Evaluate reasoning VLM JSONL records.

    Supported records are SFT message records with assistant JSON content, and
    prediction records containing predicted_reasoning_command plus an optional
    target_reasoning_command.
    """
    records = _load_jsonl(input_path)
    metrics = _empty_metrics()
    source_counts: dict[str, int] = {}
    invalid_examples: list[dict[str, Any]] = []
    target_errors: list[dict[str, Any]] = []
    valid_prediction_count = 0
    target_count = 0

    for idx, record in enumerate(records):
        predicted_raw, target_raw, source_type = _extract_prediction_and_target(record)
        source_counts[source_type] = source_counts.get(source_type, 0) + 1

        predicted, pred_error = _parse_command(predicted_raw)
        if predicted is not None:
            valid_prediction_count += 1
        elif len(invalid_examples) < int(max_error_examples):
            invalid_examples.append(
                {
                    "index": idx,
                    "source_type": source_type,
                    "record_id": record.get("id") or record.get("record_id"),
                    "error": pred_error,
                }
            )

        if target_raw is None:
            continue

        target, target_error = _parse_command(target_raw)
        if target is None:
            if len(target_errors) < int(max_error_examples):
                target_errors.append(
                    {
                        "index": idx,
                        "source_type": source_type,
                        "record_id": record.get("id") or record.get("record_id"),
                        "error": target_error,
                    }
                )
            continue

        target_count += 1
        _score_record(predicted, target, metrics)

    total = len(records)
    result = {
        "input_path": str(input_path),
        "num_records": int(total),
        "source_type_counts": source_counts,
        "json_valid_count": int(valid_prediction_count),
        "json_validity_rate": None if total == 0 else float(valid_prediction_count / total),
        "target_label_count": int(target_count),
        "metrics": {name: _finalize_counter(counter) for name, counter in metrics.items()},
        "invalid_prediction_examples": invalid_examples,
        "invalid_target_examples": target_errors,
    }
    if out_path is not None:
        _save_json(result, out_path)
    return result
