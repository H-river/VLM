"""Build multimodal LLM API SFT records for profile2setup."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable

from profile2setup.schema import VARIABLE_ORDER, validate_setup_dict

from .validator import validate_llm_output

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
_UNKNOWN_PROFILE_CHANGE = "unknown"


def image_file_to_data_url(path) -> str:
    """Convert a PNG image file to a base64 data URL."""
    image_path = Path(path)
    if image_path.suffix.lower() != ".png":
        raise ValueError(f"expected a PNG image file, got: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _json_or_null(value: Any) -> str:
    return json.dumps(value if value is not None else None, sort_keys=True)


def _clean_setup(value: Any, *, field_name: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object or null")
    if not validate_setup_dict(value):
        raise ValueError(f"{field_name} must use exactly the canonical 7 variables")
    return {key: float(value[key]) for key in CANONICAL_VARIABLE_ORDER}


def _target_delta(record: dict) -> dict | None:
    explicit = _clean_setup(record.get("target_delta"), field_name="target_delta")
    if explicit is not None:
        return explicit
    return None


def _changed_variables(delta: dict | None) -> list[str]:
    if delta is None:
        return []
    return [variable for variable in CANONICAL_VARIABLE_ORDER if float(delta[variable]) != 0.0]


def _change_direction(delta: dict | None) -> dict[str, str]:
    directions: dict[str, str] = {}
    for variable in CANONICAL_VARIABLE_ORDER:
        value = 0.0 if delta is None else float(delta[variable])
        if value > 0.0:
            directions[variable] = "increase"
        elif value < 0.0:
            directions[variable] = "decrease"
        else:
            directions[variable] = "unchanged"
    return directions


def _metric_value(metrics: dict | None, *names: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _compare_delta(delta: float | None, threshold: float, negative: str, positive: str) -> str:
    if delta is None:
        return _UNKNOWN_PROFILE_CHANGE
    if delta > threshold:
        return positive
    if delta < -threshold:
        return negative
    return "approximately_unchanged"


def _compare_relative(
    current: float | None,
    target: float | None,
    *,
    relative_threshold: float = 0.05,
    absolute_threshold: float = 1e-12,
) -> str:
    if current is None or target is None:
        return _UNKNOWN_PROFILE_CHANGE
    delta = target - current
    threshold = max(absolute_threshold, abs(current) * relative_threshold)
    if delta > threshold:
        return "increases"
    if delta < -threshold:
        return "decreases"
    return "approximately_unchanged"


def _observed_profile_change(record: dict) -> dict[str, str]:
    current = record.get("current_metrics")
    target = record.get("target_metrics")

    current_x = _metric_value(current, "centroid_x_px", "centroid_x")
    target_x = _metric_value(target, "centroid_x_px", "centroid_x")
    current_y = _metric_value(current, "centroid_y_px", "centroid_y")
    target_y = _metric_value(target, "centroid_y_px", "centroid_y")
    current_sx = _metric_value(current, "sigma_x_px", "sigma_x")
    target_sx = _metric_value(target, "sigma_x_px", "sigma_x")
    current_sy = _metric_value(current, "sigma_y_px", "sigma_y")
    target_sy = _metric_value(target, "sigma_y_px", "sigma_y")

    centroid_threshold = 0.5 if current_x is not None and target_x is not None else 1e-6
    sigma_threshold = 0.3 if current_sx is not None and target_sx is not None else 1e-6

    return {
        "centroid_x": _compare_delta(
            None if current_x is None or target_x is None else target_x - current_x,
            centroid_threshold,
            "moves_left",
            "moves_right",
        ),
        "centroid_y": _compare_delta(
            None if current_y is None or target_y is None else target_y - current_y,
            centroid_threshold,
            "moves_up",
            "moves_down",
        ),
        "beam_width_x": _compare_delta(
            None if current_sx is None or target_sx is None else target_sx - current_sx,
            sigma_threshold,
            "decreases",
            "increases",
        ),
        "beam_width_y": _compare_delta(
            None if current_sy is None or target_sy is None else target_sy - current_sy,
            sigma_threshold,
            "decreases",
            "increases",
        ),
        "peak_intensity": _compare_relative(
            _metric_value(current, "peak_intensity", "peak"),
            _metric_value(target, "peak_intensity", "peak"),
        ),
        "total_intensity": _compare_relative(
            _metric_value(current, "total_intensity", "total"),
            _metric_value(target, "total_intensity", "total"),
        ),
    }


def build_assistant_label(record: dict) -> dict:
    """Build and validate the assistant strict-JSON label for one record."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")

    current_setup = _clean_setup(record.get("current_setup"), field_name="current_setup")
    target_setup = _clean_setup(record.get("target_setup"), field_name="target_setup")
    predicted_delta = _target_delta(record)
    predicted_setup = target_setup
    valid = predicted_delta is not None or predicted_setup is not None

    label = {
        "valid": valid,
        "task_type": record.get("task_type"),
        "observed_profile_change": _observed_profile_change(record),
        "setup_understanding": {
            "current_setup": current_setup,
            "target_setup": target_setup,
            "changed_variables": _changed_variables(predicted_delta),
            "change_direction": _change_direction(predicted_delta),
            "notes": (
                "Use the rendered intensity profiles and canonical profile2setup variables only."
            ),
        },
        "predicted_delta": predicted_delta,
        "predicted_setup": predicted_setup,
        "confidence": 1.0 if valid else 0.0,
        "reasoning_summary": (
            "Predicted setup fields are supervised labels from the existing profile2setup record."
        ),
        "rejection_reason": "" if valid else "missing target setup and target delta labels",
    }
    return validate_llm_output(label)


def _image_url_for_path(
    path,
    *,
    image_mode: str,
    image_base_dir=None,
    public_prefix: str | None = None,
) -> str:
    image_path = Path(path)
    if image_mode == "base64":
        return image_file_to_data_url(image_path)
    if image_mode == "relative":
        base_dir = Path(image_base_dir) if image_base_dir is not None else Path.cwd()
        return os.path.relpath(image_path.resolve(), base_dir.resolve())
    if image_mode == "public":
        relative = image_path.name
        if image_base_dir is not None:
            relative = os.path.relpath(image_path.resolve(), Path(image_base_dir).resolve())
        prefix = (public_prefix or "").rstrip("/")
        return f"{prefix}/{relative}" if prefix else relative
    raise ValueError("image_mode must be one of: base64, relative, public")


def _image_part(
    path,
    *,
    image_mode: str,
    image_detail: str,
    image_base_dir=None,
    public_prefix: str | None = None,
) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": _image_url_for_path(
                path,
                image_mode=image_mode,
                image_base_dir=image_base_dir,
                public_prefix=public_prefix,
            ),
            "detail": image_detail,
        },
    }


def build_user_message(
    record: dict,
    image_paths: dict,
    *,
    image_mode: str = "base64",
    image_detail: str = "low",
    image_base_dir=None,
    public_prefix: str | None = None,
) -> dict:
    """Build the multimodal user message for one SFT example."""
    current_setup = record.get("current_setup") if isinstance(record.get("current_setup"), dict) else None
    text = "\n".join(
        [
            "Task: " + str(record.get("task_type")),
            "Prompt: " + str(record.get("prompt")),
            "Canonical variables: " + ", ".join(CANONICAL_VARIABLE_ORDER),
            "Current setup: " + _json_or_null(current_setup),
            "Use the profile images to output setup_understanding, predicted_delta, predicted_setup, confidence, and reasoning_summary.",
            "Return only strict JSON with the requested schema.",
        ]
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for key in ("current_profile", "target_profile", "difference_profile", "composite_profile"):
        if key in image_paths:
            content.append(
                _image_part(
                    image_paths[key],
                    image_mode=image_mode,
                    image_detail=image_detail,
                    image_base_dir=image_base_dir,
                    public_prefix=public_prefix,
                )
            )

    return {"role": "user", "content": content}


def build_sft_record(
    record: dict,
    image_paths: dict,
    *,
    image_mode: str = "base64",
    image_detail: str = "low",
    image_base_dir=None,
    public_prefix: str | None = None,
) -> dict:
    """Build one API-compatible multimodal SFT JSONL record."""
    assistant_label = build_assistant_label(record)
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an optical setup understanding model. "
                    "Output only valid JSON using the canonical 7 variables: "
                    + ", ".join(CANONICAL_VARIABLE_ORDER)
                    + "."
                ),
            },
            build_user_message(
                record,
                image_paths,
                image_mode=image_mode,
                image_detail=image_detail,
                image_base_dir=image_base_dir,
                public_prefix=public_prefix,
            ),
            {"role": "assistant", "content": json.dumps(assistant_label, sort_keys=True)},
        ]
    }


def write_jsonl(records: Iterable[dict], out_path) -> int:
    """Write SFT records to JSONL and return the number of rows written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count
