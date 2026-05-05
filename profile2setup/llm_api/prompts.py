"""Prompt builders for profile2setup multimodal LLM API inference."""

from __future__ import annotations

import json
from typing import Any

from profile2setup.schema import VARIABLE_ORDER

from .sft_records import image_file_to_data_url

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)


def build_system_prompt() -> str:
    """Build the strict JSON system instruction for the API model."""
    return (
        "You are an optical setup understanding and setup prediction model. "
        "Output only strict JSON for profile2setup. "
        "Use exactly these canonical variables: "
        + ", ".join(CANONICAL_VARIABLE_ORDER)
        + ". Do not introduce any non-canonical setup variable names. "
        "Include setup_understanding.changed_variables and "
        "setup_understanding.change_direction. Do not write chain-of-thought; "
        "include only a short reasoning_summary."
    )


def _json_or_null(value: Any) -> str:
    return json.dumps(value if value is not None else None, sort_keys=True)


def _image_content(path, *, image_detail: str) -> dict:
    return {
        "type": "input_image",
        "image_url": image_file_to_data_url(path),
        "detail": image_detail,
    }


def build_user_content(
    record: dict,
    image_paths: dict,
    *,
    image_detail: str = "low",
) -> list[dict[str, Any]]:
    """Build Responses-API style multimodal user content."""
    current_setup = record.get("current_setup") if isinstance(record.get("current_setup"), dict) else None
    text = "\n".join(
        [
            "Predict the profile2setup strict JSON response from these inputs.",
            f"Task: {record.get('task_type')}",
            f"Prompt: {record.get('prompt')}",
            "Canonical variables: " + ", ".join(CANONICAL_VARIABLE_ORDER),
            "Current setup: " + _json_or_null(current_setup),
            "Images are provided in this order when available: current profile, target profile, target-current difference, composite.",
            "Return only strict JSON with setup_understanding, predicted_delta, predicted_setup, confidence, and reasoning_summary.",
            "Do not include markdown, code fences, or hidden reasoning.",
        ]
    )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for key in ("current_profile", "target_profile", "difference_profile", "composite_profile"):
        if key in image_paths:
            content.append(_image_content(image_paths[key], image_detail=image_detail))
    return content


def build_messages(
    record: dict,
    image_paths: dict,
    *,
    image_detail: str = "low",
) -> list[dict[str, Any]]:
    """Build the complete system/user message list for an API request."""
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": build_system_prompt()}],
        },
        {
            "role": "user",
            "content": build_user_content(record, image_paths, image_detail=image_detail),
        },
    ]
