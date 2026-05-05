"""VLM prompt payload and JSON parsing helpers for profile2setup reasoning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profile2setup.schema import VARIABLE_ORDER

from .sft_dataset import build_reasoning_command
from .validator import validate_reasoning_command


def build_vlm_user_payload(
    prompt: str,
    current_image_path,
    target_image_path,
    difference_image_path,
    current_setup: dict | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build a structured VLM request payload for reasoning JSON extraction."""
    setup_text = "current_setup: null"
    if current_setup is not None:
        setup_text = "current_setup: " + json.dumps(current_setup, sort_keys=True)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a profile2setup reasoning front end. Output only valid JSON. "
                    "Use canonical variables only. Do not include long chain-of-thought; "
                    "write only a short reasoning_summary."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Build a profile2setup reasoning command for this request.\n"
                            f"prompt: {prompt}\n"
                            "canonical_variables: " + ", ".join(VARIABLE_ORDER) + "\n"
                            f"{setup_text}\n"
                            "Return only JSON with the required reasoning command fields."
                        ),
                    },
                    {"type": "image", "image_path": str(current_image_path)},
                    {"type": "image", "image_path": str(target_image_path)},
                    {"type": "image", "image_path": str(difference_image_path)},
                ],
            },
        ]
    }


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_first_json_object(text: str) -> str:
    candidate = _strip_code_fence(text)
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    start = candidate.find("{")
    if start < 0:
        raise ValueError("VLM response does not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : idx + 1]
    raise ValueError("VLM response contains an incomplete JSON object")


def parse_vlm_json(text: str) -> dict:
    """Extract, parse, and validate a reasoning command from VLM text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("VLM response text must be a non-empty string")
    json_text = _extract_first_json_object(text)
    try:
        command = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM response is not valid JSON: {exc}") from exc
    return validate_reasoning_command(command)


def get_reasoning_command(
    *,
    mode: str = "mock_rule_based",
    record: dict | None = None,
    rendered_image_paths: dict | None = None,
    json_file: str | Path | None = None,
    manual_text: str | None = None,
) -> dict:
    """Resolve a validated reasoning command without requiring a real VLM API."""
    if mode == "mock_rule_based":
        if record is None:
            raise ValueError("record is required for mode=mock_rule_based")
        return build_reasoning_command(record, rendered_image_paths=rendered_image_paths)
    if mode == "json_file":
        if json_file is None:
            raise ValueError("json_file is required for mode=json_file")
        with open(json_file, "r") as f:
            data = json.load(f)
        return validate_reasoning_command(data)
    if mode == "manual_text":
        if manual_text is None:
            raise ValueError("manual_text is required for mode=manual_text")
        return parse_vlm_json(manual_text)
    raise ValueError("mode must be one of: mock_rule_based, json_file, manual_text")
