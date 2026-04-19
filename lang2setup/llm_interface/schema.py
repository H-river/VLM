"""
schema.py
JSON schema definition and validation for structured LLM outputs.
"""
from __future__ import annotations

import json
from typing import Dict, Any, Optional

SETUP_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "id":        {"type": "integer", "minimum": 0, "maximum": 0},
        "x_bin":     {"type": "integer", "minimum": 0, "maximum": 20},
        "y_bin":     {"type": "integer", "minimum": 0, "maximum": 20},
        "angle_bin": {"type": "integer", "minimum": 0, "maximum": 20},
    },
    "required": ["id", "x_bin", "y_bin", "angle_bin"],
    "additionalProperties": False,
}

# Ranges for clamping
_RANGES = {
    "id": (0, 0),
    "x_bin": (0, 20),
    "y_bin": (0, 20),
    "angle_bin": (0, 20),
}


def validate_and_clamp(raw: Dict[str, Any]) -> Dict[str, int]:
    """
    Validate a parsed dict against the schema.
    Clamps out-of-range values. Raises ValueError if required keys missing.
    """
    result = {}
    for key in ["id", "x_bin", "y_bin", "angle_bin"]:
        if key not in raw:
            raise ValueError(f"Missing required key: {key}")
        val = int(raw[key])
        lo, hi = _RANGES[key]
        result[key] = max(lo, min(hi, val))
    return result


def parse_llm_output(raw_text: str) -> Optional[Dict[str, int]]:
    """
    Extract JSON from LLM output text, validate, and return clamped dict.
    Returns None if parsing fails entirely.
    """
    # Strip markdown code fences
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try to find JSON object
    try:
        # Find first { ... }
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        return validate_and_clamp(obj)
    except (ValueError, json.JSONDecodeError):
        return None
