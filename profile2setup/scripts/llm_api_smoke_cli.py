"""Smoke checks for profile2setup LLM API output validation."""

from __future__ import annotations

import json
from copy import deepcopy

from profile2setup.llm_api import (
    CANONICAL_VARIABLE_ORDER,
    default_output_schema,
    parse_llm_json_text,
    validate_llm_output,
)


def _zero_variables() -> dict[str, float]:
    return {variable: 0.0 for variable in CANONICAL_VARIABLE_ORDER}


def _good_output() -> dict:
    predicted_setup = {
        "source_to_lens": 120.0,
        "lens_to_camera": 220.0,
        "focal_length": 50.0,
        "lens_x": 0.1,
        "lens_y": -0.2,
        "camera_x": 0.0,
        "camera_y": 0.3,
    }
    predicted_delta = deepcopy(_zero_variables())
    predicted_delta["lens_y"] = -0.1
    predicted_delta["camera_y"] = 0.2
    return {
        "valid": True,
        "task_type": "paired_no_setup",
        "observed_profile_change": {
            "centroid_x": "approximately_unchanged",
            "centroid_y": "moves_up",
            "sigma_x": "approximately_unchanged",
            "sigma_y": "approximately_unchanged",
            "intensity": "approximately_unchanged",
        },
        "setup_understanding": {
            "current_setup": _zero_variables(),
            "target_setup": predicted_setup,
            "changed_variables": ["lens_y", "camera_y"],
            "change_direction": {
                "source_to_lens": "unchanged",
                "lens_to_camera": "unchanged",
                "focal_length": "unchanged",
                "lens_x": "unchanged",
                "lens_y": "decrease",
                "camera_x": "unchanged",
                "camera_y": "increase",
            },
            "notes": "The target profile should be reached by a small vertical correction.",
        },
        "predicted_delta": predicted_delta,
        "predicted_setup": predicted_setup,
        "confidence": 0.82,
        "reasoning_summary": "Small vertical setup changes are sufficient.",
        "rejection_reason": "",
    }


def _expect_value_error(obj: dict, expected_fragment: str) -> None:
    try:
        validate_llm_output(obj)
    except ValueError as exc:
        if expected_fragment not in str(exc):
            raise AssertionError(
                f"expected error containing {expected_fragment!r}, got {exc!s}"
            ) from exc
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    schema = default_output_schema()
    if schema["properties"]["predicted_setup"]["anyOf"][0]["required"] != CANONICAL_VARIABLE_ORDER:
        raise AssertionError("default_output_schema must require the canonical variable order")

    good = _good_output()
    validate_llm_output(good)
    parse_llm_json_text(json.dumps(good))

    bad = deepcopy(good)
    legacy_root = "alignment"
    bad["predicted_setup"][f"{legacy_root}_x"] = 1.0
    bad["predicted_setup"].pop("camera_x")
    _expect_value_error(bad, "legacy variable")

    print("LLM API smoke test passed")
    print(f"canonical_variables: {CANONICAL_VARIABLE_ORDER}")


if __name__ == "__main__":
    main()
