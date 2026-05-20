"""Prompt builder for physics-aware mixed-task SFT rows."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any, Mapping

from optics_sft.physics.metadata_policy import (
    SAFE_PROMPT_METADATA_KEYS,
    find_leakage_fields,
)


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
TEMPLATE_BY_SAMPLE_TYPE = {
    "inverse_control": "inverse_control.txt",
    "forward_transition": "forward_transition.txt",
    "counterfactual_pair": "counterfactual_pair.txt",
    "trajectory": "trajectory.txt",
}
COUNTERFACTUAL_METADATA_WRAPPERS = frozenset({"scenario_a", "scenario_b"})
ACTION_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)


def _json_block(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def _load_template(sample_type: str) -> Template:
    try:
        filename = TEMPLATE_BY_SAMPLE_TYPE[sample_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported sample_type: {sample_type!r}. Expected one of {sorted(TEMPLATE_BY_SAMPLE_TYPE)}"
        ) from exc

    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return Template(path.read_text(encoding="utf-8"))


def _prompt_inputs(row: Mapping[str, Any]) -> Mapping[str, Any]:
    prompt_inputs = row.get("prompt_inputs")
    if not isinstance(prompt_inputs, Mapping):
        raise ValueError("Row must contain object prompt_inputs")
    return prompt_inputs


def _images(row: Mapping[str, Any]) -> Mapping[str, Any]:
    images = _prompt_inputs(row).get("images")
    if not isinstance(images, Mapping):
        raise ValueError("prompt_inputs.images must be an object")
    return images


def _safe_setup_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _prompt_inputs(row).get("safe_setup_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("prompt_inputs.safe_setup_metadata must be an object")
    return metadata


def _unknown_metadata_keys(metadata: Mapping[str, Any], prefix: str = "safe_setup_metadata") -> list[str]:
    keys = set(metadata.keys())
    if keys and keys.issubset(COUNTERFACTUAL_METADATA_WRAPPERS):
        unknown: list[str] = []
        for wrapper, nested in metadata.items():
            if not isinstance(nested, Mapping):
                unknown.append(f"{prefix}.{wrapper}")
                continue
            unknown.extend(
                f"{prefix}.{wrapper}.{key}"
                for key in nested
                if key not in SAFE_PROMPT_METADATA_KEYS
            )
        return unknown

    return [
        f"{prefix}.{key}"
        for key in metadata
        if key not in SAFE_PROMPT_METADATA_KEYS
    ]


def assert_prompt_inputs_safe(row: Mapping[str, Any]) -> None:
    """Reject prompt inputs that contain answer-leaking keys or unsafe metadata."""
    prompt_inputs = _prompt_inputs(row)
    leakage_fields = find_leakage_fields(prompt_inputs)
    leakage_fields.extend(_unknown_metadata_keys(_safe_setup_metadata(row)))
    if leakage_fields:
        joined = ", ".join(sorted(leakage_fields))
        raise ValueError(f"Prompt inputs contain unsafe or leaking fields: {joined}")


def _slot(images: Mapping[str, Any], name: str, key: str) -> dict[str, str]:
    value = images.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected prompt_inputs.images.{key} for image slot {name}")
    return {"slot": name, "path": value}


def expected_image_slots(
    row: Mapping[str, Any],
    *,
    include_trajectory_step_images: bool = False,
) -> list[dict[str, str]]:
    """Return the ordered image slots expected by a physics SFT row."""
    sample_type = row.get("sample_type")
    images = _images(row)

    if sample_type == "inverse_control":
        return [
            _slot(images, "current", "current_image_path"),
            _slot(images, "target", "target_image_path"),
        ]
    if sample_type == "forward_transition":
        return [_slot(images, "before", "before_image_path")]
    if sample_type == "counterfactual_pair":
        return [
            _slot(images, "scenario_a_current", "scenario_a_current_image_path"),
            _slot(images, "scenario_a_target", "scenario_a_target_image_path"),
            _slot(images, "scenario_b_current", "scenario_b_current_image_path"),
            _slot(images, "scenario_b_target", "scenario_b_target_image_path"),
        ]
    if sample_type == "trajectory":
        slots = [
            _slot(images, "initial", "initial_image_path"),
            _slot(images, "target", "target_image_path"),
        ]
        if include_trajectory_step_images:
            step_items = sorted(
                (key, value)
                for key, value in images.items()
                if key.startswith("step_") and key.endswith("_image_path") and isinstance(value, str)
            )
            slots.extend({"slot": key.removesuffix("_image_path"), "path": value} for key, value in step_items)
        return slots

    raise ValueError(
        f"Unsupported sample_type: {sample_type!r}. Expected one of {sorted(TEMPLATE_BY_SAMPLE_TYPE)}"
    )


def _candidate_action(row: Mapping[str, Any]) -> dict[str, Any]:
    action = _prompt_inputs(row).get("action")
    if action is None:
        return {}
    if not isinstance(action, Mapping):
        raise ValueError("prompt_inputs.action must be an object when present")
    return {key: action[key] for key in ACTION_KEYS if key in action}


def build_physics_prompt(row: Mapping[str, Any]) -> str:
    """Build the prompt text for a physics-aware SFT row."""
    sample_type = row.get("sample_type")
    if not isinstance(sample_type, str):
        raise ValueError("Row must contain string sample_type")
    assert_prompt_inputs_safe(row)

    template = _load_template(sample_type)
    candidate_action = _candidate_action(row) if sample_type == "forward_transition" else {}
    return template.substitute(
        image_slots_json=_json_block(expected_image_slots(row)),
        safe_setup_metadata_json=_json_block(_safe_setup_metadata(row)),
        candidate_action_json=_json_block(candidate_action),
    )
