"""Text prompt builder for physics-aware SFT rows."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any, Mapping

from optics_sft.physics.metadata_policy import filter_safe_prompt_metadata
from optics_sft.physics.prompt_builder import assert_prompt_inputs_safe
from optics_sft.physics.text_target import TrainingTargetMode, normalize_training_target


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
TEXT_TEMPLATE_BY_SAMPLE_TYPE = {
    "inverse_control": "inverse_control_text.txt",
}
TEXT_TEMPLATE_BY_TRAINING_TARGET: dict[TrainingTargetMode, str] = {
    "full": "inverse_control_text.txt",
    "compact": "inverse_control_text.txt",
    "control_plan_only": "inverse_control_text_control_plan.txt",
}
TEXT_OBSERVATION_FIELDS = (
    ("x_px", "centroid_x_px", "centroid_px", "x"),
    ("y_px", "centroid_y_px", "centroid_px", "y"),
    ("width_x_px", "sigma_x_px", "sigma_px", "x"),
    ("width_y_px", "sigma_y_px", "sigma_px", "y"),
)


def _json_block(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def _load_template(sample_type: str, training_target: TrainingTargetMode = "compact") -> Template:
    try:
        filename = TEXT_TEMPLATE_BY_TRAINING_TARGET.get(
            training_target,
            TEXT_TEMPLATE_BY_SAMPLE_TYPE[sample_type],
        )
    except KeyError as exc:
        raise ValueError(
            f"Unsupported text sample_type: {sample_type!r}. "
            f"Expected one of {sorted(TEXT_TEMPLATE_BY_SAMPLE_TYPE)}"
        ) from exc
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Text prompt template not found: {path}")
    return Template(path.read_text(encoding="utf-8"))


def _read_state_field(state: Mapping[str, Any], flat_key: str, nested_key: str, axis: str) -> float:
    candidate = state.get(flat_key)
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        return round(float(candidate), 4)
    nested = state.get(nested_key)
    if isinstance(nested, Mapping):
        axis_value = nested.get(axis)
        if isinstance(axis_value, (int, float)) and not isinstance(axis_value, bool):
            return round(float(axis_value), 4)
    raise ValueError(f"State is missing numeric field {flat_key}")


def _state_vector(state: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(state, Mapping):
        raise ValueError("Expected a state object for text observations")
    return {
        output_key: _read_state_field(state, flat_key, nested_key, axis)
        for output_key, flat_key, nested_key, axis in TEXT_OBSERVATION_FIELDS
    }


def build_text_observations(row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    private_eval = row.get("private_eval")
    if not isinstance(private_eval, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} is missing private_eval for text observations")
    return {
        "current": _state_vector(private_eval.get("current_state")),
        "target": _state_vector(private_eval.get("target_state")),
    }


def build_text_prompt_inputs(row: Mapping[str, Any]) -> dict[str, Any]:
    prompt_inputs = row.get("prompt_inputs")
    if not isinstance(prompt_inputs, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} is missing prompt_inputs")
    metadata = prompt_inputs.get("safe_setup_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} is missing safe_setup_metadata")
    return {
        "safe_setup_metadata": filter_safe_prompt_metadata(dict(metadata)),
        "text_observations": build_text_observations(row),
    }


def build_text_prompt(
    row: Mapping[str, Any],
    *,
    training_target: TrainingTargetMode | str | None = None,
    compact_target: bool = False,
) -> str:
    sample_type = row.get("sample_type")
    if not isinstance(sample_type, str):
        raise ValueError("Row must contain string sample_type")
    mode = normalize_training_target(training_target, compact_target=compact_target)
    text_prompt_inputs = build_text_prompt_inputs(row)
    audit_row = {
        **row,
        "prompt_inputs": text_prompt_inputs,
    }
    assert_prompt_inputs_safe(audit_row)
    template = _load_template(sample_type, mode)
    return template.substitute(
        safe_setup_metadata_json=_json_block(text_prompt_inputs["safe_setup_metadata"]),
        text_observations_json=_json_block(text_prompt_inputs["text_observations"]),
    )


def build_text_messages(
    row: Mapping[str, Any],
    *,
    training_target: TrainingTargetMode | str | None = None,
    compact_target: bool = False,
) -> list[dict[str, str]]:
    from optics_sft.physics.text_target import build_training_target, training_target_json

    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} is missing target object")
    mode = normalize_training_target(training_target, compact_target=compact_target)
    if mode == "full":
        assistant_content = json.dumps(target, sort_keys=True)
    else:
        assistant_content = training_target_json(target, mode)
    return [
        {"role": "user", "content": build_text_prompt(row, training_target=mode)},
        {"role": "assistant", "content": assistant_content},
    ]
