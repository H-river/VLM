"""Convert validated reasoning commands into numerical intent features."""

from __future__ import annotations

from .schema import CANONICAL_VARIABLE_ORDER
from .validator import validate_reasoning_command

_PRIOR_VALUES = {
    "fixed": 0.0,
    "unlikely": 0.2,
    "maybe": 0.5,
    "likely": 1.0,
    "required": 1.0,
}

INTENT_FEATURE_DIM = 24
"""Intent vector dim: 7 allowed mask + 7 fixed mask + 7 relevance prior + 3 status flags."""


def _validated(command: dict) -> dict:
    return validate_reasoning_command(command)


def _constraints(command: dict) -> dict:
    return command.get("constraints") or {}


def _control_plan(command: dict) -> dict:
    return command.get("control_plan") or {}


def _fixed_variables(command: dict) -> set[str]:
    return set(_constraints(command).get("fixed_variables") or [])


def _allowed_variables(command: dict) -> set[str]:
    return set(_constraints(command).get("allowed_variables") or [])


def build_allowed_change_mask(command: dict) -> list[float]:
    """Build a length-7 mask where 1 means the variable may change."""
    command = _validated(command)
    fixed = _fixed_variables(command)
    allowed = _allowed_variables(command)

    mask: list[float] = []
    for variable in CANONICAL_VARIABLE_ORDER:
        if variable in fixed:
            mask.append(0.0)
        elif allowed:
            mask.append(1.0 if variable in allowed else 0.0)
        else:
            mask.append(1.0)
    return mask


def build_fixed_change_mask(command: dict) -> list[float]:
    """Build a length-7 mask where 1 means the variable must stay fixed."""
    command = _validated(command)
    fixed = _fixed_variables(command)
    return [1.0 if variable in fixed else 0.0 for variable in CANONICAL_VARIABLE_ORDER]


def build_relevance_prior(command: dict) -> list[float]:
    """Build a length-7 relevance prior from VLM variable intent."""
    command = _validated(command)
    fixed = _fixed_variables(command)
    change_mask_prior = _constraints(command).get("change_mask_prior") or {}
    likely_relevant = set(_control_plan(command).get("likely_relevant_variables") or [])
    avoid = set(_control_plan(command).get("avoid_variables") or [])

    values: list[float] = []
    for variable in CANONICAL_VARIABLE_ORDER:
        prior_name = change_mask_prior.get(variable, "maybe")
        value = _PRIOR_VALUES[prior_name]

        if variable in fixed:
            value = 0.0
        elif variable in likely_relevant:
            value = 1.0

        if variable not in fixed and variable in avoid and prior_name != "required":
            value = 0.0

        values.append(float(value))
    return values


def build_intent_feature_vector(command: dict) -> list[float]:
    """Build a stable length-24 intent feature vector."""
    command = _validated(command)
    vector = (
        build_allowed_change_mask(command)
        + build_fixed_change_mask(command)
        + build_relevance_prior(command)
        + [
            1.0 if command["valid"] else 0.0,
            1.0 if command["conflict_detected"] else 0.0,
            1.0 if command["unsupported_requests"] else 0.0,
        ]
    )
    if len(vector) != INTENT_FEATURE_DIM:
        raise ValueError(f"intent feature vector must have length {INTENT_FEATURE_DIM}, got {len(vector)}")
    return vector


def extract_canonical_prompt(command: dict) -> str:
    """Return the validated canonical prompt text."""
    command = _validated(command)
    return command["canonical_prompt"]
