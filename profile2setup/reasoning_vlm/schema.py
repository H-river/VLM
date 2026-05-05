"""Schema constants for the profile2setup reasoning VLM layer.

Stage 1 defines the JSON contract only. It does not call a VLM and does not
change the numerical setup predictor.
"""

from __future__ import annotations

from copy import deepcopy

from profile2setup.schema import VARIABLE_ORDER

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)

TASK_TYPE_EDIT = "edit"
TASK_TYPE_ABSOLUTE = "absolute"
TASK_TYPE_PAIRED_NO_SETUP = "paired_no_setup"
VALID_TASK_TYPES = (
    TASK_TYPE_EDIT,
    TASK_TYPE_ABSOLUTE,
    TASK_TYPE_PAIRED_NO_SETUP,
)

PROFILE_CHANGE_MOVES_UP = "moves_up"
PROFILE_CHANGE_MOVES_DOWN = "moves_down"
PROFILE_CHANGE_MOVES_LEFT = "moves_left"
PROFILE_CHANGE_MOVES_RIGHT = "moves_right"
PROFILE_CHANGE_INCREASES = "increases"
PROFILE_CHANGE_DECREASES = "decreases"
PROFILE_CHANGE_APPROXIMATELY_UNCHANGED = "approximately_unchanged"
PROFILE_CHANGE_UNKNOWN = "unknown"
PROFILE_CHANGE_VALUES = (
    PROFILE_CHANGE_MOVES_UP,
    PROFILE_CHANGE_MOVES_DOWN,
    PROFILE_CHANGE_MOVES_LEFT,
    PROFILE_CHANGE_MOVES_RIGHT,
    PROFILE_CHANGE_INCREASES,
    PROFILE_CHANGE_DECREASES,
    PROFILE_CHANGE_APPROXIMATELY_UNCHANGED,
    PROFILE_CHANGE_UNKNOWN,
)

CHANGE_MASK_FIXED = "fixed"
CHANGE_MASK_UNLIKELY = "unlikely"
CHANGE_MASK_MAYBE = "maybe"
CHANGE_MASK_LIKELY = "likely"
CHANGE_MASK_REQUIRED = "required"
CHANGE_MASK_PRIORS = (
    CHANGE_MASK_FIXED,
    CHANGE_MASK_UNLIKELY,
    CHANGE_MASK_MAYBE,
    CHANGE_MASK_LIKELY,
    CHANGE_MASK_REQUIRED,
)

EXPECTED_TOP_LEVEL_FIELDS = (
    "valid",
    "task_type",
    "observed_profile_change",
    "requested_goal",
    "constraints",
    "control_plan",
    "canonical_prompt",
    "conflict_detected",
    "unsupported_requests",
    "reasoning_summary",
    "rejection_reason",
)


def default_reasoning_command() -> dict:
    """Return a valid, empty reasoning-command skeleton."""
    return deepcopy(
        {
            "valid": True,
            "task_type": TASK_TYPE_EDIT,
            "observed_profile_change": {
                "centroid_x": PROFILE_CHANGE_UNKNOWN,
                "centroid_y": PROFILE_CHANGE_UNKNOWN,
                "sigma_x": PROFILE_CHANGE_UNKNOWN,
                "sigma_y": PROFILE_CHANGE_UNKNOWN,
                "intensity": PROFILE_CHANGE_UNKNOWN,
            },
            "requested_goal": "",
            "constraints": {
                "fixed_variables": [],
                "allowed_variables": [],
                "change_mask_prior": {
                    variable: CHANGE_MASK_MAYBE for variable in CANONICAL_VARIABLE_ORDER
                },
            },
            "control_plan": {
                "likely_relevant_variables": [],
                "avoid_variables": [],
            },
            "canonical_prompt": "",
            "conflict_detected": False,
            "unsupported_requests": [],
            "reasoning_summary": "",
            "rejection_reason": "",
        }
    )
