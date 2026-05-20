"""Physics-aware optics SFT helpers."""

from .metadata_policy import (
    FORBIDDEN_PROMPT_PATTERNS,
    SAFE_PROMPT_METADATA_KEYS,
    assert_no_prompt_leakage,
    filter_safe_prompt_metadata,
    find_leakage_fields,
)
from .control_search import (
    ActionBounds,
    choose_control_action,
    estimate_local_jacobian,
    grid_search_action,
    refine_action_local,
    score_action,
)
from .rendering import (
    intensity_to_uint8_image,
    random_render_params,
    save_intensity_png,
)
from .sim_adapter import (
    Action,
    apply_action_to_setup,
    metrics_to_state,
    metrics_to_state_m,
    residual_error_px,
    setup_to_safe_metadata,
    simulate_and_measure,
    state_m_to_state_px,
)

__all__ = [
    "Action",
    "ActionBounds",
    "FORBIDDEN_PROMPT_PATTERNS",
    "SAFE_PROMPT_METADATA_KEYS",
    "assert_no_prompt_leakage",
    "apply_action_to_setup",
    "choose_control_action",
    "estimate_local_jacobian",
    "filter_safe_prompt_metadata",
    "find_leakage_fields",
    "grid_search_action",
    "intensity_to_uint8_image",
    "metrics_to_state",
    "metrics_to_state_m",
    "random_render_params",
    "refine_action_local",
    "residual_error_px",
    "save_intensity_png",
    "score_action",
    "setup_to_safe_metadata",
    "simulate_and_measure",
    "state_m_to_state_px",
]
