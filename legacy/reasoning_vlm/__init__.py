"""Stage 1 reasoning-layer helpers for profile2setup."""

from .intent_features import (
    build_allowed_change_mask,
    build_fixed_change_mask,
    build_intent_feature_vector,
    build_relevance_prior,
    extract_canonical_prompt,
)
from .image_rendering import (
    load_intensity,
    normalize_intensity,
    render_composite,
    render_difference_image,
    render_profile_image,
    render_reasoning_images,
)
from .schema import CANONICAL_VARIABLE_ORDER, default_reasoning_command
from .sft_dataset import (
    build_reasoning_command,
    build_sft_record,
    classify_profile_change,
    classify_setup_delta,
    compute_profile_stats,
    infer_prompt_constraints,
)
from .validator import validate_reasoning_command
from .vlm_parser import build_vlm_user_payload, get_reasoning_command, parse_vlm_json

__all__ = [
    "CANONICAL_VARIABLE_ORDER",
    "build_reasoning_command",
    "build_vlm_user_payload",
    "build_allowed_change_mask",
    "build_fixed_change_mask",
    "build_intent_feature_vector",
    "build_relevance_prior",
    "build_sft_record",
    "classify_profile_change",
    "classify_setup_delta",
    "compute_profile_stats",
    "default_reasoning_command",
    "extract_canonical_prompt",
    "get_reasoning_command",
    "infer_prompt_constraints",
    "load_intensity",
    "normalize_intensity",
    "parse_vlm_json",
    "render_composite",
    "render_difference_image",
    "render_profile_image",
    "render_reasoning_images",
    "validate_reasoning_command",
]
