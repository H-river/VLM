"""Safe prompt metadata policy for physics-aware optics SFT rows."""

from __future__ import annotations

from typing import Any


SAFE_PROMPT_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "wavelength_nm",
        "beam_waist_mm",
        "power_w",
        "lens_focal_length_mm",
        "lens_aperture_mm",
        "source_to_lens_mm",
        "lens_to_camera_mm",
        "sensor_resolution",
        "pixel_size_um",
        "actuator_limits",
        "coordinate_convention",
        "propagation_backend",
        "grid_size",
        "grid_extent_mm",
    }
)
COUNTERFACTUAL_METADATA_WRAPPERS: frozenset[str] = frozenset({"scenario_a", "scenario_b"})

FORBIDDEN_PROMPT_PATTERNS: tuple[str, ...] = (
    "centroid",
    "centroid_error",
    "residual",
    "true_control",
    "control_plan",
    "label",
    "target_state",
    "after_state",
    "post_action",
    "answer",
    "ground_truth",
    "dx_px",
    "dy_px",
)


def filter_safe_prompt_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only metadata fields that are approved for prompt visibility."""
    return {
        key: value
        for key, value in metadata.items()
        if key in SAFE_PROMPT_METADATA_KEYS
    }


def find_leakage_fields(obj: Any, prefix: str = "") -> list[str]:
    """Return nested key paths whose names match forbidden leakage patterns.

    The search intentionally checks only keys, not values. Lists are traversed
    so dictionaries nested inside arrays are still inspected.
    """
    hits: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            if any(pattern in lowered for pattern in FORBIDDEN_PROMPT_PATTERNS):
                hits.append(path)
            hits.extend(find_leakage_fields(value, path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            hits.extend(find_leakage_fields(value, path))

    return hits


def _unsafe_safe_setup_metadata_keys(prompt_inputs: dict[str, Any]) -> list[str]:
    metadata = prompt_inputs.get("safe_setup_metadata")
    if not isinstance(metadata, dict):
        return []
    keys = set(metadata.keys())
    if keys and keys.issubset(COUNTERFACTUAL_METADATA_WRAPPERS):
        unsafe: list[str] = []
        for wrapper, nested in metadata.items():
            if not isinstance(nested, dict):
                unsafe.append(f"safe_setup_metadata.{wrapper}")
                continue
            unsafe.extend(
                f"safe_setup_metadata.{wrapper}.{key}"
                for key in nested
                if key not in SAFE_PROMPT_METADATA_KEYS
            )
        return unsafe

    return [
        f"safe_setup_metadata.{key}"
        for key in metadata
        if key not in SAFE_PROMPT_METADATA_KEYS
    ]


def assert_no_prompt_leakage(prompt_inputs: dict[str, Any]) -> None:
    """Raise ValueError if prompt inputs contain answer-leaking fields."""
    leakage_fields = find_leakage_fields(prompt_inputs)
    leakage_fields.extend(_unsafe_safe_setup_metadata_keys(prompt_inputs))
    if leakage_fields:
        joined = ", ".join(sorted(set(leakage_fields)))
        raise ValueError(f"Prompt inputs contain unsafe metadata fields: {joined}")
