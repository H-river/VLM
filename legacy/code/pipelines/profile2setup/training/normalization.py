"""Setup and delta normalization helpers for profile2setup v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

CANONICAL_VARIABLE_ORDER = [
    "source_to_lens",
    "lens_to_camera",
    "focal_length",
    "lens_x",
    "lens_y",
    "camera_x",
    "camera_y",
]

_FORBIDDEN_KEYS = {"alignment", "alignment_x", "alignment_y"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_no_forbidden_keys(obj: Any, prefix: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"Forbidden legacy field found: {path}. "
                    "Use camera_x/camera_y only in profile2setup v2."
                )
            _check_no_forbidden_keys(value, prefix=path)
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _check_no_forbidden_keys(item, prefix=path)


def load_variables_config(path) -> dict:
    """Load variables YAML config."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Variables config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid variables config (must be dict): {cfg_path}")

    variables = config.get("variables")
    if not isinstance(variables, dict):
        raise ValueError(f"Invalid variables config: missing dict key 'variables' in {cfg_path}")

    for name in CANONICAL_VARIABLE_ORDER:
        if name not in variables:
            raise ValueError(f"Variables config missing canonical variable: {name}")
        spec = variables[name]
        if not isinstance(spec, dict):
            raise ValueError(f"Variable spec for {name} must be a dict")
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Variable spec for {name} must include min/max")
        min_val = spec["min"]
        max_val = spec["max"]
        if not _is_number(min_val) or not _is_number(max_val):
            raise ValueError(f"Variable range for {name} must be numeric")
        if float(max_val) <= float(min_val):
            raise ValueError(f"Variable range for {name} must satisfy max > min")

    return config


def get_variable_order(config=None) -> list[str]:
    """Return canonical variable order and reject non-canonical orders."""
    if config is None:
        return list(CANONICAL_VARIABLE_ORDER)

    order = config.get("variable_order")
    if order is None:
        return list(CANONICAL_VARIABLE_ORDER)
    if not isinstance(order, list):
        raise ValueError("variables config 'variable_order' must be a list")
    if order != CANONICAL_VARIABLE_ORDER:
        raise ValueError(
            "Variable order must be canonical: "
            f"{CANONICAL_VARIABLE_ORDER}; got {order}"
        )
    return list(order)


def _validate_setup_like_dict(values: dict, *, dict_name: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{dict_name} must be a dict")

    _check_no_forbidden_keys(values)

    keys = set(values.keys())
    expected = set(CANONICAL_VARIABLE_ORDER)
    missing = [k for k in CANONICAL_VARIABLE_ORDER if k not in values]
    extra = sorted(keys - expected)

    if missing:
        raise ValueError(f"{dict_name} missing required canonical variables: {missing}")
    if extra:
        raise ValueError(f"{dict_name} contains non-canonical variables: {extra}")

    for name in CANONICAL_VARIABLE_ORDER:
        if not _is_number(values[name]):
            raise ValueError(f"{dict_name}.{name} must be numeric")


def _range_for_var(variables_config: dict, var_name: str) -> tuple[float, float]:
    spec = variables_config["variables"][var_name]
    min_val = float(spec["min"])
    max_val = float(spec["max"])
    if max_val <= min_val:
        raise ValueError(f"Invalid range for {var_name}: min={min_val}, max={max_val}")
    return min_val, max_val


def normalize_setup_vector(setup_dict, variables_config) -> np.ndarray:
    """Normalize absolute setup dict into float32 vector in [-1, 1]."""
    _validate_setup_like_dict(setup_dict, dict_name="setup_dict")
    get_variable_order(variables_config)

    values = []
    for name in CANONICAL_VARIABLE_ORDER:
        min_val, max_val = _range_for_var(variables_config, name)
        x = float(setup_dict[name])
        x_norm = 2.0 * (x - min_val) / (max_val - min_val) - 1.0
        values.append(x_norm)

    return np.asarray(values, dtype=np.float32)


def denormalize_setup_vector(norm_vector, variables_config) -> dict:
    """Denormalize setup vector in [-1, 1] back to a setup dict."""
    get_variable_order(variables_config)
    arr = np.asarray(norm_vector, dtype=np.float32).reshape(-1)
    if arr.shape[0] != len(CANONICAL_VARIABLE_ORDER):
        raise ValueError(
            f"norm_vector must have length {len(CANONICAL_VARIABLE_ORDER)}, got {arr.shape[0]}"
        )

    out = {}
    for idx, name in enumerate(CANONICAL_VARIABLE_ORDER):
        min_val, max_val = _range_for_var(variables_config, name)
        x_norm = float(arr[idx])
        x = ((x_norm + 1.0) / 2.0) * (max_val - min_val) + min_val
        out[name] = float(x)
    return out


def normalize_delta_vector(delta_dict, variables_config) -> np.ndarray:
    """Normalize setup deltas into float32 vector using half-range scaling."""
    _validate_setup_like_dict(delta_dict, dict_name="delta_dict")
    get_variable_order(variables_config)

    values = []
    for name in CANONICAL_VARIABLE_ORDER:
        min_val, max_val = _range_for_var(variables_config, name)
        half_range = (max_val - min_val) / 2.0
        if half_range <= 0.0:
            raise ValueError(f"Invalid half-range for {name}: {half_range}")
        dx = float(delta_dict[name])
        values.append(dx / half_range)

    return np.asarray(values, dtype=np.float32)


def denormalize_delta_vector(norm_delta_vector, variables_config) -> dict:
    """Denormalize setup delta vector back to canonical delta dict."""
    get_variable_order(variables_config)
    arr = np.asarray(norm_delta_vector, dtype=np.float32).reshape(-1)
    if arr.shape[0] != len(CANONICAL_VARIABLE_ORDER):
        raise ValueError(
            "norm_delta_vector must have length "
            f"{len(CANONICAL_VARIABLE_ORDER)}, got {arr.shape[0]}"
        )

    out = {}
    for idx, name in enumerate(CANONICAL_VARIABLE_ORDER):
        min_val, max_val = _range_for_var(variables_config, name)
        half_range = (max_val - min_val) / 2.0
        if half_range <= 0.0:
            raise ValueError(f"Invalid half-range for {name}: {half_range}")
        dx = float(arr[idx]) * half_range
        out[name] = float(dx)
    return out


def clamp_setup_to_ranges(setup_dict, variables_config) -> dict:
    """Clamp setup values to configured variable min/max bounds."""
    _validate_setup_like_dict(setup_dict, dict_name="setup_dict")
    get_variable_order(variables_config)

    clamped = {}
    for name in CANONICAL_VARIABLE_ORDER:
        min_val, max_val = _range_for_var(variables_config, name)
        x = float(setup_dict[name])
        clamped[name] = float(min(max(x, min_val), max_val))
    return clamped


def make_zero_setup_vector() -> np.ndarray:
    """Return a zero setup vector with canonical length."""
    return np.zeros(len(CANONICAL_VARIABLE_ORDER), dtype=np.float32)
