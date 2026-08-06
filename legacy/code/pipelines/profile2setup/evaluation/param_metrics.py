"""Parameter metric helpers for profile2setup v2 offline evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.normalization import (
    denormalize_delta_vector,
    denormalize_setup_vector,
    get_variable_order,
)


FALLBACK_TOLERANCES = {
    "source_to_lens": 0.01,
    "lens_to_camera": 0.01,
    "focal_length": 0.005,
    "lens_x": 0.0005,
    "lens_y": 0.0005,
    "camera_x": 0.0005,
    "camera_y": 0.0005,
}


def _variable_order(variable_order=None) -> list[str]:
    order = list(variable_order or VARIABLE_ORDER)
    if order != list(VARIABLE_ORDER):
        raise ValueError(f"variable_order must be canonical v2 order: {VARIABLE_ORDER}")
    return order


def _to_numpy_matrix(value, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim != 2 or arr.shape[1] != len(VARIABLE_ORDER):
        raise ValueError(f"{name} must have shape [N, {len(VARIABLE_ORDER)}], got {arr.shape}")
    return arr


def _active_mask(mask, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    if torch.is_tensor(mask):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"mask must have shape [N] or [N, 1], got {arr.shape}")
    if arr.shape[0] != length:
        raise ValueError(f"mask length must match predictions, got {arr.shape[0]} vs {length}")
    return arr.astype(bool)


def _prepare(pred, target, mask=None):
    pred_arr = _to_numpy_matrix(pred, "pred")
    target_arr = _to_numpy_matrix(target, "target")
    if pred_arr.shape != target_arr.shape:
        raise ValueError(f"pred and target shapes must match, got {pred_arr.shape} vs {target_arr.shape}")
    return pred_arr, target_arr, _active_mask(mask, pred_arr.shape[0])


def _metric_dict(values: list[float | None], variable_order=None) -> dict:
    order = _variable_order(variable_order)
    return {name: values[idx] for idx, name in enumerate(order)}


def _filtered_errors(pred, target, mask):
    pred_arr, target_arr, active = _prepare(pred, target, mask)
    if not np.any(active):
        return None
    return pred_arr[active] - target_arr[active]


def compute_mae_per_variable(pred, target, mask=None, variable_order=None) -> dict:
    errors = _filtered_errors(pred, target, mask)
    if errors is None:
        return _metric_dict([None] * len(VARIABLE_ORDER), variable_order)
    return _metric_dict(np.mean(np.abs(errors), axis=0).astype(float).tolist(), variable_order)


def compute_rmse_per_variable(pred, target, mask=None, variable_order=None) -> dict:
    errors = _filtered_errors(pred, target, mask)
    if errors is None:
        return _metric_dict([None] * len(VARIABLE_ORDER), variable_order)
    return _metric_dict(np.sqrt(np.mean(np.square(errors), axis=0)).astype(float).tolist(), variable_order)


def compute_median_abs_error_per_variable(pred, target, mask=None, variable_order=None) -> dict:
    errors = _filtered_errors(pred, target, mask)
    if errors is None:
        return _metric_dict([None] * len(VARIABLE_ORDER), variable_order)
    return _metric_dict(np.median(np.abs(errors), axis=0).astype(float).tolist(), variable_order)


def _tolerance_array(tolerances: dict | list | tuple | np.ndarray, variable_order=None) -> np.ndarray:
    order = _variable_order(variable_order)
    if isinstance(tolerances, dict):
        missing = [name for name in order if name not in tolerances]
        if missing:
            raise ValueError(f"tolerances missing variables: {missing}")
        return np.asarray([float(tolerances[name]) for name in order], dtype=np.float64)
    arr = np.asarray(tolerances, dtype=np.float64).reshape(-1)
    if arr.shape[0] != len(order):
        raise ValueError(f"tolerances must have length {len(order)}, got {arr.shape[0]}")
    return arr


def compute_within_tolerance_per_variable(pred, target, tolerances, mask=None, variable_order=None) -> dict:
    errors = _filtered_errors(pred, target, mask)
    if errors is None:
        return _metric_dict([None] * len(VARIABLE_ORDER), variable_order)
    tol = _tolerance_array(tolerances, variable_order)
    return _metric_dict(np.mean(np.abs(errors) <= tol.reshape(1, -1), axis=0).astype(float).tolist(), variable_order)


def compute_joint_within_tolerance(pred, target, tolerances, mask=None) -> float | None:
    errors = _filtered_errors(pred, target, mask)
    if errors is None:
        return None
    tol = _tolerance_array(tolerances)
    return float(np.mean(np.all(np.abs(errors) <= tol.reshape(1, -1), axis=1)))


def summarize_prediction_metrics(
    pred,
    target,
    mask=None,
    variable_order=None,
    tolerances=None,
) -> dict:
    """Return standard per-variable and optional tolerance metrics."""
    summary = {
        "mae": compute_mae_per_variable(pred, target, mask=mask, variable_order=variable_order),
        "rmse": compute_rmse_per_variable(pred, target, mask=mask, variable_order=variable_order),
        "median_abs_error": compute_median_abs_error_per_variable(
            pred, target, mask=mask, variable_order=variable_order
        ),
    }
    if tolerances is not None:
        summary["within_tolerance"] = compute_within_tolerance_per_variable(
            pred, target, tolerances, mask=mask, variable_order=variable_order
        )
        summary["joint_within_tolerance"] = compute_joint_within_tolerance(
            pred, target, tolerances, mask=mask
        )
    return summary


def denormalize_setup_matrix(norm_matrix, variables_config) -> np.ndarray:
    """Denormalize a normalized setup matrix into physical units."""
    get_variable_order(variables_config)
    arr = _to_numpy_matrix(norm_matrix, "norm_matrix")
    rows = []
    for row in arr:
        setup = denormalize_setup_vector(row, variables_config)
        rows.append([setup[name] for name in VARIABLE_ORDER])
    return np.asarray(rows, dtype=np.float64)


def denormalize_delta_matrix(norm_delta_matrix, variables_config) -> np.ndarray:
    """Denormalize a normalized delta matrix into physical units."""
    get_variable_order(variables_config)
    arr = _to_numpy_matrix(norm_delta_matrix, "norm_delta_matrix")
    rows = []
    for row in arr:
        delta = denormalize_delta_vector(row, variables_config)
        rows.append([delta[name] for name in VARIABLE_ORDER])
    return np.asarray(rows, dtype=np.float64)


def load_tolerances(variables_config) -> dict:
    """Load physical-unit tolerances from config with canonical fallbacks."""
    get_variable_order(variables_config)
    variables = variables_config.get("variables") or {}
    tolerances = {}
    for name in VARIABLE_ORDER:
        spec = variables.get(name) or {}
        value = spec.get("tolerance", FALLBACK_TOLERANCES[name])
        tolerances[name] = float(value)
    return tolerances


def _units(variables_config) -> dict:
    variables = variables_config.get("variables") or {}
    return {name: (variables.get(name) or {}).get("unit") for name in VARIABLE_ORDER}


def compute_physical_setup_metrics(
    pred_norm_setup,
    target_norm_setup,
    variables_config,
    mask=None,
    tolerances=None,
) -> dict:
    """Compute physical-unit metrics for normalized setup predictions."""
    pred = denormalize_setup_matrix(pred_norm_setup, variables_config)
    target = denormalize_setup_matrix(target_norm_setup, variables_config)
    tolerances = tolerances or load_tolerances(variables_config)
    metrics = summarize_prediction_metrics(pred, target, mask=mask, tolerances=tolerances)
    metrics["units"] = _units(variables_config)
    metrics["tolerances"] = {name: float(tolerances[name]) for name in VARIABLE_ORDER}
    return metrics


def compute_physical_delta_metrics(
    pred_norm_delta,
    target_norm_delta,
    variables_config,
    mask=None,
    tolerances=None,
) -> dict:
    """Compute physical-unit metrics for normalized delta predictions."""
    pred = denormalize_delta_matrix(pred_norm_delta, variables_config)
    target = denormalize_delta_matrix(target_norm_delta, variables_config)
    tolerances = tolerances or load_tolerances(variables_config)
    metrics = summarize_prediction_metrics(pred, target, mask=mask, tolerances=tolerances)
    metrics["units"] = _units(variables_config)
    metrics["tolerances"] = {name: float(tolerances[name]) for name in VARIABLE_ORDER}
    return metrics
