"""Simple profile feature extraction from intensity.npy for v2 data prep."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_intensity(path) -> np.ndarray:
    """Load intensity array from .npy path."""
    arr = np.load(Path(path))
    return np.asarray(arr, dtype=np.float64)


def compute_profile_features(intensity: np.ndarray) -> dict:
    """Compute simple centroid/width/energy features in pixel coordinates."""
    arr = np.asarray(intensity, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D intensity array, got shape={arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)

    h, w = arr.shape
    total = float(np.sum(arr))
    peak = float(np.max(arr)) if arr.size else 0.0

    if total <= 0.0:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        sx = 0.0
        sy = 0.0
    else:
        yy, xx = np.indices(arr.shape, dtype=np.float64)
        cx = float(np.sum(arr * xx) / total)
        cy = float(np.sum(arr * yy) / total)

        var_x = float(np.sum(arr * (xx - cx) ** 2) / total)
        var_y = float(np.sum(arr * (yy - cy) ** 2) / total)

        sx = float(np.sqrt(max(var_x, 0.0)))
        sy = float(np.sqrt(max(var_y, 0.0)))

    return {
        "centroid_x_px": float(cx),
        "centroid_y_px": float(cy),
        "sigma_x_px": float(sx),
        "sigma_y_px": float(sy),
        "peak_intensity": float(peak),
        "total_intensity": float(total),
    }


def compute_profile_features_from_path(path) -> dict:
    """Load intensity.npy and compute simple profile features."""
    intensity = load_intensity(path)
    return compute_profile_features(intensity)
