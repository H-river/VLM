"""Lightweight profile feature helpers for offline evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
    "total_intensity",
]


def load_intensity(path) -> np.ndarray:
    """Load a .npy intensity array and sanitize invalid values."""
    intensity_path = Path(path)
    if not intensity_path.exists():
        raise FileNotFoundError(f"Intensity file not found: {intensity_path}")
    if intensity_path.suffix != ".npy":
        raise ValueError(f"Expected .npy intensity file, got: {intensity_path}")
    arr = np.asarray(np.load(intensity_path), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Intensity array must be 2D, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    return arr


def compute_profile_features(intensity) -> dict:
    """Compute simple JSON-serializable profile features."""
    arr = np.asarray(intensity, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Intensity array must be 2D, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)

    height, width = arr.shape
    total = float(arr.sum())
    peak = float(arr.max()) if arr.size else 0.0
    if total <= 0.0:
        return {
            "centroid_x_px": float((width - 1) / 2.0) if width else 0.0,
            "centroid_y_px": float((height - 1) / 2.0) if height else 0.0,
            "sigma_x_px": 0.0,
            "sigma_y_px": 0.0,
            "peak_intensity": peak,
            "total_intensity": total,
        }

    y_idx, x_idx = np.indices(arr.shape, dtype=np.float64)
    centroid_x = float((x_idx * arr).sum() / total)
    centroid_y = float((y_idx * arr).sum() / total)
    sigma_x = float(np.sqrt((((x_idx - centroid_x) ** 2) * arr).sum() / total))
    sigma_y = float(np.sqrt((((y_idx - centroid_y) ** 2) * arr).sum() / total))
    return {
        "centroid_x_px": centroid_x,
        "centroid_y_px": centroid_y,
        "sigma_x_px": sigma_x,
        "sigma_y_px": sigma_y,
        "peak_intensity": peak,
        "total_intensity": total,
    }


def compute_profile_features_from_path(path) -> dict:
    """Load a .npy profile and compute simple features."""
    return compute_profile_features(load_intensity(path))


def compare_profile_features(features_a, features_b) -> dict:
    """Return per-feature differences between two feature dicts."""
    out = {}
    for name in FEATURE_NAMES:
        a = float((features_a or {}).get(name, 0.0))
        b = float((features_b or {}).get(name, 0.0))
        out[f"{name}_diff"] = float(a - b)
        out[f"{name}_abs_diff"] = float(abs(a - b))
    return out
