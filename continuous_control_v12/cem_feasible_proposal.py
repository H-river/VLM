"""Feasibility-preserving proposal sampling for controlled CEM ablations."""

from __future__ import annotations

import numpy as np


def sample_truncated_normal_by_resampling(
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    population: int,
    lower: np.ndarray,
    upper: np.ndarray,
    resample_attempts: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Draw bounded proposals with finite rejection attempts and final clipping."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if mean.shape != std.shape or mean.shape != lower.shape or mean.shape != upper.shape:
        raise ValueError("proposal mean, std, and bounds must have identical shapes")
    if population <= 0 or resample_attempts < 0:
        raise ValueError("population must be positive and resample attempts nonnegative")
    if np.any(std < 0) or np.any(lower > upper):
        raise ValueError("proposal standard deviations/bounds are invalid")
    values = rng.normal(mean[None, ...], std[None, ...], size=(population, *mean.shape))
    invalid = (values < lower[None, ...]) | (values > upper[None, ...])
    initially_invalid = int(np.sum(invalid))
    draws_resampled = 0
    attempts_used = 0
    broadcast_mean = np.broadcast_to(mean, values.shape)
    broadcast_std = np.broadcast_to(std, values.shape)
    for attempt in range(resample_attempts):
        if not np.any(invalid):
            break
        count = int(np.sum(invalid))
        values[invalid] = rng.normal(broadcast_mean[invalid], broadcast_std[invalid])
        draws_resampled += count
        attempts_used = attempt + 1
        invalid = (values < lower[None, ...]) | (values > upper[None, ...])
    remaining_invalid = int(np.sum(invalid))
    values = np.clip(values, lower[None, ...], upper[None, ...])
    total = int(values.size)
    return values, {
        "initially_out_of_feasible_bounds_fraction": initially_invalid / total,
        "remaining_out_of_feasible_bounds_before_clip_fraction": remaining_invalid
        / total,
        "resample_attempts_requested": resample_attempts,
        "resample_attempts_used": attempts_used,
        "scalar_draws_resampled": draws_resampled,
    }
