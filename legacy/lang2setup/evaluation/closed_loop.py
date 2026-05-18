"""
closed_loop.py
Closed-loop evaluation: convert predicted bins → physical params →
run simulator → compare resulting beam profile to ground truth.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from pathlib import Path


def bins_to_physical(prediction: Dict[str, int],
                     bins_cfg: dict) -> Dict[str, float]:
    """Convert bin indices back to physical parameter values (bin centers)."""
    from ..data_prep.discretize import bin_to_value

    bx = bins_cfg["bins"]["x"]
    by = bins_cfg["bins"]["y"]
    ba = bins_cfg["bins"]["angle"]

    return {
        "x_offset": bin_to_value(prediction["x_bin"], bx["min"], bx["max"], bx["num"]),
        "y_offset": bin_to_value(prediction["y_bin"], by["min"], by["max"], by["num"]),
        "tilt_x": bin_to_value(prediction["angle_bin"], ba["min"], ba["max"], ba["num"]),
    }


def compute_beam_similarity(pred_intensity: np.ndarray,
                            true_intensity: np.ndarray) -> Dict[str, float]:
    """
    Compare two beam profile intensity arrays.

    Returns MSE, normalized MSE, and peak-signal-to-noise ratio.
    SSIM can be added if skimage is available.
    """
    # Normalize both to [0, 1]
    p = pred_intensity / (pred_intensity.max() + 1e-12)
    t = true_intensity / (true_intensity.max() + 1e-12)

    mse = float(np.mean((p - t) ** 2))
    nmse = mse / (np.var(t) + 1e-12)

    # PSNR
    psnr = -10 * np.log10(mse + 1e-12)

    result = {"profile_mse": mse, "profile_nmse": nmse, "profile_psnr": psnr}

    try:
        from skimage.metrics import structural_similarity as ssim
        result["profile_ssim"] = float(ssim(p, t, data_range=1.0))
    except ImportError:
        pass

    return result
