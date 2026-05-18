"""
discretize.py
Map continuous setup parameters to bin indices based on bins.yaml config.
"""
from __future__ import annotations

import math
from typing import Dict, Any

import yaml
from pathlib import Path

_DEFAULT_BINS_PATH = Path(__file__).parent.parent / "configs" / "bins.yaml"


def load_bins_config(path: str | Path = _DEFAULT_BINS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def value_to_bin(value: float, vmin: float, vmax: float, num_bins: int) -> int:
    """
    Map a continuous value to a bin index in [0, num_bins-1].
    Values outside [vmin, vmax] are clamped to the edge bins.
    """
    if value <= vmin:
        return 0
    if value >= vmax:
        return num_bins - 1
    # Linear mapping
    frac = (value - vmin) / (vmax - vmin)
    idx = int(math.floor(frac * num_bins))
    return min(idx, num_bins - 1)


def bin_to_value(bin_idx: int, vmin: float, vmax: float, num_bins: int) -> float:
    """Convert a bin index back to the bin-center physical value."""
    bin_width = (vmax - vmin) / num_bins
    return vmin + (bin_idx + 0.5) * bin_width


def discretize_sample(features: Dict[str, Any],
                      bins_cfg: dict | None = None) -> Dict[str, int]:
    """
    Given a flat feature dict (from extract_features), return bin indices.

    Returns dict with keys: id, x_bin, y_bin, angle_bin
    """
    if bins_cfg is None:
        bins_cfg = load_bins_config()

    bx = bins_cfg["bins"]["x"]
    by = bins_cfg["bins"]["y"]
    ba = bins_cfg["bins"]["angle"]

    return {
        "id": 0,  # v1: single Gaussian family
        "x_bin": value_to_bin(features["x_offset"], bx["min"], bx["max"], bx["num"]),
        "y_bin": value_to_bin(features["y_offset"], by["min"], by["max"], by["num"]),
        "angle_bin": value_to_bin(features["tilt_x"], ba["min"], ba["max"], ba["num"]),
    }
