"""
extract_features.py
Read a single sample's metadata.json and return a flat feature dict
suitable for discretization and text generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def extract_features(metadata_path: str | Path) -> Dict[str, Any]:
    """
    Load metadata.json and return a flat dict with the fields needed
    by the downstream pipeline.

    Returns
    -------
    dict with keys:
        run_name, wavelength, beam_waist, focal_length,
        x_offset, y_offset, tilt_x, tilt_y, defocus,
        laser_to_lens, lens_to_camera,
        centroid_x, centroid_y, sigma_x, sigma_y,
        ellipticity, peak_intensity, fwhm_x, fwhm_y
    """
    metadata_path = Path(metadata_path)
    with open(metadata_path) as f:
        meta = json.load(f)

    setup = meta["setup"]
    metrics = meta["metrics"]

    return {
        "run_name": meta["run_name"],
        # source
        "wavelength": setup["source"]["wavelength"],
        "beam_waist": setup["source"]["beam_waist"],
        # lens
        "focal_length": setup["lens"]["focal_length"],
        # geometry
        "laser_to_lens": setup["geometry"]["laser_to_lens"],
        "lens_to_camera": setup["geometry"]["lens_to_camera"],
        # alignment (ground-truth parameters we want to predict)
        "x_offset": setup["alignment"]["x_offset"],
        "y_offset": setup["alignment"]["y_offset"],
        "tilt_x": setup["alignment"]["tilt_x"],
        "tilt_y": setup["alignment"]["tilt_y"],
        "defocus": setup["alignment"]["defocus"],
        # metrics (beam characteristics on sensor)
        "centroid_x": metrics["centroid_x"],
        "centroid_y": metrics["centroid_y"],
        "sigma_x": metrics["sigma_x"],
        "sigma_y": metrics["sigma_y"],
        "fwhm_x": metrics.get("fwhm_x"),
        "fwhm_y": metrics.get("fwhm_y"),
        "ellipticity": metrics.get("ellipticity"),
        "peak_intensity": metrics.get("peak_intensity", 0.0),
    }
