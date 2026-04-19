"""
metrics.py
Beam profile analysis — centroid, widths, ellipticity, rotation.

v1 uses simple moment-based (ISO 11146 variance) computation on the
intensity image.  If laserbeamsize is installed it can be used as a
drop-in alternative.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class BeamMetrics:
    """Container for single-plane beam measurements."""
    centroid_x: float          # [m]
    centroid_y: float          # [m]
    sigma_x: float             # 1-sigma width along x [m]
    sigma_y: float             # 1-sigma width along y [m]
    width_4sigma_x: float      # D4σ (ISO 11146) width x [m]
    width_4sigma_y: float      # D4σ (ISO 11146) width y [m]
    fwhm_x: Optional[float] = None
    fwhm_y: Optional[float] = None
    ellipticity: Optional[float] = None   # minor/major ratio
    rotation_angle: Optional[float] = None  # [rad]
    peak_intensity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(intensity: np.ndarray,
                    sensor_X: np.ndarray,
                    sensor_Y: np.ndarray) -> BeamMetrics:
    """
    ISO 11146 variance-method beam metrics from a 2-D intensity array.

    Parameters
    ----------
    intensity : (H, W) array
    sensor_X  : (H, W) coordinate grid  [m]
    sensor_Y  : (H, W) coordinate grid  [m]
    """
    I = intensity.astype(np.float64)
    total = I.sum()
    if total == 0:
        # degenerate — return zeros
        return BeamMetrics(0, 0, 0, 0, 0, 0, peak_intensity=0)

    # --- centroid ---
    cx = (I * sensor_X).sum() / total
    cy = (I * sensor_Y).sum() / total

    # --- second moments ---
    dx = sensor_X - cx
    dy = sensor_Y - cy
    sigma_xx = (I * dx ** 2).sum() / total
    sigma_yy = (I * dy ** 2).sum() / total
    sigma_xy = (I * dx * dy).sum() / total

    sx = np.sqrt(max(sigma_xx, 0))
    sy = np.sqrt(max(sigma_yy, 0))

    # --- rotation angle (principal axes) ---
    theta = 0.5 * np.arctan2(2 * sigma_xy, sigma_xx - sigma_yy) if (sigma_xx - sigma_yy) != 0 else 0.0

    # principal widths
    avg = 0.5 * (sigma_xx + sigma_yy)
    diff = 0.5 * (sigma_xx - sigma_yy)
    root = np.sqrt(diff ** 2 + sigma_xy ** 2)
    sigma_major = np.sqrt(max(avg + root, 0))
    sigma_minor = np.sqrt(max(avg - root, 0))

    ellipticity = (sigma_minor / sigma_major) if sigma_major > 0 else 1.0

    # --- FWHM estimate (from Gaussian assumption: FWHM ≈ 2.355·σ) ---
    fwhm_x = 2.3548 * sx
    fwhm_y = 2.3548 * sy

    return BeamMetrics(
        centroid_x=float(cx),
        centroid_y=float(cy),
        sigma_x=float(sx),
        sigma_y=float(sy),
        width_4sigma_x=float(4 * sx),
        width_4sigma_y=float(4 * sy),
        fwhm_x=float(fwhm_x),
        fwhm_y=float(fwhm_y),
        ellipticity=float(ellipticity),
        rotation_angle=float(theta),
        peak_intensity=float(I.max()),
    )


def try_laserbeamsize(intensity: np.ndarray, pixel_pitch: float) -> Optional[dict]:
    """
    Attempt to use the `laserbeamsize` package for ISO 11146 analysis.
    Returns a dict of results or None if the package is not installed.
    TODO: map laserbeamsize outputs to BeamMetrics for consistency.
    """
    try:
        import laserbeamsize as lbs  # type: ignore
        results = lbs.beam_size(intensity)
        # results is (cy, cx, dy, dx, phi) in pixel units
        cy_px, cx_px, dy_px, dx_px, phi = results
        return {
            "centroid_x_px": cx_px,
            "centroid_y_px": cy_px,
            "diameter_x_px": dx_px,
            "diameter_y_px": dy_px,
            "diameter_x_m": dx_px * pixel_pitch,
            "diameter_y_m": dy_px * pixel_pitch,
            "rotation_rad": phi,
        }
    except ImportError:
        return None
