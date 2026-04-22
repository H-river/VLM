"""
simulator.py
Physics engine: builds the complex field at each plane and propagates it
through the optical system.

v1 uses a pure-NumPy Fresnel propagator.  The backend can be swapped to
waveprop or another library by implementing a new propagation function
with the same signature.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from .optical_elements import OpticalSetup


# ──────────────────────────────────────────────────────────────
# Grid helpers
# ──────────────────────────────────────────────────────────────

def _make_grid(N: int, extent: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (X, Y, dx) for a centred N×N grid spanning [-extent, extent]."""
    x = np.linspace(-extent, extent, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    return X, Y, dx


# ──────────────────────────────────────────────────────────────
# Source field
# ──────────────────────────────────────────────────────────────

def gaussian_source_field(setup: OpticalSetup) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create the complex E-field of a Gaussian beam at the source plane.
    Returns (E_field, X, Y, dx).
    """
    N = setup.grid_size
    ext = setup.grid_extent
    X, Y, dx = _make_grid(N, ext)
    w0 = setup.source.beam_waist
    k = setup.source.wavenumber

    # Fundamental Gaussian amplitude (normalised so peak = 1)
    r2 = X ** 2 + Y ** 2
    E = np.exp(-r2 / w0 ** 2).astype(np.complex128)

    return E, X, Y, dx


# ──────────────────────────────────────────────────────────────
# Propagation back-ends
# ──────────────────────────────────────────────────────────────

def _fresnel_numpy(E: np.ndarray, dx: float, z: float, wavelength: float) -> np.ndarray:
    """
    Single-FFT Fresnel propagation (transfer-function approach).

    U_out(x,y) = IFFT{ FFT{U_in} · H(fx,fy) }
    where H = exp(j·k·z) · exp(-j·π·λ·z·(fx²+fy²))
    """
    N = E.shape[0]
    k = 2.0 * np.pi / wavelength

    # frequency grid
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)

    # transfer function
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX ** 2 + FY ** 2))

    return np.fft.ifft2(np.fft.fft2(E) * H)


def _angular_spectrum(E: np.ndarray, dx: float, z: float, wavelength: float) -> np.ndarray:
    """
    Angular-spectrum propagation (exact within the paraxial band).

    H(fx,fy) = exp(j·k·z·sqrt(1 - (λ·fx)² - (λ·fy)²))
    Evanescent waves are zeroed.
    """
    N = E.shape[0]
    k = 2.0 * np.pi / wavelength
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)

    sq = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    propagating = sq > 0
    phase = np.zeros_like(sq)
    phase[propagating] = np.sqrt(sq[propagating])

    H = np.exp(1j * k * z * phase) * propagating  # zero evanescent

    return np.fft.ifft2(np.fft.fft2(E) * H)


def _waveprop_backend(E: np.ndarray, dx: float, z: float, wavelength: float) -> np.ndarray:
    """
    Placeholder for waveprop-library propagation.
    TODO: integrate waveprop once API is confirmed.  Example call might be:
        from waveprop import fresnel_prop
        E_out = fresnel_prop(E, dx, z, wavelength)
    """
    # Fall back to numpy Fresnel for now
    return _fresnel_numpy(E, dx, z, wavelength)


_BACKENDS = {
    "fresnel_numpy": _fresnel_numpy,
    "angular_spectrum": _angular_spectrum,
    "waveprop": _waveprop_backend,
}


# ──────────────────────────────────────────────────────────────
# Thin-lens phase screen
# ──────────────────────────────────────────────────────────────

def apply_thin_lens(E: np.ndarray, X: np.ndarray, Y: np.ndarray,
                    setup: OpticalSetup) -> np.ndarray:
    """
    Multiply the field by the thin-lens phase:  exp(-j·k/(2f)·(x²+y²))
    and apply a circular hard aperture of diameter = clear_aperture.
    """
    f = setup.lens.focal_length
    k = setup.source.wavenumber
    X_lens = X - setup.lens.x_offset
    Y_lens = Y - setup.lens.y_offset
    r2 = X_lens ** 2 + Y_lens ** 2

    # lens phase
    lens_phase = np.exp(-1j * k / (2.0 * f) * r2)

    # hard aperture mask
    R_ap = setup.lens.clear_aperture / 2.0
    aperture = (np.sqrt(r2) <= R_ap).astype(np.float64)

    return E * lens_phase * aperture


# ──────────────────────────────────────────────────────────────
# Sensor-plane extraction (with camera offsets)
# ──────────────────────────────────────────────────────────────

def _extract_sensor_region(E_cam: np.ndarray, X: np.ndarray, Y: np.ndarray,
                           setup: OpticalSetup) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop / interpolate the propagated field to the sensor pixel grid,
    applying lateral offsets.

    For v1 we do nearest-neighbour crop (no sub-pixel interpolation).
    Returns (intensity, sensor_X, sensor_Y).
    """
    H_pix, W_pix = setup.sensor.resolution
    pitch = setup.sensor.pixel_pitch
    ox = setup.camera.x_offset
    oy = setup.camera.y_offset

    # sensor coordinate arrays centred on (ox, oy)
    sx = np.linspace(ox - W_pix / 2 * pitch, ox + W_pix / 2 * pitch, W_pix)
    sy = np.linspace(oy - H_pix / 2 * pitch, oy + H_pix / 2 * pitch, H_pix)

    # map sensor coords → nearest grid indices
    grid_x = X[0, :]  # 1-D
    grid_y = Y[:, 0]

    ix = np.searchsorted(grid_x, sx).clip(0, len(grid_x) - 1)
    iy = np.searchsorted(grid_y, sy).clip(0, len(grid_y) - 1)

    intensity_full = np.abs(E_cam) ** 2
    intensity = intensity_full[np.ix_(iy, ix)]

    SX, SY = np.meshgrid(sx, sy)
    return intensity, SX, SY


# ──────────────────────────────────────────────────────────────
# Top-level simulation driver
# ──────────────────────────────────────────────────────────────

def run_simulation(setup: OpticalSetup) -> dict:
    """
    Execute the full laser → lens → camera simulation.

    Returns a dict with:
        "intensity"   : 2-D numpy array at sensor
        "sensor_X"    : X coordinate grid at sensor [m]
        "sensor_Y"    : Y coordinate grid at sensor [m]
        "field_at_cam": complex field at camera grid (full grid, before crop)
    """
    propagate = _BACKENDS.get(setup.propagation_backend, _fresnel_numpy)
    wl = setup.source.wavelength

    # 1. Source field
    E, X, Y, dx = gaussian_source_field(setup)

    # 2. Propagate source → lens
    E = propagate(E, dx, setup.laser_to_lens, wl)

    # 3. Apply thin lens + aperture
    E = apply_thin_lens(E, X, Y, setup)

    # 4. Propagate lens → camera (including defocus)
    z_cam = setup.effective_camera_distance
    E = propagate(E, dx, z_cam, wl)

    # 5. Extract sensor region
    intensity, SX, SY = _extract_sensor_region(E, X, Y, setup)

    return {
        "intensity": intensity,
        "sensor_X": SX,
        "sensor_Y": SY,
        "field_at_cam": E,
    }
