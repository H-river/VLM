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
from typing import Any, Tuple

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


def normalize_field_to_power(
    field: np.ndarray,
    grid_pitch: float,
    power_w: float,
) -> tuple[np.ndarray, float]:
    """Scale a sampled source field so its area integral equals ``power_w``.

    This helper is opt-in.  The legacy simulator path deliberately continues
    to use :func:`gaussian_source_field` without this normalization.
    """

    requested = float(power_w)
    pitch = float(grid_pitch)
    if not np.isfinite(requested) or requested < 0.0:
        raise ValueError("source power must be finite and non-negative")
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("simulation-grid pitch must be finite and positive")
    values = np.asarray(field, dtype=np.complex128)
    integrated = float(np.square(np.abs(values)).sum() * pitch**2)
    if not np.isfinite(integrated) or integrated <= 0.0:
        raise ValueError("source field has no finite positive integrated power")
    scale = 0.0 if requested == 0.0 else float(np.sqrt(requested / integrated))
    return values * scale, scale


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


def sensor_pixel_center_coordinates(
    setup: OpticalSetup,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return explicit physical sensor-pixel centres in the lab frame.

    Array columns increase with lab ``+x`` and rows increase with lab ``+y``.
    The camera pose is the centre of the sensor's pixel-edge rectangle.  This
    differs intentionally from the legacy inclusive-endpoint ``linspace``.
    """

    height, width = setup.sensor.resolution
    pitch = float(setup.sensor.pixel_pitch)
    if height < 1 or width < 1 or not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("sensor resolution and pixel pitch must be positive")
    sx = float(setup.camera.x_offset) + (
        np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    ) * pitch
    sy = float(setup.camera.y_offset) + (
        np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    ) * pitch
    sensor_x, sensor_y = np.meshgrid(sx, sy)
    return sx, sy, sensor_x, sensor_y


def _bilinear_sample_uniform_zero(
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample one uniform 2-D grid with explicit zero padding."""

    source = np.asarray(values)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("bilinear sampling requires a two-dimensional grid")
    x_axis = np.asarray(grid_x[0, :], dtype=np.float64)
    y_axis = np.asarray(grid_y[:, 0], dtype=np.float64)
    if source.shape != (len(y_axis), len(x_axis)):
        raise ValueError("field and coordinate-grid shapes differ")
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    if (
        not np.allclose(np.diff(x_axis), dx, rtol=1e-10, atol=1e-15)
        or not np.allclose(np.diff(y_axis), dy, rtol=1e-10, atol=1e-15)
        or dx <= 0.0
        or dy <= 0.0
    ):
        raise ValueError("continuous v12 sampler requires increasing uniform axes")

    fractional_x = (np.asarray(sample_x, dtype=np.float64) - x_axis[0]) / dx
    fractional_y = (np.asarray(sample_y, dtype=np.float64) - y_axis[0]) / dy
    valid_x = (fractional_x >= 0.0) & (fractional_x <= len(x_axis) - 1)
    valid_y = (fractional_y >= 0.0) & (fractional_y <= len(y_axis) - 1)

    lower_x = np.floor(fractional_x).astype(np.int64)
    lower_y = np.floor(fractional_y).astype(np.int64)
    lower_x = np.clip(lower_x, 0, len(x_axis) - 2)
    lower_y = np.clip(lower_y, 0, len(y_axis) - 2)
    weight_x = fractional_x - lower_x
    weight_y = fractional_y - lower_y

    top_left = source[np.ix_(lower_y, lower_x)]
    top_right = source[np.ix_(lower_y, lower_x + 1)]
    bottom_left = source[np.ix_(lower_y + 1, lower_x)]
    bottom_right = source[np.ix_(lower_y + 1, lower_x + 1)]
    wx = weight_x[None, :]
    wy = weight_y[:, None]
    sampled = (
        top_left * (1.0 - wx) * (1.0 - wy)
        + top_right * wx * (1.0 - wy)
        + bottom_left * (1.0 - wx) * wy
        + bottom_right * wx * wy
    )
    valid = valid_y[:, None] & valid_x[None, :]
    return np.where(valid, sampled, 0.0)


def _extract_sensor_region_continuous(
    E_cam: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    setup: OpticalSetup,
    *,
    method: str = "pixel_area_bilinear_intensity",
    quadrature_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample a continuously positioned sensor with explicit v12 semantics.

    ``pixel_area_bilinear_intensity`` is the selected camera model: it averages
    irradiance over each finite pixel using Gauss-Legendre quadrature.
    ``point_bilinear_intensity`` and ``point_bilinear_complex_field`` are
    retained for controlled diagnostics.  Wrapped phase is never interpolated.
    Samples outside the propagated field are explicitly zero padded.
    """

    allowed = {
        "pixel_area_bilinear_intensity",
        "point_bilinear_intensity",
        "point_bilinear_complex_field",
    }
    if method not in allowed:
        raise ValueError(f"unknown continuous sensor sampling method: {method}")
    sx, sy, sensor_x, sensor_y = sensor_pixel_center_coordinates(setup)
    full_intensity = np.square(np.abs(np.asarray(E_cam)))
    pitch = float(setup.sensor.pixel_pitch)

    if method == "point_bilinear_intensity":
        intensity = _bilinear_sample_uniform_zero(
            full_intensity, X, Y, sx, sy
        )
        effective_order = 1
    elif method == "point_bilinear_complex_field":
        real = _bilinear_sample_uniform_zero(E_cam.real, X, Y, sx, sy)
        imag = _bilinear_sample_uniform_zero(E_cam.imag, X, Y, sx, sy)
        intensity = np.square(np.abs(real + 1j * imag))
        effective_order = 1
    else:
        effective_order = int(quadrature_order)
        if effective_order < 1 or effective_order > 9:
            raise ValueError("pixel-area quadrature order must be in [1, 9]")
        nodes, weights = np.polynomial.legendre.leggauss(effective_order)
        intensity = np.zeros((len(sy), len(sx)), dtype=np.float64)
        for y_node, y_weight in zip(nodes, weights, strict=True):
            sample_y = sy + 0.5 * pitch * float(y_node)
            for x_node, x_weight in zip(nodes, weights, strict=True):
                sample_x = sx + 0.5 * pitch * float(x_node)
                intensity += (
                    float(x_weight)
                    * float(y_weight)
                    * _bilinear_sample_uniform_zero(
                        full_intensity,
                        X,
                        Y,
                        sample_x,
                        sample_y,
                    )
                )
        intensity *= 0.25

    grid_x_axis = np.asarray(X[0, :], dtype=np.float64)
    grid_y_axis = np.asarray(Y[:, 0], dtype=np.float64)
    half_pitch = 0.5 * pitch
    valid_x = (sx - half_pitch >= grid_x_axis[0]) & (
        sx + half_pitch <= grid_x_axis[-1]
    )
    valid_y = (sy - half_pitch >= grid_y_axis[0]) & (
        sy + half_pitch <= grid_y_axis[-1]
    )
    valid_region = valid_y[:, None] & valid_x[None, :]
    grid_pitch_x = float(grid_x_axis[1] - grid_x_axis[0])
    grid_pitch_y = float(grid_y_axis[1] - grid_y_axis[0])
    metadata: dict[str, Any] = {
        "method": method,
        "measurement_model": (
            "finite_pixel_area_average_of_bilinearly_interpolated_irradiance"
            if method == "pixel_area_bilinear_intensity"
            else "point_sample_of_bilinearly_interpolated_irradiance"
            if method == "point_bilinear_intensity"
            else "point_sample_of_bilinearly_interpolated_real_imaginary_field"
        ),
        "quadrature_order": effective_order,
        "outside_propagated_field_rule": "zero_padding",
        "simulation_grid_pitch_x_m": grid_pitch_x,
        "simulation_grid_pitch_y_m": grid_pitch_y,
        "sensor_pixel_pitch_m": pitch,
        "sensor_first_center_x_m": float(sx[0]),
        "sensor_first_center_y_m": float(sy[0]),
        "sensor_last_center_x_m": float(sx[-1]),
        "sensor_last_center_y_m": float(sy[-1]),
        "center_fractional_grid_index_x": float(
            (sx[len(sx) // 2] - grid_x_axis[0]) / grid_pitch_x
        ),
        "center_fractional_grid_index_y": float(
            (sy[len(sy) // 2] - grid_y_axis[0]) / grid_pitch_y
        ),
        "axis_0_direction": "+y_lab",
        "axis_1_direction": "+x_lab",
        "valid_region_fraction": float(valid_region.mean()),
    }
    return (
        np.asarray(intensity, dtype=np.float64),
        sensor_x,
        sensor_y,
        valid_region,
        metadata,
    )


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
