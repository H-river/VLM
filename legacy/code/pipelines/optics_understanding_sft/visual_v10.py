"""Calibrated rendering and sensor-frame evidence for visual dataset v10."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from optics_sft.physics.rendering import intensity_to_uint8_image


def sensor_frame_state(result: Mapping[str, Any], setup: Any) -> dict[str, float]:
    """Convert a simulator result to the coordinate frame of its image array."""

    metrics = result["metrics"]
    height_px, width_px = setup.sensor.resolution
    pitch = float(setup.sensor.pixel_pitch)
    return {
        "centroid_x_px": (
            float(metrics["centroid_x"]) - float(setup.camera.x_offset)
        )
        / pitch
        + (width_px - 1) / 2.0,
        "centroid_y_px": (
            float(metrics["centroid_y"]) - float(setup.camera.y_offset)
        )
        / pitch
        + (height_px - 1) / 2.0,
        "peak_intensity": float(metrics["peak_intensity"]),
        "sigma_x_px": float(metrics["sigma_x"]) / pitch,
        "sigma_y_px": float(metrics["sigma_y"]) / pitch,
    }


def shared_normalization_bounds(
    intensities: Sequence[Any], percentile_clip: Sequence[float] = (0.0, 99.99)
) -> tuple[float, float]:
    """Return one finite intensity scale shared by all images in a pair."""

    if not intensities:
        raise ValueError("At least one intensity array is required")
    low_pct, high_pct = map(float, percentile_clip)
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError("percentile_clip must satisfy 0 <= low < high <= 100")
    flattened = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in intensities]
    )
    flattened = np.nan_to_num(flattened, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(flattened, [low_pct, high_pct])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    return float(low), float(high)


def render_calibrated_images(
    named_intensities: Sequence[tuple[str, Any]],
    output_dir: Path,
    *,
    size_px: int = 384,
    percentile_clip: Sequence[float] = (0.0, 99.99),
    gamma: float = 1.0,
    sensor_crop_px: int | None = None,
    instrumented: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Render a sequence with a common linear scale and no random nuisance."""

    prepared: list[tuple[str, np.ndarray]] = []
    for name, intensity in named_intensities:
        array = np.asarray(intensity, dtype=np.float64)
        if sensor_crop_px is not None:
            if sensor_crop_px <= 0 or sensor_crop_px > min(array.shape):
                raise ValueError("sensor_crop_px must fit inside every intensity array")
            y0 = (array.shape[0] - sensor_crop_px) // 2
            x0 = (array.shape[1] - sensor_crop_px) // 2
            array = array[y0 : y0 + sensor_crop_px, x0 : x0 + sensor_crop_px]
        prepared.append((name, array))
    bounds = shared_normalization_bounds(
        [intensity for _, intensity in prepared], percentile_clip
    )
    options = {
        "normalize": False,
        "normalization_bounds": list(bounds),
        "gamma": float(gamma),
        "background": 0.0,
        "read_noise_std": 0.0,
        "blur_sigma_px": 0.0,
        "saturation": 0.0,
    }
    paths: list[str] = []
    for name, intensity in prepared:
        path = output_dir / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = intensity_to_uint8_image(intensity, copy.deepcopy(options))
        image = image.resize((size_px, size_px), Image.Resampling.LANCZOS)
        if instrumented:
            draw = ImageDraw.Draw(image)
            center = (size_px - 1) / 2.0
            scale = size_px / float(sensor_crop_px or np.asarray(intensity).shape[0])
            axis_color = (0, 220, 220)
            ring_color = (220, 220, 0)
            draw.line((center, 0, center, size_px - 1), fill=axis_color, width=1)
            draw.line((0, center, size_px - 1, center), fill=axis_color, width=1)
            for sigma_px in (110.0, 135.0):
                radius = sigma_px * scale
                draw.ellipse(
                    (center - radius, center - radius, center + radius, center + radius),
                    outline=ring_color,
                    width=1,
                )
        image.save(path, format="PNG", optimize=True)
        paths.append(path.name)
    calibration = {
        "coordinate_frame": "camera_sensor_array",
        "shared_normalization_bounds": list(bounds),
        "percentile_clip": list(map(float, percentile_clip)),
        "gamma": float(gamma),
        "size_px": int(size_px),
        "pair_shared_calibration": True,
        "sensor_crop_px": sensor_crop_px,
        "instrumented_overlay": instrumented,
    }
    return paths, calibration


def render_signed_difference(
    first: Any,
    second: Any,
    output_path: Path,
    *,
    size_px: int = 384,
    sensor_crop_px: int | None = 512,
) -> dict[str, Any]:
    """Render an amplified signed difference: red=second brighter, blue=first brighter."""

    arrays = [np.asarray(value, dtype=np.float64) for value in (first, second)]
    if sensor_crop_px is not None:
        cropped = []
        for array in arrays:
            y0 = (array.shape[0] - sensor_crop_px) // 2
            x0 = (array.shape[1] - sensor_crop_px) // 2
            cropped.append(array[y0 : y0 + sensor_crop_px, x0 : x0 + sensor_crop_px])
        arrays = cropped
    difference = arrays[1] - arrays[0]
    absolute = np.abs(difference)
    nonzero = absolute[absolute > 0.0]
    scale = float(np.percentile(nonzero, 99.0)) if nonzero.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(absolute.max()) if absolute.size else 1.0
    if scale <= 0.0:
        scale = 1.0
    normalized = np.clip(difference / scale, -1.0, 1.0)
    red = np.sqrt(np.clip(normalized, 0.0, 1.0))
    blue = np.sqrt(np.clip(-normalized, 0.0, 1.0))
    green = np.zeros_like(red)
    rgb = np.stack([red, green, blue], axis=2)
    image = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")
    image = image.resize((size_px, size_px), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    center = (size_px - 1) / 2.0
    draw.line((center, 0, center, size_px - 1), fill=(0, 160, 160), width=1)
    draw.line((0, center, size_px - 1, center), fill=(0, 160, 160), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return {
        "difference_scale": scale,
        "color_key": {"red": "second_brighter", "blue": "first_brighter"},
        "sensor_crop_px": sensor_crop_px,
        "size_px": size_px,
    }
