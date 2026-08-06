"""Shared definitions for corrected image rendering and measurement."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from specialist_rebuild_v2.common import STATE_FIELDS, raw_state_array


TRANSFORM_FIELDS = (
    "exposure",
    "gamma",
    "noise_std",
    "blur_sigma_px",
    "saturation_level",
    "crop_left_px",
    "crop_right_px",
    "crop_top_px",
    "crop_bottom_px",
)


def stable_seed(*parts: Any) -> int:
    token = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return int(token[:16], 16) % (2**32)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def measurement_tolerance(state: Mapping[str, Any] | np.ndarray) -> np.ndarray:
    values = (
        raw_state_array(state)
        if isinstance(state, Mapping)
        else np.asarray(state, dtype=np.float32)
    )
    return np.asarray(
        [1.0, 1.0, 2.0, 2.0, max(0.05 * abs(float(values[4])), 1e-6)],
        dtype=np.float32,
    )


def transform_vector(
    calibration: Mapping[str, Any], transform: Mapping[str, Any]
) -> np.ndarray:
    stored = calibration["stored_resolution_px"]
    source = calibration["source_sensor_resolution_px"]
    return np.asarray(
        [
            math.log1p(float(calibration["linear_intensity_high"])),
            float(source[0]) / float(stored[0]),
            float(transform["exposure"]),
            float(transform["gamma"]),
            float(transform["noise_std"]),
            float(transform["blur_sigma_px"]) / max(float(stored[0]), 1.0),
            float(transform["saturation_level"]),
            float(transform["crop_left_px"]) / max(float(stored[0]), 1.0),
            float(transform["crop_right_px"]) / max(float(stored[0]), 1.0),
            float(transform["crop_top_px"]) / max(float(stored[1]), 1.0),
            float(transform["crop_bottom_px"]) / max(float(stored[1]), 1.0),
        ],
        dtype=np.float32,
    )


def analytic_measurement(
    linear_image: np.ndarray,
    valid_mask: np.ndarray,
    linear_high: float,
    source_resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a moment baseline and normalized diagnostic features.

    Coordinates are mapped back to the source sensor. The stored image is
    linear intensity; gamma/exposure inversion happens before this function.
    """

    image = np.clip(np.asarray(linear_image, dtype=np.float64), 0.0, None)
    mask = np.asarray(valid_mask, dtype=np.float64)
    weighted = image * mask
    height, width = weighted.shape
    source_width, source_height = map(float, source_resolution)
    x = (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5
    y = (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5
    total = max(float(weighted.sum()), 1e-12)
    x_profile = weighted.sum(axis=0)
    y_profile = weighted.sum(axis=1)
    centroid_x = float(np.dot(x_profile, x) / total)
    centroid_y = float(np.dot(y_profile, y) / total)
    sigma_x = math.sqrt(
        max(float(np.dot(x_profile, (x - centroid_x) ** 2) / total), 1e-12)
    )
    sigma_y = math.sqrt(
        max(float(np.dot(y_profile, (y - centroid_y) ** 2) / total), 1e-12)
    )
    scale_x = source_width / width
    scale_y = source_height / height
    physical_integral = total * float(linear_high) * scale_x * scale_y
    peak_area = physical_integral / max(
        2.0 * math.pi * sigma_x * sigma_y, 1e-12
    )
    peak_sample = float(weighted.max()) * float(linear_high)
    # The clean simulator label is the sampled sensor maximum. ``peak_area`` is
    # retained as a diagnostic input for recovering clipped/saturated peaks,
    # but using it as the clean baseline biases non-Gaussian diffraction images.
    peak = peak_sample
    baseline = np.asarray(
        [centroid_x, centroid_y, sigma_x, sigma_y, peak], dtype=np.float32
    )
    valid_fraction = float(mask.mean())
    saturation_fraction = float(np.mean(image >= 0.999))
    features = np.asarray(
        [
            centroid_x / source_width,
            centroid_y / source_height,
            sigma_x / source_width,
            sigma_y / source_height,
            peak_sample / max(float(linear_high), 1e-12),
            peak_area / max(float(linear_high), 1e-12),
            math.log1p(total),
            valid_fraction,
            saturation_fraction,
        ],
        dtype=np.float32,
    )
    return baseline, features
