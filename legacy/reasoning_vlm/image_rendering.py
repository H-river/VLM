"""Controlled image rendering from profile intensity arrays for VLM reasoning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_LABEL_HEIGHT = 24
_RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR


def _validate_2d_numeric(array: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype={arr.dtype}")
    if np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"{name} must be real-valued, got dtype={arr.dtype}")
    return np.asarray(arr, dtype=np.float64)


def _finite_array(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float64)

    finite_min = float(np.min(finite))
    finite_max = float(np.max(finite))
    return np.nan_to_num(
        array,
        nan=0.0,
        posinf=finite_max,
        neginf=finite_min,
    )


def load_intensity(path) -> np.ndarray:
    """Load a 2D numeric intensity array from a .npy file."""
    file_path = Path(path)
    if file_path.suffix != ".npy":
        raise ValueError(f"Expected a .npy intensity file, got: {file_path}")
    arr = np.load(file_path, allow_pickle=False)
    arr = _validate_2d_numeric(arr, name="intensity")
    return _finite_array(arr)


def normalize_intensity(
    array,
    mode: str = "max",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Normalize an intensity array into [0, 1]."""
    arr = _finite_array(_validate_2d_numeric(array, name="array"))

    if mode == "max":
        positive = np.clip(arr, a_min=0.0, a_max=None)
        max_positive = float(np.max(positive)) if positive.size else 0.0
        if max_positive <= 0.0:
            return np.zeros_like(positive, dtype=np.float32)
        normalized = positive / max_positive
    elif mode == "percentile":
        clipped = np.clip(arr, a_min=0.0, a_max=None)
        low, high = np.percentile(clipped, [low_percentile, high_percentile])
        if float(high) <= float(low):
            return np.zeros_like(clipped, dtype=np.float32)
        normalized = (clipped - float(low)) / (float(high) - float(low))
    elif mode == "log_max":
        logged = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
        max_logged = float(np.max(logged)) if logged.size else 0.0
        if max_logged <= 0.0:
            return np.zeros_like(logged, dtype=np.float32)
        normalized = logged / max_logged
    else:
        raise ValueError("mode must be one of: max, percentile, log_max")

    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _to_uint8_image(array: np.ndarray, *, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    normalized = normalize_intensity(array, mode="max")
    pixels = np.rint(normalized * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels)
    if image.size != (size, size):
        image = image.resize((size, size), resample=_RESAMPLE)
    return image


def _with_label(image: Image.Image, title: str | None) -> Image.Image:
    if not title:
        return image

    label = str(title).strip().upper()
    canvas = Image.new("L", (image.width, image.height + _LABEL_HEIGHT), color=255)
    canvas.paste(image, (0, _LABEL_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = max((canvas.width - text_width) // 2, 0)
    y = max((_LABEL_HEIGHT - text_height) // 2 - 1, 0)
    draw.text((x, y), label, fill=0, font=font)
    return canvas


def render_profile_image(array, out_path, title: str | None = None, size: int = 256) -> str:
    """Save a deterministic grayscale PNG for one profile array."""
    image = _with_label(_to_uint8_image(array, size=size), title)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return str(path)


def _signed_difference_image(current: np.ndarray, target: np.ndarray, *, size: int) -> Image.Image:
    current_arr = _finite_array(_validate_2d_numeric(current, name="current"))
    target_arr = _finite_array(_validate_2d_numeric(target, name="target"))
    if current_arr.shape != target_arr.shape:
        raise ValueError(f"current and target must have matching shapes, got {current_arr.shape} and {target_arr.shape}")

    diff = target_arr - current_arr
    scale = float(np.max(np.abs(diff))) if diff.size else 0.0
    if scale <= 0.0:
        signed = np.full_like(diff, 0.5, dtype=np.float64)
    else:
        signed = (diff / (2.0 * scale)) + 0.5
    pixels = np.rint(np.clip(signed, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels)
    if image.size != (size, size):
        image = image.resize((size, size), resample=_RESAMPLE)
    return image


def render_difference_image(current, target, out_path, size: int = 256) -> str:
    """Render target-current difference with deterministic symmetric normalization."""
    image = _with_label(_signed_difference_image(current, target, size=size), "DIFFERENCE")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return str(path)


def render_composite(current, target, out_path, size: int = 256) -> str:
    """Render [ CURRENT | TARGET | DIFFERENCE ] as one deterministic PNG."""
    current_image = _with_label(_to_uint8_image(current, size=size), "CURRENT")
    target_image = _with_label(_to_uint8_image(target, size=size), "TARGET")
    diff_image = _with_label(_signed_difference_image(current, target, size=size), "DIFFERENCE")

    width = current_image.width + target_image.width + diff_image.width
    height = max(current_image.height, target_image.height, diff_image.height)
    canvas = Image.new("L", (width, height), color=255)
    x = 0
    for image in (current_image, target_image, diff_image):
        canvas.paste(image, (x, 0))
        x += image.width

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False)
    return str(path)


def render_reasoning_images(current_path, target_path, out_dir, prefix: str = "") -> dict[str, str]:
    """Render all controlled profile images needed by the VLM reasoning layer."""
    current = load_intensity(current_path)
    target = load_intensity(target_path)
    if current.shape != target.shape:
        raise ValueError(f"current and target must have matching shapes, got {current.shape} and {target.shape}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(prefix)

    outputs = {
        "current_profile": output_dir / f"{stem}current_profile.png",
        "target_profile": output_dir / f"{stem}target_profile.png",
        "difference_profile": output_dir / f"{stem}difference_profile.png",
        "composite_profile": output_dir / f"{stem}composite_profile.png",
    }

    return {
        "current_profile": render_profile_image(current, outputs["current_profile"], title="CURRENT"),
        "target_profile": render_profile_image(target, outputs["target_profile"], title="TARGET"),
        "difference_profile": render_difference_image(current, target, outputs["difference_profile"]),
        "composite_profile": render_composite(current, target, outputs["composite_profile"]),
    }
