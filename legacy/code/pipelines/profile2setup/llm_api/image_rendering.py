"""Deterministic PNG rendering for multimodal LLM API inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_DEFAULT_SIZE = 512
_LABEL_HEIGHT = 28
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


def _validate_png_path(path) -> Path:
    out_path = Path(path)
    if out_path.suffix.lower() != ".png":
        raise ValueError(f"output image path must end with .png: {out_path}")
    return out_path


def load_intensity_npy(path) -> np.ndarray:
    """Load a 2D numeric intensity array from an intensity.npy file."""
    file_path = Path(path)
    if file_path.name != "intensity.npy":
        raise ValueError(f"expected input file named intensity.npy, got: {file_path}")
    array = np.load(file_path, allow_pickle=False)
    return _finite_array(_validate_2d_numeric(array, name="intensity"))


def normalize_intensity(array, mode: str = "max") -> np.ndarray:
    """Normalize intensity values into [0, 1] with deterministic rules."""
    arr = _finite_array(_validate_2d_numeric(array, name="array"))
    clipped = np.clip(arr, a_min=0.0, a_max=None)

    if mode == "max":
        scale = float(np.max(clipped)) if clipped.size else 0.0
        if scale <= 0.0:
            return np.zeros_like(clipped, dtype=np.float32)
        normalized = clipped / scale
    elif mode == "percentile":
        low, high = np.percentile(clipped, [1.0, 99.0])
        if float(high) <= float(low):
            return np.zeros_like(clipped, dtype=np.float32)
        normalized = (clipped - float(low)) / (float(high) - float(low))
    elif mode == "log_max":
        logged = np.log1p(clipped)
        scale = float(np.max(logged)) if logged.size else 0.0
        if scale <= 0.0:
            return np.zeros_like(logged, dtype=np.float32)
        normalized = logged / scale
    else:
        raise ValueError("mode must be one of: max, percentile, log_max")

    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _check_size(size: int) -> int:
    size = int(size)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return size


def _profile_image(array, *, size: int, mode: str) -> Image.Image:
    normalized = normalize_intensity(array, mode=mode)
    pixels = np.rint(normalized * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), resample=_RESAMPLE)
    return image


def _difference_image(current, target, *, size: int, mode: str) -> Image.Image:
    current_arr = normalize_intensity(current, mode=mode)
    target_arr = normalize_intensity(target, mode=mode)
    if current_arr.shape != target_arr.shape:
        raise ValueError(
            "current and target must have matching shapes, got "
            f"{current_arr.shape} and {target_arr.shape}"
        )

    diff = target_arr - current_arr
    scale = float(np.max(np.abs(diff))) if diff.size else 0.0
    if scale <= 0.0:
        signed = np.zeros_like(diff, dtype=np.float32)
    else:
        signed = np.clip(diff / scale, -1.0, 1.0)

    rgb = np.zeros((*signed.shape, 3), dtype=np.uint8)
    positive = np.clip(signed, 0.0, 1.0)
    negative = np.clip(-signed, 0.0, 1.0)
    rgb[..., 0] = np.rint(positive * 255.0).astype(np.uint8)
    rgb[..., 2] = np.rint(negative * 255.0).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), resample=_RESAMPLE)
    return image


def _with_label(image: Image.Image, label: str | None) -> Image.Image:
    text = str(label or "").strip().upper()
    if not text:
        return image

    canvas = Image.new("RGB", (image.width, image.height + _LABEL_HEIGHT), color=(255, 255, 255))
    canvas.paste(image, (0, _LABEL_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = max((canvas.width - text_width) // 2, 0)
    y = max((_LABEL_HEIGHT - text_height) // 2 - 1, 0)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return canvas


def _save_png(image: Image.Image, out_path) -> str:
    path = _validate_png_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return str(path)


def render_profile_png(
    array,
    out_path,
    label: str,
    size: int = _DEFAULT_SIZE,
    mode: str = "max",
) -> str:
    """Render a single profile intensity array as a labeled PNG."""
    size = _check_size(size)
    image = _with_label(_profile_image(array, size=size, mode=mode), label)
    return _save_png(image, out_path)


def render_difference_png(
    current,
    target,
    out_path,
    size: int = _DEFAULT_SIZE,
    mode: str = "max",
) -> str:
    """Render a deterministic target-current difference PNG."""
    size = _check_size(size)
    image = _with_label(_difference_image(current, target, size=size, mode=mode), "DIFFERENCE")
    return _save_png(image, out_path)


def render_composite_png(
    current,
    target,
    out_path,
    size: int = _DEFAULT_SIZE,
    mode: str = "max",
) -> str:
    """Render [ CURRENT | TARGET | DIFFERENCE ] as one deterministic PNG."""
    size = _check_size(size)
    panels = [
        _with_label(_profile_image(current, size=size, mode=mode), "CURRENT"),
        _with_label(_profile_image(target, size=size, mode=mode), "TARGET"),
        _with_label(_difference_image(current, target, size=size, mode=mode), "DIFFERENCE"),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    return _save_png(canvas, out_path)
