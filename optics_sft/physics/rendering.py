"""Rendering helpers for simulator intensity arrays."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter


RenderOptions = Mapping[str, Any] | None


def _option(options: RenderOptions, key: str, default: Any) -> Any:
    if options is None:
        return default
    return options.get(key, default)


def _as_float_array(intensity: Any) -> np.ndarray:
    array = np.asarray(intensity, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D intensity array, got shape {array.shape}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normalize(array: np.ndarray, percentile_clip: Sequence[float] | None) -> np.ndarray:
    if percentile_clip is not None:
        if len(percentile_clip) != 2:
            raise ValueError("percentile_clip must contain exactly two values")
        lo_pct, hi_pct = float(percentile_clip[0]), float(percentile_clip[1])
        if not 0.0 <= lo_pct < hi_pct <= 100.0:
            raise ValueError("percentile_clip must satisfy 0 <= low < high <= 100")
        lo, hi = np.percentile(array, [lo_pct, hi_pct])
    else:
        lo, hi = float(array.min()), float(array.max())

    if hi <= lo:
        return np.zeros_like(array, dtype=np.float64)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def _warm_colorize(gray: np.ndarray, saturation: float) -> np.ndarray:
    saturation = float(np.clip(saturation, 0.0, 1.0))
    grayscale_rgb = np.repeat(gray[..., None], 3, axis=2)
    warm_rgb = np.stack(
        [
            np.clip(gray * 1.25, 0.0, 1.0),
            np.clip(np.power(gray, 0.75), 0.0, 1.0),
            np.clip(np.power(gray, 1.8) * 0.65, 0.0, 1.0),
        ],
        axis=2,
    )
    return (1.0 - saturation) * grayscale_rgb + saturation * warm_rgb


def intensity_to_uint8_image(intensity: Any, options: RenderOptions = None) -> Image.Image:
    """Render a simulator intensity array as an RGB PIL image."""
    array = _as_float_array(intensity)

    background = float(_option(options, "background", 0.0))
    if background:
        array = array + background

    read_noise_std = float(_option(options, "read_noise_std", 0.0))
    if read_noise_std > 0.0:
        seed = _option(options, "seed", None)
        array = array + _rng_from_seed(seed).normal(0.0, read_noise_std, size=array.shape)

    normalize = bool(_option(options, "normalize", True))
    percentile_clip = _option(options, "percentile_clip", None)
    normalization_bounds = _option(options, "normalization_bounds", None)
    if normalization_bounds is not None:
        if len(normalization_bounds) != 2:
            raise ValueError("normalization_bounds must contain exactly two values")
        lo, hi = float(normalization_bounds[0]), float(normalization_bounds[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("normalization_bounds must be finite with high > low")
        array = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    elif normalize:
        array = _normalize(array, percentile_clip)
    elif percentile_clip is not None:
        lo, hi = np.percentile(array, [float(percentile_clip[0]), float(percentile_clip[1])])
        array = np.clip(array, lo, hi)
    array = np.clip(array, 0.0, 1.0)

    gamma = float(_option(options, "gamma", 1.0))
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if gamma != 1.0:
        array = np.power(array, gamma)

    saturation = _option(options, "saturation", None)
    if saturation is None:
        rgb = np.repeat(array[..., None], 3, axis=2)
    else:
        rgb = _warm_colorize(array, float(saturation))

    uint8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(uint8, mode="RGB")

    blur_sigma_px = float(_option(options, "blur_sigma_px", 0.0))
    if blur_sigma_px > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma_px))
    return image.convert("RGB")


def save_intensity_png(intensity: Any, path: str | Path, options: RenderOptions = None) -> Path:
    """Render and save an intensity array as an RGB PNG, creating parents."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = intensity_to_uint8_image(intensity, options)
    image.save(output_path, format="PNG")
    return output_path


def _uniform(rng: Any, low: float, high: float) -> float:
    if hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    raise TypeError("rng must provide a uniform(low, high) method")


def _randint(rng: Any, low: int, high: int) -> int:
    if isinstance(rng, random.Random):
        return rng.randint(low, high)
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    if hasattr(rng, "randint"):
        return int(rng.randint(low, high))
    return int(_uniform(rng, low, high + 1))


def random_render_params(rng: Any, difficulty: str = "clean") -> dict[str, Any]:
    """Sample deterministic render options for clean, medium, or hard views."""
    if difficulty == "clean":
        return {
            "normalize": True,
            "percentile_clip": [0.0, 99.9],
            "gamma": 1.0,
            "background": 0.0,
            "read_noise_std": 0.0,
            "blur_sigma_px": 0.0,
            "saturation": 0.0,
            "seed": _randint(rng, 0, 2**31 - 1),
        }
    if difficulty == "medium":
        return {
            "normalize": True,
            "percentile_clip": [_uniform(rng, 0.2, 1.0), _uniform(rng, 98.5, 99.8)],
            "gamma": _uniform(rng, 0.75, 1.25),
            "background": _uniform(rng, 0.0, 0.015),
            "read_noise_std": _uniform(rng, 0.0, 0.01),
            "blur_sigma_px": _uniform(rng, 0.0, 0.5),
            "saturation": _uniform(rng, 0.0, 0.35),
            "seed": _randint(rng, 0, 2**31 - 1),
        }
    if difficulty == "hard":
        return {
            "normalize": True,
            "percentile_clip": [_uniform(rng, 1.0, 5.0), _uniform(rng, 95.0, 99.0)],
            "gamma": _uniform(rng, 0.55, 1.65),
            "background": _uniform(rng, 0.0, 0.05),
            "read_noise_std": _uniform(rng, 0.005, 0.035),
            "blur_sigma_px": _uniform(rng, 0.0, 1.25),
            "saturation": _uniform(rng, 0.2, 0.8),
            "seed": _randint(rng, 0, 2**31 - 1),
        }
    raise ValueError("difficulty must be one of: clean, medium, hard")
