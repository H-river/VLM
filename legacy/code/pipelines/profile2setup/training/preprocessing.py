"""Preprocessing utilities for profile2setup v2 training datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch may be unavailable in some environments
    torch = None
    F = None


_EPS = 1e-8


def load_intensity(path) -> np.ndarray:
    """Load a 2D intensity array from a .npy file as float32."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Intensity file not found: {file_path}")
    if file_path.suffix.lower() != ".npy":
        raise ValueError(
            f"Only .npy intensity files are supported (do not use beam_profile.png): {file_path}"
        )

    arr = np.load(file_path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D intensity array, got shape={arr.shape} from {file_path}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    return arr.astype(np.float32, copy=False)


def normalize_intensity(I, mode="max_log") -> np.ndarray:
    """Normalize intensity values using one of: max, max_log, none."""
    arr = np.asarray(I, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D intensity array, got shape={arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)

    if mode == "none":
        return arr.astype(np.float32, copy=False)

    if mode not in {"max", "max_log"}:
        raise ValueError(f"Unsupported normalize mode: {mode}")

    peak = float(np.max(arr)) if arr.size else 0.0
    if peak <= 0.0:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = arr / (peak + _EPS)

    if mode == "max":
        return norm.astype(np.float32, copy=False)

    # max_log mode
    norm = np.log1p(10.0 * norm) / np.log1p(10.0)
    return norm.astype(np.float32, copy=False)


def _resize_with_torch(arr: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    out = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _resize_with_numpy_bilinear(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    if h == size and w == size:
        return arr.astype(np.float32, copy=False)

    y = np.linspace(0.0, float(h - 1), size, dtype=np.float32)
    x = np.linspace(0.0, float(w - 1), size, dtype=np.float32)

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    Ia = arr[y0[:, None], x0[None, :]]
    Ib = arr[y0[:, None], x1[None, :]]
    Ic = arr[y1[:, None], x0[None, :]]
    Id = arr[y1[:, None], x1[None, :]]

    wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    wb = (1.0 - wy)[:, None] * wx[None, :]
    wc = wy[:, None] * (1.0 - wx)[None, :]
    wd = wy[:, None] * wx[None, :]

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out.astype(np.float32, copy=False)


def resize_intensity(I, size=128) -> np.ndarray:
    """Resize a 2D intensity array to [size, size] with float32 output."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"size must be a positive int, got {size!r}")

    arr = np.asarray(I, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D intensity array, got shape={arr.shape}")

    if torch is not None and F is not None:
        return _resize_with_torch(arr, size=size)
    return _resize_with_numpy_bilinear(arr, size=size)


def _load_normalize_resize(path, input_size: int, normalize_mode: str) -> np.ndarray:
    arr = load_intensity(path)
    arr = normalize_intensity(arr, mode=normalize_mode)
    arr = resize_intensity(arr, size=input_size)
    return arr.astype(np.float32, copy=False)


def make_profile_channels(
    current_path=None,
    target_path=None,
    input_size=128,
    normalize_mode="max_log",
) -> np.ndarray:
    """Create a 4-channel profile tensor with shape [4, H, W]."""
    zeros = np.zeros((input_size, input_size), dtype=np.float32)

    current = zeros
    target = zeros

    if current_path is not None:
        current = _load_normalize_resize(current_path, input_size=input_size, normalize_mode=normalize_mode)
    if target_path is not None:
        target = _load_normalize_resize(target_path, input_size=input_size, normalize_mode=normalize_mode)

    if current_path is not None and target_path is not None:
        delta = target - current
    else:
        delta = zeros

    if target_path is not None:
        target_mask = np.ones((input_size, input_size), dtype=np.float32)
    else:
        target_mask = zeros

    out = np.stack([current, target, delta, target_mask], axis=0)
    return out.astype(np.float32, copy=False)
