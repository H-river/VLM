"""Utilities for reading simulator outputs into canonical v2 sample dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_metadata(metadata_path: str | Path) -> dict:
    """Load JSON metadata from a simulator sample directory."""
    metadata_path = Path(metadata_path)
    with open(metadata_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata is not a dict: {metadata_path}")
    return data


def find_sample_dirs(sim_dir: str | Path) -> list[Path]:
    """Find sample directories containing metadata.json in deterministic order."""
    sim_dir = Path(sim_dir)
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulator directory not found: {sim_dir}")

    sample_dirs = sorted({p.parent for p in sim_dir.rglob("metadata.json")}, key=lambda p: str(p))
    return sample_dirs


def find_profile_path(sample_dir: str | Path, strict: bool = True) -> Path | None:
    """Resolve intensity.npy path for v2 profile input."""
    sample_dir = Path(sample_dir)
    intensity_path = sample_dir / "intensity.npy"
    if intensity_path.exists():
        return intensity_path

    if strict:
        raise FileNotFoundError(f"Missing required intensity.npy in sample: {sample_dir}")
    return None


def _get_nested(data: dict, path: list[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(".".join(path))
        cur = cur[key]
    return cur


def extract_setup_vector(metadata: dict, strict: bool = True) -> dict | None:
    """Extract canonical v2 setup values from metadata."""
    required_paths = {
        "source_to_lens": ["setup", "geometry", "laser_to_lens"],
        "lens_to_camera": ["setup", "geometry", "lens_to_camera"],
        "focal_length": ["setup", "lens", "focal_length"],
        "lens_x": ["setup", "lens", "x_offset"],
        "lens_y": ["setup", "lens", "y_offset"],
        "camera_x": ["setup", "camera", "x_offset"],
        "camera_y": ["setup", "camera", "y_offset"],
    }

    setup: dict[str, float] = {}
    try:
        for name, path in required_paths.items():
            value = _get_nested(metadata, path)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Non-numeric value at {'.'.join(path)}: {value!r}")
            setup[name] = float(value)
    except Exception as exc:
        if strict:
            raise ValueError(f"Invalid v2 setup metadata, missing/invalid field: {exc}") from exc
        return None

    return setup


def _json_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def extract_beam_metrics(metadata: dict) -> dict:
    """Extract beam/profile metrics from metadata into JSON-serializable values."""
    candidates: list[dict[str, Any]] = []

    for key in ("metrics", "beam_metrics", "profile_metrics"):
        sub = metadata.get(key)
        if isinstance(sub, dict):
            candidates.append(sub)

    if isinstance(metadata.get("setup"), dict):
        setup_metrics = metadata["setup"].get("metrics")
        if isinstance(setup_metrics, dict):
            candidates.append(setup_metrics)

    if not candidates:
        return {}

    key_map = {
        "centroid_x": ["centroid_x"],
        "centroid_y": ["centroid_y"],
        "sigma_x": ["sigma_x"],
        "sigma_y": ["sigma_y"],
        "fwhm_x": ["fwhm_x"],
        "fwhm_y": ["fwhm_y"],
        "ellipticity": ["ellipticity"],
        "rotation": ["rotation", "rotation_angle"],
        "peak_intensity": ["peak_intensity"],
        "total_power": ["total_power", "integrated_power"],
        "total_intensity": ["total_intensity", "sum_intensity"],
    }

    out: dict[str, float] = {}
    for out_name, aliases in key_map.items():
        for metrics in candidates:
            found = None
            for alias in aliases:
                if alias in metrics:
                    found = metrics[alias]
                    break
            if found is not None:
                value = _json_number(found)
                if value is not None:
                    out[out_name] = value
                    break

    return out


def load_sample(sample_dir: str | Path, strict: bool = True) -> dict | None:
    """Load sample from simulator output directory into v2 canonical structure."""
    sample_dir = Path(sample_dir)
    metadata_path = sample_dir / "metadata.json"

    try:
        metadata = load_metadata(metadata_path)
        profile_path = find_profile_path(sample_dir, strict=strict)
        setup = extract_setup_vector(metadata, strict=strict)

        if profile_path is None or setup is None:
            if strict:
                raise ValueError(f"Invalid sample: {sample_dir}")
            return None

        return {
            "sample_id": sample_dir.name,
            "sample_dir": str(sample_dir),
            "profile_path": str(profile_path),
            "metadata_path": str(metadata_path),
            "setup": setup,
            "metrics": extract_beam_metrics(metadata),
        }

    except Exception:
        if strict:
            raise
        return None
