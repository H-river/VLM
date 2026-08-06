"""Visualization helpers for LLM API profile2setup evaluations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from profile2setup.evaluation.profile_metrics import load_intensity
from profile2setup.llm_api.image_rendering import normalize_intensity
from profile2setup.schema import VARIABLE_ORDER

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
_LABEL_HEIGHT = 26
_RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _safe_filename(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "record")).strip("._")
    return safe or "record"


def _resolve_path(value: Any, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _first_path(record: dict, top_key: str, ref_key: str, repo_root: Path) -> Path | None:
    value = _resolve_path(record.get(top_key), repo_root)
    if value is not None:
        return value
    ref = record.get("profile_loss_reference")
    if isinstance(ref, dict):
        return _resolve_path(ref.get(ref_key), repo_root)
    return None


def _metadata_paths(record: dict, repo_root: Path) -> dict[str, Path | None]:
    return {
        "target_metadata_path": _first_path(record, "target_metadata_path", "target_metadata_path", repo_root),
        "current_metadata_path": _first_path(record, "current_metadata_path", "current_metadata_path", repo_root),
        "target_profile_path": _first_path(record, "target_profile_path", "target_profile_path", repo_root),
        "current_profile_path": _first_path(record, "current_profile_path", "current_profile_path", repo_root),
    }


def _choose_base_config(paths: dict[str, Path | None], simulation_policy: str) -> tuple[str, dict, list[str]]:
    from profile2setup.evaluation.closed_loop import (
        _choose_policy_config,
        load_base_simulator_config_from_metadata,
    )

    target_meta = paths.get("target_metadata_path")
    if target_meta is None:
        raise ValueError("missing target_metadata_path")
    target_config = load_base_simulator_config_from_metadata(target_meta)
    string_paths = {key: None if value is None else str(value) for key, value in paths.items()}
    return _choose_policy_config(string_paths, target_config, simulation_policy)


def simulate_api_prediction(
    *,
    data_record: dict,
    predicted_setup: dict,
    simulation_policy: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the optical simulator for an API-predicted physical setup."""
    from profile2setup.evaluation.closed_loop import (
        apply_predicted_setup_to_sim_config,
        compute_closed_loop_profile_metrics,
        simulate_intensity_from_config,
    )

    paths = _metadata_paths(data_record, repo_root)
    target_profile_path = paths.get("target_profile_path")
    if target_profile_path is None:
        raise ValueError("missing target_profile_path")
    if not target_profile_path.exists():
        raise FileNotFoundError(f"target_profile_path does not exist: {target_profile_path}")

    policy_used, base_config, warnings = _choose_base_config(paths, simulation_policy)
    pred_config = apply_predicted_setup_to_sim_config(base_config, predicted_setup)
    predicted_intensity = simulate_intensity_from_config(pred_config)
    target_intensity = load_intensity(target_profile_path)
    current_intensity = None
    current_profile_path = paths.get("current_profile_path")
    if current_profile_path is not None and current_profile_path.exists():
        current_intensity = load_intensity(current_profile_path)

    metrics = compute_closed_loop_profile_metrics(predicted_intensity, target_intensity)
    return {
        "predicted_intensity": predicted_intensity,
        "target_intensity": target_intensity,
        "current_intensity": current_intensity,
        "profile_metrics": metrics,
        "simulation_policy_used": policy_used,
        "warnings": warnings,
        "paths": {key: None if value is None else str(value) for key, value in paths.items()},
    }


def _resize_display(array: np.ndarray, size: int) -> np.ndarray:
    norm = normalize_intensity(array, mode="max")
    pixels = np.rint(norm * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    if image.size != (size, size):
        image = image.resize((size, size), resample=_RESAMPLE)
    return np.asarray(image, dtype=np.float32) / 255.0


def _gray_panel(array: np.ndarray, *, label: str, size: int) -> Image.Image:
    display = _resize_display(array, size)
    pixels = np.rint(display * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    return _with_label(image, label)


def _error_panel(target: np.ndarray, predicted: np.ndarray, *, size: int) -> Image.Image:
    target_display = _resize_display(target, size)
    pred_display = _resize_display(predicted, size)
    error = np.abs(target_display - pred_display)
    scale = float(np.max(error)) if error.size else 0.0
    if scale > 0.0:
        error = error / scale
    red = np.rint(np.clip(error, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.zeros((*red.shape, 3), dtype=np.uint8)
    rgb[..., 0] = red
    image = Image.fromarray(rgb, mode="RGB")
    return _with_label(image, "ABS ERROR")


def _with_label(image: Image.Image, label: str) -> Image.Image:
    text = str(label or "").strip().upper()
    if not text:
        return image
    canvas = Image.new("RGB", (image.width, image.height + _LABEL_HEIGHT), color=(255, 255, 255))
    canvas.paste(image, (0, _LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = max((canvas.width - (bbox[2] - bbox[0])) // 2, 0)
    y = max((_LABEL_HEIGHT - (bbox[3] - bbox[1])) // 2 - 1, 0)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return canvas


def save_comparison_png(
    *,
    out_path: Path,
    target_intensity: np.ndarray,
    predicted_intensity: np.ndarray,
    current_intensity: np.ndarray | None = None,
    size: int = 256,
) -> str:
    """Save current/target/predicted/error comparison image."""
    panels: list[Image.Image] = []
    if current_intensity is not None:
        panels.append(_gray_panel(current_intensity, label="CURRENT", size=size))
    panels.extend(
        [
            _gray_panel(target_intensity, label="TARGET GT", size=size),
            _gray_panel(predicted_intensity, label="PREDICTED", size=size),
            _error_panel(target_intensity, predicted_intensity, size=size),
        ]
    )

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=False)
    return str(out_path)


def _setup_abs_error(predicted_setup: dict | None, groundtruth_setup: dict | None) -> dict | None:
    if not isinstance(predicted_setup, dict) or not isinstance(groundtruth_setup, dict):
        return None
    return {
        name: float(abs(float(predicted_setup[name]) - float(groundtruth_setup[name])))
        for name in CANONICAL_VARIABLE_ORDER
        if name in predicted_setup and name in groundtruth_setup
    }


def _metric_summary(metrics: dict | None) -> dict[str, float | None]:
    if not isinstance(metrics, dict):
        return {
            "normalized_profile_mse": None,
            "centroid_error_px": None,
            "sigma_error_px": None,
        }
    centroid_error = None
    if metrics.get("centroid_x_error_px") is not None and metrics.get("centroid_y_error_px") is not None:
        centroid_error = float(
            np.mean([float(metrics["centroid_x_error_px"]), float(metrics["centroid_y_error_px"])])
        )
    sigma_error = None
    if metrics.get("sigma_x_error_px") is not None and metrics.get("sigma_y_error_px") is not None:
        sigma_error = float(np.mean([float(metrics["sigma_x_error_px"]), float(metrics["sigma_y_error_px"])]))
    return {
        "normalized_profile_mse": None if metrics.get("normalized_mse") is None else float(metrics["normalized_mse"]),
        "centroid_error_px": centroid_error,
        "sigma_error_px": sigma_error,
    }


def save_api_visualization_example(
    *,
    data_record: dict,
    prediction_row: dict,
    parsed_prediction: dict,
    predicted_setup: dict,
    predicted_delta: dict | None,
    simulator_result: dict,
    out_dir: Path,
    index: int,
    valid_json: bool,
    repo_root: Path,
    image_size: int = 256,
) -> dict[str, Any]:
    """Save one comparison PNG plus setup-comparison JSON and return index row."""
    record_id = str(data_record.get("id") or prediction_row.get("record_id") or f"row_{index}")
    prefix = f"{index:03d}_{_safe_filename(record_id)}"
    comparison_png = out_dir / f"{prefix}_comparison.png"
    setup_json = out_dir / f"{prefix}_setup_comparison.json"

    save_comparison_png(
        out_path=comparison_png,
        current_intensity=simulator_result.get("current_intensity"),
        target_intensity=simulator_result["target_intensity"],
        predicted_intensity=simulator_result["predicted_intensity"],
        size=image_size,
    )

    target_setup = data_record.get("target_setup")
    setup_payload = {
        "record_id": record_id,
        "task_type": data_record.get("task_type"),
        "prompt": data_record.get("prompt"),
        "current_setup": data_record.get("current_setup"),
        "groundtruth_setup": target_setup,
        "predicted_setup": predicted_setup,
        "predicted_delta": predicted_delta,
        "absolute_error_per_variable": _setup_abs_error(predicted_setup, target_setup),
        "simulator_metrics": simulator_result.get("profile_metrics"),
        "simulation_policy_used": simulator_result.get("simulation_policy_used"),
        "warnings": simulator_result.get("warnings") or [],
        "profile_paths": simulator_result.get("paths"),
        "valid_json": bool(valid_json),
        "parsed_prediction": parsed_prediction,
        "prediction_row_reference": {
            "line_number": prediction_row.get("line_number"),
            "status": prediction_row.get("status"),
            "provider": prediction_row.get("provider"),
            "model": prediction_row.get("model"),
        },
        "comparison_png": str(comparison_png),
    }
    setup_json.parent.mkdir(parents=True, exist_ok=True)
    with setup_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(setup_payload), f, indent=2, sort_keys=True)

    metrics = _metric_summary(simulator_result.get("profile_metrics"))
    return {
        "record_id": record_id,
        "comparison_png": str(comparison_png),
        "setup_comparison_json": str(setup_json),
        "valid_json": bool(valid_json),
        **metrics,
    }


def save_visualization_index(out_dir: Path, examples: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    payload = {
        "out_dir": str(out_dir),
        "num_saved": len(examples),
        "examples": examples,
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
    return {
        "enabled": True,
        "out_dir": str(out_dir),
        "num_saved": len(examples),
        "index_path": str(index_path),
    }
