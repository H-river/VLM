#!/usr/bin/env python3
"""Run the trained v3 measurement model on one calibrated image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measurement_rebuild_v3.common import (
    STATE_FIELDS,
    TRANSFORM_FIELDS,
    analytic_measurement,
    measurement_tolerance,
    transform_vector,
)
from measurement_rebuild_v3.models import measurement_model_v3, require_torch


CLEAN_TRANSFORM = {
    "exposure": 1.0,
    "gamma": 1.0,
    "noise_std": 0.0,
    "blur_sigma_px": 0.0,
    "saturation_level": 1.0,
    "crop_left_px": 0,
    "crop_right_px": 0,
    "crop_top_px": 0,
    "crop_bottom_px": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("calibration_json", type=Path)
    parser.add_argument("--transform-json", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def load_observed_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float32).mean(axis=2)
    else:
        array = array.astype(np.float32)
    maximum = float(array.max(initial=0.0))
    if maximum > 255.0:
        array /= 65535.0
    elif maximum > 1.0:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def inverse_camera_transfer(
    observed: np.ndarray, transform: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    missing = set(TRANSFORM_FIELDS) - set(transform)
    extra = set(transform) - set(TRANSFORM_FIELDS)
    if missing or extra:
        raise ValueError(
            f"transform fields differ: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    exposure = float(transform["exposure"])
    gamma = float(transform["gamma"])
    if exposure <= 0.0 or gamma <= 0.0:
        raise ValueError("exposure and gamma must be positive")
    valid = np.ones_like(observed, dtype=np.float32)
    left = int(transform["crop_left_px"])
    right = int(transform["crop_right_px"])
    top = int(transform["crop_top_px"])
    bottom = int(transform["crop_bottom_px"])
    if left:
        valid[:, :left] = 0.0
    if right:
        valid[:, -right:] = 0.0
    if top:
        valid[:top, :] = 0.0
    if bottom:
        valid[-bottom:, :] = 0.0
    linearized = (
        np.power(np.clip(observed, 0.0, 1.0), 1.0 / gamma) / exposure
    )
    return (linearized * valid).astype(np.float32), valid


def measure_image(
    artifact_path: Path,
    image_path: Path,
    calibration: Mapping[str, Any],
    transform: Mapping[str, Any] | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    device = torch.device(
        device_name
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=False
    )
    contract = artifact["input_contract"]
    model = measurement_model_v3(
        torch,
        calibration_dim=int(contract["calibration_dim"]),
        analytic_dim=int(contract["analytic_dim"]),
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    transform_out = dict(CLEAN_TRANSFORM if transform is None else transform)
    observed = load_observed_image(image_path)
    linearized, valid = inverse_camera_transfer(observed, transform_out)
    source_resolution = tuple(
        int(value) for value in calibration["source_sensor_resolution_px"]
    )
    baseline, analytic = analytic_measurement(
        linearized,
        valid,
        float(calibration["linear_intensity_high"]),
        source_resolution,
    )
    prediction_scale = measurement_tolerance(baseline)
    tensors = [
        torch.from_numpy(value[None]).to(device)
        for value in (
            observed[None].astype(np.float32),
            linearized[None],
            valid[None],
            transform_vector(calibration, transform_out),
            analytic,
        )
    ]
    with torch.inference_mode():
        correction = model(*tensors).float().cpu().numpy()[0]
    prediction = baseline + correction * prediction_scale
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("measurement produced a non-finite value")
    state = {
        field: float(prediction[index])
        for index, field in enumerate(STATE_FIELDS)
    }
    return {
        "beam_state": state,
        "analytic_baseline": {
            field: float(baseline[index])
            for index, field in enumerate(STATE_FIELDS)
        },
        "coordinate_frame": "camera_sensor_array",
        "model_version": str(artifact["version"]),
        "device": str(device),
        "simulator_at_inference": False,
    }


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    transform = (
        json.loads(args.transform_json.read_text(encoding="utf-8"))
        if args.transform_json
        else CLEAN_TRANSFORM
    )
    result = measure_image(
        args.artifact.resolve(),
        args.image.resolve(),
        calibration,
        transform,
        args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
