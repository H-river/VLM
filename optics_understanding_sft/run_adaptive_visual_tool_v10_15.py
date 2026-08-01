#!/usr/bin/env python3
"""Run quality-routed clean/robust deterministic visual evidence tools."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .core import read_jsonl, write_jsonl
from .visual_state_tool_v10_1 import (
    classify,
    classify_pair,
    extract_features,
    extract_pair_features,
    load_calibration,
)


def image_quality(path: Path) -> dict[str, float]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    grayscale = rgb.min(axis=2)
    border_width = max(1, min(grayscale.shape) // 10)
    border = np.concatenate(
        [
            grayscale[:border_width, :].reshape(-1),
            grayscale[-border_width:, :].reshape(-1),
            grayscale[border_width:-border_width, :border_width].reshape(-1),
            grayscale[border_width:-border_width, -border_width:].reshape(-1),
        ]
    )
    background = float(np.median(border))
    deviations = np.abs(border - background)
    noise_scale = max(
        1.4826 * float(np.median(deviations)),
        float(np.percentile(deviations, 84.0)),
    )
    return {
        "border_noise_scale": noise_scale,
        "colored_fraction": float(np.mean(np.ptp(rgb, axis=2) > 1.0)),
        "max_channel_value": float(np.max(rgb)),
    }


def quality_route(qualities: list[Mapping[str, float]]) -> str:
    if max(float(value["colored_fraction"]) for value in qualities) >= 0.02:
        return "blurred"
    if max(float(value["border_noise_scale"]) for value in qualities) >= 1.5:
        if max(float(value.get("max_channel_value", 255.0)) for value in qualities) < 200.0:
            return "dim_noisy"
        return "noisy"
    return "clean"


def state_features(path: Path, calibration: Mapping[str, Any]) -> dict[str, float]:
    return extract_features(
        path,
        sensor_crop_px=int(calibration.get("sensor_crop_px", 512)),
        overlay_handling=str(calibration.get("overlay_handling", "mask_zero")),
        noise_floor_sigma=float(calibration.get("noise_floor_sigma", 0.0)),
        relative_floor=float(calibration.get("relative_floor", 0.0)),
        denoise_passes=int(calibration.get("denoise_passes", 0)),
    )


def pair_features(
    first: Path, second: Path, calibration: Mapping[str, Any]
) -> dict[str, float]:
    return extract_pair_features(
        first,
        second,
        sensor_crop_px=int(calibration.get("sensor_crop_px", 512)),
        peak_feature=str(calibration.get("peak_feature", "energy_squared_ratio")),
        overlay_handling=str(calibration.get("overlay_handling", "mask_zero")),
        sigma_x_power=float(calibration.get("sigma_x_power", 1.0)),
        sigma_y_power=float(calibration.get("sigma_y_power", 1.0)),
        noise_floor_sigma=float(calibration.get("noise_floor_sigma", 0.0)),
        relative_floor=float(calibration.get("relative_floor", 0.0)),
        width_estimator=str(calibration.get("width_estimator", "moments_2d")),
        include_differential_features=bool(
            calibration.get("include_differential_features", False)
        ),
        differential_floor_sigma=float(
            calibration.get("differential_floor_sigma", 3.0)
        ),
        denoise_passes=int(calibration.get("denoise_passes", 0)),
    )


def adaptive_state_answer(
    path: Path,
    clean: Mapping[str, Any],
    robust: Mapping[str, Any],
    dimnoise: Mapping[str, Any] | None = None,
    noise: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    qualities = [image_quality(path)]
    route = quality_route(qualities)
    selected = (
        clean
        if route == "clean"
        else dimnoise
        if route == "dim_noisy" and dimnoise is not None
        else noise
        if route == "noisy" and noise is not None
        else robust
    )
    features = state_features(path, selected)
    return classify(features, selected), {
        "quality": qualities,
        "route": route,
        "selected": features,
    }


def adaptive_pair_answer(
    first: Path,
    second: Path,
    clean: Mapping[str, Any],
    robust: Mapping[str, Any],
    noise_width: Mapping[str, Any] | None = None,
    dimnoise_width: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    qualities = [image_quality(path) for path in (first, second)]
    route = quality_route(qualities)
    clean_features = pair_features(first, second, clean)
    clean_answer = classify_pair(clean_features, clean)
    robust_features = None
    robust_answer = None
    width_features = None
    width_answer = None
    if route != "clean":
        robust_features = pair_features(first, second, robust)
        robust_answer = classify_pair(robust_features, robust)
    selected_width = (
        noise_width
        if route == "noisy"
        else dimnoise_width
        if route == "dim_noisy"
        else None
    )
    if selected_width is not None:
        width_features = pair_features(first, second, selected_width)
        width_answer = classify_pair(width_features, selected_width)
    if route == "blurred" and robust_answer is not None:
        answer = {
            **clean_answer,
            "sigma_x": robust_answer["sigma_x"],
            "sigma_y": robust_answer["sigma_y"],
        }
    elif route in {"noisy", "dim_noisy"} and robust_answer is not None:
        answer = {
            **clean_answer,
            "centroid_x": robust_answer["centroid_x"],
            "centroid_y": robust_answer["centroid_y"],
        }
    else:
        answer = clean_answer
    if width_answer is not None:
        answer = {
            **answer,
            "sigma_x": width_answer["sigma_x"],
            "sigma_y": width_answer["sigma_y"],
        }
    return answer, {
        "quality": qualities,
        "route": route,
        "clean": clean_features,
        "robust": robust_features,
        "width": width_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("state", "pair"), required=True)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--clean-calibration", type=Path, required=True)
    parser.add_argument("--robust-calibration", type=Path, required=True)
    parser.add_argument("--dimnoise-calibration", type=Path)
    parser.add_argument("--noise-calibration", type=Path)
    parser.add_argument("--noise-pair-width-calibration", type=Path)
    parser.add_argument("--dimnoise-pair-width-calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    clean = load_calibration(args.clean_calibration)
    robust = load_calibration(args.robust_calibration)
    dimnoise = (
        load_calibration(args.dimnoise_calibration)
        if args.dimnoise_calibration
        else robust
    )
    noise = (
        load_calibration(args.noise_calibration)
        if args.noise_calibration
        else robust
    )
    noise_pair_width = (
        load_calibration(args.noise_pair_width_calibration)
        if args.noise_pair_width_calibration
        else None
    )
    dimnoise_pair_width = (
        load_calibration(args.dimnoise_pair_width_calibration)
        if args.dimnoise_pair_width_calibration
        else None
    )
    task_type = (
        "visual_state_classification"
        if args.mode == "state"
        else "visual_pair_direction_extraction"
    )
    records = [row for row in read_jsonl(args.records_jsonl) if row["task_type"] == task_type]
    predictions = []
    for row in records:
        paths = [args.image_root / value for value in row["prompt_inputs"]["images"][:2]]
        if args.mode == "state":
            answer, tool_features = adaptive_state_answer(
                paths[0], clean, robust, dimnoise, noise
            )
        else:
            answer, tool_features = adaptive_pair_answer(
                paths[0],
                paths[1],
                clean,
                robust,
                noise_pair_width,
                dimnoise_pair_width,
            )
            answer = {"observed_direction_set": answer}
        predictions.append(
            {
                "example_id": row["example_id"],
                "group_id": row["group_id"],
                "task_type": row["task_type"],
                "modality": "adaptive_visual_tool",
                "parsed_json": {"status": "answerable", "answer": answer},
                "parse_error": None,
                "tool_features": tool_features,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.jsonl", records)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    print(f"wrote {len(predictions)} adaptive {args.mode} predictions")


if __name__ == "__main__":
    main()
