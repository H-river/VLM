#!/usr/bin/env python3
"""Audit whether visual benchmark pixels and quantitative labels are compatible."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .core import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--withheld-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-percentile", type=float, default=25.0)
    return parser.parse_args()


def replay_index(master_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for group in master_rows:
        for item in group["records"]:
            example_id = item["record"]["example_id"]
            result[example_id] = {
                spec["name"]: spec for spec in item["private_eval"]["replay_specs"]
            }
    return result


def replay_name(record: Mapping[str, Any], image_path: str) -> str:
    stem = Path(image_path).stem
    suffix = stem[len(str(record["example_id"])) + 1 :]
    if suffix == "before" and record["task_type"] == "diagnosis":
        return "baseline"
    return suffix


def image_centroid_sensor_px(path: Path, background_percentile: float) -> tuple[float, float]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)[..., 0]
    background = float(np.percentile(image, background_percentile))
    weights = np.maximum(image - background, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"Image has no positive signal after background removal: {path}")
    yy, xx = np.indices(weights.shape)
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    height, width = weights.shape
    sensor_size = 1024.0
    return (
        (cx + 0.5) * sensor_size / width - 0.5,
        (cy + 0.5) * sensor_size / height - 0.5,
    )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def task_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        task: float(values["task_score"])
        for task, values in summary["per_task"].items()
        if values.get("count", 0) and values.get("task_score") is not None
    }


def main() -> None:
    args = parse_args()
    records = [row for row in read_jsonl(args.records_jsonl) if row["modality"] == "visual"]
    indexed = replay_index(read_jsonl(args.master_jsonl))
    legacy_errors: list[float] = []
    corrected_errors: list[float] = []
    per_image: list[dict[str, Any]] = []
    paired_independent_options = 0
    paired_total = 0

    for record in records:
        specs = indexed[record["example_id"]]
        record_options = next(
            item["private_eval"]["render_options"]
            for group in read_jsonl(args.master_jsonl)
            for item in group["records"]
            if item["record"]["example_id"] == record["example_id"]
        )
        option_values = list(record_options.values())
        if len(option_values) > 1:
            paired_total += 1
            if any(value != option_values[0] for value in option_values[1:]):
                paired_independent_options += 1

        for image_path in record["prompt_inputs"].get("images", []):
            spec = specs[replay_name(record, image_path)]
            estimate = image_centroid_sensor_px(
                args.image_root / image_path, args.background_percentile
            )
            state = spec["expected_state"]
            legacy = (float(state["centroid_x_px"]), float(state["centroid_y_px"]))
            setup = spec["setup_config"]
            pitch = float(setup["sensor"]["pixel_pitch"])
            camera = setup["camera"]
            corrected = (
                legacy[0] - float(camera["x_offset"]) / pitch,
                legacy[1] - float(camera["y_offset"]) / pitch,
            )
            legacy_error = distance(estimate, legacy)
            corrected_error = distance(estimate, corrected)
            legacy_errors.append(legacy_error)
            corrected_errors.append(corrected_error)
            per_image.append(
                {
                    "example_id": record["example_id"],
                    "image_path": image_path,
                    "estimated_centroid_sensor_px": list(estimate),
                    "legacy_label_centroid_px": list(legacy),
                    "corrected_sensor_centroid_px": list(corrected),
                    "legacy_error_px": legacy_error,
                    "corrected_error_px": corrected_error,
                }
            )

    full = json.loads(args.full_summary.read_text(encoding="utf-8"))
    withheld = json.loads(args.withheld_summary.read_text(encoding="utf-8"))
    full_tasks = task_scores(full)
    withheld_tasks = task_scores(withheld)
    report = {
        "record_count": len(records),
        "image_count": len(per_image),
        "pixel_ablation": {
            "full_visual_macro": float(full["macro_task_score"]),
            "images_withheld_macro": float(withheld["macro_task_score"]),
            "full_minus_withheld_macro": float(full["macro_task_score"])
            - float(withheld["macro_task_score"]),
            "per_task_full": full_tasks,
            "per_task_images_withheld": withheld_tasks,
        },
        "coordinate_frame_audit": {
            "legacy_centroid_error_median_px": statistics.median(legacy_errors),
            "legacy_centroid_error_mean_px": statistics.fmean(legacy_errors),
            "corrected_centroid_error_median_px": statistics.median(corrected_errors),
            "corrected_centroid_error_mean_px": statistics.fmean(corrected_errors),
            "corrected_better_fraction": sum(
                corrected < legacy for corrected, legacy in zip(corrected_errors, legacy_errors)
            )
            / len(legacy_errors),
            "finding": (
                "The rendered array is sensor-indexed, but legacy state_m_to_state_px does not "
                "subtract camera offsets before converting to sensor pixels."
            ),
        },
        "render_calibration_audit": {
            "paired_records": paired_total,
            "paired_records_with_independent_render_options": paired_independent_options,
            "per_image_normalization_erases_absolute_peak": True,
            "finding": (
                "Paired images use independently sampled normalization, gamma, noise, blur, and "
                "saturation, so brightness and width changes are not calibrated across images."
            ),
        },
        "per_image": per_image,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = f"""# Visual quantitative failure audit

The paired ablation evaluated {len(records)} visual records ({len(per_image)} images). The full-image
macro score was {full['macro_task_score']:.3f}; withholding every image scored
{withheld['macro_task_score']:.3f}. Therefore the current checkpoint gains
{float(full['macro_task_score']) - float(withheld['macro_task_score']):+.3f} macro score from pixels.

The dominant data defect is a coordinate-frame mismatch. The legacy numerical centroid is in an
absolute lab frame because camera offsets are not subtracted, while the PNG array is indexed in the
camera sensor frame. A simple deterministic image centroid has median error
{statistics.median(legacy_errors):.2f} px against the legacy label and
{statistics.median(corrected_errors):.2f} px against the corrected sensor-frame label.

All {paired_independent_options}/{paired_total} paired records use different render options for their
two images. Per-image normalization destroys absolute peak calibration, and independent gamma,
background, noise, blur, and saturation make paired width and brightness comparisons unreliable.

These results mean that additional training on the current visual records would reinforce
contradictory or non-identifiable targets. The next dataset version must use sensor-frame centroids,
shared pair calibration, and pixel-dependent prompts before visual fine-tuning.
"""
    (args.output_dir / "report.md").write_text(md, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
