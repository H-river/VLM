#!/usr/bin/env python3
"""Build pixel-dependent visual evidence supervision with calibrated images."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import simulate_and_measure

from .core import make_qwen_record, read_jsonl, stable_json_hash, write_jsonl
from .visual_v10 import render_calibrated_images, render_signed_difference, sensor_frame_state


TRANSFORMS = ("original", "horizontal_flip", "vertical_flip")
DIRECTION_VALUES = ("increase", "decrease", "no_change")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", type=Path, required=True)
    parser.add_argument("--train-master", type=Path, required=True)
    parser.add_argument("--dev-records", type=Path, required=True)
    parser.add_argument("--dev-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size-px", type=int, default=384)
    parser.add_argument("--dataset-version", default="visual_evidence_v10")
    parser.add_argument("--sensor-crop-px", type=int)
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--include-difference", action="store_true")
    return parser.parse_args()


def master_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group in read_jsonl(path):
        for item in group["records"]:
            result[item["record"]["example_id"]] = item
    return result


def replay_name(record: Mapping[str, Any], image_path: str) -> str:
    suffix = Path(image_path).stem[len(str(record["example_id"])) + 1 :]
    if suffix == "before" and record["task_type"] == "diagnosis":
        return "baseline"
    return suffix


def direction(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "increase"
    if delta < -threshold:
        return "decrease"
    return "no_change"


def state_regions(state: Mapping[str, float]) -> dict[str, str]:
    center = 511.5

    def center_band(value: float, low_label: str, high_label: str) -> str:
        if value < center - 5.0:
            return low_label
        if value > center + 5.0:
            return high_label
        return "centered"

    def width_band(value: float) -> str:
        if value < 110.0:
            return "narrow"
        if value > 135.0:
            return "wide"
        return "medium"

    return {
        "centroid_horizontal_region": center_band(
            float(state["centroid_x_px"]), "left_of_center", "right_of_center"
        ),
        "centroid_vertical_region": center_band(
            float(state["centroid_y_px"]), "above_center", "below_center"
        ),
        "sigma_x_band": width_band(float(state["sigma_x_px"])),
        "sigma_y_band": width_band(float(state["sigma_y_px"])),
    }


def pair_directions(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, str]:
    peak_scale = max(abs(float(before["peak_intensity"])), 1e-12)
    return {
        "centroid_x": direction(
            float(after["centroid_x_px"]) - float(before["centroid_x_px"]), 1.0
        ),
        "centroid_y": direction(
            float(after["centroid_y_px"]) - float(before["centroid_y_px"]), 1.0
        ),
        "sigma_x": direction(float(after["sigma_x_px"]) - float(before["sigma_x_px"]), 2.0),
        "sigma_y": direction(float(after["sigma_y_px"]) - float(before["sigma_y_px"]), 2.0),
        "peak_intensity": direction(
            (float(after["peak_intensity"]) - float(before["peak_intensity"])) / peak_scale,
            0.05,
        ),
    }


def swap(value: str, first: str, second: str) -> str:
    if value == first:
        return second
    if value == second:
        return first
    return value


def transformed_regions(regions: Mapping[str, str], transform: str) -> dict[str, str]:
    result = dict(regions)
    if transform == "horizontal_flip":
        result["centroid_horizontal_region"] = swap(
            result["centroid_horizontal_region"], "left_of_center", "right_of_center"
        )
    elif transform == "vertical_flip":
        result["centroid_vertical_region"] = swap(
            result["centroid_vertical_region"], "above_center", "below_center"
        )
    return result


def transformed_directions(values: Mapping[str, str], transform: str) -> dict[str, str]:
    result = dict(values)
    if transform == "horizontal_flip":
        result["centroid_x"] = swap(result["centroid_x"], "increase", "decrease")
    elif transform == "vertical_flip":
        result["centroid_y"] = swap(result["centroid_y"], "increase", "decrease")
    return result


def transform_image(source: Path, destination: Path, transform: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    if transform == "horizontal_flip":
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif transform == "vertical_flip":
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image.save(destination, format="PNG", optimize=True)


def state_prompt(calibration: Mapping[str, Any], transform: str) -> str:
    visible = {
        "coordinate_convention": "sensor x increases right; sensor y increases down",
        "image_transform": transform,
        "visual_aids": (
            "cyan center crosshair; yellow narrow/medium and medium/wide reference rings"
            if calibration.get("instrumented_overlay")
            else "none; raw calibrated sensor image"
        ),
    }
    return (
        "Read the beam image itself and classify its sensor-frame position and widths. "
        "Do not infer the answer from filenames.\n\nInput data:\n"
        + json.dumps(visible, indent=2, sort_keys=True)
        + "\n\nReturn only this strict JSON shape using literal string categories:\n"
        '{"status":"answerable","answer":{'
        '"centroid_horizontal_region":"left_of_center|centered|right_of_center",'
        '"centroid_vertical_region":"above_center|centered|below_center",'
        '"sigma_x_band":"narrow|medium|wide",'
        '"sigma_y_band":"narrow|medium|wide"}}'
    )


def pair_prompt(
    calibration: Mapping[str, Any], transform: str, include_difference: bool
) -> str:
    visible = {
        "coordinate_convention": "sensor x increases right; sensor y increases down",
        "image_transform": transform,
        "thresholds": {"centroid_px": 1.0, "sigma_px": 2.0, "peak_relative": 0.05},
        "image_order": (
            "first state, second state, then signed difference map"
            if include_difference
            else "first state followed by second state"
        ),
    }
    if include_difference:
        visible["difference_color_key"] = (
            "red means second brighter; blue means first brighter"
        )
    return (
        "Compare the two calibrated beam images and extract only the observed change directions. "
        "The prompt intentionally contains no action or candidate label; use the pixels.\n\nInput data:\n"
        + json.dumps(visible, indent=2, sort_keys=True)
        + "\n\nReturn only this strict JSON shape; every field must be a literal string category:\n"
        '{"status":"answerable","answer":{"observed_direction_set":{'
        '"centroid_x":"increase|decrease|no_change",'
        '"centroid_y":"increase|decrease|no_change",'
        '"sigma_x":"increase|decrease|no_change",'
        '"sigma_y":"increase|decrease|no_change",'
        '"peak_intensity":"increase|decrease|no_change"}}}'
    )


def build_split(
    records_path: Path,
    master_path: Path,
    output_dir: Path,
    split: str,
    size_px: int,
    dataset_version: str,
    sensor_crop_px: int | None,
    instrumented: bool,
    include_difference: bool,
    augment_train: bool = True,
) -> list[dict[str, Any]]:
    records = [row for row in read_jsonl(records_path) if row["modality"] == "visual"]
    indexed = master_index(master_path)
    generated: list[dict[str, Any]] = []
    transforms = TRANSFORMS if split == "train" and augment_train else ("original",)

    for source in records:
        item = indexed[source["example_id"]]
        specs = {spec["name"]: spec for spec in item["private_eval"]["replay_specs"]}
        results: list[tuple[str, dict[str, Any], dict[str, float]]] = []
        for index, old_image_path in enumerate(source["prompt_inputs"]["images"]):
            name = replay_name(source, old_image_path)
            spec = specs[name]
            setup = setup_from_dict(spec["setup_config"])
            result = simulate_and_measure(setup)
            results.append((f"state_{index}", result, sensor_frame_state(result, setup)))

        base_image_dir = output_dir / "images" / split / source["example_id"] / "original"
        _, calibration = render_calibrated_images(
            [(name, result["intensity"]) for name, result, _ in results],
            base_image_dir,
            size_px=size_px,
            sensor_crop_px=sensor_crop_px,
            instrumented=instrumented,
        )
        original_paths = [base_image_dir / f"{name}.png" for name, _, _ in results]
        difference_original: Path | None = None
        reverse_difference_original: Path | None = None
        if include_difference and len(results) == 2:
            difference_original = base_image_dir / "signed_difference.png"
            calibration["signed_difference"] = render_signed_difference(
                results[0][1]["intensity"],
                results[1][1]["intensity"],
                difference_original,
                size_px=size_px,
                sensor_crop_px=sensor_crop_px,
            )
            if split == "train" and augment_train:
                reverse_difference_original = base_image_dir / "signed_difference_reversed.png"
                render_signed_difference(
                    results[1][1]["intensity"],
                    results[0][1]["intensity"],
                    reverse_difference_original,
                    size_px=size_px,
                    sensor_crop_px=sensor_crop_px,
                )

        for transform in transforms:
            transformed_paths: list[str] = []
            for original_path in original_paths:
                destination = (
                    output_dir
                    / "images"
                    / split
                    / source["example_id"]
                    / transform
                    / original_path.name
                )
                if transform == "original":
                    destination = original_path
                else:
                    transform_image(original_path, destination, transform)
                transformed_paths.append(destination.relative_to(output_dir).as_posix())
            difference_path: str | None = None
            reverse_difference_path: str | None = None
            if difference_original is not None:
                destination = (
                    output_dir
                    / "images"
                    / split
                    / source["example_id"]
                    / transform
                    / difference_original.name
                )
                if transform == "original":
                    destination = difference_original
                else:
                    transform_image(difference_original, destination, transform)
                difference_path = destination.relative_to(output_dir).as_posix()
            if reverse_difference_original is not None:
                reverse_destination = (
                    output_dir
                    / "images"
                    / split
                    / source["example_id"]
                    / transform
                    / reverse_difference_original.name
                )
                if transform == "original":
                    reverse_destination = reverse_difference_original
                else:
                    transform_image(
                        reverse_difference_original, reverse_destination, transform
                    )
                reverse_difference_path = reverse_destination.relative_to(output_dir).as_posix()

            for image_index, (_, _, state) in enumerate(results):
                target = {
                    "status": "answerable",
                    "answer": transformed_regions(state_regions(state), transform),
                }
                example_id = (
                    f"{source['example_id']}__v10_state_{image_index}__{transform}"
                )
                record = {
                    "example_id": example_id,
                    "group_id": source["group_id"],
                    "modality": "visual",
                    "prompt": state_prompt(calibration, transform),
                    "prompt_inputs": {
                        "images": [transformed_paths[image_index]],
                        "render_calibration": calibration,
                    },
                    "provenance": {
                        "dataset_version": dataset_version,
                        "label_source": "simulator_sensor_frame",
                        "source_example_id": source["example_id"],
                        "source_task_type": source["task_type"],
                        "transform": transform,
                    },
                    "source_example_id": source["example_id"],
                    "source_task_type": source["task_type"],
                    "split": split,
                    "target": target,
                    "task_type": "visual_state_classification",
                }
                generated.append(record)

            if len(results) == 2:
                directions = pair_directions(results[0][2], results[1][2])
                target = {
                    "status": "answerable",
                    "answer": {
                        "observed_direction_set": transformed_directions(directions, transform)
                    },
                }
                generated.append(
                    {
                        "example_id": f"{source['example_id']}__v10_pair__{transform}",
                        "group_id": source["group_id"],
                        "modality": "visual",
                        "prompt": pair_prompt(calibration, transform, include_difference),
                        "prompt_inputs": {
                            "images": transformed_paths
                            + ([difference_path] if difference_path is not None else []),
                            "render_calibration": calibration,
                        },
                        "provenance": {
                            "dataset_version": dataset_version,
                            "label_source": "simulator_sensor_frame_difference",
                            "source_example_id": source["example_id"],
                            "source_task_type": source["task_type"],
                            "transform": transform,
                        },
                        "source_example_id": source["example_id"],
                        "source_task_type": source["task_type"],
                        "split": split,
                        "target": target,
                        "task_type": "visual_pair_direction_extraction",
                    }
                )
                if split == "train" and augment_train:
                    reverse_directions = pair_directions(results[1][2], results[0][2])
                    generated.append(
                        {
                            "example_id": f"{source['example_id']}__v10_pair_reversed__{transform}",
                            "group_id": source["group_id"],
                            "modality": "visual",
                            "prompt": pair_prompt(calibration, transform, include_difference),
                            "prompt_inputs": {
                                "images": list(reversed(transformed_paths))
                                + (
                                    [reverse_difference_path]
                                    if reverse_difference_path is not None
                                    else []
                                ),
                                "render_calibration": calibration,
                            },
                            "provenance": {
                                "dataset_version": dataset_version,
                                "label_source": "simulator_sensor_frame_difference",
                                "source_example_id": source["example_id"],
                                "source_task_type": source["task_type"],
                                "transform": transform,
                                "pair_order": "reversed",
                            },
                            "source_example_id": source["example_id"],
                            "source_task_type": source["task_type"],
                            "split": split,
                            "target": {
                                "status": "answerable",
                                "answer": {
                                    "observed_direction_set": transformed_directions(
                                        reverse_directions, transform
                                    )
                                },
                            },
                            "task_type": "visual_pair_direction_extraction",
                        }
                    )
    return generated


def main() -> None:
    args = parse_args()
    train = build_split(
        args.train_records,
        args.train_master,
        args.output_dir,
        "train",
        args.size_px,
        args.dataset_version,
        args.sensor_crop_px,
        args.instrumented,
        args.include_difference,
    )
    dev = build_split(
        args.dev_records,
        args.dev_master,
        args.output_dir,
        "dev",
        args.size_px,
        args.dataset_version,
        args.sensor_crop_px,
        args.instrumented,
        args.include_difference,
    )
    for split, rows in (("train", train), ("dev", dev)):
        write_jsonl(args.output_dir / "canonical" / f"{split}.jsonl", rows)
        write_jsonl(
            args.output_dir / "exports" / "qwen" / f"{split}.jsonl",
            (make_qwen_record(row, include_target=True) for row in rows),
        )

    direction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train:
        values = row["target"]["answer"].get("observed_direction_set", {})
        for field, value in values.items():
            direction_counts[field][value] += 1
    manifest = {
        "dataset": args.dataset_version,
        "train_records": len(train),
        "dev_records": len(dev),
        "train_groups": len({row["group_id"] for row in train}),
        "dev_groups": len({row["group_id"] for row in dev}),
        "group_overlap": sorted(
            {row["group_id"] for row in train} & {row["group_id"] for row in dev}
        ),
        "task_counts": {
            split: dict(Counter(row["task_type"] for row in rows))
            for split, rows in (("train", train), ("dev", dev))
        },
        "train_direction_counts": {
            field: dict(counts) for field, counts in sorted(direction_counts.items())
        },
        "canonical_hashes": {
            "train": stable_json_hash(train),
            "dev": stable_json_hash(dev),
        },
        "claims": {
            "pixel_dependent_prompts": True,
            "sensor_frame_labels": True,
            "shared_pair_calibration": True,
            "instrumented_overlay": args.instrumented,
            "signed_difference": args.include_difference,
            "sensor_crop_px": args.sensor_crop_px,
            "native_exact_metrology": False,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
