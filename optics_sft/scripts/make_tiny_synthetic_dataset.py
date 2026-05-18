#!/usr/bin/env python3
"""Create a tiny synthetic optics SFT dataset for pipeline smoke tests.

The generated examples are deliberately simple but visually varied. A target
Gaussian beam is optionally placed at a safe random center, a current beam is
shifted away from that target, and the label asks for actuator deltas roughly
opposite to the centroid error. This dataset is for executable pipeline
validation, not optical accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic optics SFT dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../VLM_data/optics_sft_tiny"),
        help="Directory where images, train.jsonl, and val.jsonl will be written.",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shift-min-px",
        type=float,
        default=10.0,
        help="Minimum current-vs-target centroid shift in pixels.",
    )
    parser.add_argument(
        "--shift-max-px",
        type=float,
        default=45.0,
        help="Maximum current-vs-target centroid shift in pixels.",
    )
    parser.add_argument(
        "--sigma-min-px",
        type=float,
        default=10.0,
        help="Minimum independently sampled Gaussian sigma in pixels.",
    )
    parser.add_argument(
        "--sigma-max-px",
        type=float,
        default=28.0,
        help="Maximum independently sampled Gaussian sigma in pixels.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.015,
        help="Gaussian noise std in normalized [0, 1] intensity units.",
    )
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=0.15,
        help="Per-image amplitude jitter around 1.0.",
    )
    parser.add_argument(
        "--background-max",
        type=float,
        default=0.04,
        help="Maximum weak background offset in normalized [0, 1] units.",
    )
    parser.add_argument(
        "--random-target-center",
        action="store_true",
        help="Sample a safe random target center instead of using the image center.",
    )
    parser.add_argument(
        "--make-debug-contact-sheet",
        action="store_true",
        help="Save a small current | target | absolute-difference contact sheet.",
    )
    return parser.parse_args()


def gaussian_beam(
    image_size: int,
    cx: float,
    cy: float,
    sigma_x: float,
    sigma_y: float,
    amplitude: float,
    background: float,
    noise_std: float,
    rng: Any,
) -> Any:
    import numpy as np

    y, x = np.mgrid[0:image_size, 0:image_size]
    beam = np.exp(
        -(
            ((x - cx) ** 2) / (2.0 * sigma_x**2)
            + ((y - cy) ** 2) / (2.0 * sigma_y**2)
        )
    )
    beam = background + amplitude * beam
    if noise_std > 0.0:
        beam = beam + rng.normal(0.0, noise_std, size=beam.shape)
    beam = np.clip(beam, 0.0, 1.0)
    beam = np.clip(beam * 255.0, 0, 255).astype(np.uint8)
    return np.stack([beam, beam, beam], axis=-1)


def save_image(array: Any, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def save_contact_sheet(current: Any, target: Any, path: Path) -> None:
    import numpy as np
    from PIL import Image

    diff = np.abs(current.astype(np.int16) - target.astype(np.int16)).astype(np.uint8)
    sheet = np.concatenate([current, target, diff], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, mode="RGB").save(path)


def sample_signed_shift(rng: random.Random, shift_min_px: float, shift_max_px: float) -> tuple[float, float]:
    magnitude = rng.uniform(shift_min_px, shift_max_px)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return magnitude * math.cos(angle), magnitude * math.sin(angle)


def sample_target_center(
    image_size: int,
    shift_max_px: float,
    sigma_max_px: float,
    rng: random.Random,
    random_target_center: bool,
) -> tuple[float, float]:
    center = (image_size - 1) / 2.0
    if not random_target_center:
        return center, center

    margin = max(4.0, shift_max_px + 2.0 * sigma_max_px)
    if margin >= center:
        return center, center

    return (
        rng.uniform(margin, image_size - 1 - margin),
        rng.uniform(margin, image_size - 1 - margin),
    )


def brightness_scale(rng: random.Random, brightness_jitter: float) -> float:
    low = max(0.1, 1.0 - brightness_jitter)
    high = max(low, 1.0 + brightness_jitter)
    return rng.uniform(low, high)


def make_shape_notes(dx_px: float, dy_px: float, sigma_x_ratio: float, sigma_y_ratio: float) -> str:
    horizontal = "right" if dx_px > 0 else "left"
    vertical = "below" if dy_px > 0 else "above"
    width_note = "wider" if sigma_x_ratio > 1.05 else "narrower" if sigma_x_ratio < 0.95 else "similar width"
    height_note = "taller" if sigma_y_ratio > 1.05 else "shorter" if sigma_y_ratio < 0.95 else "similar height"
    return (
        f"Current beam is shifted {horizontal} and {vertical} relative to target, "
        f"with {width_note} and {height_note}."
    )


def make_label(dx_px: float, dy_px: float, sigma_x_ratio: float, sigma_y_ratio: float) -> dict[str, Any]:
    gain_x = 0.002
    gain_y = 0.002
    return {
        "task": "beam_alignment",
        "diagnosis": {
            "centroid_error_px": {
                "x": round(dx_px, 3),
                "y": round(dy_px, 3),
            },
            "size_error": {
                "sigma_x_ratio": round(sigma_x_ratio, 4),
                "sigma_y_ratio": round(sigma_y_ratio, 4),
            },
            "shape_notes": make_shape_notes(dx_px, dy_px, sigma_x_ratio, sigma_y_ratio),
        },
        "control_plan": {
            "lens_x_delta_mm": round(-gain_x * dx_px, 5),
            "lens_y_delta_mm": round(-gain_y * dy_px, 5),
            "camera_x_delta_mm": 0.0,
            "camera_y_delta_mm": 0.0,
        },
        "confidence": 0.9,
    }


def make_metadata() -> dict[str, float]:
    return {
        "wavelength_nm": 632.8,
        "beam_waist_mm": 1.0,
        "lens_focal_length_mm": 100.0,
        "source_to_lens_mm": 200.0,
        "lens_to_camera_mm": 150.0,
    }


def make_sample(
    index: int,
    output_dir: Path,
    image_size: int,
    rng: random.Random,
    np_rng: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_x, target_y = sample_target_center(
        image_size,
        args.shift_max_px,
        args.sigma_max_px,
        rng,
        args.random_target_center,
    )
    dx_px, dy_px = sample_signed_shift(rng, args.shift_min_px, args.shift_max_px)
    current_x = target_x + dx_px
    current_y = target_y + dy_px

    current_sigma_x = rng.uniform(args.sigma_min_px, args.sigma_max_px)
    current_sigma_y = rng.uniform(args.sigma_min_px, args.sigma_max_px)
    target_sigma_x = rng.uniform(args.sigma_min_px, args.sigma_max_px)
    target_sigma_y = rng.uniform(args.sigma_min_px, args.sigma_max_px)
    current_brightness = brightness_scale(rng, args.brightness_jitter)
    target_brightness = brightness_scale(rng, args.brightness_jitter)
    current_background = rng.uniform(0.0, args.background_max)
    target_background = rng.uniform(0.0, args.background_max)

    current_name = f"current_{index:05d}.png"
    target_name = f"target_{index:05d}.png"
    current_image = gaussian_beam(
        image_size,
        current_x,
        current_y,
        current_sigma_x,
        current_sigma_y,
        current_brightness,
        current_background,
        args.noise_std,
        np_rng,
    )
    target_image = gaussian_beam(
        image_size,
        target_x,
        target_y,
        target_sigma_x,
        target_sigma_y,
        target_brightness,
        target_background,
        args.noise_std,
        np_rng,
    )
    save_image(current_image, output_dir / "images" / current_name)
    save_image(target_image, output_dir / "images" / target_name)
    if args.make_debug_contact_sheet and index < 12:
        save_contact_sheet(
            current_image,
            target_image,
            output_dir / "debug_contact_sheets" / f"sample_{index:05d}.png",
        )

    sigma_x_ratio = current_sigma_x / target_sigma_x
    sigma_y_ratio = current_sigma_y / target_sigma_y
    metadata = make_metadata()
    metadata.update(
        {
            "current_centroid_px": {
                "x": round(current_x, 3),
                "y": round(current_y, 3),
            },
            "target_centroid_px": {
                "x": round(target_x, 3),
                "y": round(target_y, 3),
            },
            "centroid_error_px": {
                "x": round(dx_px, 3),
                "y": round(dy_px, 3),
            },
            "current_sigma_px": {
                "x": round(current_sigma_x, 3),
                "y": round(current_sigma_y, 3),
            },
            "target_sigma_px": {
                "x": round(target_sigma_x, 3),
                "y": round(target_sigma_y, 3),
            },
            "image_size": image_size,
            "noise_std": args.noise_std,
            "brightness_scale": {
                "current": round(current_brightness, 4),
                "target": round(target_brightness, 4),
            },
        }
    )

    return {
        "current_image_path": current_name,
        "target_image_path": target_name,
        "metadata": metadata,
        "label": make_label(dx_px, dy_px, sigma_x_ratio, sigma_y_ratio),
    }


def split_rows(rows: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0.0, 1.0)")
    val_count = int(round(len(rows) * val_ratio))
    if rows and val_ratio > 0.0:
        val_count = max(1, val_count)
    return rows[val_count:], rows[:val_count]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")
    if args.shift_min_px < 0.0 or args.shift_max_px < args.shift_min_px:
        raise ValueError("--shift-max-px must be >= --shift-min-px and shifts must be non-negative")
    if args.sigma_min_px <= 0.0 or args.sigma_max_px < args.sigma_min_px:
        raise ValueError("--sigma-max-px must be >= --sigma-min-px and sigmas must be positive")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be non-negative")
    if args.brightness_jitter < 0.0:
        raise ValueError("--brightness-jitter must be non-negative")
    if args.background_max < 0.0:
        raise ValueError("--background-max must be non-negative")

    rng = random.Random(args.seed)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install numpy to generate synthetic beams.") from exc

    np_rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        make_sample(index, args.output_dir, args.image_size, rng, np_rng, args)
        for index in range(args.num_samples)
    ]
    rng.shuffle(rows)
    train_rows, val_rows = split_rows(rows, args.val_ratio)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)

    print(f"Wrote {len(train_rows)} train rows to {args.output_dir / 'train.jsonl'}")
    print(f"Wrote {len(val_rows)} val rows to {args.output_dir / 'val.jsonl'}")
    print(f"Wrote images to {args.output_dir / 'images'}")


if __name__ == "__main__":
    main()
