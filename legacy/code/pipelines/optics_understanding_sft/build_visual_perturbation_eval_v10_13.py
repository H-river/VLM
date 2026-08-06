#!/usr/bin/env python3
"""Build deterministic blur/noise validation views from the expanded visual panel."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .core import read_jsonl, stable_json_hash, write_jsonl


VARIANTS = (
    "gaussian_noise_2",
    "blur_1",
    "dim_0_7_noise_2",
    "saturation_clip_180",
)


def seed_for(path: str, variant: str) -> int:
    digest = hashlib.sha256(f"{variant}:{path}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def transform(source: Path, destination: Path, variant: str, relative: str) -> None:
    image = Image.open(source).convert("RGB")
    if variant == "blur_1":
        image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif variant == "saturation_clip_180":
        rgb = np.asarray(image, dtype=np.uint8)
        image = Image.fromarray(np.minimum(rgb, 180).astype(np.uint8), mode="RGB")
    else:
        rgb = np.asarray(image, dtype=np.float64)
        gain = 0.7 if variant == "dim_0_7_noise_2" else 1.0
        rng = np.random.default_rng(seed_for(relative, variant))
        noise = rng.normal(0.0, 2.0, size=rgb.shape[:2])[..., None]
        rgb = np.clip(rgb * gain + noise, 0.0, 255.0)
        image = Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Subset of deterministic perturbations to build.",
    )
    parser.add_argument(
        "--training-allowed",
        action="store_true",
        help="Mark output as training-only calibration data; never use this for validation sources.",
    )
    parser.add_argument("--dataset-name", default="visual_perturbation_eval_v10_13")
    args = parser.parse_args()
    source_rows = read_jsonl(args.records_jsonl)
    source_images = sorted(
        {
            str(image)
            for row in source_rows
            for image in row["prompt_inputs"].get("images", [])
        }
    )
    manifest: dict[str, Any] = {
        "dataset": args.dataset_name,
        "training_allowed": args.training_allowed,
        "source_records": len(source_rows),
        "source_images": len(source_images),
        "variants": {},
    }
    combined: list[dict[str, Any]] = []
    for variant in args.variants:
        path_map: dict[str, str] = {}
        for relative in source_images:
            destination_relative = f"images/{variant}/{relative}"
            transform(
                args.image_root / relative,
                args.output_dir / destination_relative,
                variant,
                relative,
            )
            path_map[relative] = destination_relative
        rows: list[dict[str, Any]] = []
        for source in source_rows:
            row = copy.deepcopy(source)
            row["example_id"] = f"{source['example_id']}__{variant}"
            row["prompt_inputs"]["images"] = [
                path_map[str(image)] for image in source["prompt_inputs"].get("images", [])
            ]
            row["provenance"]["perturbation"] = variant
            row["provenance"]["source_example_id_before_perturbation"] = source[
                "example_id"
            ]
            rows.append(row)
        write_jsonl(args.output_dir / "canonical" / f"{variant}.jsonl", rows)
        combined.extend(rows)
        manifest["variants"][variant] = {
            "record_count": len(rows),
            "image_count": len(path_map),
            "canonical_hash": stable_json_hash(rows),
        }
    write_jsonl(args.output_dir / "canonical" / "combined.jsonl", combined)
    manifest["combined_records"] = len(combined)
    manifest["combined_hash"] = stable_json_hash(combined)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
