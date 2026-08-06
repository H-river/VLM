#!/usr/bin/env python3
"""Package clean and robust calibrations behind one quality-routed interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .visual_state_tool_v10_1 import load_calibration


def package(
    clean_path: Path,
    robust_path: Path,
    kind: str,
    dimnoise_path: Path | None = None,
    noise_path: Path | None = None,
) -> dict:
    value = {
        "classifier": (
            "quality_routed_visual_v3"
            if noise_path is not None
            else "quality_routed_visual_v2"
            if dimnoise_path is not None
            else "quality_routed_visual_v1"
        ),
        "kind": kind,
        "quality_thresholds": {
            "border_noise_scale": 1.5,
            "colored_fraction_blur": 0.02,
            "dimnoise_max_channel": 200.0,
        },
        "clean_calibration_path": str(clean_path.resolve()),
        "robust_calibration_path": str(robust_path.resolve()),
        "clean_calibration": load_calibration(clean_path),
        "robust_calibration": load_calibration(robust_path),
        "policy": (
            "clean calibration for clean state; corruption-specific calibrations for noise and dim-plus-noise; robust calibration for blurred state"
            if kind == "state"
            else "clean by default; robust centroids for noise; robust widths for blur"
        ),
    }
    if dimnoise_path is not None:
        value["dimnoise_calibration_path"] = str(dimnoise_path.resolve())
        value["dimnoise_calibration"] = load_calibration(dimnoise_path)
    if noise_path is not None:
        value["noise_calibration_path"] = str(noise_path.resolve())
        value["noise_calibration"] = load_calibration(noise_path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-state", type=Path, required=True)
    parser.add_argument("--robust-state", type=Path, required=True)
    parser.add_argument("--dimnoise-state", type=Path)
    parser.add_argument("--noise-state", type=Path)
    parser.add_argument("--clean-pair", type=Path, required=True)
    parser.add_argument("--robust-pair", type=Path, required=True)
    parser.add_argument("--noise-pair-width", type=Path)
    parser.add_argument("--dimnoise-pair-width", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-version", default="v10_15")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "state": package(
            args.clean_state,
            args.robust_state,
            "state",
            args.dimnoise_state,
            args.noise_state,
        ),
        "pair": package(args.clean_pair, args.robust_pair, "pair"),
    }
    if args.noise_pair_width is not None or args.dimnoise_pair_width is not None:
        pair = outputs["pair"]
        pair["classifier"] = "quality_routed_visual_v4"
        pair["policy"] = (
            "clean by default; robust centroids for noise; robust widths for blur; "
            "corruption-specific squared-signal widths for noise and dim-plus-noise"
        )
        if args.noise_pair_width is not None:
            pair["noise_width_calibration_path"] = str(
                args.noise_pair_width.resolve()
            )
            pair["noise_width_calibration"] = load_calibration(
                args.noise_pair_width
            )
        if args.dimnoise_pair_width is not None:
            pair["dimnoise_width_calibration_path"] = str(
                args.dimnoise_pair_width.resolve()
            )
            pair["dimnoise_width_calibration"] = load_calibration(
                args.dimnoise_pair_width
            )
    for kind, value in outputs.items():
        (args.output_dir / f"{kind}_tool_calibration_adaptive_{args.output_version}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({kind: value["policy"] for kind, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
