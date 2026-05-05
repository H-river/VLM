"""Render deterministic multimodal LLM API input PNGs from intensity.npy files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile2setup.llm_api.image_rendering import (
    load_intensity_npy,
    render_composite_png,
    render_difference_png,
    render_profile_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render profile2setup LLM API images from current/target intensity.npy files."
    )
    parser.add_argument("--current-profile", required=True, help="Path to current intensity.npy")
    parser.add_argument("--target-profile", required=True, help="Path to target intensity.npy")
    parser.add_argument("--out-dir", required=True, help="Directory for rendered PNG outputs")
    parser.add_argument("--size", type=int, default=512, help="Square panel size in pixels")
    parser.add_argument(
        "--normalize-mode",
        default="max",
        choices=("max", "percentile", "log_max"),
        help="Deterministic intensity normalization mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current = load_intensity_npy(args.current_profile)
    target = load_intensity_npy(args.target_profile)
    if current.shape != target.shape:
        raise ValueError(f"current and target shapes must match, got {current.shape} and {target.shape}")

    outputs = {
        "current_profile": render_profile_png(
            current,
            out_dir / "current_profile.png",
            label="CURRENT",
            size=args.size,
            mode=args.normalize_mode,
        ),
        "target_profile": render_profile_png(
            target,
            out_dir / "target_profile.png",
            label="TARGET",
            size=args.size,
            mode=args.normalize_mode,
        ),
        "difference_profile": render_difference_png(
            current,
            target,
            out_dir / "difference_profile.png",
            size=args.size,
            mode=args.normalize_mode,
        ),
        "composite_profile": render_composite_png(
            current,
            target,
            out_dir / "composite_profile.png",
            size=args.size,
            mode=args.normalize_mode,
        ),
    }

    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
