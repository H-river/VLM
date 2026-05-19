"""Render controlled profile PNGs for the profile2setup reasoning VLM layer."""

from __future__ import annotations

import argparse
import json

from legacy.reasoning_vlm.image_rendering import render_reasoning_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render reasoning VLM profile images from intensity.npy")
    parser.add_argument("--current-profile", required=True, help="Path to current intensity.npy")
    parser.add_argument("--target-profile", required=True, help="Path to target intensity.npy")
    parser.add_argument("--out-dir", required=True, help="Directory for rendered PNG outputs")
    parser.add_argument("--prefix", default="", help="Optional filename prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_reasoning_images(
        current_path=args.current_profile,
        target_path=args.target_profile,
        out_dir=args.out_dir,
        prefix=args.prefix,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
