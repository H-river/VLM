"""CLI for building profile2setup reasoning VLM SFT data."""

from __future__ import annotations

import argparse
import json

from profile2setup.data_prep.build_vlm_sft_dataset import build_vlm_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build profile2setup reasoning VLM SFT JSONL")
    parser.add_argument("--input", required=True, help="Input profile2setup JSONL")
    parser.add_argument("--out", required=True, help="Output SFT JSONL")
    parser.add_argument("--image-out-dir", required=True, help="Directory for rendered profile images")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of positive SFT records")
    parser.add_argument(
        "--include-negative-examples",
        action="store_true",
        help="Append rule-based negative prompt examples",
    )
    parser.add_argument("--strict", action="store_true", help="Raise instead of skipping bad records")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_vlm_sft_dataset(
        input_path=args.input,
        out_path=args.out,
        image_out_dir=args.image_out_dir,
        limit=args.limit,
        include_negative_examples=args.include_negative_examples,
        strict=args.strict,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
