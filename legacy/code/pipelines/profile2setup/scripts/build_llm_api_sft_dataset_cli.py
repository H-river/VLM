"""CLI for building multimodal LLM API SFT JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile2setup.data_prep.build_llm_api_sft_dataset import build_llm_api_sft_dataset


def _default_input(split: str | None) -> str | None:
    if split is None:
        return None
    return f"profile2setup/data/all_modes/{split}.jsonl"


def _default_out(split: str | None) -> str | None:
    if split is None:
        return None
    return f"profile2setup/data/llm_api_sft/{split}.jsonl"


def _default_image_out_dir(split: str | None) -> str | None:
    if split is None:
        return None
    return f"profile2setup/data/llm_api_sft/images/{split}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build API-compatible multimodal SFT JSONL from profile2setup records."
    )
    parser.add_argument("--input", default=None, help="Input profile2setup JSONL")
    parser.add_argument("--out", default=None, help="Output LLM API SFT JSONL")
    parser.add_argument("--image-out-dir", default=None, help="Directory for rendered PNG images")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of SFT rows to write")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--include-composite", action="store_true", help="Include composite image in user messages")
    parser.add_argument(
        "--image-mode",
        choices=("base64", "relative", "public"),
        default="base64",
        help="How rendered images are referenced in the user message",
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "auto"),
        default="low",
        help="Image detail hint stored with each image_url content part",
    )
    parser.add_argument("--public-prefix", default=None, help="URL prefix for --image-mode public")
    parser.add_argument("--size", type=int, default=512, help="Square panel size in pixels")
    parser.add_argument(
        "--normalize-mode",
        choices=("max", "percentile", "log_max"),
        default="max",
        help="Deterministic image normalization mode",
    )
    parser.add_argument("--strict", action="store_true", help="Raise on the first skipped/bad record")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or _default_input(args.split)
    out_path = args.out or _default_out(args.split)
    image_out_dir = args.image_out_dir or _default_image_out_dir(args.split)
    if input_path is None or out_path is None or image_out_dir is None:
        raise ValueError("--input, --out, and --image-out-dir are required unless --split is provided")

    summary = build_llm_api_sft_dataset(
        input_path=Path(input_path),
        out_path=Path(out_path),
        image_out_dir=Path(image_out_dir),
        limit=args.limit,
        split=args.split,
        include_composite=args.include_composite,
        image_mode=args.image_mode,
        image_detail=args.image_detail,
        strict=args.strict,
        size=args.size,
        normalize_mode=args.normalize_mode,
        public_prefix=args.public_prefix,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
