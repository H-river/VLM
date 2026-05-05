"""Run multimodal LLM API inference for profile2setup records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile2setup.llm_api.inference import run_llm_api_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run profile2setup multimodal LLM API inference.")
    parser.add_argument("--provider", default="openai", choices=("openai",))
    parser.add_argument("--model", required=True, help="Provider model name")
    parser.add_argument("--data", required=True, help="Input profile2setup JSONL")
    parser.add_argument("--out", required=True, help="Output predictions JSONL")
    parser.add_argument("--image-out-dir", required=True, help="Directory for rendered PNG inputs")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of API payloads/calls")
    parser.add_argument("--dry-run", action="store_true", help="Build request payloads without sending")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--reuse-rendered-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--normalize-mode",
        choices=("max", "percentile", "log_max"),
        default="max",
    )
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Disable provider structured-output format and rely on strict prompt instructions only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_llm_api_inference(
        provider=args.provider,
        model=args.model,
        data_path=Path(args.data),
        out_path=Path(args.out),
        image_out_dir=Path(args.image_out_dir),
        limit=args.limit,
        dry_run=args.dry_run,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        image_detail=args.image_detail,
        reuse_rendered_images=args.reuse_rendered_images,
        image_size=args.image_size,
        normalize_mode=args.normalize_mode,
        use_response_format=not args.no_response_format,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
