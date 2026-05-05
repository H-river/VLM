"""CLI for reasoning VLM JSONL evaluation."""

from __future__ import annotations

import argparse
import json

from profile2setup.evaluation.reasoning_vlm_eval import evaluate_reasoning_vlm_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate profile2setup reasoning VLM JSONL outputs")
    parser.add_argument("--input", required=True, help="Input reasoning JSONL or SFT JSONL")
    parser.add_argument("--out", default="profile2setup/results/reasoning_vlm_eval.json", help="Output JSON path")
    parser.add_argument("--max-error-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_reasoning_vlm_jsonl(
        input_path=args.input,
        out_path=args.out,
        max_error_examples=args.max_error_examples,
    )
    print(json.dumps(
        {
            "input": args.input,
            "out": args.out,
            "num_records": result["num_records"],
            "json_validity_rate": result["json_validity_rate"],
            "target_label_count": result["target_label_count"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
