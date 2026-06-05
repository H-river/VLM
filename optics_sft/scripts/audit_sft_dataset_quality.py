#!/usr/bin/env python3
"""Audit physics/text SFT datasets and write a structured quality report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.data_quality.report import build_quality_report, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SFT dataset quality and write a JSON report.")
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--val-jsonl", type=Path)
    parser.add_argument("--test-jsonl", type=Path)
    parser.add_argument("--dataset-name", type=str, default="unknown")
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with code 1 when quality gates fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_jsonl is None and args.val_jsonl is None and args.test_jsonl is None:
        raise SystemExit("Provide at least one of --train-jsonl, --val-jsonl, or --test-jsonl")

    extra_jsonls = {}
    if args.test_jsonl is not None:
        extra_jsonls["test"] = args.test_jsonl

    report = build_quality_report(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        extra_jsonls=extra_jsonls or None,
        dataset_name=args.dataset_name,
    )
    write_report(args.output_report, report)

    print(json.dumps(report["quality_gates"], indent=2, sort_keys=True))
    if args.fail_on_gate and not report["quality_gates"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
