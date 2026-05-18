#!/usr/bin/env python3
"""Build optics VLM SFT JSONL files from raw samples.

This is a safe scaffold for converting simulator outputs or manually prepared
optics examples into the raw SFT row format documented in
`optics_sft/data_schema/sft_sample_schema.json`.

Current behavior is conservative: it accepts JSON files that already resemble
the target row format, skips incompatible files, and writes train/val JSONL
files. TODO sections mark where existing simulator outputs should be connected.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw optics samples into SFT train/val JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../VLM_data/optics_sft/raw"),
        help="Directory containing raw simulator or optics sample files.",
    )
    parser.add_argument(
        "--output-train-jsonl",
        type=Path,
        default=Path("../VLM_data/optics_sft/train.jsonl"),
        help="Path to write the training JSONL split.",
    )
    parser.add_argument(
        "--output-val-jsonl",
        type=Path,
        default=Path("../VLM_data/optics_sft/val.jsonl"),
        help="Path to write the validation JSONL split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to put in the validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    return parser.parse_args()


def looks_like_sft_row(row: dict[str, Any]) -> bool:
    return all(key in row for key in ("current_image_path", "target_image_path", "metadata", "label"))


def collect_samples(input_dir: Path) -> list[dict[str, Any]]:
    """Collect raw samples.

    TODO:
    - Connect this to `optical_sim/outputs/` records.
    - Convert simulator metadata and rendered beam images into the schema used
      by `optics_sft/data_schema/sft_sample_schema.json`.
    - Add optional JSON Schema validation once the dependency policy is settled.
    """
    if not input_dir.exists():
        print(f"[warn] input directory does not exist: {input_dir}", file=sys.stderr)
        return []

    samples: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] skipping invalid JSON {path}: {exc}", file=sys.stderr)
            continue

        if isinstance(row, dict) and looks_like_sft_row(row):
            samples.append(row)
        else:
            print(f"[warn] skipping non-SFT row {path}", file=sys.stderr)

    return samples


def split_samples(
    samples: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0.0, 1.0)")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    return train_rows, val_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.input_dir)
    train_rows, val_rows = split_samples(samples, args.val_ratio, args.seed)
    write_jsonl(args.output_train_jsonl, train_rows)
    write_jsonl(args.output_val_jsonl, val_rows)
    print(
        "Wrote "
        f"{len(train_rows)} train rows to {args.output_train_jsonl} and "
        f"{len(val_rows)} val rows to {args.output_val_jsonl}."
    )


if __name__ == "__main__":
    main()
