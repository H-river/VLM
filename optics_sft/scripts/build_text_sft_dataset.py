#!/usr/bin/env python3
"""Convert physics SFT rows into text-only SFT JSONL rows."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.physics.text_prompt_builder import build_text_messages, build_text_prompt_inputs
from optics_sft.physics.text_target import (
    TRAINING_TARGET_MODES,
    build_training_target,
    normalize_training_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build text-only SFT JSONL from physics rows.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--sample-types",
        nargs="+",
        default=["inverse_control"],
        help="Sample types to export. Defaults to inverse_control only.",
    )
    parser.add_argument(
        "--training-target",
        choices=TRAINING_TARGET_MODES,
        default=None,
        help="Assistant JSON + user prompt style: full, compact, or control_plan_only.",
    )
    parser.add_argument(
        "--compact-target",
        action="store_true",
        help="Legacy alias for --training-target compact.",
    )
    parser.add_argument(
        "--control-plan-only",
        action="store_true",
        help="Legacy alias for --training-target control_plan_only.",
    )
    return parser.parse_args()


def resolve_export_training_target(args: argparse.Namespace) -> str:
    if args.control_plan_only and args.compact_target:
        raise ValueError("Use only one of --compact-target and --control-plan-only.")
    if args.training_target is not None:
        if args.compact_target or args.control_plan_only:
            raise ValueError("Do not combine --training-target with legacy compact/control-plan flags.")
        return args.training_target
    if args.control_plan_only:
        return "control_plan_only"
    return normalize_training_target(compact_target=args.compact_target)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def physics_to_text_row(row: dict[str, Any], *, training_target: str) -> dict[str, Any]:
    text_prompt_inputs = build_text_prompt_inputs(row)
    full_target = copy.deepcopy(row["target"])
    messages = build_text_messages(row, training_target=training_target)
    return {
        "sample_id": row["sample_id"],
        "sample_type": row["sample_type"],
        "modality": "text",
        "target_format": training_target,
        "prompt_inputs": text_prompt_inputs,
        "target": full_target,
        "training_target": build_training_target(full_target, training_target),  # type: ignore[arg-type]
        "private_eval": copy.deepcopy(row.get("private_eval", {})),
        "split_tags": list(row.get("split_tags", [])),
        "messages": messages,
        "user_prompt": messages[0]["content"],
        "assistant_completion": messages[1]["content"],
    }


def main() -> None:
    args = parse_args()
    training_target = resolve_export_training_target(args)
    allowed_types = set(args.sample_types)
    converted: list[dict[str, Any]] = []
    skipped = 0
    for row in read_jsonl(args.input_jsonl):
        sample_type = row.get("sample_type")
        if sample_type not in allowed_types:
            skipped += 1
            continue
        converted.append(physics_to_text_row(row, training_target=training_target))
    write_jsonl(args.output_jsonl, converted)
    print(
        f"Wrote {len(converted)} text rows to {args.output_jsonl} "
        f"(training_target={training_target}, skipped {skipped} other sample types)."
    )


if __name__ == "__main__":
    main()
