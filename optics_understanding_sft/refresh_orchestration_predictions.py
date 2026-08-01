#!/usr/bin/env python3
"""Reuse exact unchanged generations and isolate changed orchestration prompts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl, stable_json_hash, write_jsonl


def key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["source_example_id"]), str(row["stage"])


def prepare(args: argparse.Namespace) -> None:
    old_records = {key(row): row for row in read_jsonl(args.old_records)}
    old_predictions = {
        str(row["example_id"]): row for row in read_jsonl(args.old_predictions)
    }
    new_records = read_jsonl(args.new_records)
    if args.max_samples is not None:
        new_records = new_records[: max(0, args.max_samples)]
    reused: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for new in new_records:
        old = old_records.get(key(new))
        old_prediction = old_predictions.get(str(old["example_id"])) if old else None
        unchanged = (
            old is not None
            and old_prediction is not None
            and old["prompt"] == new["prompt"]
            and old["target"] == new["target"]
        )
        if not unchanged:
            pending.append(new)
            continue
        prediction = copy.deepcopy(old_prediction)
        prediction["example_id"] = new["example_id"]
        prediction["group_id"] = new["group_id"]
        prediction["task_type"] = new["task_type"]
        prediction["prompt_hash"] = stable_json_hash(new["prompt"])
        prediction["reused_from_example_id"] = old["example_id"]
        reused.append(prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "reused_predictions.jsonl", reused)
    write_jsonl(args.output_dir / "pending_records.jsonl", pending)
    report = {
        "record_count": len(new_records),
        "reused_count": len(reused),
        "pending_count": len(pending),
        "exact_prompt_and_target_required": True,
    }
    (args.output_dir / "refresh_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def merge(args: argparse.Namespace) -> None:
    records = read_jsonl(args.new_records)
    if args.max_samples is not None:
        records = records[: max(0, args.max_samples)]
    predictions: dict[str, dict[str, Any]] = {}
    for path in (args.reused_predictions, args.pending_predictions):
        for row in read_jsonl(path):
            example_id = str(row["example_id"])
            if example_id in predictions:
                raise ValueError(f"Duplicate prediction: {example_id}")
            predictions[example_id] = row
    expected = [str(row["example_id"]) for row in records]
    if set(predictions) != set(expected):
        raise ValueError(
            f"Prediction IDs differ: missing={sorted(set(expected) - set(predictions))[:5]} "
            f"extra={sorted(set(predictions) - set(expected))[:5]}"
        )
    write_jsonl(args.output_predictions, (predictions[example_id] for example_id in expected))
    print(json.dumps({"merged_count": len(expected)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--old-records", type=Path, required=True)
    prep.add_argument("--old-predictions", type=Path, required=True)
    prep.add_argument("--new-records", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--max-samples", type=int)
    prep.set_defaults(func=prepare)
    combine = subparsers.add_parser("merge")
    combine.add_argument("--new-records", type=Path, required=True)
    combine.add_argument("--reused-predictions", type=Path, required=True)
    combine.add_argument("--pending-predictions", type=Path, required=True)
    combine.add_argument("--output-predictions", type=Path, required=True)
    combine.add_argument("--max-samples", type=int)
    combine.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
