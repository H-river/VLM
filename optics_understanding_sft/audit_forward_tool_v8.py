#!/usr/bin/env python3
"""Audit forward-tool v8 state-handle privacy, reconstruction, and cached results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .build_forward_tool_v8 import STAGES, derive, forward_items
from .core import read_jsonl, stable_json_hash
from .evaluate_intermediate_evidence_v6 import value_equal
from .forward_prediction_tool import materialize_forward_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-master-jsonl", type=Path, required=True)
    parser.add_argument("--dev-master-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "train": forward_items(args.train_master_jsonl),
        "dev": forward_items(args.dev_master_jsonl),
    }
    failures: list[str] = []
    reconstructed = 0
    materialized = 0
    for split, items in sources.items():
        actual = read_jsonl(args.dataset_dir / "canonical" / f"{split}.jsonl")
        expected_pairs = [derive(item, split) for item in items]
        expected = [row for records, _ in expected_pairs for row in records]
        if stable_json_hash(actual) != stable_json_hash(expected):
            failures.append(f"{split}: canonical reconstruction mismatch")
        else:
            reconstructed += len(actual)
        if Counter(row["stage"] for row in actual) != Counter({stage: len(items) for stage in STAGES}):
            failures.append(f"{split}: stage count mismatch")
        registry_rows = read_jsonl(args.dataset_dir / "private" / f"{split}_state_registry.jsonl")
        registry = {row["state_handle"]: row["setup_config"] for row in registry_rows}
        if len(registry) != len(items):
            failures.append(f"{split}: state handles are not unique")
        for item in items:
            source = item["record"]
            interpretation = next(
                row
                for row in actual
                if row["source_example_id"] == source["example_id"]
                and row["stage"] == "compact_tool_result_interpretation"
            )
            if not value_equal(materialize_forward_answer(interpretation["prompt_inputs"]["tool_result"]), source["target"]):
                failures.append(f"{split}: materialization mismatch {source['example_id']}")
            else:
                materialized += 1
        public_text = "\n".join(json.dumps(row, sort_keys=True) for row in actual)
        if "setup_config" in public_text or '"camera": {' in public_text or '"x_offset"' in public_text:
            failures.append(f"{split}: private simulator state leaked into public records")
        qwen = read_jsonl(args.dataset_dir / "exports" / "qwen" / f"{split}.jsonl")
        if len(qwen) != len(actual) or any(row.get("images") for row in qwen):
            failures.append(f"{split}: export count or image failure")
    report = {
        "passed": not failures,
        "failures": failures,
        "source_record_count": sum(map(len, sources.values())),
        "reconstructed_record_count": reconstructed,
        "materialized_answer_count": materialized,
        "private_state_leak_count": sum("private simulator state leaked" in item for item in failures),
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
