#!/usr/bin/env python3
"""Audit v7.1 reconstruction, compact decisions, and deterministic materialization."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .build_compact_decision_v7_1 import STAGES, derive_records
from .build_intermediate_evidence_v6 import visible_evidence
from .core import read_jsonl, stable_json_hash
from .decision_tools import tool_for_task
from .evaluate_intermediate_evidence_v6 import value_equal
from .tool_path_adapter import SOURCE_MAPS, compact_decision, materialize_final_answer, run_mapped_tool


SPLITS = ("train", "dev", "confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def audit(dataset_dir: Path, source_dir: Path) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    source = {
        "train": read_jsonl(source_dir / "canonical" / "train.jsonl"),
        "dev": read_jsonl(source_dir / "canonical" / "dev.jsonl"),
        "confirmation": read_jsonl(source_dir / "private" / "confirmation_records.jsonl"),
    }
    actual = {
        "train": read_jsonl(dataset_dir / "canonical" / "train.jsonl"),
        "dev": read_jsonl(dataset_dir / "canonical" / "dev.jsonl"),
        "confirmation": read_jsonl(dataset_dir / "private" / "confirmation_records.jsonl"),
    }
    failures: list[str] = []
    reconstruction_failures: list[str] = []
    compact_failures: list[str] = []
    materialization_failures: list[str] = []
    for split in SPLITS:
        expected = [row for item in source[split] for row in derive_records(item, manifest["version"])]
        expected_by_id = {row["example_id"]: row for row in expected}
        actual_by_id = {row["example_id"]: row for row in actual[split]}
        if set(expected_by_id) != set(actual_by_id) or len(actual_by_id) != len(actual[split]):
            failures.append(f"{split}: ID mismatch")
        reconstruction_failures.extend(
            example_id
            for example_id in set(expected_by_id) & set(actual_by_id)
            if stable_json_hash(expected_by_id[example_id]) != stable_json_hash(actual_by_id[example_id])
        )
        if Counter(row["stage"] for row in actual[split]) != Counter(
            {stage: len(source[split]) for stage in STAGES}
        ):
            failures.append(f"{split}: stage imbalance")

        compact_by_source = {
            row["source_example_id"]: row
            for row in actual[split]
            if row["stage"] == "compact_tool_result_interpretation"
        }
        for source_row in source[split]:
            tool = tool_for_task(str(source_row["task_type"]))
            evidence = visible_evidence(source_row)
            result = run_mapped_tool(tool, evidence, SOURCE_MAPS[tool])
            compact = compact_by_source[str(source_row["example_id"])]
            if not value_equal(compact["target"], compact_decision(tool, result)):
                compact_failures.append(str(source_row["example_id"]))
            if not value_equal(materialize_final_answer(tool, result, evidence), source_row["target"]):
                materialization_failures.append(str(source_row["example_id"]))

    confirmation_prompts = read_jsonl(dataset_dir / "canonical" / "confirmation_prompts.jsonl")
    if len(confirmation_prompts) != len(actual["confirmation"]) or any(
        "target" in row for row in confirmation_prompts
    ):
        failures.append("confirmation target leakage or count mismatch")
    for split in SPLITS:
        export = read_jsonl(dataset_dir / "exports" / "qwen" / f"{split}.jsonl")
        if len(export) != len(actual[split]) or any(row.get("images") for row in export):
            failures.append(f"{split}: export count or image failure")
        if split == "confirmation" and any("completion" in row for row in export):
            failures.append("confirmation export contains completions")
    if reconstruction_failures:
        failures.append("record reconstruction failure")
    if compact_failures:
        failures.append("compact decision failure")
    if materialization_failures:
        failures.append("deterministic final-answer materialization failure")
    return {
        "passed": not failures,
        "failures": failures,
        "record_count": sum(map(len, actual.values())),
        "split_record_counts": {split: len(rows) for split, rows in actual.items()},
        "stage_counts": dict(sorted(Counter(row["stage"] for rows in actual.values() for row in rows).items())),
        "reconstruction_failure_count": len(reconstruction_failures),
        "compact_decision_failure_count": len(compact_failures),
        "materialization_failure_count": len(materialization_failures),
        "confirmation_target_leak_count": sum("target" in row for row in confirmation_prompts),
    }


def main() -> None:
    args = parse_args()
    result = audit(args.dataset_dir, args.source_dir)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
