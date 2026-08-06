#!/usr/bin/env python3
"""Audit v7 reconstruction, compact mappings, executions, and sealed outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .build_path_mapping_v7 import STAGES, derive_records
from .core import read_jsonl, stable_json_hash
from .tool_path_adapter import run_mapped_tool


SPLITS = ("train", "dev", "confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(finite_tree(child) for child in value)
    return True


def audit(dataset_dir: Path, source_dir: Path) -> dict[str, Any]:
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
    reconstruction = []
    mapping_execution = []
    interpretation = []

    for split in SPLITS:
        expected = [
            derived
            for source_record in source[split]
            for derived in derive_records(source_record, str(manifest["version"]))
        ]
        expected_by_id = {row["example_id"]: row for row in expected}
        actual_by_id = {row["example_id"]: row for row in actual[split]}
        if len(actual_by_id) != len(actual[split]) or set(expected_by_id) != set(actual_by_id):
            failures.append(f"{split}: ID count or set mismatch")
        for example_id in set(expected_by_id) & set(actual_by_id):
            if stable_json_hash(expected_by_id[example_id]) != stable_json_hash(actual_by_id[example_id]):
                reconstruction.append(example_id)
        counts = Counter(row["stage"] for row in actual[split])
        if counts != Counter({stage: len(source[split]) for stage in STAGES}):
            failures.append(f"{split}: stage imbalance {dict(counts)}")

    all_rows = [row for split in SPLITS for row in actual[split]]
    for row in all_rows:
        if not finite_tree(row):
            failures.append(f"non-finite record: {row['example_id']}")
            break
        if row["stage"] == "tool_source_mapping":
            try:
                run_mapped_tool(
                    row["target"]["tool_name"],
                    row["prompt_inputs"]["visible_evidence"],
                    row["target"]["source_map"],
                )
            except (KeyError, TypeError, ValueError):
                mapping_execution.append(row["example_id"])
        if row["stage"] == "tool_result_interpretation":
            result = row["prompt_inputs"]["tool_result"]
            evidence = row["target"]["intermediate_evidence"]
            if row["source_task_type"] == "constrained_intervention":
                expected_evidence = {
                    "successful_action_indices": result["successful_action_indices"],
                    "selected_index": result["selected_index"],
                    "best_residual_index": result["best_residual_index"],
                }
            else:
                expected_evidence = {
                    "observed_direction_set": result["observed_direction_set"],
                    "conflicting_pair_indices": result["conflicting_pair_indices"],
                }
            if evidence != expected_evidence:
                interpretation.append(row["example_id"])

    groups = {split: {row["group_id"] for row in rows} for split, rows in actual.items()}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if groups[left] & groups[right]:
                failures.append(f"group overlap between {left} and {right}")

    confirmation_prompts = read_jsonl(dataset_dir / "canonical" / "confirmation_prompts.jsonl")
    if len(confirmation_prompts) != len(actual["confirmation"]) or any(
        "target" in row for row in confirmation_prompts
    ):
        failures.append("confirmation count or target leakage")
    for split in SPLITS:
        export = read_jsonl(dataset_dir / "exports" / "qwen" / f"{split}.jsonl")
        if len(export) != len(actual[split]) or any(row["images"] for row in export):
            failures.append(f"{split}: Qwen export count or zero-image failure")
        if split == "confirmation" and any("completion" in row for row in export):
            failures.append("confirmation export contains completions")

    if reconstruction:
        failures.append(f"reconstruction failures: {reconstruction[:3]}")
    if mapping_execution:
        failures.append(f"mapping execution failures: {mapping_execution[:3]}")
    if interpretation:
        failures.append(f"interpretation evidence failures: {interpretation[:3]}")
    return {
        "passed": not failures,
        "failures": failures,
        "dataset_version": manifest["version"],
        "record_count": len(all_rows),
        "source_record_count": sum(len(rows) for rows in source.values()),
        "split_record_counts": {split: len(rows) for split, rows in actual.items()},
        "stage_counts": dict(sorted(Counter(row["stage"] for row in all_rows).items())),
        "reconstruction_failure_count": len(reconstruction),
        "mapping_execution_failure_count": len(mapping_execution),
        "interpretation_evidence_failure_count": len(interpretation),
        "confirmation_target_leak_count": sum("target" in row for row in confirmation_prompts),
    }


def main() -> None:
    args = parse_args()
    result = audit(args.dataset_dir, args.source_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
