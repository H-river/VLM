#!/usr/bin/env python3
"""Audit v6 intermediate-evidence records against their v5A sources and tools."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_intermediate_evidence_v6 import STAGES, derive_records
from .core import read_jsonl, stable_json_hash
from .decision_tools import run_tool


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


def source_rows(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": read_jsonl(source_dir / "canonical" / "train.jsonl"),
        "dev": read_jsonl(source_dir / "canonical" / "dev.jsonl"),
        "confirmation": read_jsonl(source_dir / "private" / "confirmation_records.jsonl"),
    }


def generated_rows(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": read_jsonl(dataset_dir / "canonical" / "train.jsonl"),
        "dev": read_jsonl(dataset_dir / "canonical" / "dev.jsonl"),
        "confirmation": read_jsonl(dataset_dir / "private" / "confirmation_records.jsonl"),
    }


def interpretation_consistent(record: Mapping[str, Any]) -> bool:
    if record["stage"] != "tool_result_interpretation":
        return True
    inputs = record["prompt_inputs"]
    result = inputs["tool_result"]
    target = record["target"]
    evidence = target["intermediate_evidence"]
    visible = inputs["visible_evidence"]
    if record["source_task_type"] == "constrained_intervention":
        expected_evidence = {
            "successful_action_indices": result["successful_action_indices"],
            "selected_index": result["selected_index"],
            "best_residual_index": result["best_residual_index"],
        }
        if evidence != expected_evidence:
            return False
        selected = result["selected_index"]
        if selected is None:
            best = result["best_residual_index"]
            return (
                target["status"] == "infeasible_within_limits"
                and target["answer"]["control_plan"] is None
                and abs(
                    float(target["answer"]["best_achievable_residual_px"])
                    - float(visible["candidate_action_trials"][best]["measured_residual_px"])
                )
                <= 5e-5
            )
        trial = visible["candidate_action_trials"][selected]
        return (
            target["status"] == "feasible"
            and target["answer"]["control_plan"] == trial["action"]
            and abs(
                float(target["answer"]["expected_residual_px"])
                - float(trial["measured_residual_px"])
            )
            <= 5e-5
        )
    expected_evidence = {
        "observed_direction_set": result["observed_direction_set"],
        "conflicting_pair_indices": result["conflicting_pair_indices"],
    }
    if evidence != expected_evidence:
        return False
    indices = result["conflicting_pair_indices"]
    if indices is None:
        return (
            target["status"] == "answerable"
            and target["answer"]["centroid_x_direction"]
            == result["observed_direction_set"][0]
            and target["answer"]["visible_conflicting_witness"] is None
        )
    trials = visible["compatible_completion_trials"]
    witness = [trials[index]["hidden_value_mm"] for index in indices]
    return (
        target["status"] == "insufficient_information"
        and target["answer"]["centroid_x_direction"] is None
        and target["answer"]["visible_conflicting_witness"] == witness
    )


def audit(dataset_dir: Path, source_dir: Path) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    source = source_rows(source_dir)
    actual = generated_rows(dataset_dir)
    version = str(manifest["version"])
    failures: list[str] = []
    reconstruction_failures: list[str] = []
    executable_failures: list[str] = []
    interpretation_failures: list[str] = []

    for split in SPLITS:
        expected = [row for source_record in source[split] for row in derive_records(source_record, version)]
        expected_by_id = {row["example_id"]: row for row in expected}
        actual_by_id = {row["example_id"]: row for row in actual[split]}
        if len(actual_by_id) != len(actual[split]):
            failures.append(f"{split}: duplicate example IDs")
        if set(expected_by_id) != set(actual_by_id):
            failures.append(f"{split}: derived ID set differs from source reconstruction")
        for example_id in sorted(set(expected_by_id) & set(actual_by_id)):
            if stable_json_hash(expected_by_id[example_id]) != stable_json_hash(actual_by_id[example_id]):
                reconstruction_failures.append(example_id)

        counts = Counter(row["stage"] for row in actual[split])
        expected_per_stage = len(source[split])
        if counts != Counter({stage: expected_per_stage for stage in STAGES}):
            failures.append(f"{split}: stage imbalance {dict(counts)}")

    all_rows = [row for split in SPLITS for row in actual[split]]
    for row in all_rows:
        if not finite_tree(row):
            failures.append(f"non-finite tree: {row['example_id']}")
            break
        if row["stage"] == "tool_call_construction":
            try:
                run_tool(row["target"]["tool_name"], row["target"]["arguments"])
            except (KeyError, TypeError, ValueError):
                executable_failures.append(row["example_id"])
        if not interpretation_consistent(row):
            interpretation_failures.append(row["example_id"])

    groups = {
        split: {str(row["group_id"]) for row in rows}
        for split, rows in actual.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if groups[left] & groups[right]:
                failures.append(f"group overlap between {left} and {right}")

    confirmation_prompts = read_jsonl(dataset_dir / "canonical" / "confirmation_prompts.jsonl")
    if len(confirmation_prompts) != len(actual["confirmation"]):
        failures.append("confirmation prompt count mismatch")
    if any("target" in row for row in confirmation_prompts):
        failures.append("confirmation target leakage")
    for split in ("train", "dev", "confirmation"):
        export = read_jsonl(dataset_dir / "exports" / "qwen" / f"{split}.jsonl")
        if len(export) != len(actual[split]) or any(row["images"] for row in export):
            failures.append(f"{split}: Qwen export count or zero-image failure")
        if split == "confirmation" and any("completion" in row for row in export):
            failures.append("confirmation Qwen export contains completions")

    if reconstruction_failures:
        failures.append(f"reconstruction failures: {reconstruction_failures[:3]}")
    if executable_failures:
        failures.append(f"non-executable target calls: {executable_failures[:3]}")
    if interpretation_failures:
        failures.append(f"interpretation consistency failures: {interpretation_failures[:3]}")

    return {
        "passed": not failures,
        "failures": failures,
        "dataset_version": version,
        "record_count": len(all_rows),
        "source_record_count": sum(len(rows) for rows in source.values()),
        "split_record_counts": {split: len(rows) for split, rows in actual.items()},
        "stage_counts": dict(sorted(Counter(row["stage"] for row in all_rows).items())),
        "reconstruction_failure_count": len(reconstruction_failures),
        "tool_call_execution_failure_count": len(executable_failures),
        "interpretation_consistency_failure_count": len(interpretation_failures),
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
