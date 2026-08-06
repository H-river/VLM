#!/usr/bin/env python3
"""Build v7.1 compact decision supervision after validated mapped-tool execution."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_path_mapping_v7 import derive_records as derive_v7_records
from .core import file_sha256, load_yaml, make_messages, make_qwen_record, read_jsonl, stable_json_hash, write_jsonl
from .tool_path_adapter import compact_decision


STAGES = ("tool_choice", "tool_source_mapping", "compact_tool_result_interpretation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def output_contract(source_task: str) -> dict[str, Any]:
    if source_task == "constrained_intervention":
        return {
            "exact_keys": [
                "status",
                "successful_action_indices",
                "selected_index",
                "best_residual_index",
            ],
            "status_rule": (
                "feasible when feasible_within_tolerance is true; otherwise "
                "infeasible_within_limits"
            ),
            "copy_fields": [
                "successful_action_indices",
                "selected_index",
                "best_residual_index",
            ],
        }
    return {
        "exact_keys": ["status", "observed_direction_set", "conflicting_pair_indices"],
        "status_rule": (
            "answerable when all_directions_agree is true; otherwise insufficient_information"
        ),
        "copy_fields": ["observed_direction_set", "conflicting_pair_indices"],
    }


def compact_prompt(source_task: str, inputs: Mapping[str, Any]) -> str:
    return (
        "Interpret only the validated deterministic tool result. Apply the stated status rule and "
        "copy only the named evidence fields. Return one strict JSON object with exactly the listed "
        "keys; do not add an answer, explanation, residual table, or candidate action.\n\nInput data:\n"
        + json.dumps({"source_task": source_task, **copy.deepcopy(dict(inputs))}, indent=2, sort_keys=True)
    )


def derive_records(source: Mapping[str, Any], version: str) -> list[dict[str, Any]]:
    records = derive_v7_records(source, version)
    output: list[dict[str, Any]] = []
    for record in records:
        item = copy.deepcopy(record)
        item["example_id"] = str(item["example_id"]).replace("__v7_", "__v7_1_")
        if item["stage"] == "tool_result_interpretation":
            item["stage"] = "compact_tool_result_interpretation"
            item["task_type"] = "compact_tool_result_interpretation"
            tool_name = str(item["prompt_inputs"]["tool_name"])
            result = copy.deepcopy(item["prompt_inputs"]["tool_result"])
            inputs = {
                "tool_name": tool_name,
                "adapter_validation": "passed",
                "tool_result": result,
                "output_contract": output_contract(str(item["source_task_type"])),
            }
            item["prompt_inputs"] = inputs
            item["prompt"] = compact_prompt(str(item["source_task_type"]), inputs)
            item["target"] = compact_decision(tool_name, result)
        output.append(item)
    return output


def target_free(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result.pop("target", None)
    return result


def build(config: Mapping[str, Any], source_dir: Path, output_dir: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    version = str(dataset["version"])
    source = {
        "train": read_jsonl(source_dir / "canonical" / "train.jsonl"),
        "dev": read_jsonl(source_dir / "canonical" / "dev.jsonl"),
        "confirmation": read_jsonl(source_dir / "private" / "confirmation_records.jsonl"),
    }
    records = {
        split: [derived for row in rows for derived in derive_records(row, version)]
        for split, rows in source.items()
    }
    write_jsonl(output_dir / "canonical" / "train.jsonl", records["train"])
    write_jsonl(output_dir / "canonical" / "dev.jsonl", records["dev"])
    write_jsonl(
        output_dir / "canonical" / "confirmation_prompts.jsonl",
        (target_free(row) for row in records["confirmation"]),
    )
    write_jsonl(output_dir / "private" / "confirmation_records.jsonl", records["confirmation"])
    write_jsonl(
        output_dir / "private" / "confirmation_labels.jsonl",
        (
            {
                "example_id": row["example_id"],
                "source_example_id": row["source_example_id"],
                "stage": row["stage"],
                "target": row["target"],
            }
            for row in records["confirmation"]
        ),
    )
    for split in ("train", "dev"):
        write_jsonl(
            output_dir / "exports" / "messages" / f"{split}.jsonl",
            (make_messages(row, True) for row in records[split]),
        )
        write_jsonl(
            output_dir / "exports" / "qwen" / f"{split}.jsonl",
            (make_qwen_record(row, True) for row in records[split]),
        )
    write_jsonl(
        output_dir / "exports" / "messages" / "confirmation.jsonl",
        (make_messages(row, False) for row in records["confirmation"]),
    )
    write_jsonl(
        output_dir / "exports" / "qwen" / "confirmation.jsonl",
        (make_qwen_record(row, False) for row in records["confirmation"]),
    )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records["train"]:
        groups[str(row["group_id"])].append(row)
    selected = sorted(groups)[: int(dataset["trial_train_scenarios"])]
    mapping_groups = set(selected[: int(dataset["mapping_refresh_scenarios"])])
    trial = [
        row
        for group_id in selected
        for row in groups[group_id]
        if row["stage"] == "compact_tool_result_interpretation"
        or (group_id in mapping_groups and row["stage"] == "tool_source_mapping")
    ]
    write_jsonl(
        output_dir / "exports" / "qwen" / "train_compact_focus.jsonl",
        (make_qwen_record(row, True) for row in trial),
    )
    manifest = {
        "dataset": dataset["name"],
        "version": version,
        "source_manifest_sha256": file_sha256(source_dir / "manifest.json"),
        "stages": list(STAGES),
        "split_record_counts": {split: len(rows) for split, rows in records.items()},
        "split_stage_counts": {
            split: dict(sorted(Counter(row["stage"] for row in rows).items()))
            for split, rows in records.items()
        },
        "focus_scenario_count": len(selected),
        "mapping_refresh_scenario_count": len(mapping_groups),
        "focus_record_count": len(trial),
        "focus_stage_counts": dict(sorted(Counter(row["stage"] for row in trial).items())),
        "focus_group_ids_hash": stable_json_hash(selected),
        "canonical_hashes": {split: stable_json_hash(rows) for split, rows in records.items()},
        "promotion_gates": copy.deepcopy(config["promotion_gates"]),
        "protocol": copy.deepcopy(config["protocol"]),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(build(load_yaml(args.config), args.source_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
