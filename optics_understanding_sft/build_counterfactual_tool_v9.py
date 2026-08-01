#!/usr/bin/env python3
"""Build paired-counterfactual tool-routing data with opaque state handles."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .build_path_mapping_v7 import mapped_catalog
from .core import make_messages, make_qwen_record, read_jsonl, stable_json_hash, write_jsonl
from .counterfactual_tool import (
    COUNTERFACTUAL_SOURCE_MAP,
    COUNTERFACTUAL_TOOL,
    COUNTERFACTUAL_TOOL_SPEC,
)


STAGES = ("tool_choice", "tool_source_mapping", "compact_tool_result_interpretation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-master-jsonl", type=Path, required=True)
    parser.add_argument("--dev-master-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-train-records", type=int, default=100)
    return parser.parse_args()


def counterfactual_items(path: Path) -> list[dict[str, Any]]:
    return [
        item
        for master in read_jsonl(path)
        for item in master["records"]
        if item["record"]["task_type"] == "counterfactual_reasoning"
    ]


def setup_pair(item: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    specs = {spec["name"]: spec for spec in item["private_eval"]["replay_specs"]}
    return (
        copy.deepcopy(specs["scenario_a_before"]["setup_config"]),
        copy.deepcopy(specs["scenario_b_before"]["setup_config"]),
    )


def state_handle(setup: Mapping[str, Any]) -> str:
    return "optics-state-" + stable_json_hash(setup)[:20]


def catalog() -> dict[str, Any]:
    return {**mapped_catalog(), COUNTERFACTUAL_TOOL: copy.deepcopy(COUNTERFACTUAL_TOOL_SPEC)}


def prompt(stage: str, inputs: Mapping[str, Any]) -> str:
    if stage == "tool_choice":
        instruction = "Choose the deterministic tool for the requested operation. Return only {\"tool_name\":\"registered_name\"}."
    elif stage == "tool_source_mapping":
        instruction = "Map each required role to a visible-evidence path. Return only strict JSON with tool_name and source_map; do not expose or guess private setup state."
    else:
        instruction = "A validated deterministic paired-counterfactual result is ready. Confirm that it is authoritative. Return only strict JSON with status, result_ready, and result_fields; do not copy or recalculate numbers."
    return instruction + "\n\nInput data:\n" + json.dumps(dict(inputs), indent=2, sort_keys=True)


def derive(item: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = item["record"]
    setup_a, setup_b = setup_pair(item)
    handle_a, handle_b = state_handle(setup_a), state_handle(setup_b)
    visible = {
        "scenario_a_state_handle": handle_a,
        "scenario_b_state_handle": handle_b,
        "shared_action": copy.deepcopy(source["prompt_inputs"]["shared_action"]),
        "changed_parameter": source["prompt_inputs"]["changed_parameter"],
    }
    tool_result = copy.deepcopy(source["target"]["answer"])
    result_fields = [
        "changed_parameter",
        "response_a",
        "response_b",
        "response_difference",
        "centroid_direction_preserved",
    ]
    specifications = [
        (
            "tool_choice",
            {
                "source_task": "counterfactual_reasoning",
                "requested_operation": "compare exact sensor responses for two registered states under one shared action",
                "available_tools": catalog(),
            },
            {"tool_name": COUNTERFACTUAL_TOOL},
        ),
        (
            "tool_source_mapping",
            {
                "source_task": "counterfactual_reasoning",
                "selected_tool": COUNTERFACTUAL_TOOL,
                "required_source_roles": list(COUNTERFACTUAL_SOURCE_MAP),
                "visible_evidence": visible,
            },
            {"tool_name": COUNTERFACTUAL_TOOL, "source_map": copy.deepcopy(COUNTERFACTUAL_SOURCE_MAP)},
        ),
        (
            "compact_tool_result_interpretation",
            {
                "source_task": "counterfactual_reasoning",
                "tool_name": COUNTERFACTUAL_TOOL,
                "adapter_validation": "passed",
                "tool_result": tool_result,
            },
            {"status": "answerable", "result_ready": True, "result_fields": result_fields},
        ),
    ]
    common = {
        "group_id": source["group_id"],
        "split": split,
        "modality": "text",
        "source_modality": source["modality"],
        "source_example_id": source["example_id"],
        "source_task_type": "counterfactual_reasoning",
        "provenance": {
            "dataset_version": "counterfactual_tool_v9",
            "source_dataset": source["provenance"]["dataset_version"],
            "source_example_id": source["example_id"],
            "scenario_seed": source["provenance"]["scenario_seed"],
            "label_source": "paired_simulator_cache_with_private_state_handles",
        },
    }
    records = [
        {
            **copy.deepcopy(common),
            "example_id": f"{source['example_id']}__v9_{stage}",
            "task_type": stage,
            "stage": stage,
            "prompt_inputs": copy.deepcopy(inputs),
            "prompt": prompt(stage, inputs),
            "target": copy.deepcopy(target),
        }
        for stage, inputs, target in specifications
    ]
    registries = [
        {"state_handle": handle_a, "source_example_id": source["example_id"], "setup_config": setup_a},
        {"state_handle": handle_b, "source_example_id": source["example_id"], "setup_config": setup_b},
    ]
    return records, registries


def main() -> None:
    args = parse_args()
    source = {
        "train": counterfactual_items(args.train_master_jsonl),
        "dev": counterfactual_items(args.dev_master_jsonl),
    }
    records: dict[str, list[dict[str, Any]]] = {}
    for split, items in source.items():
        derived = [derive(item, split) for item in items]
        records[split] = [record for item_records, _ in derived for record in item_records]
        registry_by_handle = {
            entry["state_handle"]: entry
            for _, item_registries in derived
            for entry in item_registries
        }
        write_jsonl(args.output_dir / "canonical" / f"{split}.jsonl", records[split])
        write_jsonl(args.output_dir / "exports" / "messages" / f"{split}.jsonl", (make_messages(row, True) for row in records[split]))
        write_jsonl(args.output_dir / "exports" / "qwen" / f"{split}.jsonl", (make_qwen_record(row, True) for row in records[split]))
        write_jsonl(args.output_dir / "private" / f"{split}_state_registry.jsonl", registry_by_handle.values())
    focus_sources = {str(row["source_example_id"]) for row in records["train"][: args.trial_train_records * 3]}
    focus = [row for row in records["train"] if row["source_example_id"] in focus_sources]
    write_jsonl(args.output_dir / "exports" / "qwen" / "train_focus.jsonl", (make_qwen_record(row, True) for row in focus))
    manifest = {
        "dataset": "optics_understanding_counterfactual_tool_v9",
        "version": "counterfactual_tool_v9",
        "stages": list(STAGES),
        "split_source_counts": {split: len(rows) for split, rows in source.items()},
        "split_record_counts": {split: len(rows) for split, rows in records.items()},
        "split_stage_counts": {
            split: dict(sorted(Counter(row["stage"] for row in rows).items()))
            for split, rows in records.items()
        },
        "split_source_modality_counts": {
            split: dict(sorted(Counter(row["source_modality"] for row in rows[::3]).items()))
            for split, rows in records.items()
        },
        "focus_source_count": len(focus_sources),
        "focus_record_count": len(focus),
        "state_handle_policy": "two opaque handles; private registry never appears in public prompts",
        "canonical_hashes": {split: stable_json_hash(rows) for split, rows in records.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
