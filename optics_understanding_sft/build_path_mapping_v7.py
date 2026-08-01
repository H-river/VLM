#!/usr/bin/env python3
"""Build v7 compact source-mapping supervision from certified v5A evidence."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_intermediate_evidence_v6 import interpretation_target, visible_evidence
from .core import file_sha256, load_yaml, make_messages, make_qwen_record, read_jsonl, stable_json_hash, write_jsonl
from .decision_tools import CONTROL_TOOL, SUFFICIENCY_TOOL, TOOL_CATALOG, tool_for_task
from .tool_path_adapter import SOURCE_MAPS, run_mapped_tool


STAGES = ("tool_choice", "tool_source_mapping", "tool_result_interpretation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mapped_catalog() -> dict[str, Any]:
    return {
        tool: {
            "description": TOOL_CATALOG[tool]["description"],
            "required_source_roles": list(SOURCE_MAPS[tool]),
        }
        for tool in (CONTROL_TOOL, SUFFICIENCY_TOOL)
    }


def stage_prompt(stage: str, source_task: str, inputs: Mapping[str, Any]) -> str:
    if stage == "tool_choice":
        instruction = (
            "Choose the deterministic tool for this optics operation. Return only strict JSON "
            "matching {\"tool_name\":\"registered_name\"}."
        )
    elif stage == "tool_source_mapping":
        instruction = (
            "Map each required semantic role to a dot path or row-field name in visible_evidence. "
            "Do not copy numeric values, perform arithmetic, select a row, or invent shortcut "
            "arguments. Return only strict JSON with tool_name and source_map."
        )
    else:
        instruction = (
            "Interpret the validated deterministic tool result. Copy the requested intermediate "
            "evidence exactly, then return the physical status and answer as strict JSON with keys "
            "intermediate_evidence, status, and answer."
        )
    payload = {"source_task": source_task, **copy.deepcopy(dict(inputs))}
    return instruction + "\n\nInput data:\n" + json.dumps(payload, indent=2, sort_keys=True)


def derive_records(source: Mapping[str, Any], version: str) -> list[dict[str, Any]]:
    source_id = str(source["example_id"])
    source_task = str(source["task_type"])
    tool_name = tool_for_task(source_task)
    evidence = visible_evidence(source)
    source_map = copy.deepcopy(SOURCE_MAPS[tool_name])
    result = run_mapped_tool(tool_name, evidence, source_map)
    common = {
        "group_id": source["group_id"],
        "split": source["split"],
        "modality": "text",
        "source_example_id": source_id,
        "source_task_type": source_task,
        "provenance": {
            "dataset_version": version,
            "source_dataset": "evidence_grounded_v5a",
            "source_example_id": source_id,
            "source_match_group_id": source["provenance"]["match_group_id"],
            "scenario_seed": source["provenance"]["scenario_seed"],
            "label_source": "deterministic_mapped_tool",
        },
    }
    choice_inputs = {
        "requested_operation": (
            "exhaustively select a constrained actuator action"
            if source_task == "constrained_intervention"
            else "threshold compatible-completion changes and test agreement"
        ),
        "available_tools": mapped_catalog(),
    }
    mapping_inputs = {
        "selected_tool": tool_name,
        "required_source_roles": list(source_map),
        "visible_evidence": evidence,
    }
    interpretation_inputs = {
        "tool_name": tool_name,
        "adapter_validation": "passed",
        "tool_result": result,
        "visible_evidence": evidence,
    }
    specs = [
        ("tool_choice", choice_inputs, {"tool_name": tool_name}),
        (
            "tool_source_mapping",
            mapping_inputs,
            {"tool_name": tool_name, "source_map": source_map},
        ),
        (
            "tool_result_interpretation",
            interpretation_inputs,
            interpretation_target(source, result),
        ),
    ]
    return [
        {
            **copy.deepcopy(common),
            "example_id": f"{source_id}__v7_{stage}",
            "task_type": stage,
            "stage": stage,
            "prompt_inputs": copy.deepcopy(inputs),
            "prompt": stage_prompt(stage, source_task, inputs),
            "target": copy.deepcopy(target),
        }
        for stage, inputs, target in specs
    ]


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
    trial = [row for group_id in selected for row in groups[group_id]]
    write_jsonl(
        output_dir / "exports" / "qwen" / "train_trial155_focus.jsonl",
        (make_qwen_record(row, True) for row in trial),
    )
    (output_dir / "mapped_tool_catalog.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "mapped_tool_catalog.json").write_text(
        json.dumps(mapped_catalog(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": dataset["name"],
        "version": version,
        "source_manifest_sha256": file_sha256(source_dir / "manifest.json"),
        "stages": list(STAGES),
        "split_record_counts": {split: len(rows) for split, rows in records.items()},
        "split_source_record_counts": {split: len(rows) for split, rows in source.items()},
        "split_stage_counts": {
            split: dict(sorted(Counter(row["stage"] for row in rows).items()))
            for split, rows in records.items()
        },
        "focus_train_scenario_count": len(selected),
        "focus_train_record_count": len(trial),
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
