#!/usr/bin/env python3
"""Build tool-choice, image-role mapping, and result-interpretation supervision."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .core import make_messages, make_qwen_record, read_jsonl, stable_json_hash, write_jsonl
from .visual_state_tool_v10_1 import load_calibration
from .visual_tool_adapter_v10_2 import (
    PAIR_SOURCE_ROLES,
    PAIR_TOOL,
    STATE_SOURCE_ROLES,
    STATE_TOOL,
    TOOL_CATALOG,
    run_mapped_visual_tool,
)


STAGES = ("tool_choice", "tool_source_mapping", "compact_tool_result_interpretation")


def prompt(stage: str, inputs: Mapping[str, Any]) -> str:
    if stage == "tool_choice":
        instruction = (
            'Choose the registered deterministic tool that directly performs the requested operation. '
            'Return only {"tool_name":"registered_name"}.'
        )
    elif stage == "tool_source_mapping":
        instruction = (
            "Copy the exact prompt-visible file path into every required tool role. Return only strict "
            "JSON with tool_name and source_map. Do not shorten paths, swap first and second images, "
            "emit evidence-field names, or invent paths."
        )
    else:
        instruction = (
            "Interpret the validated deterministic tool result as the final categorical evidence. Return "
            "only strict JSON in the requested output shape; copy categories exactly and do not recalculate them."
        )
    return instruction + "\n\nInput data:\n" + json.dumps(dict(inputs), indent=2, sort_keys=True)


def derive(
    source: Mapping[str, Any],
    *,
    split: str,
    image_root: Path,
    state_calibration: Mapping[str, Any],
    pair_calibration: Mapping[str, Any],
    dataset_version: str,
    include_optional_reference: bool = True,
) -> list[dict[str, Any]]:
    images = list(source["prompt_inputs"]["images"])
    is_state = source["task_type"] == "visual_state_classification"
    tool_name = STATE_TOOL if is_state else PAIR_TOOL
    source_roles = STATE_SOURCE_ROLES if is_state else PAIR_SOURCE_ROLES
    if is_state:
        visible = {"image_path": images[0]}
    else:
        visible = {
            "first_image_path": images[0],
            "second_image_path": images[1],
        }
        if include_optional_reference:
            visible["signed_difference_reference_path"] = images[2]
    source_map = {role: visible[role] for role in source_roles}
    tool_result = run_mapped_visual_tool(
        tool_name,
        visible,
        source_map,
        image_root=image_root,
        state_calibration=state_calibration,
        pair_calibration=pair_calibration,
    )
    requested_operation = (
        "classify the beam position and width bands in one calibrated sensor image"
        if is_state
        else "classify observed changes from the first calibrated image to the second"
    )
    final_target = {"status": "answerable", "answer": copy.deepcopy(tool_result)}
    specs = [
        (
            "tool_choice",
            {
                "source_task": source["task_type"],
                "requested_operation": requested_operation,
                "available_tools": copy.deepcopy(TOOL_CATALOG),
            },
            {"tool_name": tool_name},
        ),
        (
            "tool_source_mapping",
            {
                "source_task": source["task_type"],
                "selected_tool": tool_name,
                "required_source_roles": list(source_roles),
                "visible_evidence": visible,
            },
            {"tool_name": tool_name, "source_map": copy.deepcopy(source_map)},
        ),
        (
            "compact_tool_result_interpretation",
            {
                "source_task": source["task_type"],
                "tool_name": tool_name,
                "adapter_validation": "passed",
                "tool_result": tool_result,
                "requested_output_shape": (
                    {"status": "answerable", "answer": {field: "category" for field in tool_result}}
                    if is_state
                    else {
                        "status": "answerable",
                        "answer": {"observed_direction_set": {field: "direction" for field in tool_result["observed_direction_set"]}},
                    }
                ),
            },
            final_target,
        ),
    ]
    return [
        {
            "example_id": f"{source['example_id']}__{dataset_version}_{stage}",
            "group_id": source["group_id"],
            "split": split,
            "modality": "text",
            "source_modality": "visual",
            "source_example_id": source["example_id"],
            "source_task_type": source["task_type"],
            "task_type": stage,
            "stage": stage,
            "prompt_inputs": copy.deepcopy(inputs),
            "prompt": prompt(stage, inputs),
            "target": copy.deepcopy(target),
            "provenance": {
                "dataset_version": dataset_version,
                "source_dataset": source["provenance"]["dataset_version"],
                "source_example_id": source["example_id"],
                "transform": source["provenance"]["transform"],
                "label_source": "deterministic_calibrated_image_tool",
            },
        }
        for stage, inputs, target in specs
    ]


def original_rows(path: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(path)
        if row["provenance"].get("transform") == "original"
        and row["task_type"] in {"visual_state_classification", "visual_pair_direction_extraction"}
    ]


def qwen_with_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    exported = make_qwen_record(row, True)
    exported["stage"] = row["stage"]
    exported["source_task_type"] = row["source_task_type"]
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--train-image-root", type=Path)
    parser.add_argument("--dev-image-root", type=Path)
    parser.add_argument("--state-calibration", type=Path, required=True)
    parser.add_argument("--pair-calibration", type=Path, required=True)
    parser.add_argument("--recovery-anchors", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--dataset-version", default="visual_tool_orchestration_v10_2")
    parser.add_argument("--omit-optional-difference-reference", action="store_true")
    args = parser.parse_args()
    train_image_root = args.train_image_root or args.image_root
    dev_image_root = args.dev_image_root or args.image_root
    if train_image_root is None or dev_image_root is None:
        parser.error("provide --image-root or both --train-image-root and --dev-image-root")
    state_calibration = load_calibration(args.state_calibration)
    pair_calibration = load_calibration(args.pair_calibration)
    source = {"train": original_rows(args.train_jsonl), "dev": original_rows(args.dev_jsonl)}
    image_roots = {"train": train_image_root, "dev": dev_image_root}
    records: dict[str, list[dict[str, Any]]] = {}
    for split, rows in source.items():
        records[split] = [
            derived
            for row in rows
            for derived in derive(
                row,
                split=split,
                image_root=image_roots[split],
                state_calibration=state_calibration,
                pair_calibration=pair_calibration,
                dataset_version=args.dataset_version,
                include_optional_reference=not args.omit_optional_difference_reference,
            )
        ]
        write_jsonl(args.output_dir / "canonical" / f"{split}.jsonl", records[split])
        write_jsonl(args.output_dir / "exports" / "messages" / f"{split}.jsonl", (make_messages(row, True) for row in records[split]))
        write_jsonl(args.output_dir / "exports" / "qwen" / f"{split}.jsonl", (make_qwen_record(row, True) for row in records[split]))

    qwen_train = [qwen_with_metadata(row) for row in records["train"]]
    curriculum = list(qwen_train)
    mapping_refresh = [row for row in qwen_train if row["stage"] == "tool_source_mapping"]
    state_mapping_refresh = [
        row
        for row in mapping_refresh
        if row["source_task_type"] == "visual_state_classification"
    ]
    curriculum.extend(copy.deepcopy(mapping_refresh))
    curriculum.extend(copy.deepcopy(state_mapping_refresh))
    anchor_count = 0
    if args.recovery_anchors:
        anchors = [row for row in read_jsonl(args.recovery_anchors) if not row.get("images")]
        random.Random(args.seed).shuffle(anchors)
        anchors = anchors[: min(360, len(anchors))]
        anchor_count = len(anchors)
        curriculum.extend(anchors)
    random.Random(args.seed).shuffle(curriculum)
    write_jsonl(args.output_dir / "exports" / "qwen" / "train_curriculum_v10_2.jsonl", curriculum)
    manifest = {
        "dataset": args.dataset_version,
        "seed": args.seed,
        "split_source_counts": {split: len(rows) for split, rows in source.items()},
        "split_record_counts": {split: len(rows) for split, rows in records.items()},
        "split_stage_counts": {
            split: dict(sorted(Counter(row["stage"] for row in rows).items()))
            for split, rows in records.items()
        },
        "split_source_task_counts": {
            split: dict(sorted(Counter(row["source_task_type"] for row in rows[::3]).items()))
            for split, rows in records.items()
        },
        "train_curriculum_count": len(curriculum),
        "train_curriculum_stage_counts": dict(
            sorted(
                Counter(
                    row.get("stage", "preservation_anchor")
                    for row in curriculum
                ).items()
            )
        ),
        "recovery_anchor_count": anchor_count,
        "canonical_hashes": {split: stable_json_hash(rows) for split, rows in records.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
