#!/usr/bin/env python3
"""Build v6 records for tool routing, call construction, and interpretation."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import file_sha256, load_yaml, make_messages, make_qwen_record, read_jsonl, stable_json_hash, write_jsonl
from .decision_tools import CONTROL_TOOL, SUFFICIENCY_TOOL, TOOL_CATALOG, run_tool, tool_for_task


STAGES = ("tool_choice", "tool_call_construction", "tool_result_interpretation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _allowed_index(allowed: list[Any], motion: float) -> int:
    matches = [index for index, value in enumerate(allowed) if abs(float(value) - motion) <= 1e-9]
    if len(matches) != 1:
        raise ValueError(f"motion {motion} does not uniquely match allowed_values_mm")
    return matches[0]


def tool_arguments(record: Mapping[str, Any]) -> dict[str, Any]:
    inputs = record["prompt_inputs"]
    if record["task_type"] == "constrained_intervention":
        constraints = inputs["actuator_constraints"]
        active = str(constraints["active_actuator"])
        allowed = list(constraints["allowed_values_mm"])
        trials = inputs["candidate_action_trials"]
        motions = [float(trial["action"][active]) for trial in trials]
        return {
            "candidate_residuals_px": [float(trial["measured_residual_px"]) for trial in trials],
            "active_actuator_motions_mm": motions,
            "success_tolerance_px": float(constraints["success_tolerance_px"]),
            "allowed_order_indices": [_allowed_index(allowed, motion) for motion in motions],
        }
    if record["task_type"] == "information_sufficiency":
        return {
            "measured_deltas_px": [
                float(trial["measured_delta_px"])
                for trial in inputs["compatible_completion_trials"]
            ],
            "direction_threshold_px": float(inputs["direction_threshold_px"]),
        }
    raise ValueError(f"unsupported source task: {record['task_type']}")


def visible_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    inputs = record["prompt_inputs"]
    if record["task_type"] == "constrained_intervention":
        return {
            "actuator_constraints": copy.deepcopy(inputs["actuator_constraints"]),
            "candidate_action_trials": copy.deepcopy(inputs["candidate_action_trials"]),
        }
    return {
        "hidden_action_field": inputs["hidden_action_field"],
        "questioned_output": inputs["questioned_output"],
        "direction_threshold_px": inputs["direction_threshold_px"],
        "compatible_completion_trials": copy.deepcopy(inputs["compatible_completion_trials"]),
    }


def interpretation_target(record: Mapping[str, Any], tool_result: Mapping[str, Any]) -> dict[str, Any]:
    if record["task_type"] == "constrained_intervention":
        evidence = {
            "successful_action_indices": copy.deepcopy(tool_result["successful_action_indices"]),
            "selected_index": tool_result["selected_index"],
            "best_residual_index": tool_result["best_residual_index"],
        }
    else:
        evidence = {
            "observed_direction_set": copy.deepcopy(tool_result["observed_direction_set"]),
            "conflicting_pair_indices": copy.deepcopy(tool_result["conflicting_pair_indices"]),
        }
    return {
        "intermediate_evidence": evidence,
        "status": record["target"]["status"],
        "answer": copy.deepcopy(record["target"]["answer"]),
    }


def stage_prompt(stage: str, source_task: str, inputs: Mapping[str, Any]) -> str:
    if stage == "tool_choice":
        instruction = (
            "Choose the one deterministic tool that should perform the numerical evidence operation. "
            "Return only strict JSON matching {\"tool_name\": \"registered_name\"}."
        )
    elif stage == "tool_call_construction":
        instruction = (
            "Construct the exact deterministic tool call from the visible evidence. Preserve displayed "
            "row order. Return only strict JSON with tool_name and arguments; do not solve the task yourself."
        )
    else:
        instruction = (
            "Interpret the deterministic tool result in the optics task. Copy the requested intermediate "
            "evidence exactly, then produce status and answer. Return only strict JSON with keys "
            "intermediate_evidence, status, and answer."
        )
    payload = {"source_task": source_task, **copy.deepcopy(dict(inputs))}
    return instruction + "\n\nInput data:\n" + json.dumps(payload, indent=2, sort_keys=True)


def derive_records(source: Mapping[str, Any], version: str) -> list[dict[str, Any]]:
    source_id = str(source["example_id"])
    source_task = str(source["task_type"])
    tool_name = tool_for_task(source_task)
    arguments = tool_arguments(source)
    result = run_tool(tool_name, arguments)
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
            "label_source": "deterministic_prompt_visible_tool",
        },
    }
    choice_inputs = {
        "requested_operation": (
            "exhaustively select a constrained actuator action"
            if source_task == "constrained_intervention"
            else "threshold compatible-completion changes and test agreement"
        ),
        "available_tools": copy.deepcopy(TOOL_CATALOG),
    }
    call_inputs = {"selected_tool": tool_name, "visible_evidence": visible_evidence(source)}
    interpretation_inputs = {
        "tool_name": tool_name,
        "tool_result": copy.deepcopy(result),
        "visible_evidence": visible_evidence(source),
    }
    specifications = [
        ("tool_choice", choice_inputs, {"tool_name": tool_name}),
        (
            "tool_call_construction",
            call_inputs,
            {"tool_name": tool_name, "arguments": arguments},
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
            "example_id": f"{source_id}__{stage}",
            "task_type": stage,
            "stage": stage,
            "prompt_inputs": copy.deepcopy(stage_inputs),
            "prompt": stage_prompt(stage, source_task, stage_inputs),
            "target": copy.deepcopy(target),
        }
        for stage, stage_inputs, target in specifications
    ]


def _target_free(record: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(record))
    projected.pop("target", None)
    return projected


def build(config: Mapping[str, Any], source_dir: Path, output_dir: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    version = str(dataset["version"])
    source_by_split = {
        "train": read_jsonl(source_dir / "canonical" / "train.jsonl"),
        "dev": read_jsonl(source_dir / "canonical" / "dev.jsonl"),
        "confirmation": read_jsonl(source_dir / "private" / "confirmation_records.jsonl"),
    }
    records = {
        split: [derived for source in rows for derived in derive_records(source, version)]
        for split, rows in source_by_split.items()
    }
    write_jsonl(output_dir / "canonical" / "train.jsonl", records["train"])
    write_jsonl(output_dir / "canonical" / "dev.jsonl", records["dev"])
    write_jsonl(
        output_dir / "canonical" / "confirmation_prompts.jsonl",
        (_target_free(record) for record in records["confirmation"]),
    )
    write_jsonl(output_dir / "private" / "confirmation_records.jsonl", records["confirmation"])
    write_jsonl(
        output_dir / "private" / "confirmation_labels.jsonl",
        (
            {
                "example_id": record["example_id"],
                "source_example_id": record["source_example_id"],
                "stage": record["stage"],
                "target": record["target"],
            }
            for record in records["confirmation"]
        ),
    )
    for split in ("train", "dev"):
        write_jsonl(
            output_dir / "exports" / "messages" / f"{split}.jsonl",
            (make_messages(record, True) for record in records[split]),
        )
        write_jsonl(
            output_dir / "exports" / "qwen" / f"{split}.jsonl",
            (make_qwen_record(record, True) for record in records[split]),
        )
    write_jsonl(
        output_dir / "exports" / "messages" / "confirmation.jsonl",
        (make_messages(record, False) for record in records["confirmation"]),
    )
    write_jsonl(
        output_dir / "exports" / "qwen" / "confirmation.jsonl",
        (make_qwen_record(record, False) for record in records["confirmation"]),
    )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records["train"]:
        groups[str(record["group_id"])].append(record)
    selected_groups = sorted(groups)[: int(dataset["trial_train_scenarios"])]
    trial = [record for group_id in selected_groups for record in groups[group_id]]
    trial_path = output_dir / "exports" / "qwen" / "train_trial120.jsonl"
    write_jsonl(trial_path, (make_qwen_record(record, True) for record in trial))
    (output_dir / "tool_catalog.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "tool_catalog.json").write_text(
        json.dumps(TOOL_CATALOG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "dataset": dataset["name"],
        "version": version,
        "source_manifest_sha256": file_sha256(source_dir / "manifest.json"),
        "stages": list(STAGES),
        "split_record_counts": {split: len(rows) for split, rows in records.items()},
        "split_source_record_counts": {split: len(rows) for split, rows in source_by_split.items()},
        "split_stage_counts": {
            split: dict(sorted(Counter(row["stage"] for row in rows).items()))
            for split, rows in records.items()
        },
        "trial_train_scenario_count": len(selected_groups),
        "trial_train_record_count": len(trial),
        "trial_group_ids_hash": stable_json_hash(selected_groups),
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
