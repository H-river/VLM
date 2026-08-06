#!/usr/bin/env python3
"""Build tool-choice, source-mapping, and interpretation records for direct routes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .core import read_jsonl, stable_json_hash, write_jsonl
from .direct_reasoning_tools_v11 import (
    CAUSAL_TOOL,
    DIAGNOSIS_TOOL,
    SETUP_TOOL,
    TOOL_SPECS,
    materialize_direct_answer,
    run_mapped_direct_tool,
)


TASK_TO_TOOL = {
    "setup_interpretation": SETUP_TOOL,
    "causal_effects": CAUSAL_TOOL,
    "diagnosis": DIAGNOSIS_TOOL,
}
STAGES = ("tool_choice", "tool_source_mapping", "compact_tool_result_interpretation")
CAUSAL_THRESHOLDS = {"centroid_px": 1.0, "sigma_px": 2.0, "peak_relative": 0.05}


def handle(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{stable_json_hash(value)[:20]}"


def prompt(stage: str, inputs: Mapping[str, Any]) -> str:
    if stage == "tool_choice":
        instruction = (
            'Choose the registered deterministic tool that directly performs the requested operation. '
            'Return only {"tool_name":"registered_name"}.'
        )
    elif stage == "tool_source_mapping":
        instruction = (
            "Map every required source role to one exact prompt-visible evidence path. Return only strict "
            "JSON with tool_name and source_map. Do not expose private state, omit roles, or invent paths."
        )
    else:
        instruction = (
            "Interpret the validated deterministic tool result as the final answer. Return only strict "
            "JSON in the requested output shape and copy categorical values exactly."
        )
        if inputs.get("source_task") == "causal_effects":
            instruction += (
                " Copy all five named effect fields from tool_result; never emit a placeholder key "
                "such as field."
            )
        elif inputs.get("source_task") == "diagnosis":
            instruction += (
                " Set output status to tool_result.status and output answer.plausible_causes to the "
                "entire tool_result.plausible_causes list. Never repeat the Input data wrapper."
            )
    return instruction + "\n\nInput data:\n" + json.dumps(dict(inputs), indent=2, sort_keys=True)


def private_records(master_jsonl: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case in read_jsonl(master_jsonl):
        for item in case["records"]:
            example_id = str(item["record"]["example_id"])
            result[example_id] = {
                "case_setup_config": copy.deepcopy(case["setup_config"]),
                "private_eval": copy.deepcopy(item["private_eval"]),
            }
    return result


def replay_by_name(private: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(spec["name"]): spec
        for spec in private["private_eval"].get("replay_specs", [])
    }


def source_evidence(
    source: Mapping[str, Any], private: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    task = str(source["task_type"])
    inputs = source["prompt_inputs"]
    setup_registry: dict[str, Mapping[str, Any]] = {}
    observation_registry: dict[str, Mapping[str, Any]] = {}
    if task == "setup_interpretation":
        visible = {
            "setup": copy.deepcopy(inputs["setup"]),
            "actuator_interface": copy.deepcopy(inputs["actuator_interface"]),
        }
        source_map = {
            "setup_path": "setup",
            "actuator_interface_path": "actuator_interface",
        }
        return visible, source_map, setup_registry, observation_registry
    replay = replay_by_name(private)
    if task == "causal_effects":
        before = copy.deepcopy(replay["before"]["expected_state"])
        after = copy.deepcopy(replay["after"]["expected_state"])
        before_handle = handle("optics-observation", before)
        after_handle = handle("optics-observation", after)
        observation_registry.update({before_handle: before, after_handle: after})
        visible = {
            "before_state_handle": before_handle,
            "after_state_handle": after_handle,
            "thresholds": copy.deepcopy(CAUSAL_THRESHOLDS),
        }
        source_map = {
            "before_state_handle_path": "before_state_handle",
            "after_state_handle_path": "after_state_handle",
            "thresholds_path": "thresholds",
        }
        return visible, source_map, setup_registry, observation_registry
    if task == "diagnosis":
        baseline = replay["baseline"]
        observed = copy.deepcopy(replay["observed"]["expected_state"])
        setup_config = copy.deepcopy(baseline["setup_config"])
        setup_handle = handle("optics-state", setup_config)
        observed_handle = handle("optics-observation", observed)
        setup_registry[setup_handle] = setup_config
        observation_registry[observed_handle] = observed
        visible = {
            "setup_state_handle": setup_handle,
            "observed_state_handle": observed_handle,
            "candidate_interventions": copy.deepcopy(inputs["candidate_interventions"]),
            "matching_tolerance": copy.deepcopy(inputs["matching_tolerance"]),
        }
        source_map = {
            "setup_state_handle_path": "setup_state_handle",
            "observed_state_handle_path": "observed_state_handle",
            "candidate_interventions_path": "candidate_interventions",
            "matching_tolerance_path": "matching_tolerance",
        }
        return visible, source_map, setup_registry, observation_registry
    raise ValueError(f"unsupported source task: {task}")


def requested_output_shape(task: str) -> dict[str, Any]:
    if task == "setup_interpretation":
        return {
            "status": "answerable",
            "answer": {
                "component_order": ["component_name"],
                "adjustable_parameters": ["parameter_name"],
                "total_source_to_sensor_mm": "number",
                "lens_focal_length_m": "number",
            },
        }
    if task == "causal_effects":
        return {
            "status": "answerable",
            "answer": {
                "effects": {
                    "centroid_x": "increase | decrease | no_change",
                    "centroid_y": "increase | decrease | no_change",
                    "sigma_x": "increase | decrease | no_change",
                    "sigma_y": "increase | decrease | no_change",
                    "peak_intensity": "increase | decrease | no_change",
                }
            },
        }
    return {
        "status": "unique | ambiguous | unsupported",
        "answer": {"plausible_causes": ["candidate_id"]},
    }


def derive(
    source: Mapping[str, Any],
    private: Mapping[str, Any],
    *,
    dataset_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    task = str(source["task_type"])
    tool_name = TASK_TO_TOOL[task]
    visible, source_map, setup_registry, observation_registry = source_evidence(source, private)
    tool_result = run_mapped_direct_tool(
        tool_name,
        visible,
        source_map,
        setup_registry=setup_registry,
        observation_registry=observation_registry,
    )
    final_target = materialize_direct_answer(tool_name, tool_result)
    if final_target != source["target"]:
        raise ValueError(
            f"tool replay disagrees with simulator target for {source['example_id']}: "
            f"tool={final_target} target={source['target']}"
        )
    specs = [
        (
            "tool_choice",
            {
                "source_task": task,
                "requested_operation": TOOL_SPECS[tool_name]["operation"],
                "available_tools": copy.deepcopy(TOOL_SPECS),
            },
            {"tool_name": tool_name},
        ),
        (
            "tool_source_mapping",
            {
                "source_task": task,
                "selected_tool": tool_name,
                "required_source_roles": copy.deepcopy(
                    TOOL_SPECS[tool_name]["required_source_roles"]
                ),
                "visible_evidence": copy.deepcopy(visible),
            },
            {"tool_name": tool_name, "source_map": copy.deepcopy(source_map)},
        ),
        (
            "compact_tool_result_interpretation",
            {
                "source_task": task,
                "tool_name": tool_name,
                "adapter_validation": "passed",
                "tool_result": copy.deepcopy(tool_result),
                "requested_output_shape": requested_output_shape(task),
            },
            final_target,
        ),
    ]
    records = [
        {
            "example_id": f"{source['example_id']}__{dataset_version}_{stage}",
            "group_id": source["group_id"],
            "split": "dev",
            "modality": "text",
            "source_modality": source.get("modality", "text"),
            "source_example_id": source["example_id"],
            "source_task_type": task,
            "task_type": stage,
            "stage": stage,
            "prompt_inputs": copy.deepcopy(inputs),
            "prompt": prompt(stage, inputs),
            "target": copy.deepcopy(target),
            "provenance": {
                "dataset_version": dataset_version,
                "source_example_id": source["example_id"],
                "label_source": "deterministic_registered_direct_tool",
            },
        }
        for stage, inputs, target in specs
    ]
    return records, setup_registry, observation_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", default="direct_tool_orchestration_v11_1")
    args = parser.parse_args()
    private = private_records(args.master_jsonl)
    source = [
        row for row in read_jsonl(args.records_jsonl) if row["task_type"] in TASK_TO_TOOL
    ]
    records: list[dict[str, Any]] = []
    setup_registry: dict[str, Mapping[str, Any]] = {}
    observation_registry: dict[str, Mapping[str, Any]] = {}
    for row in source:
        derived, setups, observations = derive(
            row,
            private[str(row["example_id"])],
            dataset_version=args.dataset_version,
        )
        records.extend(derived)
        setup_registry.update(setups)
        observation_registry.update(observations)
    write_jsonl(args.output_dir / "canonical" / "dev.jsonl", records)
    write_jsonl(
        args.output_dir / "private" / "setup_registry.jsonl",
        ({"state_handle": key, "setup_config": value} for key, value in sorted(setup_registry.items())),
    )
    write_jsonl(
        args.output_dir / "private" / "observation_registry.jsonl",
        ({"observation_handle": key, "state": value} for key, value in sorted(observation_registry.items())),
    )
    manifest = {
        "dataset": args.dataset_version,
        "source_count": len(source),
        "record_count": len(records),
        "setup_registry_count": len(setup_registry),
        "observation_registry_count": len(observation_registry),
        "source_task_counts": {
            task: sum(row["task_type"] == task for row in source) for task in TASK_TO_TOOL
        },
        "canonical_hash": stable_json_hash(records),
        "target_reconstruction_exact": True,
        "sealed_evaluation_opened": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
