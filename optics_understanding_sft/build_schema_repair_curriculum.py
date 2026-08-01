#!/usr/bin/env python3
"""Build a compact, schema-compatible action-first repair curriculum.

The simulator labels and matched groups come directly from action-first v3.
Only the response envelope is changed: action evidence is serialized first
inside the required ``answer`` object, followed by the derived status label.
The 200 rows are ordered as two distribution-identical 100-row halves so both
25-step and 50-step checkpoints see every supervision branch.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_corrective_curriculum import completion_target
from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


ANCHOR_TASKS = (
    "setup_interpretation",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "counterfactual_reasoning",
)

CONTRACT_MARKER = "Task output contract (all fields shown; use null when not applicable):"
CONTROL_CONTRACT = """Task output contract (all fields shown; use null when not applicable):
{
  "answer": {
    "actuator": "actuator_name",
    "signed_movement_mm": "number",
    "predicted_residual_px": "number",
    "executable_valid": "boolean",
    "best_achievable_residual_px": "number | null",
    "control_plan": "object with numeric lens_x_delta_mm, lens_y_delta_mm, camera_x_delta_mm, camera_y_delta_mm | null",
    "expected_residual_px": "number | null"
  },
  "status": "feasible | infeasible_within_limits"
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=431)
    return parser.parse_args()


def replace_completion(row: dict[str, Any], target: Mapping[str, Any]) -> None:
    row["completion"] = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps(target, sort_keys=False)}],
        }
    ]


def transform_focused(row: Mapping[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    original_id = str(item["example_id"])
    item["source_example_id"] = original_id
    item["example_id"] = f"{original_id}__schema_v3_1"
    item["curriculum_source"] = "schema_repair_v3_1"
    if item.get("match_group_id"):
        item["match_group_id"] = f"{item['match_group_id']}:schema_v3_1"

    target = completion_target(item)
    answer = dict(target["answer"])
    if item["task_type"] == "constrained_intervention":
        action = target["action"]
        repaired_answer = {
            "actuator": action["actuator"],
            "signed_movement_mm": action["signed_movement_mm"],
            "predicted_residual_px": action["predicted_residual_px"],
            "executable_valid": action["executable_valid"],
            **answer,
        }
        repaired_target = {"answer": repaired_answer, "status": target["status"]}
        text_item = item["prompt"][0]["content"][-1]
        prompt_text = str(text_item["text"])
        if CONTRACT_MARKER not in prompt_text:
            raise ValueError(f"control prompt lacks contract marker: {original_id}")
        text_item["text"] = prompt_text.split(CONTRACT_MARKER, 1)[0] + CONTROL_CONTRACT
    elif item["task_type"] == "information_sufficiency":
        repaired_target = {"answer": answer, "status": target["status"]}
    else:
        raise ValueError(f"unexpected focused task: {item['task_type']}")
    replace_completion(item, repaired_target)
    return item


def prepare_anchor(row: Mapping[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    original_id = str(item["example_id"])
    item["example_id"] = f"{original_id}__schema_v3_1_anchor"
    item["source_example_id"] = str(item.get("source_example_id", original_id))
    item["curriculum_source"] = "schema_preservation_anchor"
    # Selected anchors are independent preservation examples.  Do not let a
    # prior curriculum match ID merge them into a partial sampling unit.
    item.pop("match_group_id", None)
    return item


def select_groups(
    rows: list[dict[str, Any]], *, task: str, sizes: Counter[int], seed: int
) -> dict[int, list[list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["task_type"] == task:
            groups[str(row.get("match_group_id"))].append(row)
    by_size: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
    for members in groups.values():
        by_size[len(members)].append(sorted(members, key=lambda value: value["example_id"]))
    selected: dict[int, list[list[dict[str, Any]]]] = {}
    for size, count in sizes.items():
        candidates = sorted(by_size[size], key=lambda members: members[0]["match_group_id"])
        random.Random(seed + size * 997).shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f"need {count} size-{size} {task} groups, got {len(candidates)}")
        selected[size] = candidates[:count]
    return selected


def select_control_groups(rows: list[dict[str, Any]], seed: int) -> dict[int, list[list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["task_type"] == "constrained_intervention":
            groups[str(row["match_group_id"])].append(row)
    selected: dict[int, list[list[dict[str, Any]]]] = {2: [], 3: []}
    for size in (2, 3):
        for actuator_index, actuator in enumerate(("lens_x_delta_mm", "lens_y_delta_mm")):
            candidates = [
                sorted(members, key=lambda value: value["example_id"])
                for group_id, members in sorted(groups.items())
                if len(members) == size and f":{actuator}:" in group_id
            ]
            random.Random(seed + size * 997 + actuator_index * 10007).shuffle(candidates)
            if len(candidates) < 10:
                raise ValueError(f"need ten size-{size} groups for {actuator}")
            selected[size].extend(candidates[:10])
        # Alternate actuators after deterministic within-actuator selection.
        left, right = selected[size][:10], selected[size][10:]
        selected[size] = [member for pair in zip(left, right) for member in pair]
    return selected


def select_anchors(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for index, task in enumerate(ANCHOR_TASKS):
        candidates = sorted(
            (row for row in rows if row.get("curriculum_source") == "preservation_anchor" and row["task_type"] == task),
            key=lambda row: row["example_id"],
        )
        random.Random(seed + index * 1013).shuffle(candidates)
        if len(candidates) < 8:
            raise ValueError(f"need eight preservation anchors for {task}")
        selected[task] = candidates[:8]
    return selected


def build_rows(
    focused_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    controls = select_control_groups(focused_rows, seed)
    sufficiency = select_groups(
        focused_rows,
        task="information_sufficiency",
        sizes=Counter({2: 30}),
        seed=seed + 17,
    )[2]
    anchors = select_anchors(anchor_rows, seed)

    triplets = controls[3]
    pairs = controls[2]
    anchor_stream = [
        anchors[ANCHOR_TASKS[index % len(ANCHOR_TASKS)]][index // len(ANCHOR_TASKS)]
        for index in range(40)
    ]
    rows: list[dict[str, Any]] = []
    for cycle in range(20):
        units = [triplets[cycle], sufficiency[cycle], [anchor_stream[cycle * 2]], pairs[cycle]]
        # Five extra sufficiency pairs per ten-cycle half make each half
        # exactly 100 rows with the same task distribution.
        if cycle < 5:
            units.append(sufficiency[20 + cycle])
        elif 10 <= cycle < 15:
            units.append(sufficiency[15 + cycle])
        units.append([anchor_stream[cycle * 2 + 1]])
        for unit in units:
            rows.extend(transform_focused(row) for row in unit if row["task_type"] in {"constrained_intervention", "information_sufficiency"})
            rows.extend(prepare_anchor(row) for row in unit if row["task_type"] not in {"constrained_intervention", "information_sufficiency"})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        target = completion_target(row)
        counts[f"task:{row['task_type']}"] += 1
        counts[f"status:{row['task_type']}:{target['status']}"] += 1
        counts[f"source:{row['curriculum_source']}"] += 1
        counts[f"modality:{'visual' if row.get('images') else 'text'}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    focused = read_jsonl(args.focused_jsonl)
    anchors = read_jsonl(args.anchor_jsonl)
    rows = build_rows(focused, anchors, args.seed)
    if len(rows) != 200 or len({row["source_example_id"] for row in rows}) != 200:
        raise ValueError("repair curriculum must contain 200 unique source examples")
    expected_half = {
        "constrained_intervention": 50,
        "information_sufficiency": 30,
        **{task: 4 for task in ANCHOR_TASKS},
    }
    for offset in (0, 100):
        observed = Counter(row["task_type"] for row in rows[offset : offset + 100])
        if observed != Counter(expected_half):
            raise ValueError(f"unbalanced curriculum half at {offset}: {dict(observed)}")
    write_jsonl(args.output_jsonl, rows)
    manifest = {
        "name": "optics_understanding_schema_repair_curriculum_v3_1",
        "seed": args.seed,
        "record_count": len(rows),
        "unique_source_examples": len({row["source_example_id"] for row in rows}),
        "repetition_count": 0,
        "focused_sha256": file_sha256(args.focused_jsonl),
        "anchor_pool_sha256": file_sha256(args.anchor_jsonl),
        "example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
        "first_half_counts": dict(sorted(Counter(row["task_type"] for row in rows[:100]).items())),
        "second_half_counts": dict(sorted(Counter(row["task_type"] for row in rows[100:]).items())),
        "counts": summarize(rows),
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
