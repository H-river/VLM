#!/usr/bin/env python3
"""Build the frozen per-field goal/history/budget dev interventions."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_vl_supervisor_v1.export_sft import validate_export_row

BASE = ROOT / "supervisor_v1_1_candidate"
PREFIX = "Current supervisor state (all coordinates and units are explicit):\n"
ZERO_GOAL = {
    "coordinate_frame": "lab_sensor_1024px_and_raw_peak",
    "centroid_x": 0.0,
    "centroid_y": 0.0,
    "width_x": 0.0,
    "width_y": 0.0,
    "peak_intensity": 0.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def state(row: dict[str, Any]) -> dict[str, Any]:
    text = row["prompt"][1]["content"][1]["text"]
    if not text.startswith(PREFIX):
        raise ValueError(row["example_id"])
    return json.loads(text[len(PREFIX):])


def set_state(row: dict[str, Any], value: dict[str, Any]) -> None:
    row["prompt"][1]["content"][1]["text"] = PREFIX + json.dumps(value, sort_keys=True, separators=(",", ":"))


def permutation(length: int, seed: int) -> list[int]:
    values = list(range(length))
    random.Random(seed).shuffle(values)
    return values


def write_condition(name: str, rows: list[dict[str, Any]], changed: int, unique_before: int) -> None:
    path = BASE / "sft/interventions" / name / "sft_dev.jsonl"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        validate_export_row(row, repository_root=ROOT)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    report = {
        "version": "supervisor_v1_1_candidate_individual_state_intervention_v1",
        "status": "NOT SEALED — FROZEN EVALUATION DISABLED",
        "name": name,
        "records": len(rows),
        "changed_rows": changed,
        "unique_source_field_values": unique_before,
        "structural_no_op": changed == 0,
        "sha256": sha256(path),
        "source_dev_sha256": sha256(BASE / "sft/sft_dev.jsonl"),
        "frozen_or_protected_used": False,
    }
    (path.parent / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    source = read_jsonl(BASE / "sft/sft_dev.jsonl")
    specs = {
        "goal_empty": ("goal_metrics", ZERO_GOAL, None),
        "history_empty": ("recent_history", [], None),
        "budget_empty": ("remaining_step_budget", 0, None),
        "goal_shuffle": ("goal_metrics", None, 2026084302),
        "history_shuffle": ("recent_history", None, 2026084303),
        "budget_shuffle": ("remaining_step_budget", None, 2026084304),
    }
    summary = {}
    for name, (field, replacement, seed) in specs.items():
        rows = deepcopy(source)
        original = [state(row)[field] for row in source]
        donors = permutation(len(rows), seed) if seed is not None else None
        changed = 0
        for index, row in enumerate(rows):
            value = state(row)
            new_value = deepcopy(original[donors[index]]) if donors is not None else deepcopy(replacement)
            changed += new_value != value[field]
            value[field] = new_value
            set_state(row, value)
        unique_before = len({json.dumps(value, sort_keys=True) for value in original})
        write_condition(name, rows, changed, unique_before)
        summary[name] = {"changed_rows": changed, "unique_source_field_values": unique_before}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
