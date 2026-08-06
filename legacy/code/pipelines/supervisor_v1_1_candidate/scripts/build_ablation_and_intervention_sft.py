#!/usr/bin/env python3
"""Build preregistered train/dev modality ablations and dev interventions."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_vl_supervisor_v1.contracts import parse_target_strict
from qwen_vl_supervisor_v1.export_sft import validate_export_row

BASE = ROOT / "supervisor_v1_1_candidate"
STATE_PREFIX = "Current supervisor state (all coordinates and units are explicit):\n"
ZERO_METRICS = {
    "coordinate_frame": "diagnostic_image_128px",
    "centroid_x": 0.0,
    "centroid_y": 0.0,
    "width_x": 0.0,
    "width_y": 0.0,
    "peak_intensity": 0.0,
}
ZERO_GOAL = {**ZERO_METRICS, "coordinate_frame": "lab_sensor_1024px_and_raw_peak"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def user_state(row: dict[str, Any]) -> dict[str, Any]:
    text = row["prompt"][1]["content"][1]["text"]
    if not text.startswith(STATE_PREFIX):
        raise ValueError(f"unexpected state prefix for {row['example_id']}")
    return json.loads(text[len(STATE_PREFIX):])


def set_user_state(row: dict[str, Any], state: dict[str, Any]) -> None:
    row["prompt"][1]["content"][1]["text"] = STATE_PREFIX + json.dumps(state, sort_keys=True, separators=(",", ":"))


def target_class(row: dict[str, Any]) -> str:
    return parse_target_strict(row["completion"][0]["content"][0]["text"])["diagnosis"]


def make_blank() -> tuple[str, str]:
    path = BASE / "assets/blank_128.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode="L").save(path)
    rel = path.relative_to(ROOT).as_posix()
    return rel, sha256(path)


def report_for(output_dir: Path, train_path: Path, dev_path: Path, mode: str) -> None:
    value = {
        "version": "supervisor_v1_1_candidate_ablation_export_v1",
        "mode": mode,
        "records": {
            "train": {"path": train_path.name, "sha256": sha256(train_path), "count": len(read_jsonl(train_path))},
            "dev": {"path": dev_path.name, "sha256": sha256(dev_path), "count": len(read_jsonl(dev_path))},
        },
        "source_full_export_sha256": sha256(BASE / "sft/export_report.json"),
        "frozen_or_protected_data_used": False,
    }
    path = output_dir / "export_report.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_ablation(mode: str, blank_rel: str, blank_hash: str) -> None:
    output_dir = BASE / "sft/ablations" / mode
    outputs = {}
    for split in ("train", "dev"):
        source = read_jsonl(BASE / f"sft/sft_{split}.jsonl")
        transformed = []
        for raw in source:
            row = deepcopy(raw)
            state = user_state(row)
            if mode == "image_only":
                state["current_metrics"] = deepcopy(ZERO_METRICS)
                state["goal_metrics"] = deepcopy(ZERO_GOAL)
                state["recent_history"] = []
                state["remaining_step_budget"] = 8
                set_user_state(row, state)
            elif mode == "metrics_only":
                row["images"] = [blank_rel]
                row["metadata"]["image_sha256"] = blank_hash
            else:
                raise ValueError(mode)
            validate_export_row(row, repository_root=ROOT)
            transformed.append(row)
        outputs[split] = output_dir / f"sft_{split}.jsonl"
        write_jsonl(outputs[split], transformed)
    report_for(output_dir, outputs["train"], outputs["dev"], mode)


def permuted(rows: list[dict[str, Any]], seed: int) -> list[int]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if all(index == donor for index, donor in enumerate(indices)) and len(indices) > 1:
        indices = indices[1:] + indices[:1]
    return indices


def cross_class_donors(rows: list[dict[str, Any]], seed: int) -> list[int]:
    nominal = [i for i, row in enumerate(rows) if target_class(row) == "nominal"]
    abnormal = [i for i, row in enumerate(rows) if target_class(row) != "nominal"]
    if len(nominal) != len(abnormal):
        raise RuntimeError("cross-class bijection requires nominal count equal to all anomalous counts")
    rng = random.Random(seed)
    rng.shuffle(nominal)
    rng.shuffle(abnormal)
    donor = [-1] * len(rows)
    for left, right in zip(nominal, abnormal, strict=True):
        donor[left] = right
        donor[right] = left
    if any(index < 0 for index in donor):
        raise RuntimeError("incomplete cross-class donor mapping")
    return donor


def build_intervention(name: str, transform: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> None:
    source = read_jsonl(BASE / "sft/sft_dev.jsonl")
    output = transform(deepcopy(source))
    if len(output) != len(source) or {row["example_id"] for row in output} != {row["example_id"] for row in source}:
        raise RuntimeError(f"intervention {name} changed dev coverage")
    path = BASE / "sft/interventions" / name / "sft_dev.jsonl"
    write_jsonl(path, output)
    report = {
        "version": "supervisor_v1_1_candidate_dev_intervention_v1",
        "name": name,
        "records": len(output),
        "sha256": sha256(path),
        "source_dev_sha256": sha256(BASE / "sft/sft_dev.jsonl"),
        "protected_data_used": False,
    }
    (path.parent / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def image_donor_transform(donors: list[int]) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        original = deepcopy(rows)
        for index, donor in enumerate(donors):
            rows[index]["images"] = deepcopy(original[donor]["images"])
            rows[index]["metadata"]["image_sha256"] = original[donor]["metadata"]["image_sha256"]
        return rows
    return transform


def main() -> None:
    blank_rel, blank_hash = make_blank()
    build_ablation("image_only", blank_rel, blank_hash)
    build_ablation("metrics_only", blank_rel, blank_hash)
    base_dev = read_jsonl(BASE / "sft/sft_dev.jsonl")

    def blank_images(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            row["images"] = [blank_rel]
            row["metadata"]["image_sha256"] = blank_hash
        return rows

    build_intervention("blank_image", blank_images)
    build_intervention("cross_class_image_shuffle", image_donor_transform(cross_class_donors(base_dev, 2026084001)))
    for seed in (2026084101, 2026084102, 2026084103):
        build_intervention(f"random_image_shuffle_{seed}", image_donor_transform(permuted(base_dev, seed)))

    def metrics_blank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            state = user_state(row)
            state["current_metrics"] = deepcopy(ZERO_METRICS)
            set_user_state(row, state)
        return rows

    build_intervention("metrics_blank", metrics_blank)

    def shuffle_state_field(field: str, seed: int) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
        def transform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            original_states = [user_state(row) for row in rows]
            donors = permuted(rows, seed)
            for index, row in enumerate(rows):
                state = user_state(row)
                state[field] = deepcopy(original_states[donors[index]][field])
                set_user_state(row, state)
            return rows
        return transform

    build_intervention("metrics_shuffle", shuffle_state_field("current_metrics", 2026084201))

    def state_empty(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            state = user_state(row)
            state["goal_metrics"] = None
            state["recent_history"] = []
            state["remaining_step_budget"] = None
            set_user_state(row, state)
        return rows

    build_intervention("goal_history_budget_empty", state_empty)

    def state_shuffle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        originals = [user_state(row) for row in rows]
        donors = permuted(rows, 2026084301)
        for index, row in enumerate(rows):
            state = user_state(row)
            for field in ("goal_metrics", "recent_history", "remaining_step_budget"):
                state[field] = deepcopy(originals[donors[index]][field])
            set_user_state(row, state)
        return rows

    build_intervention("goal_history_budget_shuffle", state_shuffle)
    index = {
        "version": "supervisor_v1_1_candidate_sft_variants_v1",
        "ablation_modes": ["image_only", "metrics_only"],
        "interventions": sorted(path.parent.name for path in (BASE / "sft/interventions").glob("*/sft_dev.jsonl")),
        "blank_image": {"path": blank_rel, "sha256": blank_hash},
        "protected_data_used": False,
    }
    (BASE / "sft/variant_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
