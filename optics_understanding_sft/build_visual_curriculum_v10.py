#!/usr/bin/env python3
"""Build a balanced visual-evidence curriculum with preservation anchors."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import read_jsonl, stable_json_hash, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-train", type=Path, required=True)
    parser.add_argument("--recovery-anchors", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--visual-only", action="store_true")
    return parser.parse_args()


def completion_target(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["completion"][0]["content"][0]["text"])


def round_robin_groups(
    rows: list[dict[str, Any]], key_fn: Any, count: int, rng: random.Random
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count and keys:
        key = keys[cursor % len(keys)]
        values = groups[key]
        if values:
            selected.append(values.pop())
        if not values:
            keys.remove(key)
            cursor = 0
        else:
            cursor += 1
    if len(selected) != count:
        raise ValueError(f"Could only select {len(selected)} of {count} requested rows")
    return selected


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    visual = read_jsonl(args.visual_train)
    states = [row for row in visual if row["task_type"] == "visual_state_classification"]
    pairs = [row for row in visual if row["task_type"] == "visual_pair_direction_extraction"]

    state_rows = round_robin_groups(
        states,
        lambda row: tuple(sorted(completion_target(row)["answer"].items())),
        220,
        rng,
    )
    nontrivial_pairs = [
        row
        for row in pairs
        if any(
            value != "no_change"
            for value in completion_target(row)["answer"]["observed_direction_set"].values()
        )
    ]
    trivial_pairs = [row for row in pairs if row not in nontrivial_pairs]
    rng.shuffle(nontrivial_pairs)
    rng.shuffle(trivial_pairs)
    pair_rows = nontrivial_pairs + trivial_pairs[: 180 - len(nontrivial_pairs)]
    if len(pair_rows) != 180:
        raise ValueError("Unable to construct 180 pair rows")

    anchors = [row for row in read_jsonl(args.recovery_anchors) if not row.get("images")]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        by_source[str(row.get("curriculum_source"))].append(row)
    for values in by_source.values():
        rng.shuffle(values)
    anchor_quotas = {
        "direct_sufficiency_recovery": 60,
        "compact_v7_1_refresh": 40,
        "forward_tool_v8_refresh": 20,
        "counterfactual_tool_v9_refresh": 20,
        "other_task_anchor": 40,
    }
    anchor_rows: list[dict[str, Any]] = []
    if not args.visual_only:
        for source, quota in anchor_quotas.items():
            if len(by_source[source]) < quota:
                raise ValueError(f"Not enough {source} anchors")
            anchor_rows.extend(by_source[source][:quota])

    tagged: list[dict[str, Any]] = []
    for source_name, rows in (
        ("visual_state_v10", state_rows),
        ("visual_pair_v10", pair_rows),
        ("preservation_anchor", anchor_rows),
    ):
        for row in rows:
            copied = dict(row)
            copied["v10_curriculum_source"] = source_name
            tagged.append(copied)
    rng.shuffle(tagged)
    if len({row["example_id"] for row in tagged}) != len(tagged):
        raise ValueError("Curriculum example IDs must be unique")
    write_jsonl(args.output_jsonl, tagged)

    pair_labels: Counter[str] = Counter()
    for row in pair_rows:
        for field, value in completion_target(row)["answer"]["observed_direction_set"].items():
            pair_labels[f"{field}:{value}"] += 1
    manifest = {
        "seed": args.seed,
        "record_count": len(tagged),
        "optimizer_steps_at_batch4": len(tagged) // 4,
        "source_counts": dict(Counter(row["v10_curriculum_source"] for row in tagged)),
        "task_counts": dict(Counter(row["task_type"] for row in tagged)),
        "pair_label_counts": dict(sorted(pair_labels.items())),
        "visual_records": len(state_rows) + len(pair_rows),
        "text_anchor_records": len(anchor_rows),
        "ordered_ids_hash": stable_json_hash([row["example_id"] for row in tagged]),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
