#!/usr/bin/env python3
"""Build balanced single-field and natural all-field direction records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from optics_understanding_sft.core import load_yaml, read_jsonl, stable_json_hash, write_jsonl


FIELD_MAP = {
    "centroid_x": "centroid_x",
    "centroid_y": "centroid_y",
    "width_x": "sigma_x",
    "width_y": "sigma_y",
    "peak_intensity": "peak_intensity",
}
CLASSES = ("decrease", "no_change", "increase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_order(rows: Iterable[Mapping[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha256(f"{salt}:{row['example_id']}".encode()).hexdigest(),
    )


def prompt_text(inputs: Mapping[str, Any], *, field: str | None) -> str:
    if field is None:
        instruction = (
            "Predict the qualitative change in all five measured beam properties caused by the proposed action."
        )
        contract: dict[str, Any] = {
            "status": "answerable",
            "answer": {"directions": {name: "decrease | no_change | increase" for name in FIELD_MAP}},
        }
    else:
        instruction = f"Predict only the qualitative change in {field} caused by the proposed action."
        contract = {
            "status": "answerable",
            "answer": {"field": field, "direction": "decrease | no_change | increase"},
        }
    visible = {key: value for key, value in inputs.items() if key != "images"}
    return (
        f"{instruction} The after-state is not provided. Use the complete initial setup, current measured beam "
        "state, and action. Treat changes of at most 1 px in each centroid, 2 px in each width, and 5% of "
        "initial peak intensity as no_change.\n\n"
        f"Input data:\n{json.dumps(visible, indent=2, sort_keys=True)}\n\n"
        "Return only strict JSON matching this contract:\n"
        f"{json.dumps(contract, indent=2, sort_keys=True)}"
    )


def base_inputs(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "setup": source["inputs"]["setup"],
        "current_beam_state": source["inputs"]["current_beam_state"],
        "action": source["inputs"]["action"],
        "images": [],
    }


def single_record(source: Mapping[str, Any], split: str, field: str) -> dict[str, Any]:
    label_field = FIELD_MAP[field]
    inputs = base_inputs(source)
    target = {
        "status": "answerable",
        "answer": {"field": field, "direction": source["target"]["directions"][label_field]},
    }
    return {
        "example_id": f"dirv1_{source['example_id']}_{field}",
        "group_id": source["group_id"],
        "split": split,
        "task_type": "direction_single_field",
        "modality": "text",
        "source_transition_id": source["example_id"],
        "prompt_inputs": inputs,
        "prompt": prompt_text(inputs, field=field),
        "target": target,
    }


def all_record(source: Mapping[str, Any], split: str) -> dict[str, Any]:
    inputs = base_inputs(source)
    labels = source["target"]["directions"]
    target = {
        "status": "answerable",
        "answer": {"directions": {field: labels[label_field] for field, label_field in FIELD_MAP.items()}},
    }
    return {
        "example_id": f"dirv1_{source['example_id']}_all",
        "group_id": source["group_id"],
        "split": split,
        "task_type": "direction_all_fields",
        "modality": "text",
        "source_transition_id": source["example_id"],
        "prompt_inputs": inputs,
        "prompt": prompt_text(inputs, field=None),
        "target": target,
    }


def balanced_single_records(
    rows: list[dict[str, Any]], split: str, quota: int, seed: int
) -> list[dict[str, Any]]:
    selected = []
    for field, label_field in FIELD_MAP.items():
        for label in CLASSES:
            candidates = [
                row for row in rows if row["target"]["directions"][label_field] == label
            ]
            candidates = stable_order(candidates, f"{seed}:{split}:{field}:{label}")
            if len(candidates) < quota:
                raise ValueError(
                    f"not enough {split} {field}={label}: need {quota}, found {len(candidates)}"
                )
            selected.extend(single_record(row, split, field) for row in candidates[:quota])
    rng = random.Random(seed + (0 if split == "train" else 1))
    rng.shuffle(selected)
    return selected


def qwen_export(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "task_type": record["task_type"],
        "images": [],
        "prompt": [{"role": "user", "content": [{"type": "text", "text": record["prompt"]}]}],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(record["target"], sort_keys=False)}],
            }
        ],
    }


def class_counts(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {field: Counter() for field in FIELD_MAP}
    for record in records:
        answer = record["target"]["answer"]
        if record["task_type"] == "direction_single_field":
            counts[answer["field"]][answer["direction"]] += 1
        else:
            for field, label in answer["directions"].items():
                counts[field][label] += 1
    return {field: dict(counter) for field, counter in counts.items()}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    source_root = Path(cfg["sources"]["transition_root"])
    seed = int(cfg["seed"])
    source_paths = {
        "train": source_root / "train.jsonl",
        "val": source_root / "val.jsonl",
        "eval_iid": source_root / "eval_iid.jsonl",
        "eval_ood": source_root / "eval_ood.jsonl",
    }
    sources = {split: read_jsonl(path) for split, path in source_paths.items()}
    quotas = cfg["direction"]["balanced_quota_per_field_class"]
    single = {
        split: balanced_single_records(sources[split], split, int(quotas[split]), seed)
        for split in ("train", "val")
    }
    all_records = {}
    for split, rows in sources.items():
        limit = cfg["direction"].get(f"all_field_{split}_limit")
        ordered = stable_order(rows, f"{seed}:{split}:all")
        if limit is not None:
            ordered = ordered[: int(limit)]
        all_records[split] = [all_record(row, split) for row in ordered]

    for split, records in single.items():
        write_jsonl(args.output_dir / "direction/single_field" / f"{split}.jsonl", records)
    for split, records in all_records.items():
        write_jsonl(args.output_dir / "direction/all_fields" / f"{split}.jsonl", records)
    curriculum = single["train"] + all_records["train"]
    random.Random(seed + 99).shuffle(curriculum)
    write_jsonl(args.output_dir / "exports/qwen/direction_train.jsonl", map(qwen_export, curriculum))
    write_jsonl(args.output_dir / "exports/qwen/direction_val.jsonl", map(qwen_export, all_records["val"]))

    group_sets = {split: {row["group_id"] for row in rows} for split, rows in sources.items()}
    overlap = {
        f"{a}:{b}": len(group_sets[a] & group_sets[b])
        for i, a in enumerate(group_sets)
        for b in list(group_sets)[i + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"source scenario overlap: {overlap}")
    audit = {
        "version": cfg["version"],
        "seed": seed,
        "passed": True,
        "simulator_at_inference": False,
        "single_field_counts": {split: len(rows) for split, rows in single.items()},
        "single_field_class_counts": {split: class_counts(rows) for split, rows in single.items()},
        "all_field_counts": {split: len(rows) for split, rows in all_records.items()},
        "all_field_class_counts": {split: class_counts(rows) for split, rows in all_records.items()},
        "group_counts": {split: len(groups) for split, groups in group_sets.items()},
        "group_overlap": overlap,
        "prompt_forbidden_token_hits": {
            token: sum(token in row["prompt"] for rows in single.values() for row in rows)
            + sum(token in row["prompt"] for rows in all_records.values() for row in rows)
            for token in ("setup_state_handle", "after_state", "simulator_result", "target_change")
        },
        "record_hashes": {
            f"single_{split}": stable_json_hash(rows) for split, rows in single.items()
        }
        | {f"all_{split}": stable_json_hash(rows) for split, rows in all_records.items()},
    }
    if any(audit["prompt_forbidden_token_hits"].values()):
        raise RuntimeError(f"prompt leakage detected: {audit['prompt_forbidden_token_hits']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "direction/audit_report.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
