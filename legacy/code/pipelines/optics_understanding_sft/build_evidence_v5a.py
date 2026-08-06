#!/usr/bin/env python3
"""Build fresh, scenario-disjoint v5A evidence aggregation data."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .build_evidence_v5_probe import control_pair_overrides, convert_item
from .core import (
    file_sha256,
    load_yaml,
    make_messages,
    make_qwen_record,
    read_jsonl,
    stable_json_hash,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def target_free(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result.pop("target", None)
    return result


def split_groups(
    group_ids: list[str], split_counts: Mapping[str, Any], seed: int
) -> tuple[dict[str, str], dict[str, list[str]]]:
    expected = sum(int(value) for value in split_counts.values())
    if len(group_ids) != expected:
        raise ValueError(f"expected {expected} source scenarios, found {len(group_ids)}")
    shuffled = sorted(group_ids)
    random.Random(seed).shuffle(shuffled)
    assignments: dict[str, str] = {}
    groups_by_split: dict[str, list[str]] = {}
    cursor = 0
    for split in ("train", "dev", "confirmation"):
        count = int(split_counts[split])
        selected = shuffled[cursor : cursor + count]
        cursor += count
        groups_by_split[split] = selected
        assignments.update({group_id: split for group_id in selected})
    return assignments, groups_by_split


def convert_case(
    source_case: Mapping[str, Any], split: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    suffix = str(dataset_cfg["record_suffix"])
    seed = int(dataset_cfg["seed"])
    source_group = str(source_case["group_id"])
    new_group = source_group + suffix
    controls = [
        item
        for item in source_case["records"]
        if item["record"]["task_type"] == "constrained_intervention"
    ]
    overrides = control_pair_overrides(controls, seed)
    items = []
    for source_item in source_case["records"]:
        source_id = str(source_item["record"]["example_id"])
        item = convert_item(
            source_item,
            new_group,
            seed,
            overrides.get(source_id),
            record_suffix=suffix,
            split=split,
            dataset_version=str(dataset_cfg["version"]),
            design_only=False,
            source_dataset="evidence_v5a_fresh_source",
            include_witness=True,
        )
        record = item["record"]
        source_match = str(record["provenance"]["match_group_id"])
        new_match = source_match + suffix
        record["provenance"].update(
            {
                "match_group_id": new_match,
                "future_evaluation_eligible": split in {"dev", "confirmation"},
                "distribution": f"fresh_{split}_iid",
            }
        )
        item["private_eval"]["match_group_id"] = new_match
        items.append(item)
    return {
        **{key: copy.deepcopy(value) for key, value in source_case.items() if key != "records"},
        "group_id": new_group,
        "split": split,
        "distribution": f"fresh_{split}_iid",
        "source_group_id": source_group,
        "records": items,
    }


def status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["target"]["status"]) for record in records).items()))


def build(config: Mapping[str, Any], source_dir: Path, output_dir: Path) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    source_cases = read_jsonl(source_dir / "master" / "cases.jsonl")
    source_by_group = {str(case["group_id"]): case for case in source_cases}
    assignments, groups_by_split = split_groups(
        list(source_by_group), dataset_cfg["split_scenarios"], int(dataset_cfg["seed"])
    )
    cases = [convert_case(source_by_group[group_id], assignments[group_id], config) for group_id in sorted(source_by_group)]
    records_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in groups_by_split}
    cases_by_group = {str(case["source_group_id"]): case for case in cases}
    for split, source_groups in groups_by_split.items():
        for source_group in source_groups:
            records_by_split[split].extend(
                [copy.deepcopy(item["record"]) for item in cases_by_group[source_group]["records"]]
            )

    trial_group_count = int(dataset_cfg["trial_train_scenarios"])
    trial_source_groups = groups_by_split["train"][:trial_group_count]
    trial_records = [
        copy.deepcopy(item["record"])
        for source_group in trial_source_groups
        for item in cases_by_group[source_group]["records"]
    ]

    write_jsonl(output_dir / "master" / "cases.jsonl", cases)
    write_jsonl(output_dir / "canonical" / "train.jsonl", records_by_split["train"])
    write_jsonl(output_dir / "canonical" / "dev.jsonl", records_by_split["dev"])
    write_jsonl(
        output_dir / "canonical" / "confirmation_prompts.jsonl",
        (target_free(record) for record in records_by_split["confirmation"]),
    )
    write_jsonl(
        output_dir / "private" / "confirmation_records.jsonl",
        records_by_split["confirmation"],
    )
    write_jsonl(
        output_dir / "private" / "confirmation_labels.jsonl",
        (
            {
                "example_id": record["example_id"],
                "group_id": record["group_id"],
                "target": record["target"],
                "provenance": record["provenance"],
            }
            for record in records_by_split["confirmation"]
        ),
    )
    for split in ("train", "dev"):
        write_jsonl(
            output_dir / "exports" / "messages" / f"{split}.jsonl",
            (make_messages(record, True) for record in records_by_split[split]),
        )
        write_jsonl(
            output_dir / "exports" / "qwen" / f"{split}.jsonl",
            (make_qwen_record(record, True) for record in records_by_split[split]),
        )
    write_jsonl(
        output_dir / "exports" / "messages" / "confirmation.jsonl",
        (make_messages(record, False) for record in records_by_split["confirmation"]),
    )
    write_jsonl(
        output_dir / "exports" / "qwen" / "confirmation.jsonl",
        (make_qwen_record(record, False) for record in records_by_split["confirmation"]),
    )
    trial_path = output_dir / "exports" / "qwen" / "train_trial50.jsonl"
    write_jsonl(trial_path, (make_qwen_record(record, True) for record in trial_records))

    manifest = {
        "dataset": str(dataset_cfg["name"]),
        "version": str(dataset_cfg["version"]),
        "seed": int(dataset_cfg["seed"]),
        "source_manifest_sha256": file_sha256(source_dir / "manifest.json"),
        "scenario_count": len(cases),
        "record_count": sum(len(records) for records in records_by_split.values()),
        "split_scenario_counts": {split: len(groups) for split, groups in groups_by_split.items()},
        "split_record_counts": {split: len(records) for split, records in records_by_split.items()},
        "split_status_counts": {split: status_counts(records) for split, records in records_by_split.items()},
        "trial_train_scenario_count": len(trial_source_groups),
        "trial_train_record_count": len(trial_records),
        "trial_train_group_ids_hash": stable_json_hash(trial_source_groups),
        "split_group_ids_hash": {split: stable_json_hash(groups) for split, groups in groups_by_split.items()},
        "canonical_hashes": {
            split: stable_json_hash(records) for split, records in records_by_split.items()
        },
        "promotion_gates": copy.deepcopy(config["promotion_gates"]),
        "protocol": copy.deepcopy(config["protocol"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trial_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "name": "evidence_v5a_seed42_trial50",
                "record_count": len(trial_records),
                "physical_group_count": len(trial_source_groups),
                "example_ids_hash": stable_json_hash([record["example_id"] for record in trial_records]),
                "status_counts": status_counts(trial_records),
                "dataset_manifest_sha256": file_sha256(output_dir / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    result = build(load_yaml(args.config), args.source_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
