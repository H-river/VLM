#!/usr/bin/env python3
"""Check scenario, record, seed, image, and match-group overlap across datasets."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from .core import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def inventory(path: Path) -> dict[str, set[Any]]:
    masters = read_jsonl(path / "master/cases.jsonl")
    records = [item["record"] for master in masters for item in master["records"]]
    return {
        "group_ids": {master["group_id"] for master in masters},
        "scenario_seeds": {int(master["scenario_seed"]) for master in masters},
        "example_ids": {record["example_id"] for record in records},
        "images": {
            image
            for record in records
            for image in record["prompt_inputs"].get("images", [])
        },
        "match_group_ids": {
            record["provenance"]["match_group_id"]
            for record in records
            if record.get("provenance", {}).get("match_group_id")
        },
    }


def main() -> None:
    args = parse_args()
    datasets: dict[str, Path] = {}
    for raw in args.dataset:
        if "=" not in raw:
            raise ValueError(f"invalid dataset: {raw}")
        name, path = raw.split("=", 1)
        datasets[name] = Path(path)
    inventories = {name: inventory(path) for name, path in datasets.items()}
    comparisons = {}
    passed = True
    for left, right in itertools.combinations(sorted(datasets), 2):
        overlaps = {
            key: len(inventories[left][key] & inventories[right][key])
            for key in inventories[left]
        }
        comparisons[f"{left}__{right}"] = overlaps
        passed = passed and not any(overlaps.values())
    result = {
        "passed": passed,
        "datasets": {
            name: {key: len(values) for key, values in inventory.items()}
            for name, inventory in inventories.items()
        },
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
