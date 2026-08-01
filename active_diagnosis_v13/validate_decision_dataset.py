#!/usr/bin/env python3
"""Validate group separation and hidden-gain-independent prompt serialization."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from active_diagnosis_v13.contracts import assert_policy_visible


VERSION = "active_diagnosis_v13_decision_dataset_validation_v1"
EXPECTED_GAINS = {0.5, 0.75, 1.0, 1.25, 1.5}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _request(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row["prompt"][0]["content"][0]["text"])
    if "evaluator_only" in text or str(row["example_id"]) in text:
        raise ValueError("hidden evaluator metadata or example ID leaked into prompt")
    request = json.loads(text.split("\n\n", 1)[1])
    assert_policy_visible(request)
    return request


def _truth(row: dict[str, Any]) -> float:
    text = row["completion"][0]["content"][0]["text"]
    return float(json.loads(text)["estimated_gain"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = {split: _jsonl(root / f"{split}.jsonl") for split in ("train", "validation")}
    groups = {
        split: {str(row["group_id"]) for row in values}
        for split, values in rows.items()
    }
    errors = []
    if groups["train"] & groups["validation"]:
        errors.append("train and validation groups overlap")
    if any(int(group.rsplit("_", 1)[1]) >= 10 for values in groups.values() for group in values):
        errors.append("protected group present")
    if manifest.get("candidate_order_seed_basis") != "root_seed_and_case_id_only_hidden_gain_independent":
        errors.append("candidate-order seed attestation missing")
    if manifest.get("counts") != {split: len(values) for split, values in rows.items()}:
        errors.append("manifest counts differ from files")
    order_by_group: defaultdict[str, set[tuple[float, ...]]] = defaultdict(set)
    labels_by_group: defaultdict[str, list[float]] = defaultdict(list)
    prompt_violations = []
    for split, values in rows.items():
        for row in values:
            try:
                request = _request(row)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                prompt_violations.append(f"{row.get('example_id')}: {error}")
                continue
            candidates = tuple(map(float, request["candidate_gains_randomized"]))
            if set(candidates) != EXPECTED_GAINS or len(candidates) != 5:
                prompt_violations.append(f"{row['example_id']}: invalid candidate set")
            order_by_group[str(row["group_id"])].add(candidates)
            labels_by_group[str(row["group_id"])].append(_truth(row))
            if row.get("images") != []:
                prompt_violations.append(f"{row['example_id']}: expected zero images")
    inconsistent_orders = {
        group: [list(order) for order in orders]
        for group, orders in order_by_group.items()
        if len(orders) != 1
    }
    incomplete_labels = {
        group: sorted(labels)
        for group, labels in labels_by_group.items()
        if set(labels) != EXPECTED_GAINS or len(labels) != 5
    }
    if inconsistent_orders:
        errors.append("candidate order varies with gain inside a case group")
    if incomplete_labels:
        errors.append("one or more groups lack exactly one row per gain")
    if prompt_violations:
        errors.append("one or more prompt contracts failed")
    unique_orders = {
        next(iter(orders)) for orders in order_by_group.values() if len(orders) == 1
    }
    report: dict[str, Any] = {
        "version": VERSION,
        "protected_set_used": False,
        "counts": {split: len(values) for split, values in rows.items()},
        "groups": {split: len(values) for split, values in groups.items()},
        "group_overlap": sorted(groups["train"] & groups["validation"]),
        "label_counts": {
            split: dict(sorted(Counter(f"{_truth(row):g}" for row in values).items()))
            for split, values in rows.items()
        },
        "candidate_order_independent_of_gain_within_group": not inconsistent_orders,
        "unique_candidate_orders_across_groups": len(unique_orders),
        "inconsistent_candidate_orders": inconsistent_orders,
        "incomplete_group_labels": incomplete_labels,
        "prompt_violations": prompt_violations,
        "passes": not errors,
        "errors": errors,
    }
    output = root / "dataset_validation.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
