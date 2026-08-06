#!/usr/bin/env python3
"""Build an all-field-only direction curriculum with balanced label marginals."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from optics_understanding_sft.core import read_jsonl, write_jsonl
from optics_understanding_sft.direction_inverse_v1.build_direction import (
    FIELD_MAP, all_record, qwen_export,
)

CLASSES = ("decrease", "no_change", "increase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=173)
    return parser.parse_args()


def ipf_weights(rows: list[dict], iterations: int = 500) -> np.ndarray:
    labels = np.asarray([[CLASSES.index(row["target"]["directions"][source])
                          for source in FIELD_MAP.values()] for row in rows])
    weights = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
    for _ in range(iterations):
        previous = weights.copy()
        for field_index in range(labels.shape[1]):
            mass = np.asarray([weights[labels[:, field_index] == value].sum() for value in range(3)])
            factors = (1.0 / 3.0) / np.maximum(mass, 1e-15)
            weights *= factors[labels[:, field_index]]
            weights /= weights.sum()
        if np.max(np.abs(weights - previous)) < 1e-12:
            break
    return weights / weights.sum()


def deterministic_counts(weights: np.ndarray, total: int) -> np.ndarray:
    expected = weights * total
    counts = np.floor(expected).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def class_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    result = {field: Counter() for field in FIELD_MAP}
    for row in rows:
        for field, label in row["target"]["answer"]["directions"].items():
            result[field][label] += 1
    return {field: dict(counter) for field, counter in result.items()}


def main() -> None:
    args = parse_args(); source = read_jsonl(args.source_dir / "train.jsonl")
    weights = ipf_weights(source); repeats = deterministic_counts(weights, args.records)
    records = []
    for source_row, count in zip(source, repeats, strict=True):
        for repeat in range(int(count)):
            row = all_record(source_row, "train")
            row["example_id"] = f"{row['example_id']}_balanced_{repeat:03d}"
            records.append(row)
    rng = np.random.default_rng(args.seed); rng.shuffle(records)
    val_source = read_jsonl(args.source_dir / "val.jsonl")
    val = [all_record(row, "val") for row in val_source]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", map(qwen_export, records))
    write_jsonl(args.output_dir / "val.jsonl", map(qwen_export, val))
    counts = class_counts(records)
    deviation = max(abs(counts[field].get(label, 0) / len(records) - 1.0 / 3.0)
                    for field in FIELD_MAP for label in CLASSES)
    # Exact simultaneous one-third marginals are not feasible because the five
    # physical labels are correlated. A 10-point cap still removes the severe
    # natural no-change collapse while retaining over one thousand sources.
    audit = {"version": "direction_inverse_v1_balanced_allfield", "passed": deviation <= 0.10,
             "train_records": len(records), "validation_records": len(val),
             "unique_source_transitions": int(np.sum(repeats > 0)),
             "maximum_source_repeat": int(repeats.max()),
             "effective_sample_size": float(1.0 / np.sum(np.square(weights))),
             "class_counts": counts, "target_marginal_fraction": 1.0 / 3.0,
             "maximum_allowed_marginal_fraction_deviation": 0.10,
             "maximum_marginal_fraction_deviation": deviation}
    (args.output_dir / "audit_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]: raise RuntimeError("all-field balancing audit failed")


if __name__ == "__main__":
    main()
