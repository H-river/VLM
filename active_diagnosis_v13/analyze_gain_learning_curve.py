#!/usr/bin/env python3
"""Group-disjoint data-size learning curve for the selected safe probe."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "active_diagnosis_v13_gain_learning_curve_v1"


def _rows(root: Path) -> list[dict[str, Any]]:
    by_id = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                by_id[row["record_id"]] = row
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080106)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    selected = json.loads(
        (root / "probes" / "selected_probe.json").read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in _rows(root)
        if str(row["design"]) == str(selected["selected_design"])
        and float(row["fraction"]) == float(selected["selected_fraction"])
    ]
    X = np.asarray(
        [row["policy_record"]["feature_vector"] for row in rows], dtype=np.float64
    )
    y = np.asarray(
        [f"{float(row['evaluator_only_true_gain']):g}" for row in rows], dtype=np.str_
    )
    groups = np.asarray([str(row["group_id"]) for row in rows], dtype=np.str_)
    if len(rows) != 150 or len(set(groups.tolist())) != 30:
        raise ValueError("learning curve requires 150 records from 30 development groups")
    outer = GroupKFold(n_splits=5)
    sizes = (6, 12, 18, 24)
    measurements: defaultdict[int, list[float]] = defaultdict(list)
    details = []
    for fold, (outer_train, validation) in enumerate(outer.split(X, y, groups)):
        train_groups = np.asarray(sorted(set(groups[outer_train].tolist())))
        for size in sizes:
            for repeat in range(args.repeats):
                rng = np.random.default_rng(args.seed + fold * 10000 + size * 100 + repeat)
                chosen_groups = set(rng.choice(train_groups, size=size, replace=False).tolist())
                train = np.asarray(
                    [index for index in outer_train if groups[index] in chosen_groups],
                    dtype=np.int64,
                )
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=3000,
                                random_state=args.seed + fold * 100 + repeat,
                            ),
                        ),
                    ]
                )
                model.fit(X[train], y[train])
                accuracy = float(np.mean(model.predict(X[validation]) == y[validation]))
                measurements[size].append(accuracy)
                details.append(
                    {
                        "fold": fold,
                        "repeat": repeat,
                        "training_groups": size,
                        "training_records": len(train),
                        "validation_groups": len(set(groups[validation].tolist())),
                        "validation_records": len(validation),
                        "accuracy": accuracy,
                    }
                )
    curve = []
    for size in sizes:
        values = np.asarray(measurements[size], dtype=np.float64)
        low, high = np.quantile(values, (0.025, 0.975))
        curve.append(
            {
                "training_groups": size,
                "training_records": size * 5,
                "evaluations": len(values),
                "mean_accuracy": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "empirical_95_low": float(low),
                "empirical_95_high": float(high),
            }
        )
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only_nested_group_disjoint",
        "protected_set_used": False,
        "selected_probe": selected,
        "outer_folds": 5,
        "repeats_per_fold_and_size": args.repeats,
        "curve": curve,
        "details": details,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "gain_learning_curve.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
