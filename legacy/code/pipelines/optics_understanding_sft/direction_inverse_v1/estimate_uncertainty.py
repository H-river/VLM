#!/usr/bin/env python3
"""Compute deterministic bootstrap/Wilson uncertainty for final component metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score

from optics_understanding_sft.core import read_jsonl
from optics_understanding_sft.direction_inverse_v1.train_direction_small import CLASSES, FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count == 0: return [0.0, 0.0]
    p = successes / count; denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return [center - radius, center + radius]


def direction_macro(rows: Sequence[dict[str, Any]]) -> float:
    values = []
    for field in FIELDS:
        target = [CLASSES.index(row["target"][field]) for row in rows]
        predicted = [CLASSES.index(row["prediction"][field]) for row in rows]
        values.append(f1_score(target, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
    return float(np.mean(values))


def bootstrap_direction(rows: list[dict[str, Any]], samples: int, rng: np.random.Generator) -> dict[str, Any]:
    estimates = []
    for _ in range(samples):
        index = rng.integers(0, len(rows), size=len(rows))
        estimates.append(direction_macro([rows[int(i)] for i in index]))
    return {"point": direction_macro(rows), "bootstrap_95_percent": [float(v) for v in np.quantile(estimates, [0.025, 0.975])],
            "records": len(rows), "bootstrap_samples": samples}


def main() -> None:
    args = parse_args(); results = args.package_dir / "results"; rng = np.random.default_rng(args.seed)
    direction_rows = read_jsonl(results / "direction_small_v1/details.jsonl")
    output: dict[str, Any] = {"seed": args.seed, "direction_small_macro_f1": {}}
    for split in ("eval_iid", "eval_ood"):
        rows = [row for row in direction_rows if row["split"] == split]
        output["direction_small_macro_f1"][split] = bootstrap_direction(rows, args.bootstrap_samples, rng)
    inverse_rows = read_jsonl(results / "inverse_ensemble_v1/details.jsonl")
    output["numeric_inverse_target_success"] = {}
    for split in ("eval_iid", "eval_ood"):
        rows = [row for row in inverse_rows if row["split"] == split
                and row["target_status"] != "infeasible_within_limits"]
        success = sum(bool(row["selected_action_reaches_target"]) for row in rows)
        output["numeric_inverse_target_success"][split] = {
            "successes": success, "cases": len(rows), "point": success / len(rows),
            "wilson_95_percent": wilson(success, len(rows))}
    visual_rows = read_jsonl(results / "visual_pipeline_sensor_v1/details.jsonl")
    output["visual_measurement_all_five"] = {}
    for split in ("eval_iid", "eval_ood"):
        passed = []
        for row in visual_rows:
            if row["split"] != split: continue
            for side in ("A", "B"):
                measured, true = row[f"measured_{side}"], row[f"true_{side}"]
                passed.append(abs(measured["centroid_x_px"] - true["centroid_x_px"]) <= 1.0
                              and abs(measured["centroid_y_px"] - true["centroid_y_px"]) <= 1.0
                              and abs(measured["sigma_x_px"] - true["sigma_x_px"]) <= 2.0
                              and abs(measured["sigma_y_px"] - true["sigma_y_px"]) <= 2.0
                              and abs(measured["peak_intensity"] - true["peak_intensity"])
                              <= 0.05 * max(abs(true["peak_intensity"]), 1e-12))
        success = sum(passed)
        output["visual_measurement_all_five"][split] = {
            "successes": success, "states": len(passed), "point": success / len(passed),
            "wilson_95_percent": wilson(success, len(passed))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
