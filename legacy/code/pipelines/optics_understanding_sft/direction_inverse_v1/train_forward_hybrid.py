#!/usr/bin/env python3
"""Blend the neural forward ensemble with an engineered-feature gradient model."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from optics_understanding_sft.core import read_jsonl, write_jsonl
from optics_understanding_sft.direction_inverse_v1.train_forward_small import (
    engineered_features, evaluate_split, input_arrays, predict_bundle, report_markdown,
    strict_metrics, target_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--neural-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {split: read_jsonl(args.source_dir / f"{split}.jsonl")
              for split in ("train", "val", "eval_iid", "eval_ood")}
    x = {split: input_arrays(rows) for split, rows in splits.items()}
    scaled = {split: target_arrays(rows)[1] for split, rows in splits.items()}
    with args.neural_bundle.open("rb") as stream:
        neural = pickle.load(stream)
    hist = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=500, max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1.0,
        early_stopping=True, random_state=42,
    ))
    hist.fit(engineered_features(x["train"]), scaled["train"])
    neural_val, logits_val = predict_bundle(neural, x["val"])
    tree_val = hist.predict(engineered_features(x["val"]))
    choices = []
    for alpha in np.linspace(0.0, 1.0, 21):
        metrics = strict_metrics(splits["val"], alpha * neural_val + (1.0 - alpha) * tree_val, logits_val)
        choices.append({"neural_weight": float(alpha), "strict_all_five_success": metrics["strict_all_five_success"],
                        "mae_in_tolerance_units": metrics["mae_in_tolerance_units"]})
    chosen = max(choices, key=lambda row: (row["strict_all_five_success"],
                                           -row["mae_in_tolerance_units"], -row["neural_weight"]))
    bundle: dict[str, Any] = {"version": "direction_inverse_v1_forward_hybrid",
                              "model_kind": "neural_hist_blend", "neural_bundle": neural,
                              "hist_model": hist, "neural_weight": chosen["neural_weight"],
                              "simulator_at_inference": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "forward_hybrid.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    summary: dict[str, Any] = {"version": bundle["version"], "simulator_at_inference": False,
                               "validation_blend_search": choices, "selected_blend": chosen,
                               "strict_success_definition": "all five errors within 1px centroid, 2px width, and 5% initial peak"}
    details = []
    for split in ("val", "eval_iid", "eval_ood"):
        summary[split], rows = evaluate_split(split, splits[split], bundle); details.extend(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "details.jsonl", details)
    (args.output_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
