#!/usr/bin/env python3
"""Evaluate a v12 learned model on the backward-compatible 81-action grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    Bounds,
    apply_action,
    legacy_action_array,
    metrics_vector,
    position_vector,
    tolerance_vector,
)
from continuous_control_v12.evaluation import forward_metrics, group_bootstrap_ci
from continuous_control_v12.schema import read_jsonl
from continuous_control_v12.simulator import build_optical_setup
from continuous_control_v12.world_model import load_forward_ensemble
from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid

DEFAULT_CONFIG = Path(__file__).with_name("config_v12.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development", "test"), default="development")
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    model = load_forward_ensemble(args.checkpoint, device_name="cpu")
    bounds = Bounds.from_config(config)
    first_rows = {}
    for row in read_jsonl(data_dir / "transitions" / f"{args.split}.jsonl"):
        first_rows.setdefault(str(row["group_id"]), row)
        if len(first_rows) >= int(args.max_groups):
            break
    actions = legacy_action_array()
    predictions, targets, group_ids, regimes, sampling, no_op, uncertainty = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    group_regimes = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )["group_regimes"]
    for group_id, row in first_rows.items():
        setup = build_optical_setup(
            row["setup_context"],
            row["positions_mm"],
            row["simulator_fixed"],
            str((REPO_ROOT / config["simulator"]["base_config"]).resolve()),
        )
        truth_surface = simulate_fixed_action_grid(setup)
        result = model.predict(
            row["setup_context"],
            row["positions_mm"],
            row["metrics"],
            actions,
        )
        current = metrics_vector(row["metrics"])
        tolerance = tolerance_vector(current)
        positions = position_vector(row["positions_mm"])
        for index, action in enumerate(actions):
            try:
                apply_action(positions, action, bounds)
            except ValueError:
                continue
            truth = np.asarray(
                [
                    float(truth_surface[index]["state"][field])
                    for field in OUTPUT_FIELDS
                ],
                dtype=np.float64,
            )
            predictions.append(result["next_metric_residual"][index])
            targets.append((truth - current) / tolerance)
            uncertainty.append(result["uncertainty"][index])
            group_ids.append(group_id)
            regimes.append(group_regimes[group_id])
            sampling.append("legacy_grid")
            no_op.append(index == 40)
    prediction_array = np.asarray(predictions)
    target_array = np.asarray(targets)
    metric = forward_metrics(
        prediction_array,
        target_array,
        group_ids=group_ids,
        regimes=regimes,
        sampling=sampling,
        uncertainty=np.asarray(uncertainty),
        no_op=np.asarray(no_op),
    )
    strict = np.all(np.abs(prediction_array - target_array) <= 1.0, axis=1)
    metric["legacy_81_action_score_group_bootstrap_95"] = group_bootstrap_ci(
        strict.astype(np.float64),
        group_ids,
        seed=int(config["seed"]),
        samples=500,
    )
    result = {
        "version": "continuous_control_v12_legacy_grid_evaluation",
        "split": args.split,
        "groups": len(first_rows),
        "metrics": metric,
        "legacy_action_order_preserved": True,
        "scientific_claim": "small_evaluation_only"
        if len(first_rows) < 30
        else "group_level_evaluation",
    }
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise RuntimeError(f"refusing to overwrite evaluation: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

