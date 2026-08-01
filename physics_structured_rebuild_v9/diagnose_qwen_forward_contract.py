#!/usr/bin/env python3
"""Compare stored direct-forward labels with the current scoring simulator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import tolerance_from_current
from control_rebuild_v4.evaluate_orchestrated_system import (
    simulator_forward_truth,
)
from specialist_rebuild_v2.common import STATE_FIELDS, read_jsonl

DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_ADAPTATION = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "qwen_forward_contract_diagnostic.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--adaptation", type=Path, default=DEFAULT_ADAPTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-samples", type=int, default=200)
    return parser.parse_args()


def compare(
    stored: np.ndarray,
    fresh: np.ndarray,
    tolerance: np.ndarray,
) -> dict[str, Any]:
    absolute = np.abs(stored - fresh)
    normalized = absolute / tolerance
    return {
        "count": int(len(stored)),
        "maximum_absolute_change_difference": float(absolute.max()),
        "mean_absolute_change_difference": float(absolute.mean()),
        "maximum_tolerance_normalized_difference": float(normalized.max()),
        "mean_tolerance_normalized_difference": float(normalized.mean()),
        "requests_over_0_01_tolerance": int(
            np.any(normalized > 0.01, axis=1).sum()
        ),
        "requests_over_0_10_tolerance": int(
            np.any(normalized > 0.10, axis=1).sum()
        ),
        "requests_over_1_00_tolerance": int(
            np.any(normalized > 1.00, axis=1).sum()
        ),
        "per_field_mean_tolerance_normalized_difference": (
            normalized.mean(axis=0).tolist()
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    qwen = args.qwen_data.resolve()
    adaptation = read_jsonl(args.adaptation.resolve())
    private_train = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen / "private/source_cases/train.jsonl")
    }
    sample_count = min(int(args.train_samples), len(adaptation))
    sample_indices = np.linspace(
        0,
        len(adaptation) - 1,
        sample_count,
        dtype=np.int64,
    )
    train_stored = []
    train_fresh = []
    train_tolerance = []
    for completed, index in enumerate(sample_indices, start=1):
        row = adaptation[int(index)]
        private = private_train[str(row["group_id"])]
        truth = simulator_forward_truth(private, row["action"])["change"]
        train_stored.append(
            [float(row["truth_change"][field]) for field in STATE_FIELDS]
        )
        train_fresh.append(
            [float(truth[field]) for field in STATE_FIELDS]
        )
        train_tolerance.append(
            tolerance_from_current(row["current_beam_state"])
        )
        if completed % 50 == 0 or completed == sample_count:
            print(
                json.dumps(
                    {"train_recomputed": completed, "count": sample_count},
                    sort_keys=True,
                ),
                flush=True,
            )

    canonical_val = [
        row
        for row in read_jsonl(qwen / "canonical/val.jsonl")
        if row["target_decision"].get("route_name")
        == "predict_forward_from_state_v1"
    ]
    private_val = {
        str(row["group_id"]): row
        for row in read_jsonl(qwen / "private/source_cases/val.jsonl")
    }
    val_fresh = []
    val_tolerance = []
    for completed, row in enumerate(canonical_val, start=1):
        arguments = row["target_decision"]["arguments"]
        private = private_val[str(row["group_id"])]
        truth = simulator_forward_truth(private, arguments["action"])[
            "change"
        ]
        val_fresh.append(
            [float(truth[field]) for field in STATE_FIELDS]
        )
        val_tolerance.append(
            tolerance_from_current(arguments["current_beam_state"])
        )
        if completed % 50 == 0:
            print(
                json.dumps(
                    {"validation_recomputed": completed, "count": len(canonical_val)},
                    sort_keys=True,
                ),
                flush=True,
            )
    system_cache = np.load(
        REPO_ROOT.parent
        / "VLM_runs/physics_structured_rebuild_v9_one_seed"
        / "direct_forward_expert_selector_system_validation.npz",
        allow_pickle=False,
    )
    report = {
        "version": "qwen_forward_contract_diagnostic_v9",
        "train_stored_vs_current_simulator": compare(
            np.asarray(train_stored, dtype=np.float32),
            np.asarray(train_fresh, dtype=np.float32),
            np.asarray(train_tolerance, dtype=np.float32),
        ),
        "validation_cache_vs_current_simulator": compare(
            np.asarray(system_cache["truth_change"], dtype=np.float32),
            np.asarray(val_fresh, dtype=np.float32),
            np.asarray(val_tolerance, dtype=np.float32),
        ),
        "source_contract": {
            "diagnostic_only": True,
            "new_training_examples_generated": 0,
            "held_out_test_files_opened": [],
        },
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
