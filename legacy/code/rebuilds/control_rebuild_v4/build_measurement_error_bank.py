#!/usr/bin/env python3
"""Build a train-only empirical error bank for inverse-control augmentation."""

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

from measurement_rebuild_v3.common import (
    iter_jsonl,
    measurement_tolerance,
    read_json,
)
from measurement_rebuild_v3.models import require_torch
from measurement_rebuild_v4.models import measurement_calibrator_v4
from measurement_rebuild_v4.train_calibrator import (
    arrays,
    load_cache,
    predict,
)

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
DEFAULT_CACHE = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_CALIBRATOR = (
    REPO_ROOT.parent
    / "VLM_runs/measurement_rebuild_v4_one_seed/measurement_calibrator_v4.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v4_one_seed/measurement_error_bank_v4.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--calibrator-artifact", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def condition_statistics(
    errors: np.ndarray,
    condition_indices: np.ndarray,
    conditions: list[str],
) -> dict[str, Any]:
    output = {}
    for index, condition in enumerate(conditions):
        values = errors[condition_indices == index]
        output[condition] = {
            "count": int(len(values)),
            "mean_normalized_error": values.mean(axis=0).tolist(),
            "mean_absolute_normalized_error": np.abs(values).mean(axis=0).tolist(),
            "p95_absolute_normalized_error": np.quantile(
                np.abs(values), 0.95, axis=0
            ).tolist(),
        }
    return output


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    calibrator_path = args.calibrator_artifact.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = read_json(data_dir / "config.json")
    conditions = list(config["conditions"])
    parameters = config["condition_parameters"]
    # Deliberately load only the training split.  Validation and all three
    # held-out splits remain unavailable to this augmentation artifact.
    rows = list(iter_jsonl(data_dir / "states/train.jsonl"))
    cache = load_cache(cache_dir / "measurement_predictions_train.npz")
    features, baseline, target, condition_indices = arrays(
        rows, cache, conditions, parameters
    )

    torch = require_torch()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    artifact = torch.load(calibrator_path, map_location="cpu", weights_only=False)
    model = measurement_calibrator_v4(torch, int(artifact["input_dim"])).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    calibrated = predict(
        torch,
        model,
        features,
        baseline,
        np.asarray(artifact["feature_mean"], dtype=np.float32),
        np.asarray(artifact["feature_scale"], dtype=np.float32),
        device,
    )
    tolerance = np.asarray(
        [measurement_tolerance(value) for value in target],
        dtype=np.float32,
    )
    normalized_error = ((calibrated - target) / tolerance).astype(np.float32)
    lower = np.quantile(normalized_error, 0.005, axis=0)
    upper = np.quantile(normalized_error, 0.995, axis=0)
    clipped = np.clip(normalized_error, lower, upper).astype(np.float32)
    if not np.isfinite(clipped).all():
        raise RuntimeError("measurement error bank contains non-finite values")

    np.savez_compressed(
        output_path,
        normalized_errors=clipped,
        condition_indices=condition_indices.astype(np.int8),
        conditions_json=json.dumps(conditions),
        lower_clip=lower.astype(np.float32),
        upper_clip=upper.astype(np.float32),
    )
    summary = {
        "version": "measurement_error_bank_v4_train_only",
        "artifact": str(output_path),
        "source_calibrator": str(calibrator_path),
        "source_split": "train",
        "sample_count": int(len(clipped)),
        "conditions": conditions,
        "field_order": [
            "centroid_x_px",
            "centroid_y_px",
            "sigma_x_px",
            "sigma_y_px",
            "peak_intensity",
        ],
        "error_units": "multiples of measurement tolerance",
        "clip_quantiles": [0.005, 0.995],
        "condition_statistics": condition_statistics(
            clipped, condition_indices, conditions
        ),
        "held_out_test_used": False,
        "validation_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
