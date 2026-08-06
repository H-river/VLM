#!/usr/bin/env python3
"""Evaluate bounded-memory averaging of independently measured frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measurement_rebuild_v3.common import (
    analytic_measurement,
    iter_jsonl,
    measurement_tolerance,
    read_json,
    stable_seed,
    transform_vector,
)
from measurement_rebuild_v3.models import measurement_model_v3, require_torch
from measurement_rebuild_v3.train import (
    apply_transform,
    load_linear_image,
    metric_block,
)
from specialist_rebuild_v2.common import raw_state_array

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
DEFAULT_ARTIFACT = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v3_one_seed/measurement_v3.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v4_one_seed/burst_diagnostic.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--conditions", nargs="+", default=("noise", "dim_noise"))
    parser.add_argument("--frame-counts", nargs="+", type=int, default=(1, 4))
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def prepared_frame(
    row: Mapping[str, Any],
    base: np.ndarray,
    condition: str,
    transform: Mapping[str, Any],
    frame_index: int,
) -> tuple[np.ndarray, ...]:
    seed = (
        stable_seed(row["state_id"], condition, "view")
        if frame_index == 0
        else stable_seed(row["state_id"], condition, "burst", frame_index)
    )
    observed, linearized, valid = apply_transform(base, transform, seed)
    calibration = row["image_calibration"]
    baseline, analytic = analytic_measurement(
        linearized,
        valid,
        float(calibration["linear_intensity_high"]),
        tuple(calibration["source_sensor_resolution_px"]),
    )
    return (
        observed[None].astype(np.float32),
        linearized[None].astype(np.float32),
        valid[None].astype(np.float32),
        transform_vector(calibration, transform),
        analytic,
        baseline,
        measurement_tolerance(baseline),
    )


def stream_mean_predictions(
    torch: Any,
    model: Any,
    device: Any,
    data_dir: Path,
    rows: list[dict[str, Any]],
    condition: str,
    transform: Mapping[str, Any],
    frame_count: int,
    batch_size: int,
) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    sums = np.zeros((len(rows), 5), dtype=np.float64)
    counts = np.zeros(len(rows), dtype=np.int64)
    items: list[tuple[np.ndarray, ...]] = []
    owners: list[int] = []

    def flush() -> None:
        if not items:
            return
        columns = [np.stack([item[index] for item in items]) for index in range(7)]
        with torch.inference_mode():
            correction = model(
                *[
                    torch.as_tensor(
                        columns[index],
                        dtype=torch.float32,
                        device=device,
                    )
                    for index in range(5)
                ]
            )
            prediction = (
                (
                    torch.as_tensor(columns[5], dtype=torch.float32, device=device)
                    + correction
                    * torch.as_tensor(columns[6], dtype=torch.float32, device=device)
                )
                .float()
                .cpu()
                .numpy()
            )
        for local_index, owner in enumerate(owners):
            sums[owner] += prediction[local_index]
            counts[owner] += 1
        items.clear()
        owners.clear()

    for row_index, row in enumerate(rows):
        base = load_linear_image(data_dir / str(row["base_image"]))
        for frame_index in range(frame_count):
            items.append(prepared_frame(row, base, condition, transform, frame_index))
            owners.append(row_index)
            if len(items) == batch_size:
                flush()
    flush()
    if not np.all(counts == frame_count):
        raise AssertionError("not every row received every frame")
    return (sums / counts[:, None]).astype(np.float32)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = read_json(data_dir / "config.json")
    rows = list(iter_jsonl(data_dir / "states" / f"{args.split}.jsonl"))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    target = np.asarray(
        [raw_state_array(row["target_state"]) for row in rows],
        dtype=np.float32,
    )
    torch = require_torch()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    artifact = torch.load(
        args.artifact.resolve(), map_location="cpu", weights_only=False
    )
    model = measurement_model_v3(torch).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    started = time.perf_counter()
    results = {}
    for condition in args.conditions:
        condition_results = {}
        for frame_count in args.frame_counts:
            prediction = stream_mean_predictions(
                torch,
                model,
                device,
                data_dir,
                rows,
                condition,
                config["condition_parameters"][condition],
                int(frame_count),
                int(args.batch_size),
            )
            metrics = metric_block(target, prediction)
            condition_results[str(frame_count)] = metrics
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "frame_count": frame_count,
                        **metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        results[condition] = condition_results
    result = {
        "version": "measurement_rebuild_v4_burst_diagnostic",
        "source_artifact": str(args.artifact.resolve()),
        "split": args.split,
        "row_count": len(rows),
        "conditions": list(args.conditions),
        "frame_counts": list(args.frame_counts),
        "aggregation": "mean_of_independent_v3_state_predictions",
        "streaming_batch_size": int(args.batch_size),
        "results": results,
        "seconds": time.perf_counter() - started,
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
