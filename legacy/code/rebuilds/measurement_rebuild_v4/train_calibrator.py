#!/usr/bin/env python3
"""Train a condition-aware residual calibration above frozen v3 outputs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measurement_rebuild_v3.common import (
    iter_jsonl,
    measurement_tolerance,
    read_json,
    transform_vector,
)
from measurement_rebuild_v3.models import require_torch
from measurement_rebuild_v3.train import metric_block
from measurement_rebuild_v4.models import measurement_calibrator_v4
from specialist_rebuild_v2.common import raw_state_array

DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
DEFAULT_CACHE = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v4_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def load_cache(
    path: Path,
) -> dict[tuple[str, str], np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    keys = json.loads(str(loaded["keys_json"].item()))
    values = np.asarray(loaded["predictions"], dtype=np.float32)
    return {(str(key[0]), str(key[1])): values[index] for index, key in enumerate(keys)}


def feature_vector(
    prediction: np.ndarray,
    calibration: Mapping[str, Any],
    transform: Mapping[str, Any],
    condition_index: int,
    condition_count: int,
) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float32)
    high = max(float(calibration["linear_intensity_high"]), 1e-6)
    one_hot = np.zeros(condition_count, dtype=np.float32)
    one_hot[condition_index] = 1.0
    derived = np.asarray(
        [
            np.log1p(max(float(values[4]), 0.0)),
            float(values[0]) / 1024.0,
            float(values[1]) / 1024.0,
            float(values[2]) / 1024.0,
            float(values[3]) / 1024.0,
            float(values[2]) / max(abs(float(values[3])), 1e-6),
            float(values[4]) / high,
            float(transform["noise_std"]) / max(float(values[4]) / high, 1e-6),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            values,
            transform_vector(calibration, transform),
            one_hot,
            derived,
        ]
    ).astype(np.float32)


def arrays(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[tuple[str, str], np.ndarray],
    conditions: Sequence[str],
    parameters: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, predictions, targets, condition_indices = [], [], [], []
    for row in rows:
        calibration = row["image_calibration"]
        for condition_index, condition in enumerate(conditions):
            prediction = np.asarray(
                cache[(str(row["state_id"]), condition)],
                dtype=np.float32,
            )
            features.append(
                feature_vector(
                    prediction,
                    calibration,
                    parameters[condition],
                    condition_index,
                    len(conditions),
                )
            )
            predictions.append(prediction)
            targets.append(raw_state_array(row["target_state"]))
            condition_indices.append(condition_index)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(predictions, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(condition_indices, dtype=np.int64),
    )


def predict(
    torch: Any,
    model: Any,
    features: np.ndarray,
    baseline: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int = 2048,
) -> np.ndarray:
    normalized = (features - mean) / scale
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            correction = (
                model(
                    torch.as_tensor(
                        normalized[start : start + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                .float()
                .cpu()
                .numpy()
            )
            base = baseline[start : start + batch_size]
            tolerance = np.asarray(
                [measurement_tolerance(row) for row in base],
                dtype=np.float32,
            )
            parts.append(base + correction * tolerance)
    return np.concatenate(parts)


def evaluation(
    target: np.ndarray,
    predicted: np.ndarray,
    condition_indices: np.ndarray,
    conditions: Sequence[str],
) -> dict[str, Any]:
    return {
        "all_conditions": metric_block(target, predicted),
        "by_condition": {
            condition: metric_block(
                target[condition_indices == index],
                predicted[condition_indices == index],
            )
            for index, condition in enumerate(conditions)
        },
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(data_dir / "config.json")
    conditions = list(config["conditions"])
    parameters = config["condition_parameters"]
    train_rows = list(iter_jsonl(data_dir / "states/train.jsonl"))
    val_rows = list(iter_jsonl(data_dir / "states/val.jsonl"))
    train_cache = load_cache(cache_dir / "measurement_predictions_train.npz")
    val_cache = load_cache(cache_dir / "measurement_predictions_val.npz")
    xt, bt, yt, condition_t = arrays(train_rows, train_cache, conditions, parameters)
    xv, bv, yv, condition_v = arrays(val_rows, val_cache, conditions, parameters)
    mean = xt.mean(axis=0)
    scale = xt.std(axis=0)
    scale[scale < 1e-7] = 1.0
    xt = ((xt - mean) / scale).astype(np.float32)
    target_tolerance = np.asarray(
        [measurement_tolerance(row) for row in yt], dtype=np.float32
    )
    normalized_target = (yt - bt) / target_tolerance

    torch = require_torch()
    torch.manual_seed(20260726)
    np.random.seed(20260726)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260726)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = measurement_calibrator_v4(torch, xt.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    condition_weights = np.asarray(
        [1.0, 2.0, 1.0, 2.0, 1.5, 1.0, 1.0],
        dtype=np.float32,
    )
    rng = np.random.default_rng(20260726 + 501)
    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    trace = []
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = rng.permutation(len(xt))
        running = 0.0
        for start in range(0, len(order), int(args.batch_size)):
            index = order[start : start + int(args.batch_size)]
            correction = model(
                torch.as_tensor(xt[index], dtype=torch.float32, device=device)
            )
            target = torch.as_tensor(
                normalized_target[index],
                dtype=torch.float32,
                device=device,
            )
            absolute = (correction - target).abs()
            weights = torch.as_tensor(
                condition_weights[condition_t[index]],
                dtype=torch.float32,
                device=device,
            )
            smooth = torch.nn.functional.smooth_l1_loss(
                correction, target, reduction="none"
            ).mean(dim=1)
            worst = absolute.max(dim=1).values
            outside = torch.nn.functional.softplus(absolute - 1.0).mean(dim=1)
            loss = (weights * (smooth + 0.2 * worst + 0.1 * outside)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)
        predicted = predict(torch, model, xv, bv, mean, scale, device)
        metrics = evaluation(yv, predicted, condition_v, conditions)
        score = (
            metrics["all_conditions"]["strict_all_five_success"]
            + 0.25 * metrics["by_condition"]["noise"]["strict_all_five_success"]
            + 0.25 * metrics["by_condition"]["dim_noise"]["strict_all_five_success"]
            - 0.01 * metrics["all_conditions"]["mae_in_tolerance_units"]
        )
        row = {
            "epoch": epoch,
            "train_loss": running / len(xt),
            "score": score,
            "validation": metrics,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("measurement calibration produced no checkpoint")
    model.load_state_dict(best_state)
    artifact_path = output_dir / "measurement_calibrator_v4.pt"
    torch.save(
        {
            "version": "measurement_rebuild_v4_one_seed",
            "seed": 20260726,
            "model": "measurement_residual_calibrator_v4",
            "state_dict": best_state,
            "input_dim": int(xt.shape[1]),
            "feature_mean": mean,
            "feature_scale": scale,
            "conditions": conditions,
            "condition_parameters": parameters,
            "source_measurement_artifact": str(
                (
                    REPO_ROOT.parent
                    / "VLM_runs/measurement_rebuild_v3_one_seed/measurement_v3.pt"
                ).resolve()
            ),
            "input_contract": {
                "uses_v3_prediction": True,
                "uses_image_calibration": True,
                "uses_transform_metadata": True,
                "uses_setup": False,
                "uses_action": False,
                "uses_simulator": False,
            },
        },
        artifact_path,
    )
    summary = {
        "version": "measurement_rebuild_v4_one_seed",
        "artifact": str(artifact_path.resolve()),
        "device": str(device),
        "train_views": len(xt),
        "val_views": len(xv),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": int(args.epochs),
        "best_epoch": best_epoch,
        "validation": trace[best_epoch - 1]["validation"],
        "trace": trace,
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "measurement_calibrator_v4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
