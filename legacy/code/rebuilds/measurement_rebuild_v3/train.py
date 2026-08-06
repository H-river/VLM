#!/usr/bin/env python3
"""Train and evaluate the spatial analytic-plus-neural measurement model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from measurement_rebuild_v3.common import (
    STATE_FIELDS,
    analytic_measurement,
    iter_jsonl,
    measurement_tolerance,
    read_json,
    stable_seed,
    transform_vector,
)
from measurement_rebuild_v3.models import measurement_model_v3, require_torch


DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v3_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=DEFAULT_DATA)
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def load_linear_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float32)
    if float(array.max(initial=0.0)) > 1.0:
        array /= 65535.0
    return np.clip(array, 0.0, 1.0)


def apply_transform(
    base: np.ndarray,
    transform: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = np.asarray(base, dtype=np.float32)
    blur = float(transform["blur_sigma_px"])
    if blur > 0.0:
        linear = gaussian_filter(linear, sigma=blur, mode="nearest")
    exposure = max(float(transform["exposure"]), 1e-6)
    gamma = max(float(transform["gamma"]), 1e-6)
    saturation = float(transform["saturation_level"])
    camera = np.clip(linear * exposure, 0.0, saturation)
    observed = np.power(np.clip(camera, 0.0, 1.0), gamma)
    noise = float(transform["noise_std"])
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        observed = observed + rng.normal(0.0, noise, observed.shape)
    observed = np.clip(observed, 0.0, 1.0).astype(np.float32)
    valid = np.ones_like(observed, dtype=np.float32)
    left = int(transform["crop_left_px"])
    right = int(transform["crop_right_px"])
    top = int(transform["crop_top_px"])
    bottom = int(transform["crop_bottom_px"])
    if left:
        valid[:, :left] = 0.0
    if right:
        valid[:, -right:] = 0.0
    if top:
        valid[:top, :] = 0.0
    if bottom:
        valid[-bottom:, :] = 0.0
    observed *= valid
    linearized = (
        np.power(np.clip(observed, 0.0, 1.0), 1.0 / gamma) / exposure
    )
    linearized *= valid
    return observed, linearized.astype(np.float32), valid


def prepare_view(
    torch: Any,
    data_dir: Path,
    row: Mapping[str, Any],
    condition: str,
    transform: Mapping[str, Any],
) -> tuple[Any, ...]:
    base = load_linear_image(data_dir / row["base_image"])
    seed = stable_seed(row["state_id"], condition, "view")
    observed, linearized, valid = apply_transform(base, transform, seed)
    calibration = row["image_calibration"]
    baseline, analytic = analytic_measurement(
        linearized,
        valid,
        float(calibration["linear_intensity_high"]),
        tuple(calibration["source_sensor_resolution_px"]),
    )
    target = np.asarray(
        [float(row["target_state"][field]) for field in STATE_FIELDS],
        dtype=np.float32,
    )
    prediction_scale = measurement_tolerance(baseline)
    target_tolerance = measurement_tolerance(target)
    return (
        torch.from_numpy(observed[None, :]),
        torch.from_numpy(linearized[None, :]),
        torch.from_numpy(valid[None, :]),
        torch.from_numpy(transform_vector(calibration, transform)),
        torch.from_numpy(analytic),
        torch.from_numpy(baseline),
        torch.from_numpy(prediction_scale),
        torch.from_numpy(target),
        torch.from_numpy(target_tolerance),
    )


class CyclingConditionDataset:
    def __init__(
        self,
        torch: Any,
        data_dir: Path,
        rows: Sequence[Mapping[str, Any]],
        conditions: Sequence[str],
        parameters: Mapping[str, Mapping[str, Any]],
        seed: int,
    ) -> None:
        self.torch = torch
        self.data_dir = data_dir
        self.rows = list(rows)
        self.conditions = list(conditions)
        self.parameters = parameters
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        row = self.rows[index]
        offset = stable_seed(self.seed, row["state_id"], "condition")
        condition = self.conditions[(offset + self.epoch) % len(self.conditions)]
        return (
            *prepare_view(
                self.torch,
                self.data_dir,
                row,
                condition,
                self.parameters[condition],
            ),
            self.torch.tensor(
                self.conditions.index(condition), dtype=self.torch.long
            ),
        )


class AllConditionDataset:
    def __init__(
        self,
        torch: Any,
        data_dir: Path,
        rows: Sequence[Mapping[str, Any]],
        conditions: Sequence[str],
        parameters: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.torch = torch
        self.data_dir = data_dir
        self.rows = list(rows)
        self.conditions = list(conditions)
        self.parameters = parameters

    def __len__(self) -> int:
        return len(self.rows) * len(self.conditions)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        row_index, condition_index = divmod(index, len(self.conditions))
        row = self.rows[row_index]
        condition = self.conditions[condition_index]
        return (
            *prepare_view(
                self.torch,
                self.data_dir,
                row,
                condition,
                self.parameters[condition],
            ),
            self.torch.tensor(condition_index, dtype=self.torch.long),
        )


def metric_block(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    errors = np.abs(predicted - target)
    tolerance = np.column_stack(
        [
            np.ones(len(target)),
            np.ones(len(target)),
            np.full(len(target), 2.0),
            np.full(len(target), 2.0),
            np.maximum(0.05 * np.abs(target[:, 4]), 1e-6),
        ]
    )
    normalized = errors / tolerance
    passed = normalized <= 1.0
    return {
        "count": int(len(target)),
        "strict_all_five_success": float(np.mean(np.all(passed, axis=1))),
        "mae_in_tolerance_units": float(np.mean(normalized)),
        "worst_field_error_in_tolerance_units": float(
            np.mean(np.max(normalized, axis=1))
        ),
        "per_field_pass": {
            field: float(np.mean(passed[:, index]))
            for index, field in enumerate(STATE_FIELDS)
        },
        "mae": {
            field: float(errors[:, index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    conditions: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    target_parts, predicted_parts, condition_parts = [], [], []
    with torch.inference_mode():
        for batch in loader:
            (
                observed,
                linearized,
                valid,
                calibration,
                analytic,
                baseline,
                prediction_scale,
                target,
                target_tolerance,
                condition,
            ) = batch
            correction = model(
                observed.to(device, non_blocking=True),
                linearized.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
                calibration.to(device, non_blocking=True),
                analytic.to(device, non_blocking=True),
            )
            predicted = (
                baseline.to(device)
                + correction * prediction_scale.to(device)
            )
            target_parts.append(target.numpy())
            predicted_parts.append(predicted.cpu().numpy())
            condition_parts.append(condition.numpy())
    target = np.concatenate(target_parts)
    predicted = np.concatenate(predicted_parts)
    condition_index = np.concatenate(condition_parts)
    by_condition = {}
    for index, condition in enumerate(conditions):
        mask = condition_index == index
        by_condition[condition] = metric_block(target[mask], predicted[mask])
    return {
        "all_conditions": metric_block(target, predicted),
        "by_condition": by_condition,
    }


def train_loss(torch: Any, normalized_error: Any) -> Any:
    absolute = normalized_error.abs()
    smooth = torch.nn.functional.smooth_l1_loss(
        normalized_error, torch.zeros_like(normalized_error)
    )
    worst = absolute.max(dim=1).values.mean()
    outside = torch.nn.functional.softplus(absolute - 1.0).mean()
    return smooth + 0.20 * worst + 0.10 * outside


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(data_dir / "config.json")
    manifest = read_json(data_dir / "manifest.json")
    if manifest["dataset_version"] != config["version"]:
        raise RuntimeError("manifest/config version mismatch")
    torch = require_torch()
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = list(config["conditions"])
    parameters = config["condition_parameters"]
    train_rows = list(iter_jsonl(data_dir / "states/train.jsonl"))
    val_rows = list(iter_jsonl(data_dir / "states/val.jsonl"))
    train_dataset = CyclingConditionDataset(
        torch, data_dir, train_rows, conditions, parameters, seed
    )
    val_dataset = AllConditionDataset(
        torch, data_dir, val_rows, conditions, parameters
    )
    workers = int(
        args.num_workers
        if args.num_workers is not None
        else config["training"]["num_workers"]
    )
    if workers < 0 or workers > 2:
        raise ValueError("num-workers must be between 0 and 2")
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(config["training"]["validation_batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    model = measurement_model_v3(torch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(args.epochs or config["training"]["epochs"])
    started = time.perf_counter()
    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    trace = []
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        model.train()
        running = 0.0
        for batch in loader:
            (
                observed,
                linearized,
                valid,
                calibration,
                analytic,
                baseline,
                prediction_scale,
                target,
                target_tolerance,
                _,
            ) = batch
            correction = model(
                observed.to(device, non_blocking=True),
                linearized.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
                calibration.to(device, non_blocking=True),
                analytic.to(device, non_blocking=True),
            )
            normalized_error = (
                baseline.to(device)
                + correction * prediction_scale.to(device)
                - target.to(device)
            ) / target_tolerance.to(device)
            loss = train_loss(torch, normalized_error)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(observed)
        metrics = evaluate(torch, model, val_loader, device, conditions)
        clean = metrics["by_condition"]["clean"]["strict_all_five_success"]
        all_conditions = metrics["all_conditions"]["strict_all_five_success"]
        score = (
            float(config["training"]["clean_score_weight"]) * clean
            + float(config["training"]["all_condition_score_weight"])
            * all_conditions
            - 0.005
            * metrics["all_conditions"]["mae_in_tolerance_units"]
        )
        row = {
            "epoch": epoch,
            "train_loss": running / len(train_dataset),
            "score": score,
            **metrics,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    evaluation = {}
    for split in ("test_iid", "test_visual_stress", "test_ood_physics"):
        rows = list(iter_jsonl(data_dir / "states" / f"{split}.jsonl"))
        dataset = AllConditionDataset(
            torch, data_dir, rows, conditions, parameters
        )
        split_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(config["training"]["validation_batch_size"]),
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        evaluation[split] = evaluate(
            torch, model, split_loader, device, conditions
        )
    artifact = {
        "version": config["version"],
        "seed": seed,
        "model": "spatial_analytic_measurement_v3",
        "state_dict": best_state,
        "state_fields": STATE_FIELDS,
        "conditions": conditions,
        "condition_parameters": parameters,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_contract": {
            "stored_transfer": "linear",
            "stored_bit_depth": 16,
            "calibration_dim": 11,
            "analytic_dim": 9,
            "correction_scale": "derived_from_analytic_baseline_only",
        },
    }
    artifact_path = output_dir / "measurement_v3.pt"
    torch.save(artifact, artifact_path)
    summary = {
        "version": config["version"],
        "device": str(device),
        "seed": seed,
        "artifact": str(artifact_path),
        "parameter_count": artifact["parameter_count"],
        "train_base_states": len(train_rows),
        "val_base_states": len(val_rows),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "validation": trace[best_epoch - 1],
        "evaluation": evaluation,
        "trace": trace,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "measurement_v3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in summary if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
