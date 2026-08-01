#!/usr/bin/env python3
"""Train all version-2 specialists once with one frozen seed."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    CLASSES,
    DIRECTION_FIELDS,
    MATCHING_TOLERANCE,
    SETUP_FIELDS,
    STATE_FIELDS,
    STATUSES,
    action_array,
    direction_feature,
    fixed_action_grid,
    forward_feature,
    inverse_context,
    raw_state_array,
    read_json,
    read_jsonl,
    setup_array,
)
from specialist_rebuild_v2.models import (
    direction_model,
    forward_model,
    inverse_ranker,
    measurement_model,
    require_torch,
    visual_inverse_ranker,
)


CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
STATUS_INDEX = {name: index for index, name in enumerate(STATUSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--only",
        choices=("direction", "forward", "inverse", "measurement", "visual_inverse"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--epoch-limit",
        type=int,
        help="Cap every epoch count; intended for end-to-end smoke tests.",
    )
    return parser.parse_args()


def configure(seed: int, device_name: str) -> tuple[Any, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch, device


def direction_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    by_field = {
        field: float(
            f1_score(
                target[:, index],
                predicted[:, index],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )
        for index, field in enumerate(DIRECTION_FIELDS)
    }
    return {
        "equal_field_macro_f1": float(np.mean(list(by_field.values()))),
        "joint_exact": float(np.mean(np.all(target == predicted, axis=1))),
        "field_macro_f1": by_field,
    }


def class_weight_matrix(labels: np.ndarray, classes: int) -> np.ndarray:
    weights = np.ones((labels.shape[1], classes), dtype=np.float32)
    for column in range(labels.shape[1]):
        counts = Counter(labels[:, column].tolist())
        for index in range(classes):
            weights[column, index] = math.sqrt(
                len(labels) / max(classes * counts.get(index, 0), 1)
            )
    return weights


def iter_batches(
    length: int, batch_size: int, rng: np.random.Generator
) -> Any:
    order = rng.permutation(length)
    for start in range(0, length, batch_size):
        yield order[start : start + batch_size]


def load_transition_arrays(
    data_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_jsonl(data_dir / "grids" / f"{split}.jsonl")
    direction_x, forward_x, changes, directions = [], [], [], []
    for row in rows:
        for candidate in row["candidates"]:
            direction_x.append(
                direction_feature(
                    row["setup"], row["current_beam_state"], candidate["action"]
                )
            )
            forward_x.append(
                forward_feature(
                    row["setup"], row["current_beam_state"], candidate["action"]
                )
            )
            raw = np.asarray(
                [float(candidate["change"][key]) for key in STATE_FIELDS],
                dtype=np.float32,
            )
            tolerance = np.asarray(
                [
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    max(
                        0.05
                        * abs(
                            float(row["current_beam_state"]["peak_intensity"])
                        ),
                        1e-6,
                    ),
                ],
                dtype=np.float32,
            )
            changes.append(raw / tolerance)
            directions.append(
                [
                    CLASS_INDEX[candidate["directions"][field]]
                    for field in DIRECTION_FIELDS
                ]
            )
    return (
        np.asarray(direction_x, dtype=np.float32),
        np.asarray(forward_x, dtype=np.float32),
        np.asarray(changes, dtype=np.float32),
        np.asarray(directions, dtype=np.int64),
    )


def save_torch_artifact(
    torch: Any, path: Path, artifact: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(artifact), path)


def train_direction(
    data_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    x_train, _, _, y_train = load_transition_arrays(data_dir, "train")
    x_val, _, _, y_val = load_transition_arrays(data_dir, "val")
    mean, scale = x_train.mean(0), x_train.std(0)
    scale[scale < 1e-8] = 1.0
    x_train = (x_train - mean) / scale
    x_val = (x_val - mean) / scale
    model = direction_model(torch, x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = torch.as_tensor(
        class_weight_matrix(y_train, 3), dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(int(config["training"]["seed"]))
    best_state = None
    best_score = -1.0
    best_epoch = 0
    trace = []
    epochs = int(config["training"]["direction_epochs"])
    batch_size = int(config["training"]["batch_size"])
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for index in iter_batches(len(x_train), batch_size, rng):
            xb = torch.as_tensor(x_train[index], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[index], dtype=torch.long, device=device)
            logits = model(xb)
            loss = torch.stack(
                [
                    torch.nn.functional.cross_entropy(
                        logits[:, field], yb[:, field], weight=weights[field]
                    )
                    for field in range(5)
                ]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            model.eval()
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(x_val), 4096):
                    values = torch.as_tensor(
                        x_val[start : start + 4096],
                        dtype=torch.float32,
                        device=device,
                    )
                    predictions.append(
                        model(values).argmax(-1).cpu().numpy()
                    )
            metrics = direction_metrics(y_val, np.concatenate(predictions))
            score = metrics["equal_field_macro_f1"]
            trace.append(
                {
                    "epoch": epoch,
                    "train_loss": running / len(x_train),
                    **metrics,
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("direction training produced no checkpoint")
    model.load_state_dict(best_state)
    artifact = output_dir / "direction_v2.pt"
    save_torch_artifact(
        torch,
        artifact,
        {
            "version": config["version"],
            "seed": int(config["training"]["seed"]),
            "model": "direction_v2",
            "input_dim": x_train.shape[1],
            "state_dict": best_state,
            "mean": mean,
            "scale": scale,
            "fields": DIRECTION_FIELDS,
            "classes": CLASSES,
        },
    )
    result = {
        "artifact": str(artifact.resolve()),
        "train_records": len(x_train),
        "val_records": len(x_val),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "validation": trace[-1] if best_epoch == trace[-1]["epoch"] else next(
            row for row in trace if row["epoch"] == best_epoch
        ),
        "trace": trace,
        "seconds": time.perf_counter() - started,
    }
    return result


def forward_strict_metrics(
    target: np.ndarray, predicted: np.ndarray, directions: np.ndarray, logits: np.ndarray
) -> dict[str, Any]:
    errors = np.abs(predicted - target)
    return {
        "mae_in_tolerance_units": float(errors.mean()),
        "per_field_tolerance_pass": {
            field: float(np.mean(errors[:, index] <= 1.0))
            for index, field in enumerate(STATE_FIELDS)
        },
        "strict_all_five_success": float(np.mean(np.all(errors <= 1.0, axis=1))),
        "direction": direction_metrics(directions, logits.argmax(-1)),
    }


def train_forward(
    data_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge

    started = time.perf_counter()
    _, x_train, y_train, d_train = load_transition_arrays(data_dir, "train")
    _, x_val, y_val, d_val = load_transition_arrays(data_dir, "val")
    mean, scale = x_train.mean(0), x_train.std(0)
    scale[scale < 1e-8] = 1.0
    xt, xv = (x_train - mean) / scale, (x_val - mean) / scale
    baseline = Ridge(alpha=1.0).fit(xt, y_train)
    base_train = baseline.predict(xt).astype(np.float32)
    base_val = baseline.predict(xv).astype(np.float32)
    residual_target = y_train - base_train
    model = forward_model(torch, xt.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    direction_weights = torch.as_tensor(
        class_weight_matrix(d_train, 3), dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(int(config["training"]["seed"]) + 1)
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    trace = []
    epochs = int(config["training"]["forward_epochs"])
    batch_size = int(config["training"]["batch_size"])
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for index in iter_batches(len(xt), batch_size, rng):
            xb = torch.as_tensor(xt[index], dtype=torch.float32, device=device)
            rb = torch.as_tensor(
                residual_target[index], dtype=torch.float32, device=device
            )
            db = torch.as_tensor(d_train[index], dtype=torch.long, device=device)
            predicted, log_variance, direction_logits = model(xb)
            log_variance = torch.clamp(log_variance, -4.0, 4.0)
            residual = predicted - rb
            regression = (
                0.5 * torch.exp(-log_variance) * residual.square()
                + 0.5 * log_variance
            ).mean()
            direction_loss = torch.stack(
                [
                    torch.nn.functional.cross_entropy(
                        direction_logits[:, field],
                        db[:, field],
                        weight=direction_weights[field],
                    )
                    for field in range(5)
                ]
            ).mean()
            loss = regression + 0.25 * direction_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)
        if epoch == 1 or epoch % 2 == 0 or epoch == epochs:
            model.eval()
            predicted, logits = [], []
            with torch.inference_mode():
                for start in range(0, len(xv), 4096):
                    values = torch.as_tensor(
                        xv[start : start + 4096],
                        dtype=torch.float32,
                        device=device,
                    )
                    change, _, direction = model(values)
                    predicted.append(change.cpu().numpy())
                    logits.append(direction.cpu().numpy())
            final = base_val + np.concatenate(predicted)
            metrics = forward_strict_metrics(
                y_val, final, d_val, np.concatenate(logits)
            )
            score = metrics["mae_in_tolerance_units"]
            trace.append(
                {
                    "epoch": epoch,
                    "train_loss": running / len(xt),
                    **metrics,
                }
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("forward training produced no checkpoint")
    model.load_state_dict(best_state)
    artifact = output_dir / "forward_v2.pt"
    save_torch_artifact(
        torch,
        artifact,
        {
            "version": config["version"],
            "seed": int(config["training"]["seed"]),
            "model": "forward_v2",
            "input_dim": xt.shape[1],
            "state_dict": best_state,
            "mean": mean,
            "scale": scale,
            "baseline_coef": baseline.coef_,
            "baseline_intercept": baseline.intercept_,
            "fields": STATE_FIELDS,
            "direction_fields": DIRECTION_FIELDS,
            "classes": CLASSES,
        },
    )
    validation = next(row for row in trace if row["epoch"] == best_epoch)
    return {
        "artifact": str(artifact.resolve()),
        "train_records": len(xt),
        "val_records": len(xv),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "validation": validation,
        "trace": trace,
        "seconds": time.perf_counter() - started,
    }


def load_forward_predictor(
    output_dir: Path, torch: Any, device: Any
) -> tuple[Any, dict[str, Any]]:
    artifact = torch.load(
        output_dir / "forward_v2.pt", map_location="cpu", weights_only=False
    )
    model = forward_model(torch, int(artifact["input_dim"])).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact


def predict_grid_states(
    rows: Sequence[Mapping[str, Any]],
    forward: Any,
    artifact: Mapping[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, np.ndarray]:
    output = {}
    mean = np.asarray(artifact["mean"], dtype=np.float32)
    scale = np.asarray(artifact["scale"], dtype=np.float32)
    coef = np.asarray(artifact["baseline_coef"], dtype=np.float32)
    intercept = np.asarray(artifact["baseline_intercept"], dtype=np.float32)
    actions = fixed_action_grid()
    for row in rows:
        x = np.asarray(
            [
                forward_feature(row["setup"], row["current_beam_state"], action)
                for action in actions
            ],
            dtype=np.float32,
        )
        normalized = (x - mean) / scale
        baseline = normalized @ coef.T + intercept
        with torch.inference_mode():
            residual, _, _ = forward(
                torch.as_tensor(normalized, dtype=torch.float32, device=device)
            )
        scaled_change = baseline + residual.cpu().numpy()
        tolerance = np.asarray(
            [
                1.0,
                1.0,
                2.0,
                2.0,
                max(
                    0.05
                    * abs(float(row["current_beam_state"]["peak_intensity"])),
                    1e-6,
                ),
            ],
            dtype=np.float32,
        )
        output[row["group_id"]] = (
            raw_state_array(row["current_beam_state"])[None, :]
            + scaled_change * tolerance[None, :]
        )
    return output


def normalized_residuals(states: np.ndarray, desired: Mapping[str, Any]) -> np.ndarray:
    target = raw_state_array(desired)
    centroid = np.hypot(states[:, 0] - target[0], states[:, 1] - target[1]) / 0.5
    width_x = np.abs(states[:, 2] - target[2]) / 1.0
    width_y = np.abs(states[:, 3] - target[3]) / 1.0
    peak = (
        np.abs(states[:, 4] - target[4])
        / max(abs(float(target[4])), 1e-12)
        / 0.02
    )
    return np.sqrt(
        (centroid**2 + width_x**2 + width_y**2 + peak**2) / 4.0
    ).astype(np.float32)


def inverse_arrays(
    data_dir: Path,
    split: str,
    grids: Mapping[str, Mapping[str, Any]],
    predicted_states: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = read_jsonl(data_dir / "inverse" / f"{split}.jsonl")
    actions = np.asarray(
        [[action[key] for key in ACTION_FIELDS] for action in fixed_action_grid()],
        dtype=np.float32,
    )
    contexts, action_features, positives, statuses = [], [], [], []
    for pair in pairs:
        grid = grids[pair["group_id"]]
        contexts.append(
            inverse_context(
                grid["setup"],
                grid["current_beam_state"],
                pair["desired_beam_state"],
            )
        )
        residual = normalized_residuals(
            predicted_states[pair["group_id"]], pair["desired_beam_state"]
        )
        action_features.append(
            np.concatenate([actions, residual[:, None]], axis=1)
        )
        mask = np.zeros(81, dtype=np.bool_)
        mask[pair["matching_indices"]] = True
        positives.append(mask)
        statuses.append(STATUS_INDEX[pair["status"]])
    return (
        np.asarray(contexts, dtype=np.float32),
        np.asarray(action_features, dtype=np.float32),
        np.asarray(positives, dtype=np.bool_),
        np.asarray(statuses, dtype=np.int64),
    )


def inverse_metrics(
    scores: np.ndarray,
    status_logits: np.ndarray,
    positives: np.ndarray,
    statuses: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    selected = scores.argmax(1)
    feasible = statuses != STATUS_INDEX["infeasible_within_limits"]
    success = positives[np.arange(len(positives)), selected]
    predicted_status = status_logits.argmax(1)
    return {
        "target_success_feasible": float(np.mean(success[feasible])),
        "target_success_all": float(np.mean(success)),
        "status_accuracy": float(np.mean(predicted_status == statuses)),
        "status_macro_f1": float(
            f1_score(
                statuses,
                predicted_status,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_inverse(
    data_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_rows = read_jsonl(data_dir / "grids/train.jsonl")
    val_rows = read_jsonl(data_dir / "grids/val.jsonl")
    train_grids = {row["group_id"]: row for row in train_rows}
    val_grids = {row["group_id"]: row for row in val_rows}
    forward, forward_artifact = load_forward_predictor(output_dir, torch, device)
    predicted_train = predict_grid_states(
        train_rows, forward, forward_artifact, torch, device
    )
    predicted_val = predict_grid_states(
        val_rows, forward, forward_artifact, torch, device
    )
    ct, at, pt, st = inverse_arrays(
        data_dir, "train", train_grids, predicted_train
    )
    cv, av, pv, sv = inverse_arrays(data_dir, "val", val_grids, predicted_val)
    mean, scale = ct.mean(0), ct.std(0)
    scale[scale < 1e-8] = 1.0
    ct, cv = (ct - mean) / scale, (cv - mean) / scale
    action_mean, action_scale = at.reshape(-1, 5).mean(0), at.reshape(-1, 5).std(0)
    action_scale[action_scale < 1e-8] = 1.0
    at = (at - action_mean) / action_scale
    av = (av - action_mean) / action_scale
    model = inverse_ranker(torch, ct.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    status_counts = Counter(st.tolist())
    status_weights = torch.as_tensor(
        [
            math.sqrt(len(st) / max(3 * status_counts.get(index, 0), 1))
            for index in range(3)
        ],
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(int(config["training"]["seed"]) + 2)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    trace = []
    epochs = int(config["training"]["inverse_epochs"])
    batch_size = min(128, int(config["training"]["batch_size"]))
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for index in iter_batches(len(ct), batch_size, rng):
            context = torch.as_tensor(ct[index], dtype=torch.float32, device=device)
            actions = torch.as_tensor(at[index], dtype=torch.float32, device=device)
            positive = torch.as_tensor(pt[index], dtype=torch.bool, device=device)
            status = torch.as_tensor(st[index], dtype=torch.long, device=device)
            scores, status_logits = model(context, actions)
            feasible = positive.any(1)
            if feasible.any():
                positive_scores = scores[feasible].masked_fill(
                    ~positive[feasible], -1e9
                )
                rank_loss = (
                    torch.logsumexp(scores[feasible], dim=1)
                    - torch.logsumexp(positive_scores, dim=1)
                ).mean()
            else:
                rank_loss = scores.sum() * 0.0
            status_loss = torch.nn.functional.cross_entropy(
                status_logits, status, weight=status_weights
            )
            loss = rank_loss + 0.5 * status_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)
        model.eval()
        score_parts, status_parts = [], []
        with torch.inference_mode():
            for start in range(0, len(cv), 512):
                scores, status_logits = model(
                    torch.as_tensor(
                        cv[start : start + 512],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        av[start : start + 512],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                score_parts.append(scores.cpu().numpy())
                status_parts.append(status_logits.cpu().numpy())
        metrics = inverse_metrics(
            np.concatenate(score_parts), np.concatenate(status_parts), pv, sv
        )
        trace.append(
            {"epoch": epoch, "train_loss": running / len(ct), **metrics}
        )
        score = (
            metrics["target_success_feasible"]
            + 0.10 * metrics["status_macro_f1"]
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("inverse training produced no checkpoint")
    model.load_state_dict(best_state)
    artifact = output_dir / "inverse_ranker_v2.pt"
    save_torch_artifact(
        torch,
        artifact,
        {
            "version": config["version"],
            "seed": int(config["training"]["seed"]),
            "model": "inverse_ranker_v2",
            "context_dim": ct.shape[1],
            "state_dict": best_state,
            "context_mean": mean,
            "context_scale": scale,
            "action_mean": action_mean,
            "action_scale": action_scale,
            "statuses": STATUSES,
            "action_grid": fixed_action_grid(),
            "forward_artifact": str((output_dir / "forward_v2.pt").resolve()),
        },
    )
    return {
        "artifact": str(artifact.resolve()),
        "train_pairs": len(ct),
        "val_pairs": len(cv),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "validation": trace[best_epoch - 1],
        "trace": trace,
        "seconds": time.perf_counter() - started,
    }


class ImageStateDataset:
    def __init__(
        self,
        torch: Any,
        data_dir: Path,
        records: Sequence[Mapping[str, Any]],
        target_mean: np.ndarray,
        target_scale: np.ndarray,
        calibration_mean: np.ndarray,
        calibration_scale: np.ndarray,
    ) -> None:
        self.torch = torch
        self.data_dir = data_dir
        self.records = list(records)
        self.target_mean = target_mean
        self.target_scale = target_scale
        self.calibration_mean = calibration_mean
        self.calibration_scale = calibration_scale
        self.image_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        row = self.records[index]
        relative = str(row["image"])
        if relative not in self.image_cache:
            with Image.open(self.data_dir / relative) as image:
                self.image_cache[relative] = np.asarray(
                    image.convert("L"), dtype=np.uint8
                ).copy()
        array = self.image_cache[relative].astype(np.float32) / 255.0
        calibration = calibration_vector(row["image_calibration"])
        target = raw_state_array(row["target_state"])
        return (
            self.torch.from_numpy(array[None, :]),
            self.torch.from_numpy(
                (calibration - self.calibration_mean) / self.calibration_scale
            ),
            self.torch.from_numpy(
                (target - self.target_mean) / self.target_scale
            ),
        )


def calibration_vector(calibration: Mapping[str, Any]) -> np.ndarray:
    resolution = calibration["source_sensor_resolution_px"]
    return np.asarray(
        [
            math.log1p(float(calibration["linear_intensity_high"])),
            float(calibration["linear_intensity_low"]),
            float(calibration["gamma"]),
            math.log(float(resolution[0]) * float(resolution[1])),
        ],
        dtype=np.float32,
    )


def measurement_metrics(
    target: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    errors = np.abs(predicted - target)
    peak_relative = errors[:, 4] / np.maximum(np.abs(target[:, 4]), 1e-12)
    tolerance = np.column_stack(
        [
            np.ones(len(target)),
            np.ones(len(target)),
            np.full(len(target), 2.0),
            np.full(len(target), 2.0),
            np.maximum(0.05 * np.abs(target[:, 4]), 1e-12),
        ]
    )
    passed = np.column_stack(
        [
            errors[:, 0] <= 1.0,
            errors[:, 1] <= 1.0,
            errors[:, 2] <= 2.0,
            errors[:, 3] <= 2.0,
            peak_relative <= 0.05,
        ]
    )
    return {
        "strict_all_five_success": float(np.mean(np.all(passed, axis=1))),
        "mae_in_tolerance_units": float(np.mean(errors / tolerance)),
        "per_field_pass": {
            field: float(np.mean(passed[:, index]))
            for index, field in enumerate(STATE_FIELDS)
        },
        "mae": {
            field: float(errors[:, index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def train_measurement(
    data_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_rows = read_jsonl(data_dir / "measurement/train.jsonl")
    val_rows = read_jsonl(data_dir / "measurement/val.jsonl")
    train_targets = np.asarray(
        [raw_state_array(row["target_state"]) for row in train_rows],
        dtype=np.float32,
    )
    target_mean, target_scale = train_targets.mean(0), train_targets.std(0)
    target_scale[target_scale < 1e-8] = 1.0
    train_calibration = np.asarray(
        [calibration_vector(row["image_calibration"]) for row in train_rows],
        dtype=np.float32,
    )
    calibration_mean = train_calibration.mean(0)
    calibration_scale = train_calibration.std(0)
    calibration_scale[calibration_scale < 1e-8] = 1.0
    train_dataset = ImageStateDataset(
        torch,
        data_dir,
        train_rows,
        target_mean,
        target_scale,
        calibration_mean,
        calibration_scale,
    )
    val_dataset = ImageStateDataset(
        torch,
        data_dir,
        val_rows,
        target_mean,
        target_scale,
        calibration_mean,
        calibration_scale,
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=1,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=1,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    model = measurement_model(torch).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    trace = []
    for epoch in range(1, int(config["training"]["measurement_epochs"]) + 1):
        model.train()
        running = 0.0
        for image, calibration, target in loader:
            image = image.to(device, non_blocking=True)
            calibration = calibration.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            predicted, log_variance = model(image, calibration)
            log_variance = torch.clamp(log_variance, -4.0, 4.0)
            residual = predicted - target
            loss = (
                0.5 * torch.exp(-log_variance) * residual.square()
                + 0.5 * log_variance
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(image)
        model.eval()
        predicted_parts, target_parts = [], []
        with torch.inference_mode():
            for image, calibration, target in val_loader:
                predicted, _ = model(
                    image.to(device, non_blocking=True),
                    calibration.to(device, non_blocking=True),
                )
                predicted_parts.append(predicted.cpu().numpy())
                target_parts.append(target.numpy())
        predicted_raw = np.concatenate(predicted_parts) * target_scale + target_mean
        target_raw = np.concatenate(target_parts) * target_scale + target_mean
        metrics = measurement_metrics(target_raw, predicted_raw)
        trace.append(
            {"epoch": epoch, "train_loss": running / len(train_dataset), **metrics}
        )
        score = (
            metrics["strict_all_five_success"]
            - 0.01 * metrics["mae_in_tolerance_units"]
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("measurement training produced no checkpoint")
    model.load_state_dict(best_state)
    artifact = output_dir / "measurement_v2.pt"
    save_torch_artifact(
        torch,
        artifact,
        {
            "version": config["version"],
            "seed": int(config["training"]["seed"]),
            "model": "measurement_v2",
            "state_dict": best_state,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "calibration_mean": calibration_mean,
            "calibration_scale": calibration_scale,
            "state_fields": STATE_FIELDS,
        },
    )
    return {
        "artifact": str(artifact.resolve()),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "validation": trace[best_epoch - 1],
        "trace": trace,
        "seconds": time.perf_counter() - started,
    }


class VisualPairDataset:
    def __init__(
        self,
        torch: Any,
        data_dir: Path,
        records: Sequence[Mapping[str, Any]],
        context_mean: np.ndarray,
        context_scale: np.ndarray,
    ) -> None:
        self.torch = torch
        self.data_dir = data_dir
        self.records = list(records)
        self.context_mean = context_mean
        self.context_scale = context_scale
        self.image_cache: dict[str, np.ndarray] = {}
        self.actions = np.asarray(
            [[action[key] for key in ACTION_FIELDS] for action in fixed_action_grid()],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, relative: str) -> Any:
        if relative not in self.image_cache:
            with Image.open(self.data_dir / relative) as image:
                self.image_cache[relative] = np.asarray(
                    image.convert("L"), dtype=np.uint8
                ).copy()
        array = self.image_cache[relative].astype(np.float32) / 255.0
        return self.torch.from_numpy(array[None, :])

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        row = self.records[index]
        context = np.concatenate(
            [setup_array(row["setup"]), calibration_vector(row["image_calibration"])]
        )
        positive = np.zeros(81, dtype=np.bool_)
        positive[row["matching_indices"]] = True
        return (
            self.load_image(row["current_image"]),
            self.load_image(row["desired_image"]),
            self.torch.from_numpy(
                (context - self.context_mean) / self.context_scale
            ),
            self.torch.from_numpy(self.actions),
            self.torch.from_numpy(positive),
            self.torch.tensor(STATUS_INDEX[row["status"]], dtype=self.torch.long),
        )


def train_visual_inverse(
    data_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_rows = read_jsonl(data_dir / "visual/train.jsonl")
    val_rows = read_jsonl(data_dir / "visual/val.jsonl")
    contexts = np.asarray(
        [
            np.concatenate(
                [
                    setup_array(row["setup"]),
                    calibration_vector(row["image_calibration"]),
                ]
            )
            for row in train_rows
        ],
        dtype=np.float32,
    )
    context_mean, context_scale = contexts.mean(0), contexts.std(0)
    context_scale[context_scale < 1e-8] = 1.0
    train_dataset = VisualPairDataset(
        torch, data_dir, train_rows, context_mean, context_scale
    )
    val_dataset = VisualPairDataset(
        torch, data_dir, val_rows, context_mean, context_scale
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=1,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    model = visual_inverse_ranker(torch).to(device)
    measurement_path = output_dir / "measurement_v2.pt"
    if measurement_path.is_file():
        measurement_artifact = torch.load(
            measurement_path, map_location="cpu", weights_only=False
        )
        compatible = {
            key.removeprefix("encoder."): value
            for key, value in measurement_artifact["state_dict"].items()
            if key.startswith("encoder.")
        }
        model.encoder.load_state_dict(compatible, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]) * 0.7,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    visual_status_counts = Counter(
        STATUS_INDEX[row["status"]] for row in train_rows
    )
    visual_status_weights = torch.as_tensor(
        [
            math.sqrt(
                len(train_rows)
                / max(3 * visual_status_counts.get(index, 0), 1)
            )
            for index in range(3)
        ],
        dtype=torch.float32,
        device=device,
    )
    best_state = None
    best_score = -1.0
    best_epoch = 0
    trace = []
    for epoch in range(1, int(config["training"]["visual_inverse_epochs"]) + 1):
        model.train()
        running = 0.0
        for current, desired, context, actions, positive, status in loader:
            scores, status_logits = model(
                current.to(device, non_blocking=True),
                desired.to(device, non_blocking=True),
                context.to(device, non_blocking=True),
                actions.to(device, non_blocking=True),
            )
            positive = positive.to(device, non_blocking=True)
            feasible = positive.any(1)
            if feasible.any():
                positive_scores = scores[feasible].masked_fill(
                    ~positive[feasible], -1e9
                )
                rank_loss = (
                    torch.logsumexp(scores[feasible], dim=1)
                    - torch.logsumexp(positive_scores, dim=1)
                ).mean()
            else:
                rank_loss = scores.sum() * 0.0
            status_loss = torch.nn.functional.cross_entropy(
                status_logits,
                status.to(device, non_blocking=True),
                weight=visual_status_weights,
            )
            loss = rank_loss + 0.25 * status_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(current)
        model.eval()
        successes, feasible_flags, status_target, status_predicted = [], [], [], []
        with torch.inference_mode():
            for current, desired, context, actions, positive, status in val_loader:
                scores, status_logits = model(
                    current.to(device, non_blocking=True),
                    desired.to(device, non_blocking=True),
                    context.to(device, non_blocking=True),
                    actions.to(device, non_blocking=True),
                )
                selected = scores.argmax(1).cpu()
                successes.extend(
                    positive[torch.arange(len(selected)), selected].numpy().tolist()
                )
                feasible_flags.extend(positive.any(1).numpy().tolist())
                status_target.extend(status.numpy().tolist())
                status_predicted.extend(status_logits.argmax(1).cpu().numpy().tolist())
        successes_array = np.asarray(successes, dtype=np.bool_)
        feasible_array = np.asarray(feasible_flags, dtype=np.bool_)
        from sklearn.metrics import f1_score

        metrics = {
            "target_success_feasible": float(
                np.mean(successes_array[feasible_array])
            ),
            "target_success_all": float(np.mean(successes_array)),
            "status_accuracy": float(
                np.mean(np.asarray(status_target) == np.asarray(status_predicted))
            ),
            "status_macro_f1": float(
                f1_score(
                    status_target,
                    status_predicted,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        trace.append(
            {"epoch": epoch, "train_loss": running / len(train_dataset), **metrics}
        )
        score = (
            metrics["target_success_feasible"]
            + 0.10 * metrics["status_macro_f1"]
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("visual inverse training produced no checkpoint")
    model.load_state_dict(best_state)
    artifact = output_dir / "visual_inverse_v2.pt"
    save_torch_artifact(
        torch,
        artifact,
        {
            "version": config["version"],
            "seed": int(config["training"]["seed"]),
            "model": "visual_inverse_v2",
            "state_dict": best_state,
            "context_mean": context_mean,
            "context_scale": context_scale,
            "action_grid": fixed_action_grid(),
            "statuses": STATUSES,
        },
    )
    return {
        "artifact": str(artifact.resolve()),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "train_status_counts": {
            status: int(
                visual_status_counts.get(STATUS_INDEX[status], 0)
            )
            for status in STATUSES
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "validation": trace[best_epoch - 1],
        "trace": trace,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(data_dir / "config.json")
    if args.epoch_limit is not None:
        if args.epoch_limit < 1:
            raise ValueError("--epoch-limit must be positive")
        config = copy.deepcopy(config)
        for name, value in list(config["training"].items()):
            if name.endswith("_epochs"):
                config["training"][name] = min(
                    int(value), args.epoch_limit
                )
    torch, device = configure(int(config["training"]["seed"]), args.device)
    order = (
        [args.only]
        if args.only
        else ["direction", "forward", "inverse", "measurement", "visual_inverse"]
    )
    results = {}
    for name in order:
        print(f"starting {name} on {device}", flush=True)
        if name == "direction":
            result = train_direction(data_dir, output_dir, config, torch, device)
        elif name == "forward":
            result = train_forward(data_dir, output_dir, config, torch, device)
        elif name == "inverse":
            if not (output_dir / "forward_v2.pt").is_file():
                raise RuntimeError("inverse training requires forward_v2.pt")
            result = train_inverse(data_dir, output_dir, config, torch, device)
        elif name == "measurement":
            result = train_measurement(
                data_dir, output_dir, config, torch, device
            )
        elif name == "visual_inverse":
            result = train_visual_inverse(
                data_dir, output_dir, config, torch, device
            )
        else:
            raise AssertionError(name)
        results[name] = result
        (output_dir / f"{name}_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({name: result["validation"]}, indent=2), flush=True)
    combined = {
        "version": config["version"],
        "seed": int(config["training"]["seed"]),
        "device": str(device),
        "models": results,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
