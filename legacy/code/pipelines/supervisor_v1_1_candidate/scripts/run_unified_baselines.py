#!/usr/bin/env python3
"""Run fixed unified three-class baselines on candidate train/dev only."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "supervisor_v1_1_candidate"
LABELS = ("nominal", "sensor_saturation", "secondary_reflection")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
METRICS = ("centroid_x", "centroid_y", "width_x", "width_y", "peak_intensity")
SEEDS = (2026080101, 2026080102, 2026080103)
ROUTING = {
    "nominal": ("standard", "execute"),
    "sensor_saturation": ("lower_exposure_reacquire", "reacquire"),
    "secondary_reflection": ("primary_spot", "switch_measurement"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_split(split: str) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = read_jsonl(BASE / f"manifests/manifest_{split}.jsonl")
    metrics = np.asarray(
        [[float(row["model_input"]["current_metrics"][field]) for field in METRICS] for row in rows],
        dtype=np.float32,
    )
    labels = np.asarray([LABEL_TO_INDEX[row["target"]["diagnosis"]] for row in rows], dtype=np.int64)
    images = []
    for row in rows:
        path = ROOT / row["assets"]["current_image_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["assets"]["current_image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {path}")
        with Image.open(path) as image:
            image = image.convert("L")
            if image.size != (128, 128):
                image = image.resize((128, 128), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.float32)[None, :, :] / 255.0)
    return rows, metrics, np.asarray(images, dtype=np.float32), labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class TinyImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image).flatten(1)


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = TinyImageEncoder()
        self.head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, image: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        del metrics
        return self.head(self.image(image))


class TinyFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = TinyImageEncoder()
        self.head = nn.Sequential(nn.Linear(517, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, image: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.image(image), metrics), dim=1))


class MetricsMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, image: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        del image
        return self.net(metrics)


def train_neural(
    model: nn.Module,
    *,
    seed: int,
    train_images: np.ndarray,
    train_metrics: np.ndarray,
    train_labels: np.ndarray,
    dev_images: np.ndarray,
    dev_metrics: np.ndarray,
    epochs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_everything(seed)
    model.train()
    counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    weights = (len(train_labels) / (3.0 * counts)).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
    dataset = TensorDataset(
        torch.from_numpy(train_images), torch.from_numpy(train_metrics), torch.from_numpy(train_labels)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator, num_workers=0)
    started = time.perf_counter()
    epoch_losses: list[float] = []
    for _ in range(epochs):
        total = 0.0
        for images, metrics, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images, metrics), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        epoch_losses.append(total / len(dataset))
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(dev_images), torch.from_numpy(dev_metrics)).argmax(1).numpy()
    return prediction, {
        "wall_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_train_loss": epoch_losses[-1],
        "train_loss_curve": epoch_losses,
        "epochs": epochs,
    }


def diagnostic_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(3), zero_division=0
    )
    return {
        "coverage": 1.0,
        "accuracy": float(np.mean(labels == prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=np.arange(3), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=np.arange(3)).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }


def write_predictions(name: str, seed: int, rows: list[dict[str, Any]], prediction: np.ndarray) -> Path:
    path = BASE / "artifacts" / "predictions" / "baselines" / f"{name}_seed_{seed}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8") as handle:
        for row, class_index in zip(rows, prediction, strict=True):
            diagnosis = LABELS[int(class_index)]
            policy, action = ROUTING[diagnosis]
            text = json.dumps(
                {"diagnosis": diagnosis, "measurement_policy": policy, "supervisor_action": action},
                separators=(",", ":"), sort_keys=True,
            )
            handle.write(json.dumps({"sample_id": row["sample_id"], "prediction": text, "seed": seed}, sort_keys=True) + "\n")
    return path


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    protocol = BASE / "protocol" / "baseline_protocol.json"
    train_rows, train_metrics, train_images, train_labels = load_split("train")
    dev_rows, dev_metrics, dev_images, dev_labels = load_split("dev")
    mean = train_metrics.mean(axis=0)
    std = train_metrics.std(axis=0)
    std[std == 0] = 1.0
    train_standard = ((train_metrics - mean) / std).astype(np.float32)
    dev_standard = ((dev_metrics - mean) / std).astype(np.float32)
    output: dict[str, Any] = {
        "version": "supervisor_v1_1_candidate_unified_baselines_v1",
        "status": "NOT SEALED — FROZEN EVALUATION DISABLED",
        "protocol": {"path": str(protocol.relative_to(ROOT)), "sha256": sha256(protocol)},
        "data": {
            "train_manifest_sha256": sha256(BASE / "manifests/manifest_train.jsonl"),
            "dev_manifest_sha256": sha256(BASE / "manifests/manifest_dev.jsonl"),
            "train_records": len(train_rows), "dev_records": len(dev_rows),
            "frozen_or_protected_used": False,
        },
        "input_fields": {"metrics": list(METRICS), "images": "128x128 grayscale"},
        "models": {},
    }

    started = time.perf_counter()
    logistic = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEEDS[0]
    )
    logistic.fit(train_standard, train_labels)
    prediction = logistic.predict(dev_standard)
    elapsed = time.perf_counter() - started
    path = write_predictions("logistic_metrics", SEEDS[0], dev_rows, prediction)
    output["models"]["logistic_metrics"] = {
        "runs": [{"seed": SEEDS[0], "wall_seconds": elapsed, "parameter_count": int(logistic.coef_.size + logistic.intercept_.size), "metrics": diagnostic_metrics(dev_labels, prediction), "predictions": str(path.relative_to(ROOT)), "predictions_sha256": sha256(path)}],
        "configuration": "standardized metrics; multinomial lbfgs; C=1; balanced weights; max_iter=2000",
    }

    definitions = (
        ("metrics_mlp", MetricsMLP, 300),
        ("tiny_cnn_image", TinyCNN, 40),
        ("tiny_cnn_metrics_fusion", TinyFusion, 40),
    )
    for name, constructor, epochs in definitions:
        runs = []
        for seed in SEEDS:
            seed_everything(seed)
            prediction, training = train_neural(
                constructor(), seed=seed,
                train_images=train_images, train_metrics=train_standard, train_labels=train_labels,
                dev_images=dev_images, dev_metrics=dev_standard, epochs=epochs,
            )
            path = write_predictions(name, seed, dev_rows, prediction)
            runs.append({
                "seed": seed, **training, "metrics": diagnostic_metrics(dev_labels, prediction),
                "predictions": str(path.relative_to(ROOT)), "predictions_sha256": sha256(path),
            })
        output["models"][name] = {"runs": runs, "configuration": "fixed in baseline_protocol.json; no dev tuning"}

    destination = BASE / "reports" / "unified_baselines.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "models": {
            name: [round(run["metrics"]["balanced_accuracy"], 6) for run in model["runs"]]
            for name, model in output["models"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
