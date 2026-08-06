#!/usr/bin/env python3
"""Train and evaluate a shared five-head direction classifier."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from optics_understanding_sft.core import read_jsonl, write_jsonl
from optics_understanding_sft.direction_inverse_v1.build_direction import FIELD_MAP


CLASSES = ("decrease", "no_change", "increase")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
FIELDS = tuple(FIELD_MAP)
FEATURE_FIELDS = (
    "wavelength_nm",
    "beam_waist_mm",
    "power_w",
    "lens_focal_length_mm",
    "lens_aperture_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
    "lens_x_offset_mm",
    "lens_y_offset_mm",
    "camera_x_offset_mm",
    "camera_y_offset_mm",
    "pixel_size_um",
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="17,42,91")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def feature_vector(record: Mapping[str, Any]) -> list[float]:
    inputs = record["prompt_inputs"]
    merged = {**inputs["setup"], **inputs["current_beam_state"], **inputs["action"]}
    return [
        math.log1p(max(float(merged[field]), 0.0)) if field == "peak_intensity" else float(merged[field])
        for field in FEATURE_FIELDS
    ]


def labels(record: Mapping[str, Any]) -> list[int]:
    answer = record["target"]["answer"]
    result = [-100] * len(FIELDS)
    if record["task_type"] == "direction_single_field":
        result[FIELDS.index(answer["field"])] = CLASS_TO_INDEX[answer["direction"]]
    else:
        for index, field in enumerate(FIELDS):
            result[index] = CLASS_TO_INDEX[answer["directions"][field]]
    return result


def arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([feature_vector(record) for record in records], dtype=np.float32),
        np.asarray([labels(record) for record in records], dtype=np.int64),
    )


def macro_f1(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    by_field = {
        field: float(
            f1_score(
                target[:, index],
                predicted[:, index],
                labels=list(range(len(CLASSES))),
                average="macro",
                zero_division=0,
            )
        )
        for index, field in enumerate(FIELDS)
    }
    joint = float(np.mean(np.all(target == predicted, axis=1)))
    return {
        "field_macro_f1": by_field,
        "equal_field_macro_f1": float(np.mean(list(by_field.values()))),
        "joint_exact": joint,
        "target_distribution": {
            field: {CLASSES[key]: int(value) for key, value in Counter(target[:, index].tolist()).items()}
            for index, field in enumerate(FIELDS)
        },
        "predicted_distribution": {
            field: {CLASSES[key]: int(value) for key, value in Counter(predicted[:, index].tolist()).items()}
            for index, field in enumerate(FIELDS)
        },
    }


def require_torch() -> Any:
    import torch

    return torch


def make_model(torch: Any, input_dim: int) -> Any:
    class DirectionNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.LayerNorm(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(128, 64),
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(),
            )
            self.heads = torch.nn.ModuleList(
                [torch.nn.Linear(64, len(CLASSES)) for _ in FIELDS]
            )

        def forward(self, values: Any) -> Any:
            hidden = self.encoder(values)
            return torch.stack([head(hidden) for head in self.heads], dim=1)

    return DirectionNet()


def class_weights(y: np.ndarray) -> np.ndarray:
    result = np.ones((len(FIELDS), len(CLASSES)), dtype=np.float32)
    for field_index in range(len(FIELDS)):
        valid = y[:, field_index]
        valid = valid[valid >= 0]
        counts = Counter(valid.tolist())
        total = sum(counts.values())
        for class_index in range(len(CLASSES)):
            count = counts.get(class_index, 0)
            result[field_index, class_index] = math.sqrt(total / max(count * len(CLASSES), 1))
    return result


def predict_logits(torch: Any, model: Any, x: np.ndarray, device: Any) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            values = torch.as_tensor(x[start : start + 512], dtype=torch.float32, device=device)
            chunks.append(model(values).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_member(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xt = (x_train - mean) / scale
    xv = (x_val - mean) / scale
    model = make_model(torch, xt.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    weights = torch.as_tensor(class_weights(y_train), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best_state = None
    best_metric = -1.0
    best_epoch = 0
    patience = 30
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(xt))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            xb = torch.as_tensor(xt[indices], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[indices], dtype=torch.long, device=device)
            logits = model(xb)
            losses = []
            for field_index in range(len(FIELDS)):
                valid = yb[:, field_index] >= 0
                if bool(valid.any()):
                    losses.append(
                        torch.nn.functional.cross_entropy(
                            logits[valid, field_index],
                            yb[valid, field_index],
                            weight=weights[field_index],
                        )
                    )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if epoch == 1 or epoch % 5 == 0:
            val_logits = predict_logits(torch, model, xv, device)
            metrics = macro_f1(y_val, val_logits.argmax(axis=-1))
            score = metrics["equal_field_macro_f1"]
            trace.append({"epoch": epoch, "train_loss": total_loss / batches, "val_macro_f1": score})
            if score > best_metric + 1e-6:
                best_metric = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= patience:
                break
    model.load_state_dict(best_state)
    return {
        "seed": seed,
        "mean": mean,
        "scale": scale,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "trace": trace,
    }


def member_logits(bundle: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = []
    for member in bundle["members"]:
        model = make_model(torch, x.shape[1]).to(device)
        model.load_state_dict(member["state_dict"])
        scaled = (x - member["mean"]) / member["scale"]
        outputs.append(predict_logits(torch, model, scaled, device))
    return np.mean(np.stack(outputs), axis=0)


def majority_prediction(train_y: np.ndarray, count: int) -> np.ndarray:
    labels_out = []
    for field_index in range(len(FIELDS)):
        valid = train_y[:, field_index]
        valid = valid[valid >= 0]
        labels_out.append(Counter(valid.tolist()).most_common(1)[0][0])
    return np.tile(np.asarray(labels_out, dtype=np.int64), (count, 1))


def evaluate_split(
    name: str,
    records: list[dict[str, Any]],
    bundle: Mapping[str, Any],
    train_y: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x, y = arrays(records)
    logits = member_logits(bundle, x)
    predicted = logits.argmax(axis=-1)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    majority = majority_prediction(train_y, len(records))
    summary = {
        "count": len(records),
        "shared_mlp": macro_f1(y, predicted),
        "majority": macro_f1(y, majority),
    }
    details = []
    for row_index, record in enumerate(records):
        details.append(
            {
                "example_id": record["example_id"],
                "group_id": record["group_id"],
                "split": name,
                "target": {field: CLASSES[y[row_index, index]] for index, field in enumerate(FIELDS)},
                "prediction": {
                    field: CLASSES[predicted[row_index, index]] for index, field in enumerate(FIELDS)
                },
                "probabilities": {
                    field: {
                        label: float(probabilities[row_index, index, class_index])
                        for class_index, label in enumerate(CLASSES)
                    }
                    for index, field in enumerate(FIELDS)
                },
            }
        )
    return summary, details


def report_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Dedicated small direction classifier",
        "",
        "A shared numerical encoder predicts five independent three-class direction heads. Training combines exactly balanced single-field records with natural-distribution all-field records. The simulator is unavailable at inference.",
        "",
        "| Split | Model | Equal-field macro-F1 | All-five exact |",
        "|---|---|---:|---:|",
    ]
    for split in ("val", "eval_iid", "eval_ood"):
        for model_name in ("majority", "shared_mlp"):
            metrics = summary[split][model_name]
            lines.append(
                f"| {split} | {model_name} | {metrics['equal_field_macro_f1']:.3f} | {metrics['joint_exact']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    single_train = read_jsonl(args.data_dir / "direction/single_field/train.jsonl")
    all_train = read_jsonl(args.data_dir / "direction/all_fields/train.jsonl")
    val = read_jsonl(args.data_dir / "direction/all_fields/val.jsonl")
    eval_iid = read_jsonl(args.data_dir / "direction/all_fields/eval_iid.jsonl")
    eval_ood = read_jsonl(args.data_dir / "direction/all_fields/eval_ood.jsonl")
    train_records = single_train + all_train
    x_train, y_train = arrays(train_records)
    x_val, y_val = arrays(val)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    members = [
        train_member(
            x_train,
            y_train,
            x_val,
            y_val,
            seed,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        for seed in seeds
    ]
    bundle = {
        "version": "direction_inverse_v1",
        "feature_fields": FEATURE_FIELDS,
        "fields": FIELDS,
        "classes": CLASSES,
        "members": members,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "direction_small_ensemble.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    summaries = {}
    all_details = []
    for split, records in (("val", val), ("eval_iid", eval_iid), ("eval_ood", eval_ood)):
        split_summary, details = evaluate_split(split, records, bundle, y_train)
        summaries[split] = split_summary
        all_details.extend(details)
    summary = {
        "version": "direction_inverse_v1",
        "simulator_at_inference": False,
        "train_records": len(train_records),
        "train_groups": len({row["group_id"] for row in train_records}),
        "seeds": seeds,
        "member_training": [
            {
                "seed": member["seed"],
                "best_epoch": member["best_epoch"],
                "best_val_macro_f1": member["best_val_macro_f1"],
                "trace": member["trace"],
            }
            for member in members
        ],
        **summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(args.output_dir / "details.jsonl", all_details)
    (args.output_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
