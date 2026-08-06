#!/usr/bin/env python3
"""Train a simulator-free multi-task model for quantitative beam changes."""

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
from optics_understanding_sft.direction_inverse_v1.train_direction_small import (
    CLASSES, CLASS_TO_INDEX, FEATURE_FIELDS, FIELDS, feature_vector,
)

CHANGE_KEYS = ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity")
DIRECTION_SOURCE_KEYS = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")
BASE_TOLERANCES = np.asarray([1.0, 1.0, 2.0, 2.0, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="17,42,91")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--direction-loss-weight", type=float, default=0.35)
    return parser.parse_args()


def peak_tolerance(record: Mapping[str, Any]) -> float:
    peak = float(record["inputs"]["current_beam_state"]["peak_intensity"])
    return max(0.05 * abs(peak), 1e-6)


def target_arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(
        [[float(record["target"]["change"][key]) for key in CHANGE_KEYS] for record in records],
        dtype=np.float32,
    )
    tolerance = np.tile(BASE_TOLERANCES, (len(records), 1))
    tolerance[:, 4] = np.asarray([peak_tolerance(record) for record in records], dtype=np.float32)
    directions = np.asarray(
        [[CLASS_TO_INDEX[record["target"]["directions"][key]] for key in DIRECTION_SOURCE_KEYS]
         for record in records], dtype=np.int64,
    )
    return raw, raw / tolerance, directions


def input_arrays(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    wrapped = [
        {"prompt_inputs": {"setup": row["inputs"]["setup"],
                           "current_beam_state": row["inputs"]["current_beam_state"],
                           "action": row["inputs"]["action"]}}
        for row in records
    ]
    return np.asarray([feature_vector(row) for row in wrapped], dtype=np.float32)


def engineered_features(x: np.ndarray) -> np.ndarray:
    """Add final actuator positions and low-order optics interaction terms."""
    column = {name: index for index, name in enumerate(FEATURE_FIELDS)}
    lx, ly = x[:, column["lens_x_offset_mm"]], x[:, column["lens_y_offset_mm"]]
    cx, cy = x[:, column["camera_x_offset_mm"]], x[:, column["camera_y_offset_mm"]]
    dlx, dly = x[:, column["lens_x_delta_mm"]], x[:, column["lens_y_delta_mm"]]
    dcx, dcy = x[:, column["camera_x_delta_mm"]], x[:, column["camera_y_delta_mm"]]
    pitch_mm = x[:, column["pixel_size_um"]] / 1000.0
    ratio = (x[:, column["lens_to_camera_mm"]] / x[:, column["lens_focal_length_mm"]]) / pitch_mm
    derived = np.column_stack([
        lx + dlx, ly + dly, cx + dcx, cy + dcy,
        np.square(lx + dlx) - np.square(lx), np.square(ly + dly) - np.square(ly),
        np.square(cx + dcx) - np.square(cx), np.square(cy + dcy) - np.square(cy),
        np.hypot(lx + dlx, ly + dly), np.hypot(lx, ly),
        np.hypot(cx + dcx, cy + dcy), np.hypot(cx, cy),
        dlx * ratio, dly * ratio, dcx / pitch_mm, dcy / pitch_mm, ratio,
        dlx * lx, dly * ly, dcx * cx, dcy * cy,
        np.abs(lx + dlx) - np.abs(lx), np.abs(ly + dly) - np.abs(ly),
        np.abs(cx + dcx) - np.abs(cx), np.abs(cy + dcy) - np.abs(cy),
    ]).astype(np.float32)
    return np.concatenate([x, derived], axis=1)


def make_model(torch: Any, input_dim: int) -> Any:
    class ForwardNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128), torch.nn.LayerNorm(128), torch.nn.SiLU(),
                torch.nn.Dropout(0.08), torch.nn.Linear(128, 64), torch.nn.LayerNorm(64),
                torch.nn.SiLU(),
            )
            self.regression_head = torch.nn.Sequential(
                torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, len(CHANGE_KEYS))
            )
            self.direction_heads = torch.nn.ModuleList(
                [torch.nn.Linear(64, len(CLASSES)) for _ in FIELDS]
            )

        def forward(self, values: Any) -> tuple[Any, Any]:
            hidden = self.encoder(values)
            return self.regression_head(hidden), torch.stack(
                [head(hidden) for head in self.direction_heads], dim=1
            )

    return ForwardNet()


def _direction_weights(y: np.ndarray) -> np.ndarray:
    result = np.ones((len(FIELDS), len(CLASSES)), dtype=np.float32)
    for field_index in range(len(FIELDS)):
        counts = Counter(y[:, field_index].tolist())
        for class_index in range(len(CLASSES)):
            result[field_index, class_index] = math.sqrt(
                len(y) / max(len(CLASSES) * counts.get(class_index, 0), 1)
            )
    return result


def _predict(torch: Any, model: Any, x: np.ndarray, device: Any) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    changes, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            xb = torch.as_tensor(x[start:start + 1024], dtype=torch.float32, device=device)
            change, direction = model(xb)
            changes.append(change.float().cpu().numpy())
            logits.append(direction.float().cpu().numpy())
    return np.concatenate(changes), np.concatenate(logits)


def train_member(
    x_train: np.ndarray, y_train_scaled: np.ndarray, y_train_direction: np.ndarray,
    x_val: np.ndarray, y_val_scaled: np.ndarray, *, seed: int, epochs: int,
    batch_size: int, learning_rate: float, direction_loss_weight: float,
) -> dict[str, Any]:
    import torch

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xt, xv = (x_train - mean) / scale, (x_val - mean) / scale
    model = make_model(torch, xt.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    weights = torch.as_tensor(_direction_weights(y_train_direction), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best_state, best_epoch, best_score = None, 0, float("inf")
    trace: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(xt)); total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            xb = torch.as_tensor(xt[index], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train_scaled[index], dtype=torch.float32, device=device)
            db = torch.as_tensor(y_train_direction[index], dtype=torch.long, device=device)
            predicted, direction_logits = model(xb)
            regression_loss = torch.nn.functional.smooth_l1_loss(predicted, yb, beta=1.0)
            classification_loss = torch.stack([
                torch.nn.functional.cross_entropy(direction_logits[:, i], db[:, i], weight=weights[i])
                for i in range(len(FIELDS))
            ]).mean()
            loss = regression_loss + direction_loss_weight * classification_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach().cpu()) * len(index)
        if epoch == 1 or epoch % 5 == 0:
            predicted, _ = _predict(torch, model, xv, device)
            val_error = float(np.mean(np.abs(predicted - y_val_scaled)))
            trace.append({"epoch": epoch, "train_loss": total / len(xt),
                          "val_tolerance_mae": val_error})
            if val_error < best_score - 1e-5:
                best_score, best_epoch, best_state = val_error, epoch, copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= 50:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    return {"seed": seed, "mean": mean, "scale": scale,
            "state_dict": {key: value.cpu() for key, value in best_state.items()},
            "best_epoch": best_epoch, "best_val_tolerance_mae": best_score, "trace": trace}


def predict_bundle(bundle: Mapping[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if bundle.get("model_kind") == "neural_hist_blend":
        neural_change, neural_logits = predict_bundle(bundle["neural_bundle"], x)
        tree_change = np.asarray(bundle["hist_model"].predict(engineered_features(x)), dtype=np.float32)
        alpha = float(bundle["neural_weight"])
        return alpha * neural_change + (1.0 - alpha) * tree_change, neural_logits
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    changes, logits = [], []
    for member in bundle["members"]:
        model = make_model(torch, x.shape[1]).to(device)
        model.load_state_dict(member["state_dict"])
        change, direction = _predict(torch, model, (x - member["mean"]) / member["scale"], device)
        changes.append(change); logits.append(direction)
    return np.mean(np.stack(changes), axis=0), np.mean(np.stack(logits), axis=0)


def strict_metrics(records: Sequence[Mapping[str, Any]], predicted_scaled: np.ndarray,
                   direction_logits: np.ndarray) -> dict[str, Any]:
    raw, target_scaled, target_direction = target_arrays(records)
    tolerance = np.tile(BASE_TOLERANCES, (len(records), 1))
    tolerance[:, 4] = np.asarray([peak_tolerance(row) for row in records], dtype=np.float32)
    predicted_raw = predicted_scaled * tolerance
    error_scaled, zero_error = np.abs(predicted_scaled - target_scaled), np.abs(target_scaled)
    passed = error_scaled <= 1.0
    predicted_direction = direction_logits.argmax(axis=-1)
    from sklearn.metrics import f1_score
    direction_f1 = {
        field: float(f1_score(target_direction[:, i], predicted_direction[:, i],
                              labels=list(range(len(CLASSES))), average="macro", zero_division=0))
        for i, field in enumerate(FIELDS)
    }
    return {
        "count": len(records),
        "strict_all_five_success": float(np.mean(np.all(passed, axis=1))),
        "zero_baseline_strict_all_five_success": float(
            np.mean(np.all(zero_error <= 1.0, axis=1))
        ),
        "per_field_tolerance_pass": {key: float(np.mean(passed[:, i])) for i, key in enumerate(CHANGE_KEYS)},
        "mae_raw": {key: float(np.mean(np.abs(predicted_raw[:, i] - raw[:, i])))
                    for i, key in enumerate(CHANGE_KEYS)},
        "mae_in_tolerance_units": float(np.mean(error_scaled)),
        "zero_baseline_mae_in_tolerance_units": float(np.mean(zero_error)),
        "skill_over_zero": float(1.0 - np.mean(error_scaled) / max(np.mean(zero_error), 1e-12)),
        "direction_macro_f1": direction_f1,
        "equal_field_direction_macro_f1": float(np.mean(list(direction_f1.values()))),
        "direction_joint_exact": float(np.mean(np.all(target_direction == predicted_direction, axis=1))),
    }


def evaluate_split(name: str, records: list[dict[str, Any]], bundle: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted_scaled, direction_logits = predict_bundle(bundle, input_arrays(records))
    raw, target_scaled, target_direction = target_arrays(records)
    tolerance = np.tile(BASE_TOLERANCES, (len(records), 1))
    tolerance[:, 4] = np.asarray([peak_tolerance(row) for row in records], dtype=np.float32)
    predicted_raw, predicted_direction = predicted_scaled * tolerance, direction_logits.argmax(axis=-1)
    details = [{
        "example_id": row["example_id"], "group_id": row["group_id"], "split": name,
        "target_change": {key: float(raw[n, i]) for i, key in enumerate(CHANGE_KEYS)},
        "predicted_change": {key: float(predicted_raw[n, i]) for i, key in enumerate(CHANGE_KEYS)},
        "field_pass": {key: bool(abs(predicted_scaled[n, i] - target_scaled[n, i]) <= 1.0)
                       for i, key in enumerate(CHANGE_KEYS)},
        "target_directions": {field: CLASSES[target_direction[n, i]] for i, field in enumerate(FIELDS)},
        "predicted_directions": {field: CLASSES[predicted_direction[n, i]] for i, field in enumerate(FIELDS)},
    } for n, row in enumerate(records)]
    return strict_metrics(records, predicted_scaled, direction_logits), details


def report_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Joint quantitative forward model", "",
             "A compact local network predicts five numerical beam changes and five direction classes without simulator access at inference. A record succeeds only when all five numerical errors are inside their declared sensor tolerances.", "",
             "| Split | Strict all-five | Zero strict | Tolerance MAE | Skill over zero | Direction macro-F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for split in ("val", "eval_iid", "eval_ood"):
        m = summary[split]
        lines.append(f"| {split} | {m['strict_all_five_success']:.3f} | {m['zero_baseline_strict_all_five_success']:.3f} | {m['mae_in_tolerance_units']:.3f} | {m['skill_over_zero']:.3f} | {m['equal_field_direction_macro_f1']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    splits = {split: read_jsonl(args.source_dir / f"{split}.jsonl")
              for split in ("train", "val", "eval_iid", "eval_ood")}
    groups = {split: {row["group_id"] for row in rows} for split, rows in splits.items()}
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if groups[left] & groups[right]:
                raise RuntimeError(f"group leakage between {left} and {right}")
    x_train, x_val = input_arrays(splits["train"]), input_arrays(splits["val"])
    _, y_train_scaled, y_train_direction = target_arrays(splits["train"])
    _, y_val_scaled, _ = target_arrays(splits["val"])
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    members = [train_member(x_train, y_train_scaled, y_train_direction, x_val, y_val_scaled,
                            seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            direction_loss_weight=args.direction_loss_weight) for seed in seeds]
    bundle = {"version": "direction_inverse_v1_forward_joint", "feature_fields": FEATURE_FIELDS,
              "change_keys": CHANGE_KEYS, "direction_fields": FIELDS, "classes": CLASSES,
              "members": members, "simulator_at_inference": False}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "forward_small_ensemble.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    summary: dict[str, Any] = {
        "version": bundle["version"], "simulator_at_inference": False,
        "strict_success_definition": "all five errors within 1px centroid, 2px width, and 5% initial peak",
        "train_records": len(splits["train"]), "train_groups": len(groups["train"]), "seeds": seeds,
        "member_training": [{key: m[key] for key in ("seed", "best_epoch", "best_val_tolerance_mae", "trace")} for m in members],
    }
    all_details = []
    for split in ("val", "eval_iid", "eval_ood"):
        summary[split], details = evaluate_split(split, splits[split], bundle); all_details.extend(details)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "details.jsonl", all_details)
    (args.output_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
