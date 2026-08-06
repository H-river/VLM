#!/usr/bin/env python3
"""Train one balanced five-head direction classifier on old and new grids."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure, iter_batches
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from direction_rebuild_v4.data import (
    CLASSES,
    DirectionArrays,
    balance_table,
    category_metrics,
    class_count_summary,
    concatenate_direction_arrays,
    direction_metrics,
    labels_from_normalized_change,
    load_grid_arrays,
)
from direction_rebuild_v4.models import direction_classifier_v4
from Qwen_orchestration.runtime.specialists import (
    DIRECTION_BUNDLE,
    _direction_member_logits,
    _load_pickle,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_NEW_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_CONTROL_RUN = (
    REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_quickcheck_12h"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_runs/direction_rebuild_v4_quickcheck_one_seed"
)
ACCEPTANCE_THRESHOLDS = {
    "old_iid_joint_exact": 0.60,
    "difficult_joint_exact": 0.60,
    "old_iid_equal_field_macro_f1": 0.80,
    "difficult_equal_field_macro_f1": 0.80,
    "old_iid_joint_delta_over_v1": 0.10,
    "difficult_joint_delta_over_v1": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument("--new-data", type=Path, default=DEFAULT_NEW_DATA)
    parser.add_argument("--control-run", type=Path, default=DEFAULT_CONTROL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument(
        "--balance-strength",
        type=float,
        default=0.50,
        help="Mixture weight for class-by-boundary-stratum balancing.",
    )
    parser.add_argument(
        "--worst-field-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--joint-surrogate-weight",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--fuse-forward-v4-features",
        action="store_true",
        help="Append the frozen forward-v4 five-change prediction to each input.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-old-train-groups", type=int)
    parser.add_argument("--max-new-train-groups", type=int)
    parser.add_argument("--max-old-val-groups", type=int)
    parser.add_argument("--max-new-val-groups", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_model(
    torch: Any,
    model: Any,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int = 8192,
) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            normalized = (
                features[start : start + batch_size] - mean
            ) / scale
            values = torch.as_tensor(
                normalized,
                dtype=torch.float32,
                device=device,
            )
            parts.append(
                model(values).argmax(dim=-1).to(torch.int64).cpu().numpy()
            )
    return np.concatenate(parts)


def frozen_v1_predictions(arrays: DirectionArrays) -> np.ndarray:
    if arrays.legacy_features is None:
        raise ValueError("frozen-v1 comparison requires legacy features")
    bundle = _load_pickle(str(DIRECTION_BUNDLE))
    return _direction_member_logits(bundle, arrays.legacy_features).argmax(axis=-1)


def read_json_groups(
    path: Path,
    maximum: int | None,
) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if maximum is not None and len(rows) >= maximum:
                break
    return rows


def forward_changes(
    path: Path,
    maximum: int | None,
    runtime: Any,
) -> np.ndarray:
    rows = read_json_groups(path, maximum)
    outputs = []
    for start in range(0, len(rows), 64):
        predicted_change = runtime.predict_changes(rows[start : start + 64])
        outputs.append(
            predicted_change.reshape(-1, predicted_change.shape[-1])
        )
    return np.concatenate(outputs)


def with_additional_features(
    arrays: DirectionArrays,
    additional: np.ndarray,
) -> DirectionArrays:
    if additional.shape != (arrays.transition_count, 5):
        raise ValueError("forward-v4 direction features must have shape N x 5")
    return replace(
        arrays,
        features=np.concatenate(
            [arrays.features, additional.astype(np.float32)],
            axis=1,
        ),
    )


def gate_summary(
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    old_v1: dict[str, Any],
    new_v1: dict[str, Any],
) -> dict[str, Any]:
    observed = {
        "old_iid_joint_exact": float(old_metrics["joint_exact"]),
        "difficult_joint_exact": float(new_metrics["joint_exact"]),
        "old_iid_equal_field_macro_f1": float(
            old_metrics["equal_field_macro_f1"]
        ),
        "difficult_equal_field_macro_f1": float(
            new_metrics["equal_field_macro_f1"]
        ),
        "old_iid_joint_delta_over_v1": float(
            old_metrics["joint_exact"] - old_v1["joint_exact"]
        ),
        "difficult_joint_delta_over_v1": float(
            new_metrics["joint_exact"] - new_v1["joint_exact"]
        ),
    }
    margins = {
        name: observed[name] - threshold
        for name, threshold in ACCEPTANCE_THRESHOLDS.items()
    }
    passed = {name: margin >= 0.0 for name, margin in margins.items()}
    return {
        "thresholds": dict(ACCEPTANCE_THRESHOLDS),
        "observed": observed,
        "margins": margins,
        "passed": passed,
        "passed_count": sum(passed.values()),
        "gate_count": len(passed),
        "all_passed": all(passed.values()),
        "minimum_margin": min(margins.values()),
        "mean_margin": sum(margins.values()) / len(margins),
    }


def selection_key(
    gates: dict[str, Any],
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    validation_loss: float,
) -> tuple[float, ...]:
    return (
        float(gates["passed_count"]),
        float(gates["minimum_margin"]),
        float(gates["mean_margin"]),
        0.5 * float(old_metrics["joint_exact"])
        + 0.5 * float(new_metrics["joint_exact"]),
        0.5 * float(old_metrics["equal_field_macro_f1"])
        + 0.5 * float(new_metrics["equal_field_macro_f1"]),
        -float(validation_loss),
    )


def validation_loss(
    torch: Any,
    model: Any,
    arrays: DirectionArrays,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, arrays.transition_count, 8192):
            stop = min(start + 8192, arrays.transition_count)
            values = torch.as_tensor(
                (arrays.features[start:stop] - mean) / scale,
                dtype=torch.float32,
                device=device,
            )
            targets = torch.as_tensor(
                arrays.labels[start:stop],
                dtype=torch.long,
                device=device,
            )
            logits = model(values)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3),
                targets.reshape(-1),
                reduction="sum",
            )
            total += float(loss.cpu())
            count += int(targets.numel())
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if not 0.0 <= args.balance_strength <= 1.0:
        raise ValueError("--balance-strength must be between zero and one")
    if args.worst_field_weight < 0.0 or args.joint_surrogate_weight < 0.0:
        raise ValueError("loss component weights must be nonnegative")
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(args.seed, args.device)

    old_train = load_grid_arrays(
        args.old_data.resolve() / "grids/train.jsonl",
        include_legacy_features=False,
        max_groups=args.max_old_train_groups,
    )
    new_train = load_grid_arrays(
        args.new_data.resolve() / "grids/train.jsonl",
        include_legacy_features=False,
        max_groups=args.max_new_train_groups,
    )
    old_val = load_grid_arrays(
        args.old_data.resolve() / "grids/val.jsonl",
        include_legacy_features=True,
        max_groups=args.max_old_val_groups,
    )
    new_val = load_grid_arrays(
        args.new_data.resolve() / "grids/val.jsonl",
        include_legacy_features=True,
        max_groups=args.max_new_val_groups,
    )
    forward_runtime, _ = load_forward_runtime_v4(
        args.control_run.resolve() / "forward_physics_residual_v4.pt",
        torch,
        device,
    )
    forward_old_val_changes = forward_changes(
        args.old_data.resolve() / "grids/val.jsonl",
        args.max_old_val_groups,
        forward_runtime,
    )
    forward_new_val_changes = forward_changes(
        args.new_data.resolve() / "grids/val.jsonl",
        args.max_new_val_groups,
        forward_runtime,
    )
    if args.fuse_forward_v4_features:
        old_train = with_additional_features(
            old_train,
            forward_changes(
                args.old_data.resolve() / "grids/train.jsonl",
                args.max_old_train_groups,
                forward_runtime,
            ),
        )
        new_train = with_additional_features(
            new_train,
            forward_changes(
                args.new_data.resolve() / "grids/train.jsonl",
                args.max_new_train_groups,
                forward_runtime,
            ),
        )
        old_val = with_additional_features(old_val, forward_old_val_changes)
        new_val = with_additional_features(new_val, forward_new_val_changes)
    del forward_runtime
    old_train_group_count = old_train.group_count
    new_train_group_count = new_train.group_count
    train = concatenate_direction_arrays([old_train, new_train])
    del old_train, new_train

    balance, balance_report = balance_table(
        train.labels,
        train.distance_bins,
    )
    feature_mean = train.features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = train.features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-8] = 1.0
    np.subtract(train.features, feature_mean, out=train.features)
    np.divide(train.features, feature_scale, out=train.features)

    frozen_old_predictions = frozen_v1_predictions(old_val)
    frozen_new_predictions = frozen_v1_predictions(new_val)
    frozen_old_metrics = direction_metrics(
        old_val.labels,
        frozen_old_predictions,
        old_val.distance_bins,
    )
    frozen_new_metrics = direction_metrics(
        new_val.labels,
        frozen_new_predictions,
        new_val.distance_bins,
    )

    forward_old_predictions = labels_from_normalized_change(
        forward_old_val_changes
    )
    forward_new_predictions = labels_from_normalized_change(
        forward_new_val_changes
    )
    forward_old_metrics = direction_metrics(
        old_val.labels,
        forward_old_predictions,
        old_val.distance_bins,
    )
    forward_new_metrics = direction_metrics(
        new_val.labels,
        forward_new_predictions,
        new_val.distance_bins,
    )
    model = direction_classifier_v4(torch, train.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.08,
    )
    balance_tensor = torch.as_tensor(
        balance,
        dtype=torch.float32,
        device=device,
    )
    field_indices = torch.arange(5, device=device)[None, :]
    rng = np.random.default_rng(args.seed + 907)
    best_key: tuple[float, ...] | None = None
    best_state = None
    best_epoch = 0
    best_record: dict[str, Any] | None = None
    trace = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for indices in iter_batches(
            train.transition_count,
            args.batch_size,
            rng,
        ):
            values = torch.as_tensor(
                train.features[indices],
                dtype=torch.float32,
                device=device,
            )
            targets = torch.as_tensor(
                train.labels[indices],
                dtype=torch.long,
                device=device,
            )
            bins = torch.as_tensor(
                train.distance_bins[indices],
                dtype=torch.long,
                device=device,
            )
            logits = model(values)
            raw_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3),
                targets.reshape(-1),
                reduction="none",
                label_smoothing=0.02,
            ).reshape(-1, 5)
            sample_weights = balance_tensor[
                field_indices,
                targets,
                bins,
            ]
            balanced_field_losses = (raw_loss * sample_weights).mean(dim=0)
            unweighted_field_losses = raw_loss.mean(dim=0)
            field_losses = (
                args.balance_strength * balanced_field_losses
                + (1.0 - args.balance_strength) * unweighted_field_losses
            )
            joint_surrogate = raw_loss.max(dim=1).values.mean()
            loss = (
                field_losses.mean()
                + args.worst_field_weight * field_losses.max()
                + args.joint_surrogate_weight * joint_surrogate
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        scheduler.step()

        old_predictions = predict_model(
            torch,
            model,
            old_val.features,
            feature_mean,
            feature_scale,
            device,
        )
        new_predictions = predict_model(
            torch,
            model,
            new_val.features,
            feature_mean,
            feature_scale,
            device,
        )
        old_metrics = direction_metrics(
            old_val.labels,
            old_predictions,
            old_val.distance_bins,
        )
        new_metrics = direction_metrics(
            new_val.labels,
            new_predictions,
            new_val.distance_bins,
        )
        combined_val_loss = 0.5 * validation_loss(
            torch,
            model,
            old_val,
            feature_mean,
            feature_scale,
            device,
        ) + 0.5 * validation_loss(
            torch,
            model,
            new_val,
            feature_mean,
            feature_scale,
            device,
        )
        gates = gate_summary(
            old_metrics,
            new_metrics,
            frozen_old_metrics,
            frozen_new_metrics,
        )
        key = selection_key(
            gates,
            old_metrics,
            new_metrics,
            combined_val_loss,
        )
        record = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train_loss": running / max(seen, 1),
            "validation_loss": combined_val_loss,
            "old_iid": {
                "joint_exact": old_metrics["joint_exact"],
                "equal_field_macro_f1": old_metrics["equal_field_macro_f1"],
                "mean_field_accuracy": old_metrics["mean_field_accuracy"],
            },
            "difficult": {
                "joint_exact": new_metrics["joint_exact"],
                "equal_field_macro_f1": new_metrics["equal_field_macro_f1"],
                "mean_field_accuracy": new_metrics["mean_field_accuracy"],
            },
            "selection_gates": gates,
            "selection_key": list(key),
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_record = record

    if best_state is None or best_record is None or best_key is None:
        raise RuntimeError("direction training produced no checkpoint")
    model.load_state_dict(best_state)
    final_old_predictions = predict_model(
        torch,
        model,
        old_val.features,
        feature_mean,
        feature_scale,
        device,
    )
    final_new_predictions = predict_model(
        torch,
        model,
        new_val.features,
        feature_mean,
        feature_scale,
        device,
    )
    final_old_metrics = direction_metrics(
        old_val.labels,
        final_old_predictions,
        old_val.distance_bins,
    )
    final_new_metrics = direction_metrics(
        new_val.labels,
        final_new_predictions,
        new_val.distance_bins,
    )
    final_gates = gate_summary(
        final_old_metrics,
        final_new_metrics,
        frozen_old_metrics,
        frozen_new_metrics,
    )
    artifact_path = output_dir / "direction_classifier_v4.pt"
    artifact = {
        "version": "direction_rebuild_v4_quickcheck_one_seed",
        "model": "balanced_five_head_direction_v4",
        "seed": int(args.seed),
        "input_dim": int(train.features.shape[1]),
        "direction_fields": list(DIRECTION_FIELDS),
        "classes": list(CLASSES),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "feature_mode": (
            "engineered_46_plus_forward_v4_change_5"
            if args.fuse_forward_v4_features
            else "engineered_46"
        ),
        "state_dict": {
            key: value.detach().cpu() for key, value in best_state.items()
        },
        "balance_contract": balance_report,
        "checkpoint_selection": (
            "validation_only_gate_count_then_minimum_margin_then_joint_exact"
        ),
        "best_epoch": best_epoch,
        "acceptance_gates": final_gates,
        "held_out_test_used": False,
    }
    torch.save(artifact, artifact_path)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "seed": int(args.seed),
        "device": str(device),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "input_dim": int(train.features.shape[1]),
        "epochs": int(args.epochs),
        "training_objective": {
            "balance_strength": float(args.balance_strength),
            "worst_field_weight": float(args.worst_field_weight),
            "joint_surrogate_weight": float(args.joint_surrogate_weight),
            "label_smoothing": 0.02,
            "fuse_forward_v4_features": bool(
                args.fuse_forward_v4_features
            ),
        },
        "best_epoch": best_epoch,
        "best_selection_key": list(best_key),
        "training": {
            "old_iid_groups": old_train_group_count,
            "difficult_groups": new_train_group_count,
            "total_groups": train.group_count,
            "total_transitions": train.transition_count,
            "class_counts": class_count_summary(train.labels),
            "balance_contract": balance_report,
        },
        "validation": {
            "old_iid_groups": old_val.group_count,
            "difficult_groups": new_val.group_count,
            "old_iid_transitions": old_val.transition_count,
            "difficult_transitions": new_val.transition_count,
            "frozen_v1": {
                "old_iid": frozen_old_metrics,
                "difficult": frozen_new_metrics,
            },
            "thresholded_forward_v4": {
                "old_iid": forward_old_metrics,
                "difficult": forward_new_metrics,
            },
            "direction_v4": {
                "old_iid": final_old_metrics,
                "difficult": final_new_metrics,
                "difficult_by_category": category_metrics(
                    new_val,
                    final_new_predictions,
                ),
            },
            "acceptance_gates": final_gates,
        },
        "trace": trace,
        "source_contract": {
            "old_train": str(
                args.old_data.resolve() / "grids/train.jsonl"
            ),
            "old_val": str(args.old_data.resolve() / "grids/val.jsonl"),
            "new_train": str(
                args.new_data.resolve() / "grids/train.jsonl"
            ),
            "new_val": str(args.new_data.resolve() / "grids/val.jsonl"),
            "forward_artifact_sha256": sha256(
                args.control_run.resolve() / "forward_physics_residual_v4.pt"
            ),
            "frozen_direction_v1_sha256": sha256(DIRECTION_BUNDLE),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "direction_classifier_v4_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "summary": str(summary_path),
                "best_epoch": best_epoch,
                "parameter_count": summary["parameter_count"],
                "old_iid_joint_exact": final_old_metrics["joint_exact"],
                "difficult_joint_exact": final_new_metrics["joint_exact"],
                "old_iid_macro_f1": final_old_metrics[
                    "equal_field_macro_f1"
                ],
                "difficult_macro_f1": final_new_metrics[
                    "equal_field_macro_f1"
                ],
                "all_gates_passed": final_gates["all_passed"],
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
