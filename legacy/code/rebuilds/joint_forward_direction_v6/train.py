#!/usr/bin/env python3
"""Train one shared encoder for numerical forward and direction prediction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure, forward_metrics
from control_rebuild_v5.forward_runtime import (
    ZERO_ACTION_INDEX,
    load_forward_runtime_v5,
)
from direction_rebuild_v4.data import (
    DirectionArrays,
    balance_table,
    category_metrics,
    concatenate_direction_arrays,
    direction_metrics,
    labels_from_normalized_change,
    load_grid_arrays,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    CLASSES,
    DIRECTION_FIELDS,
    STATE_FIELDS,
)

from joint_forward_direction_v6.models import shared_forward_direction_model

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_BASE_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v5_one_seed/forward_tree_v5.pkl"
)
DEFAULT_DIRECTION = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed/direction_tree_v4.pkl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_runs/joint_forward_direction_v6_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument(
        "--difficult-data", type=Path, default=DEFAULT_DIFFICULT_DATA
    )
    parser.add_argument(
        "--additional-data", type=Path, default=DEFAULT_ADDITIONAL_DATA
    )
    parser.add_argument(
        "--base-forward", type=Path, default=DEFAULT_BASE_FORWARD
    )
    parser.add_argument(
        "--direction-reference", type=Path, default=DEFAULT_DIRECTION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--direction-loss-weight", type=float, default=0.80)
    parser.add_argument("--joint-loss-weight", type=float, default=0.15)
    parser.add_argument("--residual-loss-weight", type=float, default=0.005)
    parser.add_argument("--balance-strength", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-old-train-groups", type=int)
    parser.add_argument("--max-difficult-train-groups", type=int)
    parser.add_argument("--max-additional-train-groups", type=int)
    parser.add_argument("--max-old-val-groups", type=int)
    parser.add_argument("--max-difficult-val-groups", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_predictions(
    base_runtime: Any,
    arrays: DirectionArrays,
) -> np.ndarray:
    return np.stack(
        [
            model.predict(arrays.features)
            for model in base_runtime.models
        ],
        axis=1,
    ).astype(np.float32)


def direction_reference_predictions(
    path: Path,
    arrays: DirectionArrays,
) -> np.ndarray:
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    if (
        artifact.get("model")
        != "balanced_five_head_hist_gradient_boosting_direction_v4"
    ):
        raise ValueError("direction reference is not the v4 tree model")
    return np.stack(
        [model.predict(arrays.features) for model in artifact["models"]],
        axis=1,
    ).astype(np.int64)


def action_complexity_weight() -> np.ndarray:
    actions = np.asarray(
        [
            [float(action[field]) for field in ACTION_FIELDS]
            for action in ACTION_GRID
        ],
        dtype=np.float32,
    )
    complexity = np.count_nonzero(actions, axis=1)
    counts = np.bincount(complexity, minlength=5)
    return np.asarray(
        [
            0.02 if value == 0 else 0.245 / counts[value]
            for value in complexity
        ],
        dtype=np.float32,
    )


def source_sample_weights(
    group_counts: list[int],
    source_mass: list[float],
) -> np.ndarray:
    if len(group_counts) != len(source_mass):
        raise ValueError("group counts and source masses differ")
    action_weight = action_complexity_weight()
    parts = []
    for groups, mass in zip(group_counts, source_mass, strict=True):
        group_weight = np.full(
            groups,
            float(mass) / groups,
            dtype=np.float32,
        )
        parts.append(
            (group_weight[:, None] * action_weight[None, :]).reshape(-1)
        )
    result = np.concatenate(parts)
    result *= len(result) / result.sum()
    return result.astype(np.float32)


def predict_arrays(
    torch: Any,
    model: Any,
    combined: np.ndarray,
    prior: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    change_parts = []
    direction_parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(combined), batch_size):
            values = torch.as_tensor(
                (combined[start : start + batch_size] - mean) / scale,
                dtype=torch.float32,
                device=device,
            )
            prior_tensor = torch.as_tensor(
                prior[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            changes, logits = model(values, prior_tensor)
            change_parts.append(changes.cpu().numpy())
            direction_parts.append(logits.argmax(dim=-1).cpu().numpy())
    changes = np.concatenate(change_parts).astype(np.float32)
    directions = np.concatenate(direction_parts).astype(np.int64)
    zero = np.arange(len(changes)) % len(ACTION_GRID) == ZERO_ACTION_INDEX
    changes[zero] = 0.0
    directions[zero] = 1
    return changes, directions


def validation_metrics(
    arrays: DirectionArrays,
    changes: np.ndarray,
    directions: np.ndarray,
) -> dict[str, Any]:
    groups = arrays.group_count
    target = arrays.normalized_changes.reshape(groups, len(ACTION_GRID), 5)
    predicted = changes.reshape(groups, len(ACTION_GRID), 5)
    return {
        "forward": forward_metrics(target, predicted),
        "direction": direction_metrics(
            arrays.labels,
            directions,
            arrays.distance_bins,
        ),
        "direction_by_category": category_metrics(arrays, directions),
    }


def geometric_score(metrics: list[float]) -> float:
    return float(
        math.exp(np.mean(np.log(np.clip(metrics, 1e-8, 1.0))))
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if not 0.0 <= args.balance_strength <= 1.0:
        raise ValueError("balance strength must be between zero and one")
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(args.seed, args.device)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    old_train = load_grid_arrays(
        args.old_data.resolve() / "grids/train.jsonl",
        include_legacy_features=False,
        max_groups=args.max_old_train_groups,
    )
    difficult_train = load_grid_arrays(
        args.difficult_data.resolve() / "grids/train.jsonl",
        include_legacy_features=False,
        max_groups=args.max_difficult_train_groups,
    )
    additional_train = load_grid_arrays(
        args.additional_data.resolve() / "grids/train.jsonl",
        include_legacy_features=False,
        max_groups=args.max_additional_train_groups,
    )
    old_val = load_grid_arrays(
        args.old_data.resolve() / "grids/val.jsonl",
        include_legacy_features=False,
        max_groups=args.max_old_val_groups,
    )
    difficult_val = load_grid_arrays(
        args.difficult_data.resolve() / "grids/val.jsonl",
        include_legacy_features=False,
        max_groups=args.max_difficult_val_groups,
    )
    group_counts = [
        old_train.group_count,
        difficult_train.group_count,
        additional_train.group_count,
    ]
    train = concatenate_direction_arrays(
        [old_train, difficult_train, additional_train]
    )
    del old_train, difficult_train, additional_train

    base_forward, _ = load_forward_runtime_v5(args.base_forward.resolve())
    prior_train = prior_predictions(base_forward, train)
    prior_old = prior_predictions(base_forward, old_val)
    prior_difficult = prior_predictions(base_forward, difficult_val)
    combined_train = np.concatenate(
        [train.features, prior_train],
        axis=1,
    ).astype(np.float32)
    combined_old = np.concatenate(
        [old_val.features, prior_old],
        axis=1,
    ).astype(np.float32)
    combined_difficult = np.concatenate(
        [difficult_val.features, prior_difficult],
        axis=1,
    ).astype(np.float32)
    input_mean = combined_train.mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    input_scale = combined_train.std(axis=0, dtype=np.float64).astype(
        np.float32
    )
    input_scale = np.maximum(input_scale, 1e-6)
    base_weight = source_sample_weights(
        group_counts,
        [0.30, 0.30, 0.40],
    )
    balance, balance_report = balance_table(
        train.labels,
        train.distance_bins,
    )
    direction_weight = np.empty_like(
        train.normalized_changes,
        dtype=np.float32,
    )
    for field in range(5):
        stratum = balance[
            field,
            train.labels[:, field],
            train.distance_bins[:, field],
        ]
        direction_weight[:, field] = (
            args.balance_strength * stratum
            + (1.0 - args.balance_strength)
        )

    model_config = {
        "input_dim": int(combined_train.shape[1]),
        "width": int(args.width),
        "hidden_dim": int(args.hidden_dim),
        "residual_blocks": int(args.residual_blocks),
        "dropout": float(args.dropout),
    }
    model = shared_forward_direction_model(torch, **model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.learning_rate) * 0.08,
    )
    field_weight = torch.as_tensor(
        [1.20, 1.20, 0.80, 0.80, 1.50],
        dtype=torch.float32,
        device=device,
    )
    use_amp = device.type == "cuda"
    autocast_dtype = torch.bfloat16
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_key = (-math.inf, -math.inf)
    trace = []
    stale = 0

    baseline = {
        "forward_v5": {
            "old_iid": forward_metrics(
                old_val.normalized_changes.reshape(
                    old_val.group_count, len(ACTION_GRID), 5
                ),
                prior_old.reshape(
                    old_val.group_count, len(ACTION_GRID), 5
                ),
            ),
            "difficult": forward_metrics(
                difficult_val.normalized_changes.reshape(
                    difficult_val.group_count, len(ACTION_GRID), 5
                ),
                prior_difficult.reshape(
                    difficult_val.group_count, len(ACTION_GRID), 5
                ),
            ),
        },
        "thresholded_forward_v5": {
            "old_iid": direction_metrics(
                old_val.labels,
                labels_from_normalized_change(prior_old),
                old_val.distance_bins,
            ),
            "difficult": direction_metrics(
                difficult_val.labels,
                labels_from_normalized_change(prior_difficult),
                difficult_val.distance_bins,
            ),
        },
        "direction_tree_v4": {
            "old_iid": direction_metrics(
                old_val.labels,
                direction_reference_predictions(
                    args.direction_reference.resolve(), old_val
                ),
                old_val.distance_bins,
            ),
            "difficult": direction_metrics(
                difficult_val.labels,
                direction_reference_predictions(
                    args.direction_reference.resolve(), difficult_val
                ),
                difficult_val.distance_bins,
            ),
        },
    }
    baseline_primary = [
        baseline["forward_v5"]["old_iid"]["strict_all_five_success"],
        baseline["forward_v5"]["difficult"]["strict_all_five_success"],
        baseline["thresholded_forward_v5"]["old_iid"]["joint_exact"],
        baseline["thresholded_forward_v5"]["difficult"]["joint_exact"],
    ]
    best_key = (
        geometric_score(baseline_primary),
        float(np.mean(baseline_primary)),
    )

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(combined_train))
        sums = {
            "loss": 0.0,
            "regression": 0.0,
            "direction": 0.0,
            "joint": 0.0,
            "residual": 0.0,
        }
        samples = 0
        for start in range(0, len(order), int(args.batch_size)):
            selected = order[start : start + int(args.batch_size)]
            values = torch.as_tensor(
                (combined_train[selected] - input_mean) / input_scale,
                dtype=torch.float32,
                device=device,
            )
            prior = torch.as_tensor(
                prior_train[selected],
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(
                train.normalized_changes[selected],
                dtype=torch.float32,
                device=device,
            )
            labels = torch.as_tensor(
                train.labels[selected],
                dtype=torch.long,
                device=device,
            )
            sample_weight = torch.as_tensor(
                base_weight[selected],
                dtype=torch.float32,
                device=device,
            )
            class_weight = torch.as_tensor(
                direction_weight[selected],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_amp,
            ):
                changes, logits = model(values, prior)
                reg_components = torch.nn.functional.smooth_l1_loss(
                    changes,
                    target,
                    beta=0.5,
                    reduction="none",
                )
                regression = (
                    reg_components
                    * field_weight[None, :]
                    * sample_weight[:, None]
                ).mean()
                direction_components = torch.stack(
                    [
                        torch.nn.functional.cross_entropy(
                            logits[:, field],
                            labels[:, field],
                            reduction="none",
                        )
                        for field in range(5)
                    ],
                    dim=1,
                )
                direction = (
                    direction_components
                    * class_weight
                    * sample_weight[:, None]
                ).mean()
                maximum_error = (changes - target).abs().max(dim=1).values
                joint = (
                    torch.relu(maximum_error - 1.0) * sample_weight
                ).mean()
                residual = (
                    torch.square(changes - prior).mean(dim=1)
                    * sample_weight
                ).mean()
                loss = (
                    regression
                    + float(args.direction_loss_weight) * direction
                    + float(args.joint_loss_weight) * joint
                    + float(args.residual_loss_weight) * residual
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(selected)
            sums["loss"] += float(loss.detach()) * count
            sums["regression"] += float(regression.detach()) * count
            sums["direction"] += float(direction.detach()) * count
            sums["joint"] += float(joint.detach()) * count
            sums["residual"] += float(residual.detach()) * count
            samples += count
        scheduler.step()

        old_change, old_direction = predict_arrays(
            torch,
            model,
            combined_old,
            prior_old,
            input_mean,
            input_scale,
            device,
            int(args.batch_size) * 2,
        )
        difficult_change, difficult_direction = predict_arrays(
            torch,
            model,
            combined_difficult,
            prior_difficult,
            input_mean,
            input_scale,
            device,
            int(args.batch_size) * 2,
        )
        old_metrics = validation_metrics(
            old_val, old_change, old_direction
        )
        difficult_metrics = validation_metrics(
            difficult_val,
            difficult_change,
            difficult_direction,
        )
        primary = [
            old_metrics["forward"]["strict_all_five_success"],
            difficult_metrics["forward"]["strict_all_five_success"],
            old_metrics["direction"]["joint_exact"],
            difficult_metrics["direction"]["joint_exact"],
        ]
        key = (geometric_score(primary), float(np.mean(primary)))
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train": {
                name: value / samples for name, value in sums.items()
            },
            "validation": {
                "old_iid_forward_all_five": primary[0],
                "difficult_forward_all_five": primary[1],
                "old_iid_direction_joint": primary[2],
                "difficult_direction_joint": primary[3],
                "geometric_score": key[0],
            },
            "selected": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break

    model.load_state_dict(best_state)
    old_change, old_direction = predict_arrays(
        torch,
        model,
        combined_old,
        prior_old,
        input_mean,
        input_scale,
        device,
        int(args.batch_size) * 2,
    )
    difficult_change, difficult_direction = predict_arrays(
        torch,
        model,
        combined_difficult,
        prior_difficult,
        input_mean,
        input_scale,
        device,
        int(args.batch_size) * 2,
    )
    final_validation = {
        "old_iid": validation_metrics(
            old_val,
            old_change,
            old_direction,
        ),
        "difficult": validation_metrics(
            difficult_val,
            difficult_change,
            difficult_direction,
        ),
    }
    artifact_path = output_dir / "shared_forward_direction_v6.pt"
    artifact = {
        "version": "joint_forward_direction_v6_one_seed",
        "model": "shared_forward_direction_residual_v6",
        "seed": int(args.seed),
        "state_fields": list(STATE_FIELDS),
        "direction_fields": list(DIRECTION_FIELDS),
        "classes": list(CLASSES),
        "action_grid": ACTION_GRID,
        "zero_action_index": ZERO_ACTION_INDEX,
        "base_forward_artifact": str(args.base_forward.resolve()),
        "base_forward_sha256": sha256(args.base_forward.resolve()),
        "model_config": model_config,
        "input_mean": input_mean,
        "input_scale": input_scale,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "held_out_test_used": False,
    }
    torch.save(artifact, artifact_path)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "model": artifact["model"],
        "seed": int(args.seed),
        "model_config": model_config,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training": {
            "old_iid_groups": group_counts[0],
            "difficult_groups": group_counts[1],
            "additional_v5_groups": group_counts[2],
            "total_groups": train.group_count,
            "total_transitions": train.transition_count,
            "source_weighting": {
                "old_iid": 0.30,
                "difficult": 0.30,
                "additional_v5": 0.40,
            },
            "direction_balance": balance_report,
            "best_epoch": best_epoch,
            "trace": trace,
        },
        "validation": {
            "references": baseline,
            "shared_forward_direction_v6": final_validation,
        },
        "loss": {
            "regression": "field-weighted smooth_l1 in tolerance units",
            "regression_field_weights": {
                field: float(value)
                for field, value in zip(
                    STATE_FIELDS,
                    [1.20, 1.20, 0.80, 0.80, 1.50],
                    strict=True,
                )
            },
            "direction_loss_weight": float(args.direction_loss_weight),
            "joint_loss_weight": float(args.joint_loss_weight),
            "residual_loss_weight": float(args.residual_loss_weight),
            "balance_strength": float(args.balance_strength),
        },
        "source_contract": {
            "old_train": str(
                args.old_data.resolve() / "grids/train.jsonl"
            ),
            "old_val": str(args.old_data.resolve() / "grids/val.jsonl"),
            "difficult_train": str(
                args.difficult_data.resolve() / "grids/train.jsonl"
            ),
            "difficult_val": str(
                args.difficult_data.resolve() / "grids/val.jsonl"
            ),
            "additional_train": str(
                args.additional_data.resolve() / "grids/train.jsonl"
            ),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "shared_forward_direction_v6_summary.json"
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
                "validation": {
                    split: {
                        "forward_all_five": values["forward"][
                            "strict_all_five_success"
                        ],
                        "direction_joint_exact": values["direction"][
                            "joint_exact"
                        ],
                    }
                    for split, values in final_validation.items()
                },
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
