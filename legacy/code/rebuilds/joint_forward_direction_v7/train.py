#!/usr/bin/env python3
"""Train cross-fitted shared v7 with conservative direction correction."""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    direction_metrics,
    labels_from_normalized_change,
    load_grid_arrays,
    concatenate_direction_arrays,
)
from joint_forward_direction_v6.train import (
    direction_reference_predictions,
    geometric_score,
    prior_predictions,
    sha256,
    source_sample_weights,
)
from joint_forward_direction_v7.models import (
    shared_forward_direction_model_v7,
)
from joint_forward_direction_v7.runtime import gated_direction_indices
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    CLASSES,
    DIRECTION_FIELDS,
    STATE_FIELDS,
)

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_OOF_DATA = (
    REPO_ROOT.parent / "VLM_data/joint_forward_direction_v7"
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
    REPO_ROOT.parent / "VLM_runs/joint_forward_direction_v7_one_seed"
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
    parser.add_argument("--oof-data", type=Path, default=DEFAULT_OOF_DATA)
    parser.add_argument(
        "--targeted-data",
        type=Path,
        help=(
            "Optional new training-only grids that the frozen v5 base model "
            "has never seen; direct v5 predictions are valid priors for them"
        ),
    )
    parser.add_argument(
        "--base-forward", type=Path, default=DEFAULT_BASE_FORWARD
    )
    parser.add_argument(
        "--direction-reference", type=Path, default=DEFAULT_DIRECTION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--version",
        default="joint_forward_direction_v7_one_seed",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--direction-loss-weight", type=float, default=0.80)
    parser.add_argument("--joint-loss-weight", type=float, default=0.15)
    parser.add_argument("--residual-loss-weight", type=float, default=0.02)
    parser.add_argument("--balance-strength", type=float, default=0.20)
    parser.add_argument("--base-error-weight", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


def action_complexities() -> np.ndarray:
    actions = np.asarray(
        [
            [float(action[field]) for field in ACTION_FIELDS]
            for action in ACTION_GRID
        ],
        dtype=np.float32,
    )
    return np.count_nonzero(actions, axis=1).astype(np.int64)


def forward_by_complexity(
    target: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    complexity = action_complexities()
    result = {}
    for value in range(5):
        selected = complexity == value
        result[str(value)] = forward_metrics(
            target[:, selected],
            predicted[:, selected],
        )
    high = complexity >= 3
    result["three_or_four"] = forward_metrics(
        target[:, high],
        predicted[:, high],
    )
    return result


def direction_by_complexity(
    arrays: DirectionArrays,
    predicted: np.ndarray,
) -> dict[str, Any]:
    complexity = np.tile(action_complexities(), arrays.group_count)
    result = {}
    for value in range(5):
        selected = complexity == value
        result[str(value)] = direction_metrics(
            arrays.labels[selected],
            predicted[selected],
            arrays.distance_bins[selected],
        )
    high = complexity >= 3
    result["three_or_four"] = direction_metrics(
        arrays.labels[high],
        predicted[high],
        arrays.distance_bins[high],
    )
    return result


def predict_raw(
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
    logit_parts = []
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
            logit_parts.append(logits.cpu().numpy())
    return (
        np.concatenate(change_parts).astype(np.float32),
        np.concatenate(logit_parts).astype(np.float32),
    )


def forward_metric_bundle(
    arrays: DirectionArrays,
    changes: np.ndarray,
) -> dict[str, Any]:
    target = arrays.normalized_changes.reshape(
        arrays.group_count,
        len(ACTION_GRID),
        5,
    )
    predicted = changes.reshape(target.shape)
    return {
        "overall": forward_metrics(target, predicted),
        "by_action_complexity": forward_by_complexity(target, predicted),
    }


def direction_metric_bundle(
    arrays: DirectionArrays,
    directions: np.ndarray,
) -> dict[str, Any]:
    return {
        "overall": direction_metrics(
            arrays.labels,
            directions,
            arrays.distance_bins,
        ),
        "by_action_complexity": direction_by_complexity(
            arrays,
            directions,
        ),
    }


def calibrate(
    old_arrays: DirectionArrays,
    difficult_arrays: DirectionArrays,
    old_prior: np.ndarray,
    difficult_prior: np.ndarray,
    old_raw: np.ndarray,
    difficult_raw: np.ndarray,
    old_logits: np.ndarray,
    difficult_logits: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any], tuple[float, float]]:
    blends = (0.0, 0.25, 0.50, 0.75, 1.0)
    blend_rows = []
    for blend in blends:
        old_change = old_prior + blend * (old_raw - old_prior)
        difficult_change = difficult_prior + blend * (
            difficult_raw - difficult_prior
        )
        old_change[
            np.arange(len(old_change)) % len(ACTION_GRID)
            == ZERO_ACTION_INDEX
        ] = 0.0
        difficult_change[
            np.arange(len(difficult_change)) % len(ACTION_GRID)
            == ZERO_ACTION_INDEX
        ] = 0.0
        old_metrics = forward_metric_bundle(old_arrays, old_change)
        difficult_metrics = forward_metric_bundle(
            difficult_arrays,
            difficult_change,
        )
        values = [
            old_metrics["overall"]["strict_all_five_success"],
            difficult_metrics["overall"]["strict_all_five_success"],
            old_metrics["by_action_complexity"]["three_or_four"][
                "strict_all_five_success"
            ],
            difficult_metrics["by_action_complexity"]["three_or_four"][
                "strict_all_five_success"
            ],
        ]
        blend_rows.append(
            {
                "blend": float(blend),
                "score": geometric_score(values),
                "mean": float(np.mean(values)),
                "old": old_metrics,
                "difficult": difficult_metrics,
                "old_change": old_change,
                "difficult_change": difficult_change,
            }
        )
    selected_blend = max(
        blend_rows,
        key=lambda row: (row["score"], row["mean"], -row["blend"]),
    )

    gate_rows = []
    probabilities = (0.34, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 1.01)
    margins = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
    for probability in probabilities:
        for margin in margins:
            old_direction = gated_direction_indices(
                selected_blend["old_change"],
                old_logits,
                minimum_probability=probability,
                minimum_margin=margin,
            )
            difficult_direction = gated_direction_indices(
                selected_blend["difficult_change"],
                difficult_logits,
                minimum_probability=probability,
                minimum_margin=margin,
            )
            old_metrics = direction_metric_bundle(
                old_arrays,
                old_direction,
            )
            difficult_metrics = direction_metric_bundle(
                difficult_arrays,
                difficult_direction,
            )
            values = [
                old_metrics["overall"]["joint_exact"],
                difficult_metrics["overall"]["joint_exact"],
                old_metrics["by_action_complexity"]["three_or_four"][
                    "joint_exact"
                ],
                difficult_metrics["by_action_complexity"]["three_or_four"][
                    "joint_exact"
                ],
            ]
            gate_rows.append(
                {
                    "probability": float(probability),
                    "margin": float(margin),
                    "score": geometric_score(values),
                    "mean": float(np.mean(values)),
                    "old": old_metrics,
                    "difficult": difficult_metrics,
                }
            )
    selected_gate = max(
        gate_rows,
        key=lambda row: (
            row["score"],
            row["mean"],
            row["probability"],
            row["margin"],
        ),
    )
    calibration = {
        "residual_blend": float(selected_blend["blend"]),
        "direction_minimum_probability": float(
            selected_gate["probability"]
        ),
        "direction_minimum_margin": float(selected_gate["margin"]),
    }
    metrics = {
        "forward": {
            "old_iid": selected_blend["old"],
            "difficult": selected_blend["difficult"],
        },
        "direction": {
            "old_iid": selected_gate["old"],
            "difficult": selected_gate["difficult"],
        },
        "calibration_search": {
            "blend_candidates": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"old_change", "difficult_change"}
                }
                for row in blend_rows
            ],
            "gate_candidates": gate_rows,
        },
    }
    key = (
        geometric_score([selected_blend["score"], selected_gate["score"]]),
        float(np.mean([selected_blend["mean"], selected_gate["mean"]])),
    )
    return calibration, metrics, key


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(args.seed, args.device)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base_train_parts = [
        load_grid_arrays(
            args.old_data.resolve() / "grids/train.jsonl",
            include_legacy_features=False,
        ),
        load_grid_arrays(
            args.difficult_data.resolve() / "grids/train.jsonl",
            include_legacy_features=False,
        ),
        load_grid_arrays(
            args.additional_data.resolve() / "grids/train.jsonl",
            include_legacy_features=False,
        ),
    ]
    base_group_counts = [part.group_count for part in base_train_parts]
    base_train = concatenate_direction_arrays(base_train_parts)
    old_val = load_grid_arrays(
        args.old_data.resolve() / "grids/val.jsonl",
        include_legacy_features=False,
    )
    difficult_val = load_grid_arrays(
        args.difficult_data.resolve() / "grids/val.jsonl",
        include_legacy_features=False,
    )
    oof_path = args.oof_data.resolve() / "oof_forward_v5_predictions.npy"
    oof_metadata_path = (
        args.oof_data.resolve() / "oof_forward_v5_metadata.json"
    )
    oof_metadata = json.loads(oof_metadata_path.read_text())
    oof_prior = np.load(oof_path, allow_pickle=False)
    if (
        oof_metadata.get("complete") is not True
        or tuple(oof_prior.shape)
        != tuple(base_train.normalized_changes.shape)
        or sha256(oof_path) != oof_metadata["prediction_sha256"]
    ):
        raise ValueError("out-of-fold artifact contract differs")

    base_forward, _ = load_forward_runtime_v5(args.base_forward.resolve())
    targeted = None
    targeted_prior = None
    targeted_path = None
    if args.targeted_data is not None:
        targeted_path = args.targeted_data.resolve() / "grids/train.jsonl"
        targeted = load_grid_arrays(
            targeted_path,
            include_legacy_features=False,
        )
        targeted_prior = prior_predictions(base_forward, targeted)
        train = concatenate_direction_arrays([base_train, targeted])
        oof_prior = np.concatenate(
            [oof_prior, targeted_prior],
            axis=0,
        ).astype(np.float32)
        group_counts = [*base_group_counts, targeted.group_count]
        source_masses = [0.20, 0.20, 0.30, 0.30]
    else:
        train = base_train
        group_counts = base_group_counts
        source_masses = [0.30, 0.30, 0.40]
    del base_train_parts, base_train
    old_prior = prior_predictions(base_forward, old_val)
    difficult_prior = prior_predictions(base_forward, difficult_val)
    combined_train = np.concatenate(
        [train.features, oof_prior],
        axis=1,
    ).astype(np.float32)
    combined_old = np.concatenate(
        [old_val.features, old_prior],
        axis=1,
    ).astype(np.float32)
    combined_difficult = np.concatenate(
        [difficult_val.features, difficult_prior],
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
        source_masses,
    )
    balance, balance_report = balance_table(
        train.labels,
        train.distance_bins,
    )
    base_classes = labels_from_normalized_change(oof_prior)
    correction_weight = np.empty_like(
        train.normalized_changes,
        dtype=np.float32,
    )
    for field in range(5):
        stratum = balance[
            field,
            train.labels[:, field],
            train.distance_bins[:, field],
        ]
        balanced = (
            float(args.balance_strength) * stratum
            + (1.0 - float(args.balance_strength))
        )
        wrong = base_classes[:, field] != train.labels[:, field]
        correction_weight[:, field] = balanced * np.where(
            wrong,
            float(args.base_error_weight),
            1.0,
        )

    model_config = {
        "input_dim": int(combined_train.shape[1]),
        "width": int(args.width),
        "hidden_dim": int(args.hidden_dim),
        "residual_blocks": int(args.residual_blocks),
        "dropout": float(args.dropout),
    }
    model = shared_forward_direction_model_v7(
        torch,
        **model_config,
    ).to(device)
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
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    baseline_old_logits = np.zeros(
        (*old_prior.shape, 3),
        dtype=np.float32,
    )
    baseline_difficult_logits = np.zeros(
        (*difficult_prior.shape, 3),
        dtype=np.float32,
    )
    best_calibration, best_metrics, best_key = calibrate(
        old_val,
        difficult_val,
        old_prior,
        difficult_prior,
        old_prior,
        difficult_prior,
        baseline_old_logits,
        baseline_difficult_logits,
    )
    trace = []
    stale = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(combined_train))
        totals = {
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
                oof_prior[selected],
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
            direction_weight = torch.as_tensor(
                correction_weight[selected],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
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
                    * direction_weight
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
            for name, value in (
                ("loss", loss),
                ("regression", regression),
                ("direction", direction),
                ("joint", joint),
                ("residual", residual),
            ):
                totals[name] += float(value.detach()) * count
            samples += count
        scheduler.step()
        old_raw, old_logits = predict_raw(
            torch,
            model,
            combined_old,
            old_prior,
            input_mean,
            input_scale,
            device,
            int(args.batch_size) * 2,
        )
        difficult_raw, difficult_logits = predict_raw(
            torch,
            model,
            combined_difficult,
            difficult_prior,
            input_mean,
            input_scale,
            device,
            int(args.batch_size) * 2,
        )
        calibration, metrics, key = calibrate(
            old_val,
            difficult_val,
            old_prior,
            difficult_prior,
            old_raw,
            difficult_raw,
            old_logits,
            difficult_logits,
        )
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_calibration = calibration
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train": {
                name: value / samples for name, value in totals.items()
            },
            "calibration": calibration,
            "validation": {
                "forward_old": metrics["forward"]["old_iid"]["overall"][
                    "strict_all_five_success"
                ],
                "forward_difficult": metrics["forward"]["difficult"][
                    "overall"
                ]["strict_all_five_success"],
                "direction_old": metrics["direction"]["old_iid"]["overall"][
                    "joint_exact"
                ],
                "direction_difficult": metrics["direction"]["difficult"][
                    "overall"
                ]["joint_exact"],
                "selection_score": key[0],
            },
            "selected": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break

    model.load_state_dict(best_state)
    old_raw, old_logits = predict_raw(
        torch,
        model,
        combined_old,
        old_prior,
        input_mean,
        input_scale,
        device,
        int(args.batch_size) * 2,
    )
    difficult_raw, difficult_logits = predict_raw(
        torch,
        model,
        combined_difficult,
        difficult_prior,
        input_mean,
        input_scale,
        device,
        int(args.batch_size) * 2,
    )
    final_calibration, final_metrics, final_key = calibrate(
        old_val,
        difficult_val,
        old_prior,
        difficult_prior,
        old_raw,
        difficult_raw,
        old_logits,
        difficult_logits,
    )
    if final_calibration != best_calibration:
        raise RuntimeError("restored checkpoint calibration differs")

    direction_reference = {
        "old_iid": direction_metric_bundle(
            old_val,
            direction_reference_predictions(
                args.direction_reference.resolve(),
                old_val,
            ),
        ),
        "difficult": direction_metric_bundle(
            difficult_val,
            direction_reference_predictions(
                args.direction_reference.resolve(),
                difficult_val,
            ),
        ),
    }
    artifact_path = output_dir / "shared_forward_direction_v7.pt"
    artifact = {
        "version": str(args.version),
        "model": "cross_fitted_shared_forward_direction_v7",
        "seed": int(args.seed),
        "state_fields": list(STATE_FIELDS),
        "direction_fields": list(DIRECTION_FIELDS),
        "classes": list(CLASSES),
        "action_grid": ACTION_GRID,
        "zero_action_index": ZERO_ACTION_INDEX,
        "base_forward_artifact": str(args.base_forward.resolve()),
        "base_forward_sha256": sha256(args.base_forward.resolve()),
        "oof_metadata": str(oof_metadata_path),
        "oof_metadata_sha256": sha256(oof_metadata_path),
        "oof_predictions_sha256": sha256(oof_path),
        "targeted_training_file": (
            str(targeted_path) if targeted_path is not None else None
        ),
        "targeted_training_sha256": (
            sha256(targeted_path) if targeted_path is not None else None
        ),
        "model_config": model_config,
        "calibration": final_calibration,
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
        "seed": int(args.seed),
        "model": artifact["model"],
        "model_config": model_config,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "calibration": final_calibration,
        "training": {
            "old_iid_groups": group_counts[0],
            "difficult_groups": group_counts[1],
            "additional_v5_groups": group_counts[2],
            "targeted_groups": (
                group_counts[3] if len(group_counts) == 4 else 0
            ),
            "source_masses": source_masses,
            "total_groups": train.group_count,
            "total_transitions": train.transition_count,
            "out_of_fold_forward": oof_metadata,
            "base_error_weight": float(args.base_error_weight),
            "direction_balance": balance_report,
            "best_epoch": best_epoch,
            "trace": trace,
        },
        "validation": {
            "references": {
                "forward_v5": {
                    "old_iid": forward_metric_bundle(old_val, old_prior),
                    "difficult": forward_metric_bundle(
                        difficult_val,
                        difficult_prior,
                    ),
                },
                "thresholded_forward_v5": {
                    "old_iid": direction_metric_bundle(
                        old_val,
                        labels_from_normalized_change(old_prior),
                    ),
                    "difficult": direction_metric_bundle(
                        difficult_val,
                        labels_from_normalized_change(difficult_prior),
                    ),
                },
                "direction_tree_v4": direction_reference,
            },
            "shared_forward_direction_v7": final_metrics,
            "selection_key": list(final_key),
        },
        "source_contract": {
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "shared_forward_direction_v7_summary.json"
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
                "calibration": final_calibration,
                "validation": {
                    "forward_old": final_metrics["forward"]["old_iid"][
                        "overall"
                    ]["strict_all_five_success"],
                    "forward_difficult": final_metrics["forward"][
                        "difficult"
                    ]["overall"]["strict_all_five_success"],
                    "direction_old": final_metrics["direction"]["old_iid"][
                        "overall"
                    ]["joint_exact"],
                    "direction_difficult": final_metrics["direction"][
                        "difficult"
                    ]["overall"]["joint_exact"],
                },
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
