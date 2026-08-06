#!/usr/bin/env python3
"""Train one shared forward-direction TabM or feature-token Transformer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from control_rebuild_v3.train_forward import configure
from control_rebuild_v5.forward_runtime import ZERO_ACTION_INDEX
from direction_rebuild_v4.data import (
    DirectionArrays,
    balance_table,
    concatenate_direction_arrays,
    labels_from_normalized_change,
    load_grid_arrays,
)
from joint_forward_direction_v6.train import (
    geometric_score,
    source_sample_weights,
)
from joint_forward_direction_v7.train import (
    direction_metric_bundle,
    forward_metric_bundle,
)
from tabm_transformer_rebuild_v8.models import (
    build_forward_direction_model,
)

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_runs/tabm_transformer_rebuild_v8_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("tabm", "transformer"),
        required=True,
    )
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument(
        "--difficult-data",
        type=Path,
        default=DEFAULT_DIFFICULT_DATA,
    )
    parser.add_argument(
        "--additional-data",
        type=Path,
        default=DEFAULT_ADDITIONAL_DATA,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-old-train-groups", type=int)
    parser.add_argument("--max-difficult-train-groups", type=int)
    parser.add_argument("--max-additional-train-groups", type=int)
    parser.add_argument("--max-validation-groups", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architecture_config(name: str) -> dict[str, Any]:
    if name == "tabm":
        return {
            "k": 16,
            "n_blocks": 3,
            "d_block": 256,
            "dropout": 0.05,
        }
    return {
        "dimension": 128,
        "heads": 8,
        "feedforward": 384,
        "layers": 3,
        "dropout": 0.05,
    }


def ensemble_view(
    changes: Any,
    logits: Any,
) -> tuple[Any, Any]:
    if changes.ndim == 2:
        changes = changes[:, None, :]
        logits = logits[:, None, :, :]
    return changes, logits


def predict(
    torch: Any,
    model: Any,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    changes = []
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            values = torch.as_tensor(
                (features[start : start + batch_size] - mean) / scale,
                dtype=torch.float32,
                device=device,
            )
            change, logits = ensemble_view(*model(values))
            changes.append(change.float().mean(dim=1).cpu().numpy())
            probabilities.append(
                logits.float().softmax(dim=-1).mean(dim=1).cpu().numpy()
            )
    output_change = np.concatenate(changes).astype(np.float32)
    output_probability = np.concatenate(probabilities).astype(np.float32)
    zero = np.arange(len(output_change)) % len(ACTION_GRID) == ZERO_ACTION_INDEX
    output_change[zero] = 0.0
    output_probability[zero] = 0.0
    output_probability[zero, :, 1] = 1.0
    return output_change, output_probability


def calibrate_direction(
    old_arrays: DirectionArrays,
    difficult_arrays: DirectionArrays,
    old_changes: np.ndarray,
    difficult_changes: np.ndarray,
    old_probabilities: np.ndarray,
    difficult_probabilities: np.ndarray,
) -> tuple[float, dict[str, Any], tuple[float, float]]:
    rows = []
    for learned_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        predictions = {}
        metrics = {}
        for name, arrays, changes, learned in (
            (
                "old_iid",
                old_arrays,
                old_changes,
                old_probabilities,
            ),
            (
                "difficult",
                difficult_arrays,
                difficult_changes,
                difficult_probabilities,
            ),
        ):
            threshold = labels_from_normalized_change(changes)
            threshold_probability = np.eye(3, dtype=np.float32)[threshold]
            mixed = (
                learned_weight * learned
                + (1.0 - learned_weight) * threshold_probability
            )
            predictions[name] = mixed.argmax(axis=-1).astype(np.int64)
            metrics[name] = direction_metric_bundle(
                arrays,
                predictions[name],
            )
        values = [
            metrics["old_iid"]["overall"]["joint_exact"],
            metrics["difficult"]["overall"]["joint_exact"],
            metrics["old_iid"]["by_action_complexity"]["three_or_four"][
                "joint_exact"
            ],
            metrics["difficult"]["by_action_complexity"]["three_or_four"][
                "joint_exact"
            ],
        ]
        rows.append(
            {
                "learned_probability_weight": float(learned_weight),
                "score": geometric_score(values),
                "mean": float(np.mean(values)),
                "metrics": metrics,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["score"],
            row["mean"],
            -row["learned_probability_weight"],
        ),
    )
    return (
        float(selected["learned_probability_weight"]),
        {
            **selected["metrics"],
            "calibration_candidates": rows,
        },
        (float(selected["score"]), float(selected["mean"])),
    )


def evaluate(
    torch: Any,
    model: Any,
    old_val: DirectionArrays,
    difficult_val: DirectionArrays,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, float], tuple[float, float]]:
    old_changes, old_probabilities = predict(
        torch,
        model,
        old_val.features,
        mean,
        scale,
        device,
        batch_size,
    )
    difficult_changes, difficult_probabilities = predict(
        torch,
        model,
        difficult_val.features,
        mean,
        scale,
        device,
        batch_size,
    )
    direction_weight, direction_metrics, direction_key = calibrate_direction(
        old_val,
        difficult_val,
        old_changes,
        difficult_changes,
        old_probabilities,
        difficult_probabilities,
    )
    forward = {
        "old_iid": forward_metric_bundle(old_val, old_changes),
        "difficult": forward_metric_bundle(
            difficult_val,
            difficult_changes,
        ),
    }
    forward_values = [
        forward["old_iid"]["overall"]["strict_all_five_success"],
        forward["difficult"]["overall"]["strict_all_five_success"],
        forward["old_iid"]["by_action_complexity"]["three_or_four"][
            "strict_all_five_success"
        ],
        forward["difficult"]["by_action_complexity"]["three_or_four"][
            "strict_all_five_success"
        ],
    ]
    forward_key = (
        geometric_score(forward_values),
        float(np.mean(forward_values)),
    )
    key = (
        geometric_score([forward_key[0], direction_key[0]]),
        float(np.mean([forward_key[1], direction_key[1]])),
    )
    return (
        {
            "forward": forward,
            "direction": {
                "old_iid": direction_metrics["old_iid"],
                "difficult": direction_metrics["difficult"],
            },
            "calibration_search": direction_metrics[
                "calibration_candidates"
            ],
        },
        {
            "direction_learned_probability_weight": direction_weight,
        },
        key,
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve() / args.architecture
    summary_path = output_dir / "forward_direction_summary.json"
    if summary_path.is_file():
        raise RuntimeError(f"refusing to overwrite completed run: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(int(args.seed), args.device)
    random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    train_parts = [
        load_grid_arrays(
            args.old_data.resolve() / "grids/train.jsonl",
            include_legacy_features=False,
            max_groups=args.max_old_train_groups,
        ),
        load_grid_arrays(
            args.difficult_data.resolve() / "grids/train.jsonl",
            include_legacy_features=False,
            max_groups=args.max_difficult_train_groups,
        ),
        load_grid_arrays(
            args.additional_data.resolve() / "grids/train.jsonl",
            include_legacy_features=False,
            max_groups=args.max_additional_train_groups,
        ),
    ]
    group_counts = [part.group_count for part in train_parts]
    train = concatenate_direction_arrays(train_parts)
    del train_parts
    old_val = load_grid_arrays(
        args.old_data.resolve() / "grids/val.jsonl",
        include_legacy_features=False,
        max_groups=args.max_validation_groups,
    )
    difficult_val = load_grid_arrays(
        args.difficult_data.resolve() / "grids/val.jsonl",
        include_legacy_features=False,
        max_groups=args.max_validation_groups,
    )
    mean = train.features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    sample_weight = source_sample_weights(
        group_counts,
        [0.30, 0.30, 0.40],
    )
    direction_balance, balance_report = balance_table(
        train.labels,
        train.distance_bins,
    )
    direction_weight = np.empty_like(
        train.normalized_changes,
        dtype=np.float32,
    )
    for field in range(5):
        direction_weight[:, field] = direction_balance[
            field,
            train.labels[:, field],
            train.distance_bins[:, field],
        ]

    config = architecture_config(args.architecture)
    model = build_forward_direction_model(
        torch,
        args.architecture,
        train.features.shape[1],
        config,
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
    batch_size = int(
        args.batch_size
        or (2048 if args.architecture == "tabm" else 1024)
    )
    use_amp = device.type == "cuda"
    field_weight = torch.as_tensor(
        [1.20, 1.20, 0.80, 0.80, 1.50],
        dtype=torch.float32,
        device=device,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_key = (-float("inf"), -float("inf"))
    best_metrics = None
    best_calibration = None
    stale = 0
    trace = []

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(train.transition_count)
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            values = torch.as_tensor(
                (train.features[selected] - mean) / scale,
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
            weight = torch.as_tensor(
                sample_weight[selected],
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
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                change, logits = ensemble_view(*model(values))
                ensemble_count = change.shape[1]
                regression_components = torch.nn.functional.smooth_l1_loss(
                    change,
                    target[:, None, :].expand_as(change),
                    beta=0.5,
                    reduction="none",
                )
                regression = (
                    regression_components
                    * field_weight[None, None, :]
                    * weight[:, None, None]
                ).mean()
                repeated_labels = labels[:, None, :].expand(
                    -1,
                    ensemble_count,
                    -1,
                )
                direction_components = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 3),
                    repeated_labels.reshape(-1),
                    reduction="none",
                ).reshape(len(selected), ensemble_count, 5)
                direction = (
                    direction_components
                    * class_weight[:, None, :]
                    * weight[:, None, None]
                ).mean()
                maximum_error = (
                    change - target[:, None, :]
                ).abs().max(dim=-1).values
                joint = (
                    torch.relu(maximum_error - 1.0)
                    * weight[:, None]
                ).mean()
                loss = regression + 0.70 * direction + 0.15 * joint
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(selected)
            seen += len(selected)
        scheduler.step()
        metrics, calibration, key = evaluate(
            torch,
            model,
            old_val,
            difficult_val,
            mean,
            scale,
            device,
            batch_size * 2,
        )
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = metrics
            best_calibration = calibration
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "loss": total / seen,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "forward_old": metrics["forward"]["old_iid"]["overall"][
                "strict_all_five_success"
            ],
            "forward_difficult": metrics["forward"]["difficult"]["overall"][
                "strict_all_five_success"
            ],
            "direction_old": metrics["direction"]["old_iid"]["overall"][
                "joint_exact"
            ],
            "direction_difficult": metrics["direction"]["difficult"][
                "overall"
            ]["joint_exact"],
            "direction_weight": calibration[
                "direction_learned_probability_weight"
            ],
            "selection_score": key[0],
            "selected": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break

    model.load_state_dict(best_state)
    final_metrics, final_calibration, final_key = evaluate(
        torch,
        model,
        old_val,
        difficult_val,
        mean,
        scale,
        device,
        batch_size * 2,
    )
    if final_calibration != best_calibration:
        raise RuntimeError("restored forward-direction calibration differs")
    artifact_path = output_dir / "forward_direction.pt"
    artifact = {
        "version": f"{args.architecture}_forward_direction_v8_one_seed",
        "model": f"{args.architecture}_shared_forward_direction_v8",
        "architecture": args.architecture,
        "architecture_config": config,
        "input_dim": int(train.features.shape[1]),
        "input_mean": mean,
        "input_scale": scale,
        "calibration": final_calibration,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "held_out_test_used": False,
    }
    torch.save(artifact, artifact_path)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "architecture": args.architecture,
        "architecture_config": config,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "dependency_versions": {
            "torch": torch.__version__,
            "tabm": "0.0.3" if args.architecture == "tabm" else None,
        },
        "training": {
            "seed": int(args.seed),
            "groups": {
                "old_iid": group_counts[0],
                "difficult": group_counts[1],
                "additional_v5": group_counts[2],
                "total": train.group_count,
            },
            "transitions": train.transition_count,
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            "direction_balance": balance_report,
            "trace": trace,
        },
        "validation": final_metrics,
        "calibration": final_calibration,
        "selection_key": list(final_key),
        "source_contract": {
            "old_validation": str(
                args.old_data.resolve() / "grids/val.jsonl"
            ),
            "difficult_validation": str(
                args.difficult_data.resolve() / "grids/val.jsonl"
            ),
            "targeted_partial_shards_used": False,
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
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
