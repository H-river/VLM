#!/usr/bin/env python3
"""Train one physics-structured forward model with auxiliary direction heads."""

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
    load_grid_arrays,
)
from joint_forward_direction_v6.train import (
    action_complexity_weight,
    geometric_score,
)
from tabm_transformer_rebuild_v8.train_forward_direction import (
    evaluate,
)
from physics_structured_rebuild_v9.models import (
    build_structured_forward_model,
)

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_TARGETED_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/targeted/train.jsonl"
)
DEFAULT_TARGETED_MANIFEST = DEFAULT_TARGETED_DATA.parent / "manifest.json"
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("balanced", "hard_peak"),
        default="balanced",
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
    parser.add_argument(
        "--targeted-data",
        type=Path,
        default=DEFAULT_TARGETED_DATA,
    )
    parser.add_argument(
        "--targeted-manifest",
        type=Path,
        default=DEFAULT_TARGETED_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-old-train-groups", type=int)
    parser.add_argument("--max-difficult-train-groups", type=int)
    parser.add_argument("--max-additional-train-groups", type=int)
    parser.add_argument("--max-targeted-train-groups", type=int)
    parser.add_argument("--max-validation-groups", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architecture_config() -> dict[str, Any]:
    return {
        "dimension": 128,
        "heads": 8,
        "feedforward": 384,
        "layers": 4,
        "head_hidden": 64,
        "dropout": 0.04,
    }


def profile_config(name: str) -> dict[str, Any]:
    if name == "hard_peak":
        return {
            "source_mass": (1 / 6, 1 / 6, 1 / 6, 0.50),
            "field_weight": (1.20, 1.20, 0.90, 0.90, 2.00),
            "direction_loss_weight": 0.45,
            "joint_loss_weight": 0.35,
            "next_loss_weight": 0.10,
            "consistency_loss_weight": 0.10,
        }
    return {
        "source_mass": (0.20, 0.20, 0.20, 0.40),
        "field_weight": (1.20, 1.20, 0.80, 0.80, 1.50),
        "direction_loss_weight": 0.55,
        "joint_loss_weight": 0.25,
        "next_loss_weight": 0.10,
        "consistency_loss_weight": 0.10,
    }


def source_weights(
    parts: list[DirectionArrays],
    masses: tuple[float, float, float, float],
) -> np.ndarray:
    action_weight = action_complexity_weight()
    output = []
    for source_index, (part, mass) in enumerate(
        zip(parts, masses, strict=True)
    ):
        if source_index < 3:
            group_weight = np.full(
                part.group_count,
                float(mass) / part.group_count,
                dtype=np.float32,
            )
        else:
            categories = part.category_indices.reshape(
                part.group_count,
                len(ACTION_GRID),
            )[:, 0]
            unique = np.unique(categories)
            group_weight = np.empty(part.group_count, dtype=np.float32)
            for category in unique:
                mask = categories == category
                group_weight[mask] = (
                    float(mass) / len(unique) / int(mask.sum())
                )
        output.append(
            (group_weight[:, None] * action_weight[None, :]).reshape(-1)
        )
    weights = np.concatenate(output).astype(np.float32)
    weights *= len(weights) / weights.sum()
    return weights


def transformed_next_states(arrays: DirectionArrays) -> np.ndarray:
    current = arrays.features[:, 12:17].astype(np.float32)
    raw_peak = np.expm1(current[:, 4]).astype(np.float32)
    tolerance = np.column_stack(
        [
            np.ones(len(current), dtype=np.float32),
            np.ones(len(current), dtype=np.float32),
            np.full(len(current), 2.0, dtype=np.float32),
            np.full(len(current), 2.0, dtype=np.float32),
            np.maximum(0.05 * np.abs(raw_peak), 1e-6),
        ]
    )
    delta = arrays.normalized_changes * tolerance
    output = current.copy()
    output[:, :4] += delta[:, :4]
    output[:, 4] = np.log1p(np.maximum(raw_peak + delta[:, 4], 0.0))
    return output.astype(np.float32)


def predicted_next_from_change(
    torch: Any,
    current_transformed: Any,
    predicted_change: Any,
    next_mean: Any,
    next_scale: Any,
) -> Any:
    raw_peak = torch.expm1(current_transformed[:, 4:5])
    tolerance = torch.cat(
        [
            torch.ones_like(current_transformed[:, :2]),
            torch.full_like(current_transformed[:, 2:4], 2.0),
            torch.clamp(raw_peak.abs() * 0.05, min=1e-6),
        ],
        dim=1,
    )
    delta = predicted_change * tolerance
    first = current_transformed[:, :4] + delta[:, :4]
    peak = torch.log1p(torch.clamp(raw_peak + delta[:, 4:5], min=0.0))
    transformed = torch.cat([first, peak], dim=1)
    return (transformed - next_mean) / next_scale


def forward_selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    values = [
        metrics["forward"]["old_iid"]["overall"][
            "strict_all_five_success"
        ],
        metrics["forward"]["difficult"]["overall"][
            "strict_all_five_success"
        ],
        metrics["forward"]["old_iid"]["by_action_complexity"][
            "three_or_four"
        ]["strict_all_five_success"],
        metrics["forward"]["difficult"]["by_action_complexity"][
            "three_or_four"
        ]["strict_all_five_success"],
    ]
    return geometric_score(values), float(np.mean(values))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve() / args.profile
    summary_path = output_dir / "forward_summary.json"
    if summary_path.exists():
        raise RuntimeError(f"refusing to overwrite completed run: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        args.targeted_manifest.resolve().read_text(encoding="utf-8")
    )
    if manifest.get("complete") is not True:
        raise ValueError("targeted manifest is incomplete")
    if sha256(args.targeted_data.resolve()) != manifest["train_sha256"]:
        raise ValueError("targeted training checksum differs from manifest")
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
        load_grid_arrays(
            args.targeted_data.resolve(),
            include_legacy_features=False,
            max_groups=args.max_targeted_train_groups,
        ),
    ]
    group_counts = [part.group_count for part in train_parts]
    profile = profile_config(args.profile)
    sample_weight = source_weights(train_parts, profile["source_mass"])
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
    next_targets = transformed_next_states(train)
    next_mean_np = next_targets.mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    next_scale_np = next_targets.std(axis=0, dtype=np.float64).astype(
        np.float32
    )
    next_scale_np = np.maximum(next_scale_np, 1e-6)
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

    config = architecture_config()
    model = build_structured_forward_model(torch, config).to(device)
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
    use_amp = device.type == "cuda"
    batch_size = int(args.batch_size)
    field_weight = torch.as_tensor(
        profile["field_weight"],
        dtype=torch.float32,
        device=device,
    )
    next_mean = torch.as_tensor(
        next_mean_np,
        dtype=torch.float32,
        device=device,
    )
    next_scale = torch.as_tensor(
        next_scale_np,
        dtype=torch.float32,
        device=device,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_key = (-float("inf"), -float("inf"))
    best_calibration = None
    stale = 0
    trace = []

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(train.transition_count)
        totals = {
            "loss": 0.0,
            "regression": 0.0,
            "direction": 0.0,
            "joint": 0.0,
            "next": 0.0,
            "consistency": 0.0,
        }
        seen = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            raw_features = torch.as_tensor(
                train.features[selected],
                dtype=torch.float32,
                device=device,
            )
            values = (
                raw_features
                - torch.as_tensor(mean, device=device)
            ) / torch.as_tensor(scale, device=device)
            target = torch.as_tensor(
                train.normalized_changes[selected],
                dtype=torch.float32,
                device=device,
            )
            target_next = torch.as_tensor(
                (next_targets[selected] - next_mean_np) / next_scale_np,
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
                change, next_prediction, logits = model.forward_with_aux(values)
                regression_components = (
                    torch.nn.functional.smooth_l1_loss(
                        change,
                        target,
                        beta=0.5,
                        reduction="none",
                    )
                )
                regression = (
                    regression_components
                    * field_weight[None, :]
                    * weight[:, None]
                ).mean()
                direction_components = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 3),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape(len(selected), 5)
                direction = (
                    direction_components
                    * class_weight
                    * weight[:, None]
                ).mean()
                maximum_error = (change - target).abs().max(dim=1).values
                joint = (
                    torch.relu(maximum_error - 1.0) * weight
                ).mean()
                next_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        next_prediction,
                        target_next,
                        beta=0.5,
                        reduction="none",
                    ).mean(dim=1)
                    * weight
                ).mean()
                implied_next = predicted_next_from_change(
                    torch,
                    raw_features[:, 12:17],
                    change,
                    next_mean,
                    next_scale,
                )
                consistency = (
                    torch.nn.functional.smooth_l1_loss(
                        next_prediction,
                        implied_next,
                        beta=0.5,
                        reduction="none",
                    ).mean(dim=1)
                    * weight
                ).mean()
                loss = (
                    regression
                    + float(profile["direction_loss_weight"]) * direction
                    + float(profile["joint_loss_weight"]) * joint
                    + float(profile["next_loss_weight"]) * next_loss
                    + float(profile["consistency_loss_weight"]) * consistency
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
                ("next", next_loss),
                ("consistency", consistency),
            ):
                totals[name] += float(value.detach()) * count
            seen += count
        scheduler.step()
        metrics, calibration, _ = evaluate(
            torch,
            model,
            old_val,
            difficult_val,
            mean,
            scale,
            device,
            batch_size * 2,
        )
        key = forward_selection_key(metrics)
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_calibration = calibration
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            **{name: value / seen for name, value in totals.items()},
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "forward_old": metrics["forward"]["old_iid"]["overall"][
                "strict_all_five_success"
            ],
            "forward_difficult": metrics["forward"]["difficult"]["overall"][
                "strict_all_five_success"
            ],
            "forward_old_three_or_four": metrics["forward"]["old_iid"][
                "by_action_complexity"
            ]["three_or_four"]["strict_all_five_success"],
            "forward_difficult_three_or_four": metrics["forward"][
                "difficult"
            ]["by_action_complexity"]["three_or_four"][
                "strict_all_five_success"
            ],
            "direction_old": metrics["direction"]["old_iid"]["overall"][
                "joint_exact"
            ],
            "direction_difficult": metrics["direction"]["difficult"][
                "overall"
            ]["joint_exact"],
            "direction_probability_weight": calibration[
                "direction_learned_probability_weight"
            ],
            "selection_key": list(key),
            "selected": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break

    model.load_state_dict(best_state)
    final_metrics, final_calibration, _ = evaluate(
        torch,
        model,
        old_val,
        difficult_val,
        mean,
        scale,
        device,
        batch_size * 2,
    )
    final_key = forward_selection_key(final_metrics)
    if final_calibration != best_calibration:
        raise RuntimeError("restored calibration differs from selected epoch")
    artifact_path = output_dir / "forward.pt"
    artifact = {
        "version": f"physics_structured_forward_v9_{args.profile}_one_seed",
        "model": "physics_structured_forward_v9",
        "profile": args.profile,
        "architecture_config": config,
        "profile_config": profile,
        "input_dim": 46,
        "input_mean": mean,
        "input_scale": scale,
        "next_mean": next_mean_np,
        "next_scale": next_scale_np,
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
        "profile": args.profile,
        "architecture_config": config,
        "profile_config": profile,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training": {
            "seed": int(args.seed),
            "groups": {
                "old_iid": group_counts[0],
                "difficult": group_counts[1],
                "additional_v5": group_counts[2],
                "targeted": group_counts[3],
                "total": int(sum(group_counts)),
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
            "targeted_manifest": str(args.targeted_manifest.resolve()),
            "targeted_manifest_sha256": sha256(
                args.targeted_manifest.resolve()
            ),
            "old_validation": str(
                args.old_data.resolve() / "grids/val.jsonl"
            ),
            "difficult_validation": str(
                args.difficult_data.resolve() / "grids/val.jsonl"
            ),
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
