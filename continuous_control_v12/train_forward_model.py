#!/usr/bin/env python3
"""Train and serialize the first numerical v12 residual forward ensemble."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.contracts import Bounds, split_hash
from continuous_control_v12.evaluation import forward_metrics
from continuous_control_v12.schema import read_jsonl, validate_dataset
from continuous_control_v12.world_model import (
    ForwardEnsemble,
    build_member,
    transition_arrays,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_v12.json")
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/continuous_control_v12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--overfit", action="store_true")
    parser.add_argument(
        "--train-groups",
        type=int,
        help="Use the first N sorted train group IDs (nested learning curve).",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--image-conditioning",
        action="store_true",
        help="Clean one-step ablation using fixed pooled current-intensity features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    validation = validate_dataset(data_dir, config)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_rows = read_jsonl(data_dir / "transitions/train.jsonl")
    development_rows = read_jsonl(
        data_dir / "transitions/development.jsonl"
    )
    if args.train_groups is not None:
        if args.overfit:
            raise ValueError("--train-groups cannot be combined with --overfit")
        if args.train_groups < 1:
            raise ValueError("--train-groups must be positive")
        available_groups = sorted(
            {str(row["group_id"]) for row in train_rows}
        )
        if args.train_groups > len(available_groups):
            raise ValueError(
                f"requested {args.train_groups} train groups but dataset has "
                f"{len(available_groups)}"
            )
        selected_groups = set(available_groups[: args.train_groups])
        train_rows = [
            row
            for row in train_rows
            if str(row["group_id"]) in selected_groups
        ]
    if args.overfit:
        train_rows = train_rows[
            : int(config["smoke"]["forward_overfit_transitions"])
        ]
        development_rows = train_rows
    bounds = Bounds.from_config(config)
    image_conditioning = bool(
        args.image_conditioning or config["forward_model"]["image_conditioning"]
    )
    train = transition_arrays(
        train_rows,
        bounds,
        data_dir=data_dir,
        image_conditioning=image_conditioning,
    )
    development = transition_arrays(
        development_rows,
        bounds,
        data_dir=data_dir,
        image_conditioning=image_conditioning,
    )
    mean = train["features"].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train["features"].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x_train = ((train["features"] - mean) / scale).astype(np.float32)
    x_dev = ((development["features"] - mean) / scale).astype(np.float32)
    model_config = dict(config["forward_model"])
    if args.overfit:
        epochs = int(config["smoke"]["forward_overfit_epochs"])
        patience = int(config["smoke"]["forward_overfit_patience"])
        model_config["ensemble_members"] = 1
    elif args.smoke:
        epochs = int(config["smoke"]["forward_epochs"])
        patience = int(config["smoke"]["forward_patience"])
    else:
        epochs = int(model_config["epochs"])
        patience = int(model_config["patience"])
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        epochs = int(args.epochs)
    if args.patience is not None:
        if args.patience < 1:
            raise ValueError("--patience must be positive")
        patience = int(args.patience)
    batch_size = min(int(model_config["batch_size"]), len(x_train))
    seed = int(config["seed"] if args.seed is None else args.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    tx = torch.as_tensor(x_train, dtype=torch.float32)
    ty = torch.as_tensor(train["targets"], dtype=torch.float32)
    taux = torch.as_tensor(train["auxiliary"], dtype=torch.float32)
    tmask = torch.as_tensor(train["auxiliary_mask"], dtype=torch.bool)
    tnonzero = torch.as_tensor(~train["no_op"], dtype=torch.float32)
    dx = torch.as_tensor(x_dev, dtype=torch.float32)
    dy = torch.as_tensor(development["targets"], dtype=torch.float32)
    dnonzero = torch.as_tensor(~development["no_op"], dtype=torch.float32)
    weights = model_config["auxiliary_loss_weights"]
    member_states = []
    member_reports = []
    for member_index in range(int(model_config["ensemble_members"])):
        member_seed = seed + member_index * 1009
        torch.manual_seed(member_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)
        member = build_member(torch, x_train.shape[1], model_config).to(device)
        optimizer = torch.optim.AdamW(
            member.parameters(),
            lr=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
        )
        generator = torch.Generator().manual_seed(member_seed)
        best_state = None
        best_mae = float("inf")
        best_epoch = -1
        stale = 0
        history = []
        for epoch in range(epochs):
            member.train()
            order = torch.randperm(len(tx), generator=generator)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                metric_delta, log_variance, auxiliary = member(
                    tx[indices].to(device),
                    tnonzero[indices].to(device),
                )
                target = ty[indices].to(device)
                regression = torch.nn.functional.smooth_l1_loss(
                    metric_delta,
                    target,
                    beta=float(model_config["huber_beta"]),
                )
                uncertainty = 0.5 * (
                    (metric_delta - target).square() * torch.exp(-log_variance)
                    + log_variance
                ).mean()
                loss = regression + 0.01 * uncertainty
                mask = tmask[indices].to(device)
                aux_target = taux[indices].to(device)
                if mask[:, 0].any():
                    loss = loss + float(weights["captured_power"]) * (
                        torch.nn.functional.smooth_l1_loss(
                            auxiliary[mask[:, 0], 0],
                            aux_target[mask[:, 0], 0],
                        )
                    )
                if mask[:, 1].any():
                    loss = loss + float(weights["clipping_fraction"]) * (
                        torch.nn.functional.smooth_l1_loss(
                            auxiliary[mask[:, 1], 1],
                            aux_target[mask[:, 1], 1],
                        )
                    )
                for field, key in ((2, "camera_boundary"), (3, "actuator_limit")):
                    if mask[:, field].any():
                        loss = loss + float(weights[key]) * (
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                auxiliary[mask[:, field], field],
                                aux_target[mask[:, field], field],
                            )
                        )
                if float(model_config["soft_boundary_weight"]) > 0.0:
                    smooth_max = torch.logsumexp(
                        torch.abs(metric_delta - target) * 6.0, dim=1
                    ) / 6.0
                    loss = loss + float(
                        model_config["soft_boundary_weight"]
                    ) * torch.nn.functional.softplus(smooth_max - 1.0).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            member.eval()
            with torch.inference_mode():
                fit_prediction_epoch, _, _ = member(
                    tx.to(device), tnonzero.to(device)
                )
                prediction, _, _ = member(
                    dx.to(device), dnonzero.to(device)
                )
                fit_mae = float(
                    torch.abs(
                        fit_prediction_epoch - ty.to(device)
                    ).mean().cpu()
                )
                mae = float(torch.abs(prediction - dy.to(device)).mean().cpu())
            history.append(
                {
                    "epoch": epoch + 1,
                    "fit_normalized_mae": fit_mae,
                    "development_normalized_mae": mae,
                }
            )
            if mae < best_mae - 1e-6:
                best_mae = mae
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in member.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("ensemble member did not produce a checkpoint")
        member_states.append(best_state)
        member_reports.append(
            {
                "member": member_index,
                "seed": member_seed,
                "best_epoch": best_epoch,
                "epochs_executed": epoch + 1,
                "development_normalized_mae": best_mae,
                "history": history,
            }
        )
    artifact = {
        "version": "continuous_forward_model_v12",
        "schema_version": config["schema_version"],
        "model_config": model_config,
        "input_dim": int(x_train.shape[1]),
        "feature_mean": mean,
        "feature_scale": scale,
        "member_states": member_states,
        "bounds": {
            "action_low": bounds.action_low,
            "action_high": bounds.action_high,
            "position_low": bounds.position_low,
            "position_high": bounds.position_high,
        },
        "output_fields": list(
            (
                "centroid_x_px",
                "centroid_y_px",
                "sigma_x_px",
                "sigma_y_px",
                "peak_intensity",
            )
        ),
        "target": "tolerance_normalized_metric_delta",
        "image_conditioning": image_conditioning,
        "q_star_in_inputs": False,
        "train_group_hash": split_hash(sorted(set(train["group_ids"]))),
        "development_group_hash": split_hash(
            sorted(set(development["group_ids"]))
        ),
        "train_groups": len(set(train["group_ids"])),
        "development_groups": len(set(development["group_ids"])),
        "training_seed": seed,
    }
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "continuous_forward_v12_overfit"
        if args.overfit
        else "continuous_forward_v12_smoke"
        if args.smoke
        else (
            f"continuous_forward_v12_{args.train_groups}g"
            if args.train_groups is not None
            else "continuous_forward_v12"
        )
    )
    checkpoint_path = run_dir / f"{stem}.pt"
    report_path = run_dir / f"{stem}.json"
    if checkpoint_path.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite v12 forward run: {stem}")
    torch.save(artifact, checkpoint_path)
    runtime = ForwardEnsemble(artifact, torch, device)

    def predict(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        predictions, uncertainty = [], []
        for row in rows:
            current_image = None
            if image_conditioning:
                with np.load(
                    data_dir / str(row["image_ref"]), allow_pickle=False
                ) as image:
                    current_image = np.asarray(image["intensity"])
            result = runtime.predict(
                row["setup_context"],
                row["positions_mm"],
                row["metrics"],
                [
                    [
                        row["action_mm"]["lens_x_delta_mm"],
                        row["action_mm"]["lens_y_delta_mm"],
                        row["action_mm"]["camera_x_delta_mm"],
                        row["action_mm"]["camera_y_delta_mm"],
                    ]
                ],
                current_image=current_image,
            )
            predictions.append(result["next_metric_residual"][0])
            uncertainty.append(result["uncertainty"][0])
        return np.asarray(predictions), np.asarray(uncertainty)

    fit_prediction, fit_uncertainty = predict(train_rows)
    dev_prediction, dev_uncertainty = predict(development_rows)
    group_regimes = manifest["group_regimes"]
    fit_metrics = forward_metrics(
        fit_prediction,
        train["targets"],
        group_ids=train["group_ids"],
        regimes=[group_regimes[group] for group in train["group_ids"]],
        sampling=train["sampling"],
        uncertainty=fit_uncertainty,
        no_op=train["no_op"],
    )
    dev_metrics = forward_metrics(
        dev_prediction,
        development["targets"],
        group_ids=development["group_ids"],
        regimes=[group_regimes[group] for group in development["group_ids"]],
        sampling=development["sampling"],
        uncertainty=dev_uncertainty,
        no_op=development["no_op"],
    )
    loaded = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    reloaded = ForwardEnsemble(loaded, torch, device)
    serialization_image = None
    if image_conditioning:
        with np.load(
            data_dir / str(development_rows[0]["image_ref"]),
            allow_pickle=False,
        ) as image:
            serialization_image = np.asarray(image["intensity"])
    serialization_probe = reloaded.predict(
        development_rows[0]["setup_context"],
        development_rows[0]["positions_mm"],
        development_rows[0]["metrics"],
        [[development_rows[0]["action_mm"][field] for field in (
            "lens_x_delta_mm",
            "lens_y_delta_mm",
            "camera_x_delta_mm",
            "camera_y_delta_mm",
        )]],
        current_image=serialization_image,
    )
    report = {
        "version": "continuous_forward_model_v12",
        "scale": "overfit" if args.overfit else "smoke" if args.smoke else "full",
        "checkpoint": str(checkpoint_path),
        "member_reports": member_reports,
        "train_groups": len(set(train["group_ids"])),
        "development_groups": len(set(development["group_ids"])),
        "training_seed": seed,
        "parameter_count_per_member": int(
            sum(
                value.numel()
                for value in build_member(
                    torch, x_train.shape[1], model_config
                ).parameters()
            )
        ),
        "fit": fit_metrics,
        "development": dev_metrics,
        "dataset_validation": validation,
        "serialization_inference": {
            "passed": bool(
                np.isfinite(
                    serialization_probe["predicted_next_metrics"]
                ).all()
            ),
            "shape": list(
                serialization_probe["predicted_next_metrics"].shape
            ),
        },
        "q_star_in_inputs": False,
        "scientific_claim": "diagnostic_only_no_comparison_to_v9"
        if args.smoke or args.overfit
        else "requires_group_level_evaluation",
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
