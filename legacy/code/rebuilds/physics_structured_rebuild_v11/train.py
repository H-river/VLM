#!/usr/bin/env python3
"""Train one isolated v11 baseline/ablation on independent optical groups."""

from __future__ import annotations

import argparse
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

from physics_structured_rebuild_v11.contracts import (
    LEARNING_CURVE_SIZES,
    OVERFIT_GROUP_SIZES,
    forward_metrics,
    rows_to_arrays,
    select_group_rows,
    split_hash,
)
from physics_structured_rebuild_v11.models import build_model

DEFAULT_CONFIG = Path(__file__).with_name("config_v11.json")
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ablation", default="baseline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--group-count", type=int)
    mode.add_argument("--overfit-groups", type=int, choices=OVERFIT_GROUP_SIZES)
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--development-groups", type=int)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--training-source", type=Path, action="append")
    parser.add_argument("--development-source", type=Path, action="append")
    return parser.parse_args()


def prior_design(arrays: dict[str, Any], representation: str) -> np.ndarray:
    if representation == "structured":
        return np.asarray(arrays["features"], dtype=np.float32)
    one_hot = np.eye(81, dtype=np.float32)[arrays["action_indices"]]
    return np.concatenate([arrays["contexts"], one_hot], axis=1).astype(np.float32)


def fit_oof_prior(
    train: dict[str, Any],
    development: dict[str, Any],
    representation: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = prior_design(train, representation)
    x_development = prior_design(development, representation)
    target = np.asarray(train["targets"], dtype=np.float32)
    groups = np.asarray(train["group_indices"], dtype=np.int64)
    group_ids = train["group_ids"]
    fold_by_group = np.asarray(
        [
            int.from_bytes(
                __import__("hashlib").sha256(
                    f"{seed}:{group_id}:prior_fold".encode()
                ).digest()[:8],
                "big",
            )
            % 5
            for group_id in group_ids
        ],
        dtype=np.int64,
    )
    oof = np.zeros_like(target)
    folds_used = []
    for fold in sorted(set(fold_by_group.tolist())):
        validation_groups = np.flatnonzero(fold_by_group == fold)
        validation_mask = np.isin(groups, validation_groups)
        training_mask = ~validation_mask
        if not validation_mask.any() or not training_mask.any():
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train[training_mask], target[training_mask])
        oof[validation_mask] = model.predict(x_train[validation_mask]).astype(
            np.float32
        )
        folds_used.append(int(fold))
    if np.any(~np.isfinite(oof)):
        raise ValueError("OOF prior contains non-finite values")
    final = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    final.fit(x_train, target)
    development_prior = final.predict(x_development).astype(np.float32)
    zero = train["action_indices"] == 40
    oof[zero] = 0.0
    development_prior[development["action_indices"] == 40] = 0.0
    return oof, development_prior, {
        "kind": "five_fold_group_oof_standardized_ridge",
        "folds_used": folds_used,
        "alpha": 1.0,
    }


def standardize(
    train_values: np.ndarray,
    development_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return (
        ((train_values - mean) / scale).astype(np.float32),
        ((development_values - mean) / scale).astype(np.float32),
        mean,
        scale,
    )


def parity_gate(
    fit: dict[str, Any],
    development: dict[str, Any],
    config: dict[str, Any],
    *,
    gate_eligible: bool = True,
) -> dict[str, Any]:
    gate = config["parity_gate"]
    checks = {
        "fit_normalized_mae": (
            fit["normalized_mae"]
            <= float(gate["reference_fit_normalized_mae"])
            * float(gate["fit_normalized_mae_max_ratio"])
        ),
        "development_normalized_mae": (
            development["normalized_mae"]
            <= float(gate["reference_development_normalized_mae"])
            * float(gate["development_normalized_mae_max_ratio"])
        ),
        "fit_strict_accuracy": (
            fit["strict_all_five_accuracy"]
            >= float(gate["reference_fit_full_surface_strict"])
            * float(gate["fit_strict_accuracy_min_ratio"])
        ),
        "development_strict_accuracy": (
            development["strict_all_five_accuracy"]
            >= float(gate["reference_development_full_surface_strict"])
            * float(gate["development_strict_accuracy_min_ratio"])
        ),
    }
    passed = all(checks.values()) and gate_eligible
    return {
        "reference": gate["reference_name"],
        "checks": checks,
        "passed": passed,
        "ablation_interpretation": "interpretable" if passed else "not_interpretable",
        "gate_eligible_run": gate_eligible,
        "reason": (
            "all_parity_checks_passed"
            if passed
            else "diagnostic_mode_cannot_unlock_interpretation_gate"
            if not gate_eligible
            else "one_or_more_parity_checks_failed"
        ),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.ablation not in config["ablations"]:
        raise ValueError(f"unknown ablation: {args.ablation}")
    ablation = config["ablations"][args.ablation]
    seed = int(config["seed"])
    if args.smoke:
        group_count = int(config["smoke"]["groups"])
        development_count = int(config["smoke"]["development_groups"])
        epochs = int(config["smoke"]["epochs"])
        patience = int(config["smoke"]["patience"])
        batch_size = int(config["smoke"]["batch_size"])
        mode = "smoke"
    elif args.overfit_groups is not None:
        group_count = int(args.overfit_groups)
        development_count = group_count
        epochs = int(config["optimization"]["epochs"])
        patience = int(config["optimization"]["patience"])
        batch_size = int(config["optimization"]["batch_size"])
        mode = "overfit"
    else:
        group_count = int(args.group_count)
        if group_count not in LEARNING_CURVE_SIZES:
            raise ValueError(
                f"group-count must be one of {list(LEARNING_CURVE_SIZES)}"
            )
        development_count = int(args.development_groups or 600)
        epochs = int(config["optimization"]["epochs"])
        patience = int(config["optimization"]["patience"])
        batch_size = int(config["optimization"]["batch_size"])
        mode = "learning_curve"
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.ablation}_{mode}_{group_count}_seed{seed}"
    checkpoint_path = run_dir / f"{stem}.pt"
    report_path = run_dir / f"{stem}.json"
    if checkpoint_path.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite completed v11 run: {stem}")

    training_paths = (
        list(args.training_source)
        if args.training_source
        else [
            Path(value)
            for value in config["distribution_sources"][
                ablation["distribution"]
            ]
        ]
    )
    development_paths = (
        list(args.development_source)
        if args.development_source
        else [Path(value) for value in config["development_sources"]]
    )
    train_rows, train_provenance = select_group_rows(
        training_paths, group_count, seed
    )
    if mode == "overfit":
        development_rows = train_rows
        development_provenance = train_provenance
    else:
        development_rows, development_provenance = select_group_rows(
            development_paths, development_count, seed + 1
        )
    train = rows_to_arrays(train_rows, seed)
    development = rows_to_arrays(development_rows, seed)
    representation = str(ablation["action_representation"])
    train_prior, development_prior, prior_report = fit_oof_prior(
        train, development, representation, seed
    )
    if mode == "overfit":
        development_prior = train_prior.copy()
        prior_report["overfit_development_uses_training_oof_prior"] = True
    structured_train, structured_dev, structured_mean, structured_scale = standardize(
        train["features"], development["features"]
    )
    context_train, context_dev, context_mean, context_scale = standardize(
        train["contexts"], development["contexts"]
    )

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    model = build_model(
        torch,
        representation=representation,
        structured_input_dim=structured_train.shape[1],
        context_input_dim=context_train.shape[1],
        config=config["model"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    tensors = {
        "structured": torch.as_tensor(structured_train, dtype=torch.float32),
        "context": torch.as_tensor(context_train, dtype=torch.float32),
        "action": torch.as_tensor(train["action_indices"], dtype=torch.long),
        "prior": torch.as_tensor(train_prior, dtype=torch.float32),
        "target": torch.as_tensor(train["targets"], dtype=torch.float32),
        "raw_target": torch.as_tensor(train["raw_targets"], dtype=torch.float32),
        "tolerance": torch.as_tensor(train["tolerances"], dtype=torch.float32),
    }
    development_tensors = {
        "structured": torch.as_tensor(structured_dev, dtype=torch.float32),
        "context": torch.as_tensor(context_dev, dtype=torch.float32),
        "action": torch.as_tensor(
            development["action_indices"], dtype=torch.long
        ),
        "prior": torch.as_tensor(development_prior, dtype=torch.float32),
    }
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_value = float("inf")
    best_epoch = -1
    stale = 0

    def predict(values: dict[str, Any]) -> np.ndarray:
        model.eval()
        output = []
        with torch.inference_mode():
            for start in range(0, len(values["action"]), batch_size):
                stop = start + batch_size
                output.append(
                    model(
                        values["structured"][start:stop].to(device),
                        values["context"][start:stop].to(device),
                        values["action"][start:stop].to(device),
                        values["prior"][start:stop].to(device),
                    )
                    .cpu()
                    .numpy()
                )
        return np.concatenate(output)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tensors["action"]), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            prediction = model(
                tensors["structured"][indices].to(device),
                tensors["context"][indices].to(device),
                tensors["action"][indices].to(device),
                tensors["prior"][indices].to(device),
            )
            if ablation["loss"] == "ordinary_raw":
                predicted_value = prediction * tensors["tolerance"][indices].to(
                    device
                )
                target_value = tensors["raw_target"][indices].to(device)
            else:
                predicted_value = prediction
                target_value = tensors["target"][indices].to(device)
            loss = torch.nn.functional.smooth_l1_loss(
                predicted_value,
                target_value,
                beta=float(config["optimization"]["huber_beta"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        development_prediction = predict(development_tensors)
        value = float(
            np.abs(development_prediction - development["targets"]).mean()
        )
        if value < best_value - 1e-6:
            best_value = value
            best_epoch = epoch + 1
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("v11 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_tensors = {
        "structured": tensors["structured"],
        "context": tensors["context"],
        "action": tensors["action"],
        "prior": tensors["prior"],
    }
    fit_prediction = predict(train_tensors)
    development_prediction = predict(development_tensors)
    fit_metrics = forward_metrics(fit_prediction, train["targets"], train)
    development_metrics = forward_metrics(
        development_prediction, development["targets"], development
    )
    gate = parity_gate(
        fit_metrics,
        development_metrics,
        config,
        gate_eligible=mode == "learning_curve",
    )
    artifact = {
        "version": config["version"],
        "model_state": best_state,
        "model_config": config["model"],
        "representation": representation,
        "structured_mean": structured_mean,
        "structured_scale": structured_scale,
        "context_mean": context_mean,
        "context_scale": context_scale,
        "seed": seed,
        "q_star_in_inputs": False,
    }
    torch.save(artifact, checkpoint_path)
    report = {
        "version": config["version"],
        "ablation": args.ablation,
        "ablation_contract": ablation,
        "mode": mode,
        "seed": seed,
        "group_count": group_count,
        "development_group_count": development_count,
        "best_epoch": best_epoch,
        "epochs_executed": epoch + 1,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "prior": prior_report,
        "fit": fit_metrics,
        "development": development_metrics,
        "parity_gate": gate,
        "train_split_hash": split_hash(train["group_ids"]),
        "development_split_hash": split_hash(development["group_ids"]),
        "train_provenance": train_provenance,
        "development_provenance": development_provenance,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_claim": (
            "diagnostic_only" if mode in {"smoke", "overfit"} else "gate_required"
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
