#!/usr/bin/env python3
"""Train a leakage-free explicit 81-action forward-surface Transformer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.action_surface_transformer import (
    build_action_surface_transformer,
)
from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    group_partitions,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import sha256
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS, fixed_action_grid

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_CACHE = DEFAULT_RUN / "clean_nonoverlap_forward_training_features.npz"
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "action_surface_transformer_v9.pt"

BLENDS = np.asarray(
    [-0.25, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.25],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--base-forward", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--feedforward", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.abs(prediction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=2)
    return {
        "count": int(exact.size),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": passed.mean(axis=(0, 1)).tolist(),
    }


def selection_key(
    prior: np.ndarray,
    correction: np.ndarray,
    target: np.ndarray,
    blend: np.ndarray,
) -> tuple[int, int, float, float]:
    error = np.abs(
        prior + correction * blend[None, None, :] - target
    )
    passed = error <= 1.0
    return (
        int(np.all(passed, axis=2).sum()),
        int(passed.sum()),
        -float(error.mean()),
        -float(np.abs(blend).sum()),
    )


def choose_blend(
    prior: np.ndarray,
    correction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected = np.ones(5, dtype=np.float32)
    trace = []
    for pass_index in range(6):
        changed = False
        for field in range(5):
            candidates = []
            for value in BLENDS:
                proposal = selected.copy()
                proposal[field] = value
                candidates.append(
                    (
                        selection_key(prior, correction, target, proposal),
                        -abs(float(value)),
                        proposal,
                    )
                )
            best = max(candidates, key=lambda row: row[:2])
            if not np.array_equal(selected, best[2]):
                selected = best[2]
                changed = True
        trace.append(
            {
                "pass": pass_index + 1,
                "blend": selected.tolist(),
                "metric": metric(
                    prior + correction * selected[None, None, :],
                    target,
                ),
            }
        )
        if not changed:
            break
    return selected, trace


def predict(
    torch: Any,
    model: Any,
    context: np.ndarray,
    actions: np.ndarray,
    residual_scale: np.ndarray,
    groups: np.ndarray,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(groups), batch_size):
            indices = groups[start : start + batch_size]
            raw = model(
                torch.as_tensor(
                    context[indices], dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    actions[indices], dtype=torch.float32, device=device
                ),
            )
            output.append(raw.float().cpu().numpy())
    values = np.concatenate(output, axis=0)
    values *= residual_scale[None, None, :]
    values[:, 40, :] = 0.0
    return values.astype(np.float32)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    torch, device = configure(int(args.seed), args.device)
    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        flat_features = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        target = np.asarray(cache["grid_target_normalized"], dtype=np.float32)
        residual = np.asarray(cache["grid_residual_target"], dtype=np.float32)
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    group_count = len(group_ids)
    if len(flat_features) != group_count * 81:
        raise ValueError("cache does not contain 81 transitions per group")
    features = flat_features.reshape(group_count, 81, -1)
    prior = prior.reshape(group_count, 81, 5)
    target = target.reshape(group_count, 81, 5)
    residual = residual.reshape(group_count, 81, 5)
    training, calibration, confirmation = group_partitions(
        group_ids, int(args.seed)
    )
    training_rows = features[training].reshape(-1, features.shape[2])
    feature_mean = training_rows.mean(axis=0)
    feature_scale = training_rows.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    standardized = (
        features - feature_mean[None, None, :]
    ) / feature_scale[None, None, :]
    context = standardized[:, 0, :17].astype(np.float32)
    actions = standardized[:, :, 17:].astype(np.float32)
    residual_scale = residual[training].reshape(-1, 5).std(axis=0)
    residual_scale[residual_scale < 1e-5] = 1.0
    config = {
        "context_dim": 17,
        "action_dim": int(features.shape[2] - 17),
        "action_count": 81,
        "dimension": int(args.dimension),
        "heads": int(args.heads),
        "layers": int(args.layers),
        "feedforward": int(args.feedforward),
        "dropout": float(args.dropout),
        "runtime_batch_size": 64,
    }
    model = build_action_surface_transformer(torch, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.learning_rate) * 0.05,
    )
    field_weight = torch.as_tensor(
        [1.15, 1.15, 0.85, 0.85, 1.60],
        dtype=torch.float32,
        device=device,
    )
    complexity = np.asarray(
        [
            sum(abs(float(action[field])) > 0.0 for field in ACTION_FIELDS)
            for action in fixed_action_grid()
        ],
        dtype=np.float32,
    )
    action_weight = torch.as_tensor(
        1.0 + 0.12 * complexity,
        dtype=torch.float32,
        device=device,
    )
    residual_scale_tensor = torch.as_tensor(
        residual_scale, dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(int(args.seed) + 901)
    best_key = None
    best_state = None
    best_epoch = 0
    stale = 0
    trace = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        shuffled = rng.permutation(training)
        running = 0.0
        for start in range(0, len(shuffled), int(args.batch_size)):
            indices = shuffled[start : start + int(args.batch_size)]
            context_tensor = torch.as_tensor(
                context[indices], dtype=torch.float32, device=device
            )
            action_tensor = torch.as_tensor(
                actions[indices], dtype=torch.float32, device=device
            )
            prior_tensor = torch.as_tensor(
                prior[indices], dtype=torch.float32, device=device
            )
            target_tensor = torch.as_tensor(
                target[indices], dtype=torch.float32, device=device
            )
            raw = model(context_tensor, action_tensor)
            correction = raw * residual_scale_tensor
            prediction = prior_tensor + correction
            error = prediction - target_tensor
            smooth = torch.nn.functional.smooth_l1_loss(
                prediction, target_tensor, reduction="none", beta=0.25
            )
            boundary = torch.nn.functional.softplus(
                (error.abs() - 0.85) * 5.0
            ) / 5.0
            joint = torch.logsumexp(error.abs() * 3.0, dim=2) / 3.0
            weights = (
                action_weight[None, :, None] * field_weight[None, None, :]
            )
            loss = (
                (smooth * weights).mean()
                + 0.35 * (boundary * weights).mean()
                + 0.18 * (joint * action_weight[None, :]).mean()
                + 0.001 * raw.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
        scheduler.step()
        correction = predict(
            torch,
            model,
            context,
            actions,
            residual_scale,
            calibration,
            int(args.batch_size),
            device,
        )
        current_metrics = metric(
            prior[calibration] + correction, target[calibration]
        )
        key = (
            current_metrics["strict_all_five_count"],
            sum(current_metrics["per_field_tolerance_pass"]),
            -current_metrics["mae_in_tolerance_units"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            record = {
                "epoch": epoch,
                "train_loss": running / len(training),
                "calibration": current_metrics,
                "best_epoch": best_epoch,
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("no Transformer checkpoint was selected")
    model.load_state_dict(best_state)
    calibration_correction = predict(
        torch,
        model,
        context,
        actions,
        residual_scale,
        calibration,
        int(args.batch_size),
        device,
    )
    blend, blend_trace = choose_blend(
        prior[calibration],
        calibration_correction,
        target[calibration],
    )
    confirmation_correction = predict(
        torch,
        model,
        context,
        actions,
        residual_scale,
        confirmation,
        int(args.batch_size),
        device,
    )
    confirmation_base = metric(prior[confirmation], target[confirmation])
    confirmation_candidate = metric(
        prior[confirmation]
        + confirmation_correction * blend[None, None, :],
        target[confirmation],
    )
    artifact = {
        "version": "action_surface_transformer_v9_one_seed",
        "model": "explicit_81_action_surface_transformer_v9",
        "seed": int(args.seed),
        "state_dict": best_state,
        "config": config,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "residual_scale": residual_scale,
        "field_blend": blend,
        "base_forward": str(args.base_forward.resolve()),
        "base_forward_sha256": sha256(args.base_forward.resolve()),
        "training_cache": str(args.cache.resolve()),
        "training_cache_sha256": sha256(args.cache.resolve()),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "group_split": {
            "training": int(len(training)),
            "calibration": int(len(calibration)),
            "internal_confirmation": int(len(confirmation)),
        },
        "best_epoch": int(best_epoch),
        "selected_blend": blend.tolist(),
        "calibration": {
            "base": metric(prior[calibration], target[calibration]),
            "candidate": metric(
                prior[calibration]
                + calibration_correction * blend[None, None, :],
                target[calibration],
            ),
        },
        "internal_confirmation": {
            "base": confirmation_base,
            "candidate": confirmation_candidate,
            "strict_count_delta": int(
                confirmation_candidate["strict_all_five_count"]
                - confirmation_base["strict_all_five_count"]
            ),
        },
        "trace": trace,
        "blend_trace": blend_trace,
        "source_contract": {
            "generated_setups": 0,
            "generated_images": 0,
            "qwen_direct_group_overlap": 0,
            "protected_validation_used": False,
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
