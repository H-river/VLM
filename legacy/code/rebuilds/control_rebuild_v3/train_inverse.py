#!/usr/bin/env python3
"""Train residual-corrected numerical inverse control on all 81 actions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    ACTION_GRID,
    MOVEMENT,
    inverse_candidate_features,
    inverse_pair_arrays,
    read_json,
    read_jsonl,
)
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.models import inverse_ranker_model
from control_rebuild_v3.train_forward import configure, iter_batches


DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=DEFAULT_DATA)
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-val-pairs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def macro_f1(target: np.ndarray, predicted: np.ndarray) -> float:
    values = []
    for label in range(3):
        true_positive = np.sum((target == label) & (predicted == label))
        false_positive = np.sum((target != label) & (predicted == label))
        false_negative = np.sum((target == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(values))


def inverse_metrics(
    scores: np.ndarray,
    status_logits: np.ndarray,
    positives: np.ndarray,
    statuses: np.ndarray,
    selected_truth: np.ndarray,
) -> dict[str, Any]:
    adjusted = np.asarray(scores) - 1e-7 * MOVEMENT[None, :]
    selected = adjusted.argmax(axis=1)
    feasible = positives.any(axis=1)
    success = positives[np.arange(len(positives)), selected]
    predicted_status = status_logits.argmax(axis=1)
    exact = selected == selected_truth
    return {
        "pair_count": int(len(positives)),
        "reachable_pair_count": int(feasible.sum()),
        "target_success_feasible": float(success[feasible].mean()),
        "target_success_all": float(success.mean()),
        "minimum_movement_exact_feasible": float(exact[feasible].mean()),
        "status_accuracy": float((predicted_status == statuses).mean()),
        "status_macro_f1": macro_f1(statuses, predicted_status),
    }


def feature_statistics(
    states: np.ndarray,
    group_indices: np.ndarray,
    desired: np.ndarray,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_sum = None
    candidate_square_sum = None
    status_sum = None
    status_square_sum = None
    candidate_count = 0
    status_count = 0
    for start in range(0, len(group_indices), chunk_size):
        index = slice(start, start + chunk_size)
        candidate, _, status = inverse_candidate_features(
            states[group_indices[index]], desired[index]
        )
        flat = candidate.reshape(-1, candidate.shape[-1]).astype(np.float64)
        status64 = status.astype(np.float64)
        if candidate_sum is None:
            candidate_sum = flat.sum(axis=0)
            candidate_square_sum = np.square(flat).sum(axis=0)
            status_sum = status64.sum(axis=0)
            status_square_sum = np.square(status64).sum(axis=0)
        else:
            candidate_sum += flat.sum(axis=0)
            candidate_square_sum += np.square(flat).sum(axis=0)
            status_sum += status64.sum(axis=0)
            status_square_sum += np.square(status64).sum(axis=0)
        candidate_count += len(flat)
        status_count += len(status64)
    candidate_mean = candidate_sum / candidate_count
    candidate_variance = np.maximum(
        candidate_square_sum / candidate_count - np.square(candidate_mean), 0.0
    )
    status_mean = status_sum / status_count
    status_variance = np.maximum(
        status_square_sum / status_count - np.square(status_mean), 0.0
    )
    candidate_scale = np.sqrt(candidate_variance)
    status_scale = np.sqrt(status_variance)
    candidate_scale[candidate_scale < 1e-7] = 1.0
    status_scale[status_scale < 1e-7] = 1.0
    return (
        candidate_mean.astype(np.float32),
        candidate_scale.astype(np.float32),
        status_mean.astype(np.float32),
        status_scale.astype(np.float32),
    )


def predict_corrections(
    torch: Any,
    model: Any,
    contexts: np.ndarray,
    states: np.ndarray,
    group_indices: np.ndarray,
    desired: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_scale: np.ndarray,
    status_mean: np.ndarray,
    status_scale: np.ndarray,
    device: Any,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrections, logits, base_scores = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            index = slice(start, start + batch_size)
            candidate, cost, status = inverse_candidate_features(
                states[group_indices[index]], desired[index]
            )
            candidate = (candidate - candidate_mean) / candidate_scale
            status = (status - status_mean) / status_scale
            correction, status_logits = model(
                torch.as_tensor(
                    contexts[index], dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    candidate, dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    status, dtype=torch.float32, device=device
                ),
            )
            corrections.append(correction.float().cpu().numpy())
            logits.append(status_logits.float().cpu().numpy())
            base_scores.append(-cost)
    return (
        np.concatenate(corrections),
        np.concatenate(logits),
        np.concatenate(base_scores),
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    seed = int(config["seed"])
    torch, device = configure(seed, args.device)
    started = time.perf_counter()

    train_rows = read_jsonl(data_dir / "grids/train.jsonl")
    val_rows = read_jsonl(data_dir / "grids/val.jsonl")
    train_map = {str(row["group_id"]): row for row in train_rows}
    val_map = {str(row["group_id"]): row for row in val_rows}
    train_ids = list(train_map)
    val_ids = list(val_map)
    train_position = {group_id: index for index, group_id in enumerate(train_ids)}
    val_position = {group_id: index for index, group_id in enumerate(val_ids)}
    train_pairs = read_jsonl(data_dir / "inverse/train.jsonl")
    val_pairs = read_jsonl(data_dir / "inverse/val.jsonl")
    if args.max_train_pairs is not None:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs is not None:
        val_pairs = val_pairs[: args.max_val_pairs]

    forward_path = output_dir / "forward_control_v3_calibrated.pt"
    forward, _ = load_forward_runtime(forward_path, torch, device)
    train_states = forward.predict_states(train_rows)
    val_states = forward.predict_states(val_rows)

    gt, ct, dt, pt, st = inverse_pair_arrays(
        train_pairs, train_map, train_position
    )
    gv, cv, dv, pv, sv = inverse_pair_arrays(
        val_pairs, val_map, val_position
    )
    selected_t = np.asarray(
        [
            -1
            if row["selected_index"] is None
            else int(row["selected_index"])
            for row in train_pairs
        ],
        dtype=np.int64,
    )
    selected_v = np.asarray(
        [
            -1
            if row["selected_index"] is None
            else int(row["selected_index"])
            for row in val_pairs
        ],
        dtype=np.int64,
    )

    context_mean = ct.mean(axis=0)
    context_scale = ct.std(axis=0)
    context_scale[context_scale < 1e-7] = 1.0
    ct = ((ct - context_mean) / context_scale).astype(np.float32)
    cv = ((cv - context_mean) / context_scale).astype(np.float32)
    candidate_mean, candidate_scale, status_mean, status_scale = (
        feature_statistics(train_states, gt, dt)
    )
    sample_candidate, _, sample_status = inverse_candidate_features(
        train_states[gt[:1]], dt[:1]
    )
    model = inverse_ranker_model(
        torch,
        context_dim=ct.shape[1],
        candidate_dim=sample_candidate.shape[-1],
        status_dim=sample_status.shape[-1],
    ).to(device)
    inverse_config = config["inverse"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(inverse_config["learning_rate"]),
        weight_decay=float(inverse_config["weight_decay"]),
    )
    status_counts = Counter(st.tolist())
    status_weights = torch.as_tensor(
        [
            math.sqrt(len(st) / max(3 * status_counts.get(index, 0), 1))
            for index in range(3)
        ],
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(seed + 202)
    epochs = int(args.epochs or inverse_config["epochs"])
    batch_size = int(inverse_config["batch_pairs"])
    alpha_candidates = [
        float(value) for value in inverse_config["alpha_candidates"]
    ]
    best_score = float("-inf")
    best_state = None
    best_alpha = 0.0
    best_epoch = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for index in iter_batches(len(ct), batch_size, rng):
            candidate, cost, status_feature = inverse_candidate_features(
                train_states[gt[index]], dt[index]
            )
            candidate = (candidate - candidate_mean) / candidate_scale
            status_feature = (status_feature - status_mean) / status_scale
            context_tensor = torch.as_tensor(
                ct[index], dtype=torch.float32, device=device
            )
            candidate_tensor = torch.as_tensor(
                candidate, dtype=torch.float32, device=device
            )
            status_feature_tensor = torch.as_tensor(
                status_feature, dtype=torch.float32, device=device
            )
            positive = torch.as_tensor(
                pt[index], dtype=torch.bool, device=device
            )
            status_target = torch.as_tensor(
                st[index], dtype=torch.long, device=device
            )
            base = torch.as_tensor(-cost, dtype=torch.float32, device=device)
            correction, status_logits = model(
                context_tensor, candidate_tensor, status_feature_tensor
            )
            scores = base + correction
            feasible = positive.any(dim=1)
            positive_scores = scores[feasible].masked_fill(
                ~positive[feasible], -1e9
            )
            rank_loss = (
                torch.logsumexp(scores[feasible], dim=1)
                - torch.logsumexp(positive_scores, dim=1)
            ).mean()
            status_loss = torch.nn.functional.cross_entropy(
                status_logits, status_target, weight=status_weights
            )
            regularization = correction.square().mean()
            loss = (
                rank_loss
                + float(inverse_config["status_loss_weight"]) * status_loss
                + float(inverse_config["correction_regularization"])
                * regularization
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)

        correction_v, logits_v, base_v = predict_corrections(
            torch,
            model,
            cv,
            val_states,
            gv,
            dv,
            candidate_mean,
            candidate_scale,
            status_mean,
            status_scale,
            device,
        )
        alpha_rows = []
        for alpha in alpha_candidates:
            metrics = inverse_metrics(
                base_v + alpha * correction_v,
                logits_v,
                pv,
                sv,
                selected_v,
            )
            alpha_rows.append({"alpha": alpha, **metrics})
        selected_alpha = max(
            alpha_rows,
            key=lambda row: (
                row["target_success_feasible"],
                row["minimum_movement_exact_feasible"],
                row["status_macro_f1"],
                -row["alpha"],
            ),
        )
        checkpoint_score = (
            selected_alpha["target_success_feasible"]
            + 0.05 * selected_alpha["minimum_movement_exact_feasible"]
            + 0.10 * selected_alpha["status_macro_f1"]
        )
        row = {
            "epoch": epoch,
            "train_loss": running / len(ct),
            "score": checkpoint_score,
            "selected_alpha": selected_alpha,
            "alpha_candidates": alpha_rows,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if checkpoint_score > best_score:
            best_score = checkpoint_score
            best_epoch = epoch
            best_alpha = float(selected_alpha["alpha"])
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("inverse training produced no checkpoint")
    model.load_state_dict(best_state)
    artifact_path = output_dir / "inverse_control_v3.pt"
    torch.save(
        {
            "version": config["version"],
            "seed": seed,
            "model": "residual_corrected_inverse_control_v3",
            "state_dict": best_state,
            "context_dim": int(ct.shape[1]),
            "candidate_dim": int(sample_candidate.shape[-1]),
            "status_dim": int(sample_status.shape[-1]),
            "context_mean": context_mean,
            "context_scale": context_scale,
            "candidate_mean": candidate_mean,
            "candidate_scale": candidate_scale,
            "status_mean": status_mean,
            "status_scale": status_scale,
            "correction_alpha": best_alpha,
            "statuses": ["unique", "ambiguous", "infeasible_within_limits"],
            "action_grid": ACTION_GRID,
            "forward_artifact": str(forward_path.resolve()),
        },
        artifact_path,
    )
    validation = trace[best_epoch - 1]
    summary = {
        "version": config["version"],
        "device": str(device),
        "seed": seed,
        "artifact": str(artifact_path.resolve()),
        "forward_artifact": str(forward_path.resolve()),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "validation": validation,
        "trace": trace,
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "inverse_control_v3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
