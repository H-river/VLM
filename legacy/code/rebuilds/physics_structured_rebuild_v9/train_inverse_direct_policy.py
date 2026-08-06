#!/usr/bin/env python3
"""Train a direct multi-positive inverse policy and calibrate a safe blend."""

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

from control_rebuild_v3.common import MOVEMENT
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.inverse_direct_policy_runtime import (
    build_direct_policy,
    sha256,
)
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    normalized_per_request,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    split_groups,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_OLD = REPO_ROOT.parent / "VLM_data/tabm_transformer_inverse_v8/train.npz"
DEFAULT_NATURAL = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9"
    / "combined_natural_inverse_adaptation/train.npz"
)
DEFAULT_NATURAL_FORWARD = (
    DEFAULT_RUN / "combined_natural_inverse_forward_cache.npz"
)
DEFAULT_BASE = (
    DEFAULT_RUN / "combined_natural_inverse_ranker_adaptation/inverse.pt"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "inverse_direct_policy_v9.pt"
BLEND_WEIGHTS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--natural-data", type=Path, default=DEFAULT_NATURAL)
    parser.add_argument(
        "--natural-forward-cache",
        type=Path,
        default=DEFAULT_NATURAL_FORWARD,
    )
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--natural-weight", type=float, default=4.0)
    parser.add_argument("--target-loss-weight", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


def prediction_success(
    positives: np.ndarray,
    scores: np.ndarray,
) -> tuple[int, np.ndarray]:
    selected = (scores - 1e-7 * MOVEMENT[None, :]).argmax(axis=1)
    return (
        int(positives[np.arange(len(selected)), selected].sum()),
        selected,
    )


def policy_logits(
    torch: Any,
    model: Any,
    contexts: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            batch = torch.as_tensor(
                (contexts[start : start + batch_size] - mean) / scale,
                dtype=torch.float32,
                device=device,
            )
            parts.append(model(batch).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def calibration_candidates(
    positives_by_source: dict[str, np.ndarray],
    base_scores_by_source: dict[str, np.ndarray],
    policy_scores_by_source: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    candidates = []
    base_counts = {
        source: prediction_success(positives, base_scores_by_source[source])[0]
        for source, positives in positives_by_source.items()
    }
    for weight in BLEND_WEIGHTS:
        counts = {}
        for source, positives in positives_by_source.items():
            scores = (
                normalized_per_request(base_scores_by_source[source])
                + float(weight)
                * normalized_per_request(policy_scores_by_source[source])
            )
            counts[source] = prediction_success(positives, scores)[0]
        old_non_regression = counts["old"] >= base_counts["old"]
        key = [
            int(old_non_regression),
            counts["natural"] if old_non_regression else -1,
            counts["old"],
            -float(weight),
        ]
        candidates.append(
            {
                "policy_weight": float(weight),
                "success_count": counts,
                "base_success_count": base_counts,
                "old_non_regression": bool(old_non_regression),
                "key": key,
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    recovery = output.with_name(output.stem + "_recovery.pt")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")

    old_path = args.old_data.resolve()
    with np.load(old_path, allow_pickle=False) as arrays:
        old_contexts = np.asarray(arrays["contexts"], dtype=np.float32)
        old_positives = np.asarray(arrays["positives"], dtype=np.bool_)
        old_targets = np.asarray(arrays["selected_truth"], dtype=np.int64)
        old_desired = np.asarray(arrays["desired"], dtype=np.float32)
        old_states = np.asarray(arrays["candidate_states"], dtype=np.float32)
    feasible = old_positives.any(axis=1)
    feasible_indices = np.flatnonzero(feasible)
    rng = np.random.default_rng(int(args.seed))
    old_order = rng.permutation(feasible_indices)
    old_calibration_indices = np.sort(old_order[:8000])
    old_training_indices = np.sort(old_order[8000:])

    natural_path = args.natural_data.resolve()
    with np.load(natural_path, allow_pickle=False) as arrays:
        natural_ids = np.asarray(arrays["group_ids"], dtype=np.str_)
        natural_contexts = np.asarray(arrays["contexts"], dtype=np.float32)
        natural_desired = np.asarray(arrays["desired"], dtype=np.float32)
        natural_positives = np.asarray(arrays["positives"], dtype=np.bool_)
        natural_targets = np.asarray(arrays["target_indices"], dtype=np.int64)
    natural_training_indices, natural_calibration_indices = split_groups(
        natural_ids,
        int(args.seed),
    )
    forward_path = args.natural_forward_cache.resolve()
    with np.load(forward_path, allow_pickle=False) as arrays:
        forward_ids = np.asarray(arrays["group_ids"], dtype=np.str_)
        natural_states = np.asarray(
            arrays["primary_states"],
            dtype=np.float32,
        )
    if not np.array_equal(natural_ids, forward_ids):
        raise ValueError("natural inverse and forward cache group IDs differ")

    train_contexts = np.concatenate(
        [
            old_contexts[old_training_indices],
            natural_contexts[natural_training_indices],
        ]
    )
    train_positives = np.concatenate(
        [
            old_positives[old_training_indices],
            natural_positives[natural_training_indices],
        ]
    )
    train_targets = np.concatenate(
        [
            old_targets[old_training_indices],
            natural_targets[natural_training_indices],
        ]
    )
    train_weights = np.concatenate(
        [
            np.ones(len(old_training_indices), dtype=np.float32),
            np.full(
                len(natural_training_indices),
                float(args.natural_weight),
                dtype=np.float32,
            ),
        ]
    )
    context_mean = train_contexts.mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    context_scale = train_contexts.std(axis=0, dtype=np.float64).astype(
        np.float32
    )
    context_scale = np.maximum(context_scale, 1e-6)

    torch, device = configure(int(args.seed), args.device)
    random.seed(int(args.seed))
    model = build_direct_policy(
        torch,
        train_contexts.shape[1],
        int(args.hidden),
    ).to(device)
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

    base_path = args.base_artifact.resolve()
    base, _ = load_inverse_runtime_v8(base_path, torch, device)
    old_calibration = {
        "contexts": old_contexts[old_calibration_indices],
        "desired": old_desired[old_calibration_indices],
        "states": old_states[old_calibration_indices],
        "positives": old_positives[old_calibration_indices],
    }
    natural_calibration = {
        "contexts": natural_contexts[natural_calibration_indices],
        "desired": natural_desired[natural_calibration_indices],
        "states": natural_states[natural_calibration_indices],
        "positives": natural_positives[natural_calibration_indices],
    }
    base_scores_by_source = {
        "old": np.asarray(
            base.score_feature_arrays(
                old_calibration["contexts"],
                old_calibration["desired"],
                old_calibration["states"],
            )["scores"],
            dtype=np.float32,
        ),
        "natural": np.asarray(
            base.score_feature_arrays(
                natural_calibration["contexts"],
                natural_calibration["desired"],
                natural_calibration["states"],
            )["scores"],
            dtype=np.float32,
        ),
    }
    positives_by_source = {
        "old": old_calibration["positives"],
        "natural": natural_calibration["positives"],
    }

    best_key: tuple[Any, ...] = (-float("inf"),)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_weight = 0.0
    best_candidates: list[dict[str, Any]] = []
    trace = []
    stale = 0
    use_amp = device.type == "cuda"
    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_contexts))
        loss_total = 0.0
        seen = 0
        for start in range(0, len(order), int(args.batch_size)):
            selected = order[start : start + int(args.batch_size)]
            contexts = torch.as_tensor(
                (train_contexts[selected] - context_mean) / context_scale,
                dtype=torch.float32,
                device=device,
            )
            positives = torch.as_tensor(
                train_positives[selected],
                dtype=torch.bool,
                device=device,
            )
            targets = torch.as_tensor(
                train_targets[selected],
                dtype=torch.long,
                device=device,
            )
            weights = torch.as_tensor(
                train_weights[selected],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(contexts)
                positive_logits = logits.masked_fill(~positives, -1e9)
                listwise = torch.logsumexp(logits, dim=1) - torch.logsumexp(
                    positive_logits,
                    dim=1,
                )
                exact = torch.nn.functional.cross_entropy(
                    logits,
                    targets,
                    reduction="none",
                )
                per_row = listwise + float(args.target_loss_weight) * exact
                loss = (per_row * weights).sum() / weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(selected)
            seen += len(selected)
        scheduler.step()

        policy_scores = {
            "old": policy_logits(
                torch,
                model,
                old_calibration["contexts"],
                context_mean,
                context_scale,
                device,
                int(args.batch_size),
            ),
            "natural": policy_logits(
                torch,
                model,
                natural_calibration["contexts"],
                context_mean,
                context_scale,
                device,
                int(args.batch_size),
            ),
        }
        candidates = calibration_candidates(
            positives_by_source,
            base_scores_by_source,
            policy_scores,
        )
        selected_candidate = max(
            candidates,
            key=lambda row: tuple(row["key"]),
        )
        key = tuple(selected_candidate["key"])
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_weight = float(selected_candidate["policy_weight"])
            best_candidates = candidates
            stale = 0
            temporary = recovery.with_suffix(".tmp")
            torch.save(
                {
                    "epoch": int(epoch),
                    "state_dict": best_state,
                    "selection_key": list(key),
                    "policy_weight": best_weight,
                },
                temporary,
            )
            os.replace(temporary, recovery)
        else:
            stale += 1
        record = {
            "epoch": int(epoch),
            "loss": loss_total / max(seen, 1),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selected": bool(improved),
            "policy_weight": float(selected_candidate["policy_weight"]),
            "success_count": selected_candidate["success_count"],
            "base_success_count": selected_candidate["base_success_count"],
            "old_non_regression": selected_candidate["old_non_regression"],
            "seconds": time.perf_counter() - epoch_started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break

    model.load_state_dict(best_state)
    artifact = {
        "version": "direct_inverse_policy_v9_one_seed",
        "model": "direct_inverse_policy_v9",
        "seed": int(args.seed),
        "hidden_dimension": int(args.hidden),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in best_state.items()
        },
        "context_mean": context_mean,
        "context_scale": context_scale,
        "policy_weight": float(best_weight),
        "base_inverse_artifact": str(base_path),
        "base_inverse_artifact_sha256": sha256(base_path),
        "old_training_data": str(old_path),
        "old_training_data_sha256": sha256(old_path),
        "natural_training_data": str(natural_path),
        "natural_training_data_sha256": sha256(natural_path),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, output)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training": {
            "old_feasible_training_count": int(len(old_training_indices)),
            "natural_training_count": int(len(natural_training_indices)),
            "natural_weight": float(args.natural_weight),
            "target_loss_weight": float(args.target_loss_weight),
            "epochs_completed": int(len(trace)),
            "best_epoch": int(best_epoch),
            "trace": trace,
        },
        "internal_calibration": {
            "old_count": int(len(old_calibration_indices)),
            "natural_count": int(len(natural_calibration_indices)),
            "selected_policy_weight": float(best_weight),
            "selected_key": list(best_key),
            "candidates": best_candidates,
        },
        "source_contract": {
            "old_training_data": str(old_path),
            "natural_training_data": str(natural_path),
            "natural_forward_cache": str(forward_path),
            "base_inverse_artifact": str(base_path),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
        "held_out_validation_used": False,
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
