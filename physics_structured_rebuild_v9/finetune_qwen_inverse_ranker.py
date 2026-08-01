#!/usr/bin/env python3
"""Fine-tune the frozen Set Transformer on natural multi-positive inverse data."""

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

from control_rebuild_v3.common import MOVEMENT
from control_rebuild_v3.train_forward import configure
from tabm_transformer_rebuild_v8.inverse_features import (
    inverse_features_torch,
)
from tabm_transformer_rebuild_v8.models import build_inverse_model

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "qwen_inverse_forward_cache.npz"
DEFAULT_BASE = (
    REPO_ROOT.parent
    / "VLM_runs/tabm_transformer_rebuild_v8_one_seed"
    / "transformer/inverse.pt"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_inverse_ranker_adaptation"
WEIGHTS = (0.25, 0.50, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--distillation-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_groups(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import hashlib as hash_module

    keys = np.asarray(
        [
            int(
                hash_module.sha256(
                    f"{seed}:{str(group_id)}".encode()
                ).hexdigest()[:16],
                16,
            )
            for group_id in group_ids
        ],
        dtype=np.uint64,
    )
    order = np.argsort(keys)
    calibration = np.sort(order[:200])
    train = np.sort(order[200:])
    return train, calibration


def tensors(
    torch: Any,
    contexts: np.ndarray,
    desired: np.ndarray,
    states: np.ndarray,
    positives: np.ndarray,
    indices: np.ndarray,
    artifact: dict[str, Any],
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    context = torch.as_tensor(
        (
            contexts[indices]
            - np.asarray(artifact["context_mean"], dtype=np.float32)
        )
        / np.asarray(artifact["context_scale"], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    state = torch.as_tensor(
        states[indices],
        dtype=torch.float32,
        device=device,
    )
    desired_tensor = torch.as_tensor(
        desired[indices],
        dtype=torch.float32,
        device=device,
    )
    positive = torch.as_tensor(
        positives[indices],
        dtype=torch.bool,
        device=device,
    )
    candidate, cost, status = inverse_features_torch(
        torch,
        state,
        desired_tensor,
    )
    candidate = (
        candidate
        - torch.as_tensor(
            artifact["candidate_mean"],
            dtype=torch.float32,
            device=device,
        )
    ) / torch.as_tensor(
        artifact["candidate_scale"],
        dtype=torch.float32,
        device=device,
    )
    status = (
        status
        - torch.as_tensor(
            artifact["status_mean"],
            dtype=torch.float32,
            device=device,
        )
    ) / torch.as_tensor(
        artifact["status_scale"],
        dtype=torch.float32,
        device=device,
    )
    return context, candidate, cost, status, positive


def corrections(
    torch: Any,
    model: Any,
    contexts: np.ndarray,
    desired: np.ndarray,
    states: np.ndarray,
    artifact: dict[str, Any],
    indices: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    correction_parts = []
    cost_parts = []
    dummy_positive = np.zeros(
        (len(contexts), states.shape[1]),
        dtype=np.bool_,
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            context, candidate, cost, status, _ = tensors(
                torch,
                contexts,
                desired,
                states,
                dummy_positive,
                selected,
                artifact,
                device,
            )
            correction, _ = model(context, candidate, status)
            correction_parts.append(correction.float().cpu().numpy())
            cost_parts.append(cost.float().cpu().numpy())
    return np.concatenate(correction_parts), np.concatenate(cost_parts)


def success_count(
    correction: np.ndarray,
    cost: np.ndarray,
    positives: np.ndarray,
    indices: np.ndarray,
    weight: float,
) -> int:
    scores = (
        -cost
        + float(weight) * correction
        - 1e-7 * MOVEMENT[None, :]
    )
    selected = scores.argmax(axis=1)
    return int(positives[indices, selected].sum())


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    artifact_path = output_dir / "inverse.pt"
    summary_path = output_dir / "inverse_summary.json"
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache.resolve()
    arrays = np.load(cache_path, allow_pickle=False)
    contexts = np.asarray(arrays["contexts"], dtype=np.float32)
    desired = np.asarray(arrays["desired"], dtype=np.float32)
    positives = np.asarray(arrays["positives"], dtype=np.bool_)
    source_states = {
        "primary": np.asarray(arrays["primary_states"], dtype=np.float32),
        "secondary": np.asarray(
            arrays["secondary_states"],
            dtype=np.float32,
        ),
    }
    train_groups, calibration_groups = split_groups(
        np.asarray(arrays["group_ids"]),
        int(args.seed),
    )

    torch, device = configure(int(args.seed), args.device)
    base_path = args.base_artifact.resolve()
    base_artifact = torch.load(
        base_path,
        map_location="cpu",
        weights_only=False,
    )
    if base_artifact.get("model") != "transformer_numerical_inverse_v8":
        raise ValueError("expected frozen transformer inverse artifact")
    model = build_inverse_model(
        torch,
        "transformer",
        context_dim=len(base_artifact["context_mean"]),
        candidate_dim=len(base_artifact["candidate_mean"]),
        status_dim=len(base_artifact["status_mean"]),
        candidate_count=81,
        config=dict(base_artifact["architecture_config"]),
    ).to(device)
    model.load_state_dict(base_artifact["state_dict"])
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.learning_rate) * 0.1,
    )
    rng = np.random.default_rng(int(args.seed))
    base_weight = float(base_artifact["correction_weight"])

    baseline = {}
    for source, states in source_states.items():
        correction, cost = corrections(
            torch,
            model,
            contexts,
            desired,
            states,
            base_artifact,
            calibration_groups,
            device,
            int(args.batch_size),
        )
        baseline[source] = success_count(
            correction,
            cost,
            positives,
            calibration_groups,
            base_weight,
        )

    best_state = copy.deepcopy(model.state_dict())
    best_weight = base_weight
    best_key = (
        baseline["primary"],
        baseline["secondary"],
        -base_weight,
    )
    best_epoch = 0
    trace = []
    source_names = tuple(source_states)
    training_pairs = np.asarray(
        [
            (source_index, int(group))
            for source_index in range(len(source_names))
            for group in train_groups
        ],
        dtype=np.int64,
    )
    use_amp = device.type == "cuda"
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = rng.permutation(len(training_pairs))
        total_loss = 0.0
        total_rank = 0.0
        seen = 0
        for start in range(0, len(order), int(args.batch_size)):
            pairs = training_pairs[
                order[start : start + int(args.batch_size)]
            ]
            source_index = pairs[:, 0]
            group_index = pairs[:, 1]
            batch_states = np.stack(
                [
                    source_states[source_names[int(source_index[row])]][
                        int(group_index[row])
                    ]
                    for row in range(len(pairs))
                ]
            )
            batch_contexts = contexts[group_index]
            batch_desired = desired[group_index]
            batch_positives = positives[group_index]
            local_indices = np.arange(len(pairs))
            context, candidate, cost, status, positive = tensors(
                torch,
                batch_contexts,
                batch_desired,
                batch_states,
                batch_positives,
                local_indices,
                base_artifact,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_correction, _ = teacher(
                    context,
                    candidate,
                    status,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                correction, _ = model(context, candidate, status)
                scores = -cost + base_weight * correction
                positive_scores = scores.masked_fill(~positive, -1e9)
                rank = (
                    torch.logsumexp(scores, dim=1)
                    - torch.logsumexp(positive_scores, dim=1)
                ).mean()
                distillation = torch.nn.functional.mse_loss(
                    correction.float(),
                    teacher_correction.float(),
                )
                loss = (
                    rank
                    + float(args.distillation_weight) * distillation
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(pairs)
            total_rank += float(rank.detach()) * len(pairs)
            seen += len(pairs)
        scheduler.step()

        raw = {}
        for source, states in source_states.items():
            raw[source] = corrections(
                torch,
                model,
                contexts,
                desired,
                states,
                base_artifact,
                calibration_groups,
                device,
                int(args.batch_size),
            )
        candidates = []
        for weight in WEIGHTS:
            counts = {
                source: success_count(
                    raw[source][0],
                    raw[source][1],
                    positives,
                    calibration_groups,
                    weight,
                )
                for source in source_names
            }
            key = (
                counts["primary"],
                counts["secondary"],
                -float(weight),
            )
            candidates.append(
                {
                    "correction_weight": float(weight),
                    "success_count": counts,
                    "key": list(key),
                }
            )
        selected = max(candidates, key=lambda row: tuple(row["key"]))
        key = tuple(selected["key"])
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_weight = float(selected["correction_weight"])
            best_epoch = epoch
        record = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "rank_loss": total_rank / seen,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selected": improved,
            "calibration_candidates": candidates,
            "seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    model.load_state_dict(best_state)
    adapted_artifact = dict(base_artifact)
    adapted_artifact.update(
        {
            "version": "qwen_adapted_transformer_inverse_v9_one_seed",
            "correction_weight": float(best_weight),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "base_inverse_artifact": str(base_path),
            "base_inverse_artifact_sha256": sha256(base_path),
            "natural_training_cache": str(cache_path),
            "natural_training_cache_sha256": sha256(cache_path),
            "held_out_test_used": False,
        }
    )
    torch.save(adapted_artifact, artifact_path)
    summary = {
        "version": adapted_artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training": {
            "seed": int(args.seed),
            "train_group_count": int(len(train_groups)),
            "internal_calibration_group_count": int(
                len(calibration_groups)
            ),
            "enumerator_sources": list(source_names),
            "baseline_internal_success_count": baseline,
            "best_epoch": best_epoch,
            "best_correction_weight": best_weight,
            "best_key": list(best_key),
            "trace": trace,
        },
        "source_contract": {
            "natural_training_cache": str(cache_path),
            "natural_training_cache_sha256": sha256(cache_path),
            "base_inverse_artifact": str(base_path),
            "base_inverse_artifact_sha256": sha256(base_path),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
        "held_out_validation_used": False,
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
