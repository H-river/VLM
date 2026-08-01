#!/usr/bin/env python3
"""Train a TabM or Set-Transformer numerical inverse ranker."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.train_inverse import inverse_metrics
from control_rebuild_v3.common import inverse_candidate_features
from tabm_transformer_rebuild_v8.inverse_features import (
    feature_statistics,
    inverse_features_torch,
)
from tabm_transformer_rebuild_v8.models import build_inverse_model

DEFAULT_CACHE = (
    REPO_ROOT.parent / "VLM_data/tabm_transformer_inverse_v8"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "VLM_runs/tabm_transformer_rebuild_v8_one_seed"
)
DEFAULT_BASELINE = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v5_one_seed/inverse_tree_v5_summary.json"
)
VALIDATION_BLOCKS = (
    "iid_clean",
    "iid_measurement_augmented",
    "difficult_clean",
    "difficult_measurement_augmented",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("tabm", "transformer"),
        required=True,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tabm-candidates", type=int, default=32)
    parser.add_argument(
        "--max-train-requests",
        type=int,
        default=None,
        help="Optional prefix limit for smoke tests only.",
    )
    parser.add_argument(
        "--max-validation-requests",
        type=int,
        default=None,
        help="Optional per-block prefix limit for smoke tests only.",
    )
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
            "d_block": 128,
            "dropout": 0.05,
        }
    return {
        "dimension": 96,
        "heads": 8,
        "feedforward": 256,
        "layers": 2,
        "dropout": 0.05,
    }


def load_block(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    return {name: np.asarray(loaded[name]) for name in loaded.files}


def hard_candidate_indices(
    states: np.ndarray,
    desired: np.ndarray,
    positives: np.ndarray,
    count: int,
    chunk_size: int = 512,
) -> np.ndarray:
    if count >= states.shape[1]:
        return np.broadcast_to(
            np.arange(states.shape[1], dtype=np.int64)[None, :],
            (len(states), states.shape[1]),
        ).copy()
    output = np.empty((len(states), count), dtype=np.int64)
    for start in range(0, len(states), chunk_size):
        stop = min(start + chunk_size, len(states))
        _, cost, _ = inverse_candidate_features(
            states[start:stop],
            desired[start:stop],
        )
        for local in range(stop - start):
            positive = np.flatnonzero(positives[start + local])
            ordered = np.argsort(cost[local])
            chosen = list(int(value) for value in positive[:count])
            for index in ordered:
                if len(chosen) == count:
                    break
                value = int(index)
                if value not in chosen:
                    chosen.append(value)
            output[start + local] = chosen
    return output


def normalized_tensors(
    torch: Any,
    block: dict[str, np.ndarray],
    selected: np.ndarray,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    candidate_mean: Any,
    candidate_scale: Any,
    status_mean: Any,
    status_scale: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    context = torch.as_tensor(
        (block["contexts"][selected] - context_mean) / context_scale,
        dtype=torch.float32,
        device=device,
    )
    states = torch.as_tensor(
        block["candidate_states"][selected],
        dtype=torch.float32,
        device=device,
    )
    desired = torch.as_tensor(
        block["desired"][selected],
        dtype=torch.float32,
        device=device,
    )
    positives = torch.as_tensor(
        block["positives"][selected],
        dtype=torch.bool,
        device=device,
    )
    statuses = torch.as_tensor(
        block["statuses"][selected],
        dtype=torch.long,
        device=device,
    )
    candidate, cost, status_features = inverse_features_torch(
        torch,
        states,
        desired,
    )
    candidate = (candidate - candidate_mean) / candidate_scale
    status_features = (status_features - status_mean) / status_scale
    return context, candidate, cost, status_features, positives, statuses


def raw_predictions(
    torch: Any,
    model: Any,
    block: dict[str, np.ndarray],
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    candidate_mean: Any,
    candidate_scale: Any,
    status_mean: Any,
    status_scale: Any,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrections = []
    costs = []
    status_probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(block["contexts"]), batch_size):
            selected = np.arange(
                start,
                min(start + batch_size, len(block["contexts"])),
            )
            (
                context,
                candidate,
                cost,
                status_features,
                _,
                _,
            ) = normalized_tensors(
                torch,
                block,
                selected,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                device,
            )
            correction, status_logits = model(
                context,
                candidate,
                status_features,
            )
            if correction.ndim == 3:
                correction = correction.float().mean(dim=-1)
                status_probability = (
                    status_logits.float().softmax(dim=-1).mean(dim=1)
                )
            else:
                status_probability = status_logits.float().softmax(dim=-1)
            corrections.append(correction.float().cpu().numpy())
            costs.append(cost.float().cpu().numpy())
            status_probabilities.append(status_probability.cpu().numpy())
    return (
        np.concatenate(corrections).astype(np.float32),
        np.concatenate(costs).astype(np.float32),
        np.concatenate(status_probabilities).astype(np.float32),
    )


def calibrate(
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    blocks: dict[str, dict[str, np.ndarray]],
) -> tuple[float, dict[str, Any], tuple[Any, ...]]:
    rows = []
    for correction_weight in (0.0, 0.25, 0.50, 1.0, 1.50, 2.0):
        metrics = {}
        for name in VALIDATION_BLOCKS:
            correction, cost, status_probability = raw[name]
            block = blocks[name]
            metrics[name] = inverse_metrics(
                -cost + correction_weight * correction,
                np.log(np.clip(status_probability, 1e-8, 1.0)),
                block["positives"],
                block["statuses"],
                block["selected_truth"],
            )
        clean = [
            metrics["iid_clean"]["target_success_feasible"],
            metrics["difficult_clean"]["target_success_feasible"],
        ]
        noisy = [
            metrics["iid_measurement_augmented"][
                "target_success_feasible"
            ],
            metrics["difficult_measurement_augmented"][
                "target_success_feasible"
            ],
        ]
        key = (
            min(clean),
            float(
                np.mean(
                    [
                        value["target_success_feasible"]
                        for value in metrics.values()
                    ]
                )
            ),
            min(noisy),
            float(
                np.mean(
                    [
                        value["minimum_movement_exact_feasible"]
                        for value in metrics.values()
                    ]
                )
            ),
            -float(correction_weight),
        )
        rows.append(
            {
                "correction_weight": float(correction_weight),
                "selection_key": list(key),
                "metrics": metrics,
            }
        )
    selected = max(rows, key=lambda row: tuple(row["selection_key"]))
    return (
        float(selected["correction_weight"]),
        {
            "selected": selected["metrics"],
            "candidates": rows,
        },
        tuple(selected["selection_key"]),
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve() / args.architecture
    summary_path = output_dir / "inverse_summary.json"
    if summary_path.exists():
        raise RuntimeError(f"refusing to overwrite inverse run: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (cache_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("complete") is not True:
        raise ValueError("inverse cache is incomplete")
    train = load_block(cache_dir / "train.npz")
    validation = {
        name: load_block(cache_dir / f"{name}.npz")
        for name in VALIDATION_BLOCKS
    }
    if args.max_train_requests is not None:
        count = min(int(args.max_train_requests), len(train["contexts"]))
        train = {name: values[:count] for name, values in train.items()}
    if args.max_validation_requests is not None:
        validation_count = int(args.max_validation_requests)
        validation = {
            block_name: {
                name: values[: min(validation_count, len(values))]
                for name, values in block.items()
            }
            for block_name, block in validation.items()
        }
    torch, device = configure(int(args.seed), args.device)
    random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    context_mean = train["contexts"].mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    context_scale = train["contexts"].std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    context_scale = np.maximum(context_scale, 1e-6)
    (
        candidate_mean_np,
        candidate_scale_np,
        status_mean_np,
        status_scale_np,
    ) = feature_statistics(
        train["candidate_states"],
        train["desired"],
    )
    candidate_mean = torch.as_tensor(
        candidate_mean_np,
        dtype=torch.float32,
        device=device,
    )
    candidate_scale = torch.as_tensor(
        candidate_scale_np,
        dtype=torch.float32,
        device=device,
    )
    status_mean = torch.as_tensor(
        status_mean_np,
        dtype=torch.float32,
        device=device,
    )
    status_scale = torch.as_tensor(
        status_scale_np,
        dtype=torch.float32,
        device=device,
    )
    tabm_indices = (
        hard_candidate_indices(
            train["candidate_states"],
            train["desired"],
            train["positives"],
            int(args.tabm_candidates),
        )
        if args.architecture == "tabm"
        else None
    )
    config = architecture_config(args.architecture)
    model = build_inverse_model(
        torch,
        args.architecture,
        context_dim=train["contexts"].shape[1],
        candidate_dim=len(candidate_mean_np),
        status_dim=len(status_mean_np),
        candidate_count=len(ACTION_GRID),
        config=config,
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
    counts = Counter(int(value) for value in train["statuses"])
    class_weight = torch.as_tensor(
        [
            len(train["statuses"]) / max(3 * counts[index], 1)
            for index in range(3)
        ],
        dtype=torch.float32,
        device=device,
    )
    use_amp = device.type == "cuda"
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_key: tuple[Any, ...] = (-float("inf"),)
    best_calibration = 0.0
    best_metrics = None
    trace = []
    stale = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train["contexts"]))
        total = 0.0
        total_rank = 0.0
        total_status = 0.0
        seen = 0
        for start in range(0, len(order), int(args.batch_size)):
            selected = order[start : start + int(args.batch_size)]
            (
                context,
                candidates,
                cost,
                status_features,
                positives,
                statuses,
            ) = normalized_tensors(
                torch,
                train,
                selected,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                device,
            )
            if tabm_indices is not None:
                local_indices = torch.as_tensor(
                    tabm_indices[selected],
                    dtype=torch.long,
                    device=device,
                )
                gather = local_indices[..., None].expand(
                    -1,
                    -1,
                    candidates.shape[-1],
                )
                candidates = candidates.gather(1, gather)
                cost = cost.gather(1, local_indices)
                positives = positives.gather(1, local_indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                correction, status_logits = model(
                    context,
                    candidates,
                    status_features,
                )
                if correction.ndim == 2:
                    correction = correction[..., None]
                    status_logits = status_logits[:, None, :]
                scores = -cost[..., None] + correction
                feasible = positives.any(dim=1)
                positive_scores = scores[feasible].masked_fill(
                    ~positives[feasible, :, None],
                    -1e9,
                )
                rank = (
                    torch.logsumexp(scores[feasible], dim=1)
                    - torch.logsumexp(positive_scores, dim=1)
                ).mean()
                repeated_status = statuses[:, None].expand(
                    -1,
                    status_logits.shape[1],
                )
                status_loss = torch.nn.functional.cross_entropy(
                    status_logits.reshape(-1, 3),
                    repeated_status.reshape(-1),
                    weight=class_weight,
                )
                regularizer = correction.square().mean()
                loss = rank + 0.30 * status_loss + 0.01 * regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(selected)
            total_rank += float(rank.detach()) * len(selected)
            total_status += float(status_loss.detach()) * len(selected)
            seen += len(selected)
        scheduler.step()
        raw = {
            name: raw_predictions(
                torch,
                model,
                block,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                device,
                max(32, int(args.batch_size) // 2),
            )
            for name, block in validation.items()
        }
        correction_weight, metrics, key = calibrate(raw, validation)
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_calibration = correction_weight
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "loss": total / seen,
            "rank_loss": total_rank / seen,
            "status_loss": total_status / seen,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "correction_weight": correction_weight,
            "selection_key": list(key),
            "selected": improved,
            "validation": {
                name: metrics["selected"][name][
                    "target_success_feasible"
                ]
                for name in VALIDATION_BLOCKS
            },
            "seconds": time.perf_counter() - epoch_started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break

    model.load_state_dict(best_state)
    raw = {
        name: raw_predictions(
            torch,
            model,
            block,
            context_mean,
            context_scale,
            candidate_mean,
            candidate_scale,
            status_mean,
            status_scale,
            device,
            max(32, int(args.batch_size) // 2),
        )
        for name, block in validation.items()
    }
    calibration, final_metrics, final_key = calibrate(raw, validation)
    if calibration != best_calibration:
        raise RuntimeError("restored inverse calibration differs")
    artifact_path = output_dir / "inverse.pt"
    artifact = {
        "version": f"{args.architecture}_numerical_inverse_v8_one_seed",
        "model": f"{args.architecture}_numerical_inverse_v8",
        "architecture": args.architecture,
        "architecture_config": config,
        "context_mean": context_mean,
        "context_scale": context_scale,
        "candidate_mean": candidate_mean_np,
        "candidate_scale": candidate_scale_np,
        "status_mean": status_mean_np,
        "status_scale": status_scale_np,
        "correction_weight": calibration,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "forward_artifact": metadata["forward_artifact"],
        "held_out_test_used": False,
    }
    torch.save(artifact, artifact_path)
    baseline = json.loads(
        args.baseline_summary.resolve().read_text(encoding="utf-8")
    )["validation"]["inverse_tree_v5"]
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "architecture": args.architecture,
        "architecture_config": config,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training": {
            "seed": int(args.seed),
            "request_count": int(len(train["contexts"])),
            "measurement_augmented_count": int(train["noisy"].sum()),
            "best_epoch": best_epoch,
            "batch_size": int(args.batch_size),
            "tabm_candidates_per_request": (
                int(args.tabm_candidates)
                if args.architecture == "tabm"
                else None
            ),
            "status_class_counts": dict(sorted(counts.items())),
            "trace": trace,
        },
        "calibration": {
            "correction_weight": calibration,
            "candidates": final_metrics["candidates"],
        },
        "validation": {
            "inverse_v5_reference": baseline,
            f"{args.architecture}_inverse_v8": final_metrics["selected"],
        },
        "selection_key": list(final_key),
        "source_contract": {
            "cache_metadata": str(cache_dir / "metadata.json"),
            "cache_metadata_sha256": sha256(cache_dir / "metadata.json"),
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
                "correction_weight": calibration,
                "validation": final_metrics["selected"],
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
