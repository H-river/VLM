#!/usr/bin/env python3
"""Train a full-candidate Set Transformer with hard-negative ranking."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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
from tabm_transformer_rebuild_v8.inverse_features import feature_statistics
from tabm_transformer_rebuild_v8.models import build_inverse_model
from tabm_transformer_rebuild_v8.train_inverse import (
    VALIDATION_BLOCKS,
    calibrate,
    load_block,
    normalized_tensors,
    raw_predictions,
)

DEFAULT_CACHE = REPO_ROOT.parent / "VLM_data/physics_structured_inverse_v9"
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed/inverse"
)
DEFAULT_V5_BASELINE = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v5_one_seed/inverse_tree_v5_summary.json"
)
DEFAULT_V8_BASELINE = (
    REPO_ROOT.parent
    / "VLM_runs/tabm_transformer_rebuild_v8_one_seed/transformer"
    / "inverse_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--v5-baseline-summary",
        type=Path,
        default=DEFAULT_V5_BASELINE,
    )
    parser.add_argument(
        "--v8-baseline-summary",
        type=Path,
        default=DEFAULT_V8_BASELINE,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-negative-count", type=int, default=8)
    parser.add_argument("--ranking-margin", type=float, default=0.20)
    parser.add_argument("--margin-loss-weight", type=float, default=0.50)
    parser.add_argument("--cooling-seconds", type=float, default=5.0)
    parser.add_argument("--max-train-requests", type=int)
    parser.add_argument("--max-validation-requests", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architecture_config() -> dict[str, Any]:
    return {
        "dimension": 96,
        "heads": 8,
        "feedforward": 256,
        "layers": 2,
        "dropout": 0.05,
    }


def save_recovery(
    torch: Any,
    path: Path,
    model: Any,
    epoch: int,
    key: tuple[Any, ...],
    calibration: float,
    trace: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "best_epoch": int(epoch),
            "selection_key": list(key),
            "correction_weight": float(calibration),
            "trace": trace,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "inverse_summary.json"
    artifact_path = output_dir / "inverse.pt"
    recovery_path = output_dir / "recovery.pt"
    if summary_path.exists() or artifact_path.exists():
        raise RuntimeError(f"refusing to overwrite completed run: {output_dir}")
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
        limit = int(args.max_validation_requests)
        validation = {
            block_name: {
                name: values[: min(limit, len(values))]
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
    ) = feature_statistics(train["candidate_states"], train["desired"])
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
    config = architecture_config()
    model = build_inverse_model(
        torch,
        "transformer",
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
    trace: list[dict[str, Any]] = []
    stale = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train["contexts"]))
        totals = {
            "loss": 0.0,
            "listwise": 0.0,
            "margin": 0.0,
            "status": 0.0,
        }
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
                scores = -cost + correction
                feasible = positives.any(dim=1)
                feasible_scores = scores[feasible]
                feasible_positives = positives[feasible]
                positive_scores = feasible_scores.masked_fill(
                    ~feasible_positives,
                    -1e9,
                )
                listwise = (
                    torch.logsumexp(feasible_scores, dim=1)
                    - torch.logsumexp(positive_scores, dim=1)
                ).mean()
                best_positive = positive_scores.max(dim=1).values
                negative_scores = feasible_scores.masked_fill(
                    feasible_positives,
                    -1e9,
                )
                hard_count = min(
                    int(args.hard_negative_count),
                    negative_scores.shape[1],
                )
                hard_negative = negative_scores.topk(
                    hard_count,
                    dim=1,
                ).values
                margin_loss = torch.relu(
                    float(args.ranking_margin)
                    - best_positive[:, None]
                    + hard_negative
                ).mean()
                status_loss = torch.nn.functional.cross_entropy(
                    status_logits,
                    statuses,
                    weight=class_weight,
                )
                regularizer = correction.square().mean()
                loss = (
                    listwise
                    + float(args.margin_loss_weight) * margin_loss
                    + 0.30 * status_loss
                    + 0.01 * regularizer
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(selected)
            for name, value in (
                ("loss", loss),
                ("listwise", listwise),
                ("margin", margin_loss),
                ("status", status_loss),
            ):
                totals[name] += float(value.detach()) * count
            seen += count
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
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            **{name: value / seen for name, value in totals.items()},
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
        if improved:
            save_recovery(
                torch,
                recovery_path,
                model,
                epoch,
                key,
                correction_weight,
                trace,
            )
        if stale >= int(args.patience):
            break
        if float(args.cooling_seconds) > 0.0:
            time.sleep(float(args.cooling_seconds))

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
    artifact = {
        "version": "hard_negative_set_transformer_inverse_v9_one_seed",
        "model": "hard_negative_set_transformer_inverse_v9",
        "architecture": "transformer",
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
    v5_baseline = json.loads(
        args.v5_baseline_summary.resolve().read_text(encoding="utf-8")
    )["validation"]["inverse_tree_v5"]
    v8_baseline = json.loads(
        args.v8_baseline_summary.resolve().read_text(encoding="utf-8")
    )["validation"]["transformer_inverse_v8"]
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
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
            "hard_negative_count": int(args.hard_negative_count),
            "ranking_margin": float(args.ranking_margin),
            "margin_loss_weight": float(args.margin_loss_weight),
            "status_class_counts": dict(sorted(counts.items())),
            "trace": trace,
        },
        "calibration": {
            "correction_weight": calibration,
            "candidates": final_metrics["candidates"],
        },
        "validation": {
            "inverse_v5_reference": v5_baseline,
            "set_transformer_v8_reference": v8_baseline,
            "hard_negative_set_transformer_v9": final_metrics["selected"],
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
