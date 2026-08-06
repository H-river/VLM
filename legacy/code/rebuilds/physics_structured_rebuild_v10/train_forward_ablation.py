#!/usr/bin/env python3
"""Train one preregistered A-D forward response-surface experiment."""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v10.contracts import (
    ACTION_NORMALIZED,
    explicit_action_features,
    group_targets,
    natural_requested_action_index,
    sha256_file,
    stable_seed,
    visible_context,
)
from physics_structured_rebuild_v10.evaluate import forward_metrics
from physics_structured_rebuild_v10.models import (
    build_forward_model,
    forward_call,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_v10.json")
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v10_pilot"
EXPERIMENTS = {
    "A": {
        "data": "existing_distribution",
        "architecture": "pointwise_action_id",
        "objective": "fieldwise_huber",
    },
    "B": {
        "data": "system_aligned",
        "architecture": "pointwise_action_id",
        "objective": "fieldwise_huber",
    },
    "C": {
        "data": "system_aligned",
        "architecture": "pointwise_action_id",
        "objective": "tolerance_aware_joint",
    },
    "D": {
        "data": "system_aligned",
        "architecture": "structured_81_action",
        "objective": "tolerance_aware_joint",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def existing_distribution_sample(
    sources: list[str],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Select the globally lowest deterministic hashes without split leakage."""

    heap: list[tuple[int, int, dict[str, Any]]] = []
    serial = 0
    provenance = []
    for source_text in sources:
        source = Path(source_text).resolve()
        provenance.append({"path": str(source), "sha256": sha256_file(source)})
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                token = stable_seed(seed, source, row["group_id"], "sample")
                entry = (-token, serial, row)
                serial += 1
                if len(heap) < count:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
    rows = [entry[2] for entry in heap]
    rows.sort(key=lambda row: str(row["group_id"]))
    if len(rows) != count:
        raise ValueError("existing source sample is smaller than requested count")
    return rows, provenance


def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    contexts = np.asarray(
        [visible_context(row["setup"], row["current_beam_state"]) for row in rows],
        dtype=np.float32,
    )
    targets = np.asarray([group_targets(row)[1] for row in rows], dtype=np.float32)
    return contexts, targets


def partition(group_ids: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(
        [stable_seed(seed, group_id, "internal_split") for group_id in group_ids],
        dtype=np.uint64,
    )
    order = np.argsort(tokens)
    calibration_count = max(1, int(round(0.15 * len(order))))
    return order[calibration_count:], order[:calibration_count]


def loss_value(
    torch: Any,
    mean: Any,
    log_variance: Any | None,
    target: Any,
    objective: str,
) -> Any:
    residual = mean - target
    field_weight = torch.as_tensor(
        [1.10, 1.10, 0.90, 0.90, 1.35],
        dtype=mean.dtype,
        device=mean.device,
    )
    huber = torch.nn.functional.smooth_l1_loss(
        mean,
        target,
        reduction="none",
        beta=0.35,
    )
    value = (huber * field_weight[None, None, :]).mean()
    if objective == "fieldwise_huber":
        return value
    smooth_max = torch.logsumexp(residual.abs() * 6.0, dim=2) / 6.0
    joint = torch.nn.functional.softplus(smooth_max - 1.0).mean()
    if log_variance is None:
        uncertainty = mean.new_zeros(())
    else:
        uncertainty = 0.5 * (
            residual.square() * torch.exp(-log_variance) + log_variance
        ).mean()
    return value + 0.35 * joint + 0.02 * uncertainty


def predict(
    torch: Any,
    model: Any,
    architecture: str,
    contexts: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    explicit = torch.as_tensor(
        explicit_action_features(),
        dtype=torch.float32,
        device=device,
    )
    categories = torch.as_tensor(
        (ACTION_NORMALIZED + 1).astype(np.int64),
        dtype=torch.long,
        device=device,
    )
    predictions = []
    uncertainties = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            context = torch.as_tensor(
                (contexts[start : start + batch_size] - mean) / scale,
                dtype=torch.float32,
                device=device,
            )
            prediction, log_variance = forward_call(
                model,
                architecture,
                context,
                explicit,
                categories,
            )
            predictions.append(prediction.float().cpu().numpy())
            if log_variance is not None:
                uncertainties.append(log_variance.float().cpu().numpy())
    return (
        np.concatenate(predictions),
        None if not uncertainties else np.concatenate(uncertainties),
    )


def main() -> None:
    args = parse_args()
    experiment = EXPERIMENTS[args.experiment]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f"forward_{args.experiment}_v10.pt"
    report_path = run_dir / f"forward_{args.experiment}_v10.json"
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite experiment {args.experiment}")
    started = time.perf_counter()
    seed = int(config["model_seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    aligned_rows = read_jsonl(data_dir / "grids" / "train.jsonl")
    if args.smoke:
        aligned_rows = aligned_rows[: min(8, len(aligned_rows))]
    if experiment["data"] == "existing_distribution":
        rows, source_provenance = existing_distribution_sample(
            list(config["existing_distribution_sources"]),
            len(aligned_rows),
            seed,
        )
    else:
        rows = aligned_rows
        source_provenance = [
            {
                "path": str(data_dir / "grids" / "train.jsonl"),
                "sha256": sha256_file(data_dir / "grids" / "train.jsonl"),
            }
        ]
    category_to_regime = {
        "iid_expanded": "ordinary",
        "high_nonlinearity": "tolerance_boundary",
        "ood_boundary": "camera_boundary",
        "hard_interaction": "high_offset_interaction",
        "peak_interaction": "focusing",
    }
    for row in rows:
        row.setdefault(
            "natural_requested_action_index",
            natural_requested_action_index(str(row["group_id"]), seed),
        )
        row.setdefault(
            "regime",
            category_to_regime.get(
                str(row.get("source_category", "")),
                "ordinary",
            ),
        )
    contexts, targets = arrays(rows)
    training, calibration = partition(
        [str(row["group_id"]) for row in rows],
        seed,
    )
    context_mean = contexts[training].mean(axis=0)
    context_scale = contexts[training].std(axis=0)
    context_scale[context_scale < 1e-6] = 1.0
    training_config = config["forward_training"]
    model_config = {
        "dimension": int(training_config["dimension"]),
        "dropout": float(training_config["dropout"]),
        "heads": 4,
        "layers": 2,
        "explicit_action_dim": int(explicit_action_features().shape[1]),
    }
    model = build_forward_model(
        torch,
        str(experiment["architecture"]),
        model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = (
        int(training_config["smoke_epochs"])
        if args.smoke
        else int(training_config["epochs"])
    )
    batch_size = min(int(training_config["batch_size"]), len(training))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(training_config["learning_rate"]) * 0.05,
    )
    explicit = torch.as_tensor(
        explicit_action_features(),
        dtype=torch.float32,
        device=device,
    )
    categories = torch.as_tensor(
        (ACTION_NORMALIZED + 1).astype(np.int64),
        dtype=torch.long,
        device=device,
    )
    rng = np.random.default_rng(seed + 17)
    best_key = None
    best_state = None
    best_epoch = 0
    stale = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(training)
        running = 0.0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            context = torch.as_tensor(
                (contexts[indices] - context_mean) / context_scale,
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(
                targets[indices],
                dtype=torch.float32,
                device=device,
            )
            prediction, log_variance = forward_call(
                model,
                str(experiment["architecture"]),
                context,
                explicit,
                categories,
            )
            loss = loss_value(
                torch,
                prediction,
                log_variance,
                target,
                str(experiment["objective"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
        scheduler.step()
        calibration_prediction, calibration_uncertainty = predict(
            torch,
            model,
            str(experiment["architecture"]),
            contexts[calibration],
            context_mean,
            context_scale,
            device,
            batch_size,
        )
        calibration_rows = [rows[int(index)] for index in calibration]
        calibration_metrics, _ = forward_metrics(
            calibration_rows,
            calibration_prediction,
            targets[calibration],
            calibration_uncertainty,
            int(config["bootstrap_seed"]),
            draws=200 if args.smoke else 500,
        )
        primary = calibration_metrics["natural_requested_action"]["strict_all_five"]
        surface = calibration_metrics["full_81_action_surface"]["strict_all_five"]
        key = (
            float(primary["mean"]),
            float(surface["mean"]),
            -float(np.abs(calibration_prediction - targets[calibration]).mean()),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
            )
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            record = {
                "epoch": epoch,
                "training_loss": running / len(training),
                "calibration_natural_strict": primary,
                "calibration_surface_strict": surface,
                "best_epoch": best_epoch,
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if (
            not args.smoke
            and stale >= int(training_config["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    artifact = {
        "version": "physics_structured_rebuild_v10_forward_surface",
        "experiment": args.experiment,
        "experiment_contract": experiment,
        "architecture": experiment["architecture"],
        "objective": experiment["objective"],
        "model_config": model_config,
        "state_dict": best_state,
        "context_mean": context_mean,
        "context_scale": context_scale,
        "seed": seed,
        "training_group_ids": [str(rows[int(index)]["group_id"]) for index in training],
        "calibration_group_ids": [str(rows[int(index)]["group_id"]) for index in calibration],
        "source_provenance": source_provenance,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "development_used_for_checkpoint_selection": False,
        "locked_test_files_opened": [],
        "old_system_evaluation_files_opened": [],
        "best_epoch": best_epoch,
    }
    torch.save(artifact, output)
    report = {
        "version": artifact["version"],
        "experiment": args.experiment,
        "experiment_contract": experiment,
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "group_split": {
            "training": int(len(training)),
            "internal_calibration": int(len(calibration)),
        },
        "best_epoch": best_epoch,
        "maximum_epochs": epochs,
        "trace": trace,
        "source_provenance": source_provenance,
        "smoke": bool(args.smoke),
        "seconds": time.perf_counter() - started,
        "complete": True,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
