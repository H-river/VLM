#!/usr/bin/env python3
"""Train the joint 81-candidate forward model for inverse control."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    ACTION_GRID,
    STATE_FIELDS,
    action_basis,
    group_arrays,
    inverse_pair_arrays,
    read_json,
    read_jsonl,
    residual_cost,
    select_minimum_cost,
)
from control_rebuild_v3.models import joint_forward_model, require_torch


DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=DEFAULT_DATA)
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-groups", type=int)
    parser.add_argument("--max-val-groups", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def configure(seed: int, device_name: str | None) -> tuple[Any, Any]:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return torch, device


def iter_batches(
    length: int, batch_size: int, rng: np.random.Generator
) -> Sequence[np.ndarray]:
    order = rng.permutation(length)
    return [
        order[start : start + batch_size]
        for start in range(0, length, batch_size)
    ]


def forward_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = np.abs(np.asarray(predicted) - np.asarray(target))
    passed = error <= 1.0
    return {
        "count": int(error.shape[0] * error.shape[1]),
        "mae_in_tolerance_units": float(error.mean()),
        "worst_field_error_in_tolerance_units": float(
            error.max(axis=-1).mean()
        ),
        "strict_all_five_success": float(np.all(passed, axis=-1).mean()),
        "per_field_tolerance_pass": {
            field: float(passed[..., index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def predicted_states(
    current: np.ndarray, tolerance: np.ndarray, predicted_change: np.ndarray
) -> np.ndarray:
    return (
        current[:, None, :] + predicted_change * tolerance[:, None, :]
    ).astype(np.float32)


def inverse_selection_metrics(
    states: np.ndarray,
    group_indices: np.ndarray,
    desired: np.ndarray,
    positives: np.ndarray,
    statuses: np.ndarray,
) -> dict[str, Any]:
    costs = residual_cost(
        states[group_indices], desired[:, None, :]
    )
    selected = select_minimum_cost(costs)
    success = positives[np.arange(len(positives)), selected]
    feasible = positives.any(axis=1)
    return {
        "pair_count": int(len(positives)),
        "reachable_pair_count": int(feasible.sum()),
        "target_success_feasible": float(success[feasible].mean()),
        "target_success_all": float(success.mean()),
        "mean_selected_true_match": float(success.mean()),
        "predicted_minimum_cost_mean": float(
            costs[np.arange(len(costs)), selected].mean()
        ),
    }


def pair_schedule(
    pairs: Sequence[Mapping[str, Any]],
    group_ids: Sequence[str],
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    by_group: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for pair in pairs:
        matching = list(pair["matching_indices"])
        if not matching:
            continue
        desired = np.asarray(
            [float(pair["desired_beam_state"][field]) for field in STATE_FIELDS],
            dtype=np.float32,
        )
        positive = np.zeros(len(ACTION_GRID), dtype=np.bool_)
        positive[np.asarray(matching, dtype=np.int64)] = True
        by_group[str(pair["group_id"])].append((desired, positive))
    output = [by_group[group_id] for group_id in group_ids]
    if any(not items for items in output):
        missing = [
            group_ids[index] for index, items in enumerate(output) if not items
        ]
        raise RuntimeError(f"groups have no reachable inverse pairs: {missing[:5]}")
    return output


def ranking_batch(
    torch: Any,
    states: Any,
    scheduled: Sequence[list[tuple[np.ndarray, np.ndarray]]],
    group_indices: np.ndarray,
    epoch: int,
    device: Any,
) -> Any:
    desired = []
    positives = []
    for group_index in group_indices:
        items = scheduled[int(group_index)]
        position = (epoch + int(group_index) * 7) % len(items)
        desired.append(items[position][0])
        positives.append(items[position][1])
    desired_tensor = torch.as_tensor(
        np.asarray(desired), dtype=torch.float32, device=device
    )
    positive_tensor = torch.as_tensor(
        np.asarray(positives), dtype=torch.bool, device=device
    )
    delta = states - desired_tensor[:, None, :]
    peak_scale = torch.clamp(
        desired_tensor[:, None, 4:5].abs() * 0.02, min=1e-6
    )
    components = torch.cat(
        [
            delta[..., 0:2] / 0.5,
            delta[..., 2:4],
            delta[..., 4:5] / peak_scale,
        ],
        dim=-1,
    )
    scores = -torch.sqrt(
        torch.mean(components.square(), dim=-1) + 1e-8
    )
    positive_scores = scores.masked_fill(~positive_tensor, -1e9)
    return (
        torch.logsumexp(scores, dim=1)
        - torch.logsumexp(positive_scores, dim=1)
    ).mean()


def predict_all(
    torch: Any,
    model: Any,
    normalized_context: np.ndarray,
    basis_tensor: Any,
    device: Any,
    batch_size: int = 256,
) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(normalized_context), batch_size):
            values = torch.as_tensor(
                normalized_context[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            parts.append(model(values, basis_tensor).float().cpu().numpy())
    return np.concatenate(parts)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    source_manifest = read_json(data_dir / "manifest.json")
    if (
        source_manifest["dataset_version"]
        != config["source_dataset_version"]
    ):
        raise RuntimeError("source dataset version mismatch")
    seed = int(config["seed"])
    torch, device = configure(seed, args.device)
    started = time.perf_counter()

    train_rows = read_jsonl(data_dir / "grids/train.jsonl")
    val_rows = read_jsonl(data_dir / "grids/val.jsonl")
    if args.max_train_groups is not None:
        train_rows = train_rows[: args.max_train_groups]
    if args.max_val_groups is not None:
        val_rows = val_rows[: args.max_val_groups]
    ct, current_t, tolerance_t, target_t, train_group_ids = group_arrays(
        train_rows
    )
    cv, current_v, tolerance_v, target_v, val_group_ids = group_arrays(
        val_rows
    )
    context_mean = ct.mean(axis=0)
    context_scale = ct.std(axis=0)
    context_scale[context_scale < 1e-7] = 1.0
    ct = (ct - context_mean) / context_scale
    cv = (cv - context_mean) / context_scale

    train_group_set = set(train_group_ids)
    val_group_set = set(val_group_ids)
    train_pairs = [
        row
        for row in read_jsonl(data_dir / "inverse/train.jsonl")
        if row["group_id"] in train_group_set
    ]
    val_pairs = [
        row
        for row in read_jsonl(data_dir / "inverse/val.jsonl")
        if row["group_id"] in val_group_set
    ]
    scheduled = pair_schedule(train_pairs, train_group_ids)
    train_grid_map = {row["group_id"]: row for row in train_rows}
    val_grid_map = {row["group_id"]: row for row in val_rows}
    val_position = {
        group_id: index for index, group_id in enumerate(val_group_ids)
    }
    gv, _, dv, pv, sv = inverse_pair_arrays(
        val_pairs, val_grid_map, val_position
    )

    basis = action_basis()
    basis_tensor = torch.as_tensor(
        basis, dtype=torch.float32, device=device
    )
    model = joint_forward_model(torch, ct.shape[1], basis.shape[1]).to(device)
    forward_config = config["forward"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(forward_config["learning_rate"]),
        weight_decay=float(forward_config["weight_decay"]),
    )
    rng = np.random.default_rng(seed + 101)
    epochs = int(args.epochs or forward_config["epochs"])
    batch_size = int(forward_config["batch_groups"])
    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for index in iter_batches(len(ct), batch_size, rng):
            context = torch.as_tensor(
                ct[index], dtype=torch.float32, device=device
            )
            target = torch.as_tensor(
                target_t[index], dtype=torch.float32, device=device
            )
            current = torch.as_tensor(
                current_t[index], dtype=torch.float32, device=device
            )
            tolerance = torch.as_tensor(
                tolerance_t[index], dtype=torch.float32, device=device
            )
            predicted = model(context, basis_tensor)
            error = predicted - target
            smooth = torch.nn.functional.smooth_l1_loss(predicted, target)
            worst_field = error.abs().max(dim=-1).values.mean()
            worst_candidate = error.abs().max(dim=-1).values.max(dim=-1).values.mean()
            states = current[:, None, :] + predicted * tolerance[:, None, :]
            rank = ranking_batch(
                torch, states, scheduled, index, epoch, device
            )
            loss = (
                smooth
                + float(forward_config["worst_field_weight"]) * worst_field
                + float(forward_config["worst_candidate_weight"])
                * worst_candidate
                + float(forward_config["ranking_loss_weight"]) * rank
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)

        predicted_v = predict_all(torch, model, cv, basis_tensor, device)
        metrics = forward_metrics(target_v, predicted_v)
        states_v = predicted_states(current_v, tolerance_v, predicted_v)
        inverse = inverse_selection_metrics(states_v, gv, dv, pv, sv)
        score = (
            inverse["target_success_feasible"]
            + 0.25 * metrics["strict_all_five_success"]
            - 0.03 * metrics["mae_in_tolerance_units"]
        )
        row = {
            "epoch": epoch,
            "train_loss": running / len(ct),
            "score": score,
            "forward": metrics,
            "inverse_selection": inverse,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("forward training produced no checkpoint")
    model.load_state_dict(best_state)
    artifact_path = output_dir / "forward_control_v3.pt"
    torch.save(
        {
            "version": config["version"],
            "seed": seed,
            "model": "joint_grid_forward_control_v3",
            "context_dim": int(ct.shape[1]),
            "basis_dim": int(basis.shape[1]),
            "state_dict": best_state,
            "context_mean": context_mean,
            "context_scale": context_scale,
            "action_basis": basis,
            "action_grid": ACTION_GRID,
            "state_fields": STATE_FIELDS,
            "zero_action_exact": True,
        },
        artifact_path,
    )
    validation = trace[best_epoch - 1]
    summary = {
        "version": config["version"],
        "device": str(device),
        "seed": seed,
        "artifact": str(artifact_path),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "train_groups": len(train_rows),
        "train_transitions": len(train_rows) * len(ACTION_GRID),
        "train_inverse_pairs": len(train_pairs),
        "val_groups": len(val_rows),
        "val_transitions": len(val_rows) * len(ACTION_GRID),
        "val_inverse_pairs": len(val_pairs),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "validation": validation,
        "trace": trace,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "forward_control_v3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

