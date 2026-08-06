#!/usr/bin/env python3
"""Train a zero-anchored physics-baseline plus neural-residual forward model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    ACTION_GRID,
    group_arrays,
    inverse_pair_arrays,
    read_jsonl,
    residual_cost,
    select_minimum_cost,
)
from control_rebuild_v3.evaluate_controlled import selection_metrics
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.train_forward import (
    configure,
    forward_metrics,
    iter_batches,
)
from control_rebuild_v4.assess_validation import THRESHOLDS
from control_rebuild_v4.inverse_data import derived_inverse_pairs
from control_rebuild_v4.models import anchored_forward_residual_model
from control_rebuild_v4.selection_gates import (
    gate_aware_selection_key,
    gate_margin_summary,
)
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    forward_feature,
    matching_mask,
    raw_state_array,
    stable_token,
)

DEFAULT_V2_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_V4_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_numerical"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_V3_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v3_one_seed/forward_control_v3_calibrated.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-data", type=Path, default=DEFAULT_V2_DATA)
    parser.add_argument("--v4-data", type=Path, default=DEFAULT_V4_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-groups", type=int, default=48)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-v4-train-groups", type=int)
    parser.add_argument("--v4-train-repeat", type=int, default=1)
    parser.add_argument(
        "--v3-forward-artifact",
        type=Path,
        default=DEFAULT_V3_FORWARD,
    )
    return parser.parse_args()


def feature_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    return np.asarray(
        [
            [
                forward_feature(row["setup"], row["current_beam_state"], action)
                for action in ACTION_GRID
            ]
            for row in rows
        ],
        dtype=np.float32,
    )


def selection_schedule(
    rows: Sequence[Mapping[str, Any]], target_count: int
) -> tuple[np.ndarray, np.ndarray]:
    desired, positives = [], []
    for row in rows:
        states = np.asarray(
            [
                raw_state_array(candidate["next_state"])
                for candidate in row["candidates"]
            ],
            dtype=np.float32,
        )
        group_desired, group_positive = [], []
        used = set()
        offset = int(stable_token(row["group_id"], "v4_forward_schedule")[:8], 16)
        for target_number in range(target_count):
            index = (offset + 23 * target_number) % len(ACTION_GRID)
            while index in used:
                index = (index + 1) % len(ACTION_GRID)
            used.add(index)
            group_desired.append(states[index])
            group_positive.append(matching_mask(states, states[index]))
        desired.append(group_desired)
        positives.append(group_positive)
    return (
        np.asarray(desired, dtype=np.float32),
        np.asarray(positives, dtype=np.bool_),
    )


def predicted_changes(
    torch: Any,
    model: Any,
    features: np.ndarray,
    zero_features: np.ndarray,
    baseline: np.ndarray,
    device: Any,
    residual_alpha: float = 1.0,
    batch_size: int = 256,
) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            residual = (
                model(
                    torch.as_tensor(
                        features[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        zero_features[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                .float()
                .cpu()
                .numpy()
            )
            parts.append(residual)
    output = baseline + float(residual_alpha) * np.concatenate(parts)
    output[:, 40, :] = 0.0
    return output


def inverse_selection(
    current: np.ndarray,
    tolerance: np.ndarray,
    predicted: np.ndarray,
    desired: np.ndarray,
    positives: np.ndarray,
) -> dict[str, Any]:
    states = current[:, None, :] + predicted * tolerance[:, None, :]
    success_parts = []
    for target_index in range(desired.shape[1]):
        costs = residual_cost(states, desired[:, target_index, None, :])
        selected = select_minimum_cost(costs)
        success_parts.append(
            positives[
                np.arange(len(positives)),
                target_index,
                selected,
            ]
        )
    success = np.concatenate(success_parts)
    return {
        "request_count": int(len(success)),
        "target_success": float(success.mean()),
    }


def exact_forward_retrieval(
    rows: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    tolerance: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    predicted_states = current[:, None, :] + predicted * tolerance[:, None, :]
    pairs = derived_inverse_pairs(rows)
    grid_map = {str(row["group_id"]): row for row in rows}
    group_ids = [str(row["group_id"]) for row in rows]
    position = {group_id: index for index, group_id in enumerate(group_ids)}
    group_index, _, desired, positives, _ = inverse_pair_arrays(
        pairs,
        grid_map,
        position,
    )
    selected_truth = np.asarray(
        [
            -1 if row["selected_index"] is None else int(row["selected_index"])
            for row in pairs
        ],
        dtype=np.int64,
    )
    scores = -residual_cost(
        predicted_states[group_index],
        desired[:, None, :],
    )
    return selection_metrics(scores, positives, selected_truth)


def evaluate_split(
    torch: Any,
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    zero_features: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    tolerance: np.ndarray,
    desired: np.ndarray,
    positives: np.ndarray,
    device: Any,
    residual_alpha: float = 1.0,
) -> dict[str, Any]:
    predicted = predicted_changes(
        torch,
        model,
        features,
        zero_features,
        baseline,
        device,
        residual_alpha,
    )
    return {
        "groups": len(rows),
        "transitions": len(rows) * len(ACTION_GRID),
        "forward": forward_metrics(target, predicted),
        "forward_cost_only": exact_forward_retrieval(
            rows,
            current,
            tolerance,
            predicted,
        ),
        "inverse_selection": inverse_selection(
            current, tolerance, predicted, desired, positives
        ),
    }


def evaluate_frozen_forward(
    rows: Sequence[Mapping[str, Any]],
    runtime: Any,
) -> dict[str, Any]:
    _, current, tolerance, target, _ = group_arrays(rows)
    desired, positives = selection_schedule(rows, 3)
    predicted = runtime.predict_changes(rows)
    return {
        "groups": len(rows),
        "transitions": len(rows) * len(ACTION_GRID),
        "forward": forward_metrics(target, predicted),
        "forward_cost_only": exact_forward_retrieval(
            rows,
            current,
            tolerance,
            predicted,
        ),
        "inverse_selection": inverse_selection(
            current,
            tolerance,
            predicted,
            desired,
            positives,
        ),
    }


def main() -> None:
    from sklearn.linear_model import Ridge

    args = parse_args()
    v2_data = args.v2_data.resolve()
    v4_data = args.v4_data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(20260726, args.device)
    started = time.perf_counter()

    v2_train = read_jsonl(v2_data / "grids/train.jsonl")
    v2_val = read_jsonl(v2_data / "grids/val.jsonl")
    v4_train_unique = read_jsonl(v4_data / "grids/train.jsonl")
    v4_val = read_jsonl(v4_data / "grids/val.jsonl")
    if args.max_v4_train_groups is not None:
        v4_train_unique = v4_train_unique[: args.max_v4_train_groups]
    v4_train_repeat = int(args.v4_train_repeat)
    if v4_train_repeat < 1:
        raise ValueError("--v4-train-repeat must be positive")
    v4_train = v4_train_unique * v4_train_repeat
    train_rows = [*v2_train, *v4_train]
    ft = feature_arrays(train_rows)
    fvi = feature_arrays(v2_val)
    fvo = feature_arrays(v4_val)
    _, current_t, tolerance_t, target_t, _ = group_arrays(train_rows)
    _, current_vi, tolerance_vi, target_vi, _ = group_arrays(v2_val)
    _, current_vo, tolerance_vo, target_vo, _ = group_arrays(v4_val)

    mean = ft.reshape(-1, ft.shape[-1]).mean(axis=0)
    scale = ft.reshape(-1, ft.shape[-1]).std(axis=0)
    scale[scale < 1e-7] = 1.0
    ft = ((ft - mean) / scale).astype(np.float32)
    fvi = ((fvi - mean) / scale).astype(np.float32)
    fvo = ((fvo - mean) / scale).astype(np.float32)
    zt = ft[:, 40, :].copy()
    zvi = fvi[:, 40, :].copy()
    zvo = fvo[:, 40, :].copy()
    difference = ft - zt[:, None, :]
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    ridge.fit(difference.reshape(-1, difference.shape[-1]), target_t.reshape(-1, 5))
    coefficient = np.asarray(ridge.coef_, dtype=np.float32)

    def baseline(features: np.ndarray, zero: np.ndarray) -> np.ndarray:
        output = np.einsum(
            "gaf,kf->gak", features - zero[:, None, :], coefficient
        ).astype(np.float32)
        output[:, 40] = 0.0
        return output

    baseline_t = baseline(ft, zt)
    baseline_vi = baseline(fvi, zvi)
    baseline_vo = baseline(fvo, zvo)
    residual_target = target_t - baseline_t
    desired_t, positives_t = selection_schedule(train_rows, 4)
    desired_vi, positives_vi = selection_schedule(v2_val, 3)
    desired_vo, positives_vo = selection_schedule(v4_val, 3)
    category_indices = {
        category: np.asarray(
            [
                index
                for index, row in enumerate(v4_val)
                if str(row["source_category"]) == category
            ],
            dtype=np.int64,
        )
        for category in sorted({str(row["source_category"]) for row in v4_val})
    }
    required_hard_categories = {"ood_boundary", "high_nonlinearity"}
    if not required_hard_categories.issubset(category_indices):
        raise ValueError("v4 validation lacks a boundary or high-nonlinearity category")
    v3_forward, _ = load_forward_runtime(
        args.v3_forward_artifact.resolve(),
        torch,
        device,
    )
    v3_reference = {
        "all": evaluate_frozen_forward(v4_val, v3_forward),
        "by_category": {
            category: evaluate_frozen_forward(
                [v4_val[index] for index in indices],
                v3_forward,
            )
            for category, indices in category_indices.items()
        },
    }

    model = anchored_forward_residual_model(torch, ft.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    rng = np.random.default_rng(20260726 + 601)
    best_score = float("-inf")
    best_selection_key = None
    best_state = None
    best_epoch = 0
    best_alpha = 1.0
    trace = []
    residual_alpha_candidates = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        for index in iter_batches(len(ft), int(args.batch_groups), rng):
            features = torch.as_tensor(ft[index], dtype=torch.float32, device=device)
            zero = torch.as_tensor(zt[index], dtype=torch.float32, device=device)
            target = torch.as_tensor(
                residual_target[index],
                dtype=torch.float32,
                device=device,
            )
            residual = model(features, zero)
            error = residual - target
            smooth = torch.nn.functional.smooth_l1_loss(residual, target)
            worst = error.abs().max(dim=-1).values.mean()
            predicted = (
                torch.as_tensor(
                    baseline_t[index],
                    dtype=torch.float32,
                    device=device,
                )
                + residual
            )
            states = (
                torch.as_tensor(
                    current_t[index],
                    dtype=torch.float32,
                    device=device,
                )[:, None, :]
                + predicted
                * torch.as_tensor(
                    tolerance_t[index],
                    dtype=torch.float32,
                    device=device,
                )[:, None, :]
            )
            target_number = epoch % desired_t.shape[1]
            desired = torch.as_tensor(
                desired_t[index, target_number],
                dtype=torch.float32,
                device=device,
            )
            positive = torch.as_tensor(
                positives_t[index, target_number],
                dtype=torch.bool,
                device=device,
            )
            delta = states - desired[:, None, :]
            peak_scale = torch.clamp(desired[:, None, 4:5].abs() * 0.02, min=1e-6)
            components = torch.cat(
                [
                    delta[..., 0:2] / 0.5,
                    delta[..., 2:4],
                    delta[..., 4:5] / peak_scale,
                ],
                dim=-1,
            )
            scores = -torch.sqrt(components.square().mean(dim=-1) + 1e-8)
            positive_scores = scores.masked_fill(~positive, -1e9)
            rank = (
                torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)
            ).mean()
            loss = smooth + 0.20 * worst + 0.10 * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)
        calibration_rows = []
        for residual_alpha in residual_alpha_candidates:
            iid = evaluate_split(
                torch,
                model,
                v2_val,
                fvi,
                zvi,
                baseline_vi,
                target_vi,
                current_vi,
                tolerance_vi,
                desired_vi,
                positives_vi,
                device,
                residual_alpha,
            )
            expanded = evaluate_split(
                torch,
                model,
                v4_val,
                fvo,
                zvo,
                baseline_vo,
                target_vo,
                current_vo,
                tolerance_vo,
                desired_vo,
                positives_vo,
                device,
                residual_alpha,
            )
            expanded_by_category = {
                category: evaluate_split(
                    torch,
                    model,
                    [v4_val[index] for index in indices],
                    fvo[indices],
                    zvo[indices],
                    baseline_vo[indices],
                    target_vo[indices],
                    current_vo[indices],
                    tolerance_vo[indices],
                    desired_vo[indices],
                    positives_vo[indices],
                    device,
                    residual_alpha,
                )
                for category, indices in category_indices.items()
            }
            score = (
                0.25 * iid["forward"]["strict_all_five_success"]
                + 0.25 * iid["forward_cost_only"]["target_success_feasible"]
                + 0.25 * expanded["forward"]["strict_all_five_success"]
                + 0.25 * expanded["forward_cost_only"]["target_success_feasible"]
                - 0.01
                * (
                    iid["forward"]["mae_in_tolerance_units"]
                    + expanded["forward"]["mae_in_tolerance_units"]
                )
            )
            gate_summary = gate_margin_summary(
                {
                    "same_distribution_overall_forward_delta": (
                        expanded["forward"]["strict_all_five_success"]
                        - v3_reference["all"]["forward"]["strict_all_five_success"]
                        - THRESHOLDS["same_distribution_overall_forward_delta"]
                    ),
                    "same_distribution_overall_forward_retrieval_delta": (
                        expanded["forward_cost_only"]["target_success_feasible"]
                        - v3_reference["all"]["forward_cost_only"][
                            "target_success_feasible"
                        ]
                        - THRESHOLDS[
                            "same_distribution_overall_forward_retrieval_delta"
                        ]
                    ),
                    "ood_boundary_forward_delta": (
                        expanded_by_category["ood_boundary"]["forward"][
                            "strict_all_five_success"
                        ]
                        - v3_reference["by_category"]["ood_boundary"]["forward"][
                            "strict_all_five_success"
                        ]
                        - THRESHOLDS["hard_category_delta_floor"]
                    ),
                    "high_nonlinearity_forward_delta": (
                        expanded_by_category["high_nonlinearity"]["forward"][
                            "strict_all_five_success"
                        ]
                        - v3_reference["by_category"]["high_nonlinearity"]["forward"][
                            "strict_all_five_success"
                        ]
                        - THRESHOLDS["hard_category_delta_floor"]
                    ),
                    "old_iid_forward_floor": (
                        iid["forward"]["strict_all_five_success"]
                        - THRESHOLDS["iid_forward_floor"]
                    ),
                }
            )
            calibration_rows.append(
                {
                    "residual_alpha": residual_alpha,
                    "score": score,
                    "validation_iid": iid,
                    "validation_expanded": expanded,
                    "validation_expanded_by_category": expanded_by_category,
                    "checkpoint_selection_gates": gate_summary,
                }
            )
        selected_calibration = max(
            calibration_rows,
            key=lambda item: gate_aware_selection_key(
                item["checkpoint_selection_gates"],
                item["score"],
                -abs(float(item["residual_alpha"]) - 1.0),
            ),
        )
        score = float(selected_calibration["score"])
        selection_key = gate_aware_selection_key(
            selected_calibration["checkpoint_selection_gates"],
            score,
            -abs(float(selected_calibration["residual_alpha"]) - 1.0),
        )
        row = {
            "epoch": epoch,
            "train_loss": running / len(ft),
            "score": score,
            "selected_residual_alpha": float(selected_calibration["residual_alpha"]),
            "validation_iid": selected_calibration["validation_iid"],
            "validation_expanded": selected_calibration["validation_expanded"],
            "validation_expanded_by_category": selected_calibration[
                "validation_expanded_by_category"
            ],
            "checkpoint_selection_gates": selected_calibration[
                "checkpoint_selection_gates"
            ],
            "checkpoint_selection_key": list(selection_key),
            "residual_calibration_candidates": calibration_rows,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best_selection_key is None or selection_key > best_selection_key:
            best_score = score
            best_selection_key = selection_key
            best_epoch = epoch
            best_alpha = float(selected_calibration["residual_alpha"])
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("forward v4 produced no checkpoint")
    artifact_path = output_dir / "forward_physics_residual_v4.pt"
    torch.save(
        {
            "version": "control_rebuild_v4_one_seed",
            "seed": 20260726,
            "model": "anchored_physics_residual_forward_v4",
            "state_dict": best_state,
            "input_dim": int(ft.shape[-1]),
            "feature_mean": mean,
            "feature_scale": scale,
            "ridge_coefficient": coefficient,
            "residual_alpha": best_alpha,
            "action_grid": ACTION_GRID,
            "zero_action_exact": True,
            "training_sources": {
                "v2_train_groups": len(v2_train),
                "v4_unique_train_groups": len(v4_train_unique),
                "v4_train_repeat": v4_train_repeat,
                "v4_train_groups": len(v4_train),
            },
            "checkpoint_selection": (
                "validation_only_gate_count_then_worst_margin_then_composite"
            ),
        },
        artifact_path,
    )
    summary = {
        "version": "control_rebuild_v4_one_seed",
        "artifact": str(artifact_path.resolve()),
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "unique_train_groups": len(v2_train) + len(v4_train_unique),
        "train_groups": len(train_rows),
        "train_transitions": len(train_rows) * 81,
        "epochs": int(args.epochs),
        "best_epoch": best_epoch,
        "best_residual_alpha": best_alpha,
        "best_composite_score": best_score,
        "best_selection_key": list(best_selection_key),
        "v3_selection_validation_reference": v3_reference,
        "v4_unique_train_groups": len(v4_train_unique),
        "v4_train_repeat": v4_train_repeat,
        "validation": trace[best_epoch - 1],
        "trace": trace,
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "forward_physics_residual_v4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
