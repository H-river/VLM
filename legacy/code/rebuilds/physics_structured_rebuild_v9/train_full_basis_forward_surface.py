#!/usr/bin/env python3
"""Train a complete 80-term action-basis surface on existing clean grids."""

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

from control_rebuild_v3.common import ACTION_NORMALIZED, action_basis
from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from joint_forward_direction_v7.runtime import load_forward_direction_runtime_v7
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    full_ternary_action_basis,
)
from physics_structured_rebuild_v9.grouped_forward_surface import (
    grouped_context_features,
    grouped_forward_surface_model,
)
from physics_structured_rebuild_v9.train_boundary_direction_correction import (
    group_partitions,
)
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import sha256
from specialist_rebuild_v2.common import read_jsonl

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed/shared_forward_direction_v7.pt"
)
DEFAULT_NATURAL = DEFAULT_RUN / "clean_nonoverlap_forward_training_features.npz"
DEFAULT_DIRECT = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "full_basis_forward_surface_v9.pt"
DEFAULT_GRIDS = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v5_numerical/grids/train.jsonl",
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/targeted_bundle/grids/train.jsonl",
)
BLENDS = np.asarray(
    [-0.25, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.25],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-train", type=Path, nargs="+", default=list(DEFAULT_GRIDS)
    )
    parser.add_argument("--natural-cache", type=Path, default=DEFAULT_NATURAL)
    parser.add_argument("--direct-exclusion", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--base-forward", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--natural-weight", type=float, default=2.0)
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
                error = np.abs(
                    prior + correction * proposal[None, None, :] - target
                )
                passed = error <= 1.0
                key = (
                    int(np.all(passed, axis=2).sum()),
                    int(passed.sum()),
                    -float(error.mean()),
                    -float(np.abs(proposal).sum()),
                )
                candidates.append((key, proposal))
            best = max(candidates, key=lambda row: row[0])
            if not np.array_equal(selected, best[1]):
                selected = best[1]
                changed = True
        trace.append(
            {
                "pass": pass_index + 1,
                "blend": selected.tolist(),
                "metric": metric(
                    prior + correction * selected[None, None, :], target
                ),
            }
        )
        if not changed:
            break
    return selected, trace


def group_ids(path: Path) -> np.ndarray:
    return np.asarray(
        [str(row["group_id"]) for row in read_jsonl(path)], dtype=np.str_
    )


def model_correction(
    torch: Any,
    model: Any,
    features: np.ndarray,
    coefficient_scale: np.ndarray,
    basis: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            values = model(
                torch.as_tensor(
                    features[batch], dtype=torch.float32, device=device
                )
            )
            outputs.append(values.float().cpu().numpy())
    coefficients = np.concatenate(outputs, axis=0)
    coefficients *= coefficient_scale[None, :, :]
    correction = np.einsum("ab,gbf->gaf", basis, coefficients)
    correction[:, 40, :] = 0.0
    return correction.astype(np.float32)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    torch, device = configure(int(args.seed), args.device)
    forward_path = args.base_forward.resolve()
    forward, _ = load_forward_direction_runtime_v7(
        forward_path, torch, device
    )
    contexts_parts = []
    prior_parts = []
    target_parts = []
    id_parts = []
    source_parts = []
    source_report = []
    for source_index, raw_path in enumerate(args.grid_train):
        path = raw_path.resolve()
        arrays = load_grid_arrays(path, include_legacy_features=False)
        prior = stream_forward_changes(path, forward).reshape(
            arrays.group_count, 81, 5
        )
        contexts = arrays.features.reshape(
            arrays.group_count, 81, -1
        )[:, 40, :17]
        targets = arrays.normalized_changes.reshape(
            arrays.group_count, 81, 5
        )
        ids = group_ids(path)
        if len(ids) != arrays.group_count:
            raise ValueError("grid identifiers differ from loaded groups")
        contexts_parts.append(contexts.astype(np.float32))
        prior_parts.append(prior.astype(np.float32))
        target_parts.append(targets.astype(np.float32))
        id_parts.append(ids)
        source_parts.append(
            np.full(arrays.group_count, source_index, dtype=np.int64)
        )
        source_report.append(
            {"path": str(path), "group_count": int(arrays.group_count)}
        )
    natural_path = args.natural_cache.resolve()
    with np.load(natural_path, allow_pickle=False) as natural:
        natural_features = np.asarray(
            natural["grid_features"], dtype=np.float32
        )
        natural_groups = np.asarray(natural["group_ids"], dtype=np.str_)
        natural_count = len(natural_groups)
        contexts_parts.append(
            natural_features.reshape(natural_count, 81, -1)[:, 40, :17]
        )
        prior_parts.append(
            np.asarray(
                natural["grid_base_prediction"], dtype=np.float32
            ).reshape(natural_count, 81, 5)
        )
        target_parts.append(
            np.asarray(
                natural["grid_target_normalized"], dtype=np.float32
            ).reshape(natural_count, 81, 5)
        )
        id_parts.append(natural_groups)
        source_parts.append(
            np.full(natural_count, len(source_report), dtype=np.int64)
        )
    source_report.append(
        {"path": str(natural_path), "group_count": int(natural_count)}
    )
    contexts = np.concatenate(contexts_parts)
    prior = np.concatenate(prior_parts)
    target = np.concatenate(target_parts)
    ids = np.concatenate(id_parts)
    sources = np.concatenate(source_parts)
    if len(set(map(str, ids))) != len(ids):
        raise ValueError("complete-basis training group identifiers overlap")
    excluded = {
        str(row["group_id"])
        for row in read_jsonl(args.direct_exclusion.resolve())
    }
    overlap = excluded & set(map(str, ids))
    if overlap:
        raise ValueError(
            f"complete-basis training overlaps {len(overlap)} direct groups"
        )
    training, calibration, confirmation = group_partitions(ids, int(args.seed))
    input_basis = np.asarray(action_basis(), dtype=np.float32)
    input_pseudoinverse = np.linalg.pinv(input_basis).astype(np.float32)
    full_basis = full_ternary_action_basis()
    full_pseudoinverse = np.linalg.pinv(full_basis).astype(np.float32)
    prior_coefficients = np.einsum(
        "ba,gaf->gbf", input_pseudoinverse, prior
    )
    residual = target - prior
    residual_coefficients = np.einsum(
        "ba,gaf->gbf", full_pseudoinverse, residual
    ).astype(np.float32)
    raw_features = np.concatenate(
        [
            grouped_context_features(contexts),
            prior_coefficients.reshape(len(ids), -1),
        ],
        axis=1,
    ).astype(np.float32)
    input_mean = raw_features[training].mean(axis=0)
    input_scale = raw_features[training].std(axis=0)
    input_scale[input_scale < 1e-6] = 1.0
    features = (raw_features - input_mean) / input_scale
    coefficient_scale = residual_coefficients[training].std(axis=0)
    coefficient_scale[coefficient_scale < 1e-5] = 1.0
    config = {
        "input_dim": int(features.shape[1]),
        "coefficient_count": 80,
        "width": int(args.width),
        "blocks": int(args.blocks),
        "dropout": float(args.dropout),
        "runtime_batch_size": 128,
    }
    model = grouped_forward_surface_model(
        torch,
        config["input_dim"],
        config["coefficient_count"],
        width=config["width"],
        blocks=config["blocks"],
        dropout=config["dropout"],
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
    basis_tensor = torch.as_tensor(
        full_basis, dtype=torch.float32, device=device
    )
    scale_tensor = torch.as_tensor(
        coefficient_scale, dtype=torch.float32, device=device
    )
    field_weight = torch.as_tensor(
        [1.15, 1.15, 0.85, 0.85, 1.60],
        dtype=torch.float32,
        device=device,
    )
    complexity = np.count_nonzero(
        np.abs(ACTION_NORMALIZED) > 0.0,
        axis=1,
    ).astype(np.float32)
    action_weight = torch.as_tensor(
        1.0 + 0.12 * complexity,
        dtype=torch.float32,
        device=device,
    )
    group_weight = np.ones(len(ids), dtype=np.float32)
    group_weight[sources == len(source_report) - 1] = float(
        args.natural_weight
    )
    rng = np.random.default_rng(int(args.seed) + 1201)
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
            raw = model(
                torch.as_tensor(
                    features[indices], dtype=torch.float32, device=device
                )
            )
            coefficients = raw * scale_tensor
            correction = torch.einsum(
                "ab,gbf->gaf", basis_tensor, coefficients
            )
            prior_tensor = torch.as_tensor(
                prior[indices], dtype=torch.float32, device=device
            )
            target_tensor = torch.as_tensor(
                target[indices], dtype=torch.float32, device=device
            )
            error = prior_tensor + correction - target_tensor
            smooth = torch.nn.functional.smooth_l1_loss(
                prior_tensor + correction,
                target_tensor,
                reduction="none",
                beta=0.25,
            )
            boundary = torch.nn.functional.softplus(
                (error.abs() - 0.85) * 5.0
            ) / 5.0
            joint = torch.logsumexp(error.abs() * 3.0, dim=2) / 3.0
            weights = torch.as_tensor(
                group_weight[indices], dtype=torch.float32, device=device
            )
            loss = (
                (
                    smooth
                    * field_weight[None, None, :]
                    * action_weight[None, :, None]
                    * weights[:, None, None]
                ).mean()
                + 0.30
                * (
                    boundary
                    * field_weight[None, None, :]
                    * action_weight[None, :, None]
                    * weights[:, None, None]
                ).mean()
                + 0.15
                * (
                    joint * action_weight[None, :] * weights[:, None]
                ).mean()
                + 0.001 * raw.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
        scheduler.step()
        correction = model_correction(
            torch,
            model,
            features,
            coefficient_scale,
            full_basis,
            calibration,
            int(args.batch_size),
            device,
        )
        metrics = metric(prior[calibration] + correction, target[calibration])
        key = (
            metrics["strict_all_five_count"],
            sum(metrics["per_field_tolerance_pass"]),
            -metrics["mae_in_tolerance_units"],
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
        if epoch == 1 or epoch % 5 == 0:
            record = {
                "epoch": epoch,
                "train_loss": running / len(training),
                "calibration": metrics,
                "best_epoch": best_epoch,
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("complete-basis training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration_correction = model_correction(
        torch,
        model,
        features,
        coefficient_scale,
        full_basis,
        calibration,
        int(args.batch_size),
        device,
    )
    blend, blend_trace = choose_blend(
        prior[calibration], calibration_correction, target[calibration]
    )
    confirmation_correction = model_correction(
        torch,
        model,
        features,
        coefficient_scale,
        full_basis,
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
        "version": "full_basis_forward_surface_v9_one_seed",
        "model": "complete_ternary_basis_forward_surface_v9",
        "seed": int(args.seed),
        "state_dict": best_state,
        "config": config,
        "input_mean": input_mean,
        "input_scale": input_scale,
        "coefficient_scale": coefficient_scale,
        "field_blend": blend,
        "base_forward": str(forward_path),
        "base_forward_sha256": sha256(forward_path),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "source_groups": source_report,
        "group_count": int(len(ids)),
        "transition_count": int(len(ids) * 81),
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
