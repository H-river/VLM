#!/usr/bin/env python3
"""Train a strict-all-five neural correction on the frozen existing cache."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import tolerance_from_current
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.strict_forward_runtime import (
    build_strict_forward_model,
    sha256,
)
from physics_structured_rebuild_v9.train_forward_selector_extension import (
    ensemble_prediction,
    load_ensemble,
)
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    forward_feature,
    read_jsonl,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "qwen_combined_round2_forward_training_features.npz"
DEFAULT_CURRENT = DEFAULT_RUN / "forward_selector_ensemble_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "strict_forward_correction_v9.pt"
DEFAULT_DIRECT_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
BLENDS = np.asarray(
    [0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.25],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--current-selector", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument(
        "--direct-data",
        type=Path,
        help=(
            "Optional existing direct-request training data to add without "
            "generating new targets."
        ),
    )
    parser.add_argument(
        "--direct-repeat",
        type=int,
        default=32,
        help="Training/calibration sampling weight for each direct request.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    return parser.parse_args()


def group_partitions(
    group_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bucket = np.asarray(
        [
            int(
                hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16],
                16,
            )
            % 10
            for group_id in group_ids
        ],
        dtype=np.int64,
    )
    return (
        np.flatnonzero(bucket >= 2),
        np.flatnonzero(bucket == 1),
        np.flatnonzero(bucket == 0),
    )


def rows(groups: np.ndarray) -> np.ndarray:
    offsets = groups[:, None] * 81 + np.arange(81)[None, :]
    return offsets.reshape(-1)


def direct_arrays(
    direct_path: Path,
    current_path: Path,
    torch: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct = read_jsonl(direct_path)
    runtime_rows = [
        {
            "group_id": str(row["group_id"]),
            "setup": row["setup"],
            "current_beam_state": row["current_beam_state"],
        }
        for row in direct
    ]
    runtime, _ = load_forward_selector_ensemble_runtime_v9(
        current_path,
        torch,
        device,
    )
    current_grid = runtime.predict_changes(runtime_rows)
    prior_grid = runtime.base.predict_changes(runtime_rows)
    selected_actions = np.asarray(
        [action_index(row["action"]) for row in direct],
        dtype=np.int64,
    )
    row_index = np.arange(len(direct), dtype=np.int64)
    current = current_grid[row_index, selected_actions].astype(np.float32)
    prior = prior_grid[row_index, selected_actions].astype(np.float32)
    physical = np.asarray(
        [
            forward_feature(
                row["setup"],
                row["current_beam_state"],
                row["action"],
            )
            for row in direct
        ],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            np.asarray(
                [
                    float(row["truth_change"][field])
                    for field in STATE_FIELDS
                ],
                dtype=np.float32,
            )
            / tolerance_from_current(row["current_beam_state"])
            for row in direct
        ],
        dtype=np.float32,
    )
    values = np.concatenate([physical, prior, current], axis=1).astype(
        np.float32
    )
    group_ids = np.asarray(
        [str(row["group_id"]) for row in direct],
        dtype=np.str_,
    )
    return values, current, target, group_ids


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    error = np.abs(prediction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return {
        "count": int(len(exact)),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "per_field_tolerance_pass": passed.mean(axis=0).tolist(),
        "mae_in_tolerance_units": float(error.mean()),
        "p90_maximum_error": float(
            np.quantile(np.max(error, axis=1), 0.90)
        ),
    }


def selection_key(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[int, int, float, float]:
    error = np.abs(prediction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return (
        int(exact.sum()),
        int(passed.sum()),
        -float(error.mean()),
        -float(np.max(error, axis=1).mean()),
    )


def calibrate_blends(
    current: np.ndarray,
    correction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, float, float]]:
    starts = (0.0, 0.25, 0.50, 1.0)
    finals = []
    for start in starts:
        selected = np.full(5, start, dtype=np.float32)
        for _ in range(4):
            changed = False
            for field in range(5):
                best = None
                for blend in BLENDS:
                    proposal = selected.copy()
                    proposal[field] = float(blend)
                    prediction = current + correction * proposal[None, :]
                    candidate = (
                        selection_key(prediction, target),
                        -float(np.abs(proposal).sum()),
                        -float(blend),
                        proposal,
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
                assert best is not None
                if not np.array_equal(selected, best[3]):
                    selected = best[3]
                    changed = True
            if not changed:
                break
        prediction = current + correction * selected[None, :]
        finals.append(
            (
                selection_key(prediction, target),
                -float(np.abs(selected).sum()),
                selected,
            )
        )
    best = max(finals, key=lambda value: value[:2])
    return best[2], best[0]


def predict_correction(
    torch: Any,
    model: Any,
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    correction_scale: np.ndarray,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            correction = model(
                (
                    batch
                    - torch.as_tensor(mean, device=device)
                )
                / torch.as_tensor(scale, device=device)
            )
            correction = correction * torch.as_tensor(
                correction_scale,
                device=device,
            )
            output.append(correction.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    if int(args.direct_repeat) < 1:
        raise ValueError("direct-repeat must be positive")
    started = time.perf_counter()
    torch, device = configure(int(args.seed), args.device)
    random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    cache_path = args.training_cache.resolve()
    with np.load(cache_path, allow_pickle=False) as cache:
        physical_plus_prior = np.asarray(
            cache["grid_features"],
            dtype=np.float32,
        )
        prior = np.asarray(
            cache["grid_base_prediction"],
            dtype=np.float32,
        )
        target = np.asarray(
            cache["grid_target_normalized"],
            dtype=np.float32,
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    current_path = args.current_selector.resolve()
    current_artifact = load_ensemble(current_path)
    current = ensemble_prediction(
        current_artifact,
        physical_plus_prior,
        prior,
    )
    values = np.concatenate(
        [physical_plus_prior, current],
        axis=1,
    ).astype(np.float32)
    surface_count = len(values)
    train_groups, calibration_groups, test_groups = group_partitions(
        group_ids,
        int(args.seed),
    )
    surface_train_indices = rows(train_groups)
    surface_calibration_indices = rows(calibration_groups)
    surface_test_indices = rows(test_groups)
    direct_path = (
        None if args.direct_data is None else args.direct_data.resolve()
    )
    direct_splits: dict[str, np.ndarray] | None = None
    if direct_path is not None:
        (
            direct_values,
            direct_current,
            direct_target,
            direct_group_ids,
        ) = direct_arrays(
            direct_path,
            current_path,
            torch,
            device,
        )
        if direct_values.shape[1] != values.shape[1]:
            raise ValueError("direct and surface feature widths differ")
        direct_train, direct_calibration, direct_test = group_partitions(
            direct_group_ids,
            int(args.seed),
        )
        offset = len(values)
        direct_splits = {
            "train": offset + direct_train,
            "calibration": offset + direct_calibration,
            "internal_test": offset + direct_test,
        }
        values = np.concatenate([values, direct_values], axis=0)
        current = np.concatenate([current, direct_current], axis=0)
        target = np.concatenate([target, direct_target], axis=0)
        train_indices = np.concatenate(
            [
                surface_train_indices,
                np.repeat(
                    direct_splits["train"], int(args.direct_repeat)
                ),
            ]
        )
        calibration_indices = np.concatenate(
            [
                surface_calibration_indices,
                np.repeat(
                    direct_splits["calibration"],
                    int(args.direct_repeat),
                ),
            ]
        )
    else:
        train_indices = surface_train_indices
        calibration_indices = surface_calibration_indices
    mean = values[train_indices].mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    scale = values[train_indices].std(axis=0, dtype=np.float64).astype(
        np.float32
    )
    scale = np.maximum(scale, 1e-6)
    target_correction = target - current
    correction_scale = target_correction[train_indices].std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    correction_scale = np.maximum(correction_scale, 1e-4)
    architecture = {
        "input_dim": int(values.shape[1]),
        "width": int(args.width),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
    }
    model = build_strict_forward_model(torch, architecture).to(device)
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
    baseline_calibration = metrics(
        current[calibration_indices],
        target[calibration_indices],
    )
    baseline_test = metrics(
        current[surface_test_indices],
        target[surface_test_indices],
    )
    best_state = copy.deepcopy(model.state_dict())
    best_blend = np.zeros(5, dtype=np.float32)
    best_key = selection_key(
        current[calibration_indices],
        target[calibration_indices],
    )
    best_epoch = 0
    stale = 0
    trace = []
    use_amp = device.type == "cuda"
    batch_size = int(args.batch_size)
    mean_tensor = torch.as_tensor(mean, device=device)
    scale_tensor = torch.as_tensor(scale, device=device)
    correction_scale_tensor = torch.as_tensor(
        correction_scale,
        device=device,
    )

    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(train_indices)
        totals = {
            "loss": 0.0,
            "component": 0.0,
            "joint": 0.0,
            "boundary": 0.0,
            "preservation": 0.0,
        }
        seen = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch_values = torch.as_tensor(
                values[selected],
                dtype=torch.float32,
                device=device,
            )
            batch_current = torch.as_tensor(
                current[selected],
                dtype=torch.float32,
                device=device,
            )
            batch_target = torch.as_tensor(
                target[selected],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                correction = (
                    model(
                        (batch_values - mean_tensor) / scale_tensor
                    )
                    * correction_scale_tensor[None, :]
                )
                prediction = batch_current + correction
                error = (prediction - batch_target).abs()
                baseline_error = (batch_current - batch_target).abs()
                component = torch.nn.functional.smooth_l1_loss(
                    prediction,
                    batch_target,
                    beta=0.25,
                    reduction="none",
                ).mean()
                maximum = error.max(dim=1).values
                joint = (
                    torch.nn.functional.softplus(
                        8.0 * (maximum - 0.90)
                    )
                    / 8.0
                ).mean()
                boundary = torch.relu(error - 0.75).square().mean()
                baseline_success = (
                    baseline_error.max(dim=1).values <= 1.0
                )
                if bool(baseline_success.any()):
                    preservation = torch.relu(
                        maximum[baseline_success] - 0.90
                    ).square().mean()
                else:
                    preservation = maximum.new_zeros(())
                loss = (
                    0.45 * component
                    + 1.00 * joint
                    + 0.30 * boundary
                    + 0.75 * preservation
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            count = len(selected)
            for name, value in (
                ("loss", loss),
                ("component", component),
                ("joint", joint),
                ("boundary", boundary),
                ("preservation", preservation),
            ):
                totals[name] += float(value.detach()) * count
            seen += count
        scheduler.step()

        calibration_correction = predict_correction(
            torch,
            model,
            values[calibration_indices],
            mean,
            scale,
            correction_scale,
            device,
            batch_size * 2,
        )
        blend, key = calibrate_blends(
            current[calibration_indices],
            calibration_correction,
            target[calibration_indices],
        )
        selected_prediction = (
            current[calibration_indices]
            + calibration_correction * blend[None, :]
        )
        improved = key > best_key
        if improved:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_blend = blend.copy()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            **{name: value / seen for name, value in totals.items()},
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "calibration": metrics(
                selected_prediction,
                target[calibration_indices],
            ),
            "field_blend": blend.tolist(),
            "selected": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break

    model.load_state_dict(best_state)
    calibration_correction = predict_correction(
        torch,
        model,
        values[calibration_indices],
        mean,
        scale,
        correction_scale,
        device,
        batch_size * 2,
    )
    test_correction = predict_correction(
        torch,
        model,
        values[surface_test_indices],
        mean,
        scale,
        correction_scale,
        device,
        batch_size * 2,
    )
    selected_calibration = (
        current[calibration_indices]
        + calibration_correction * best_blend[None, :]
    )
    selected_test = (
        current[surface_test_indices]
        + test_correction * best_blend[None, :]
    )
    artifact = {
        "version": "strict_forward_correction_v9_one_seed",
        "model": "strict_forward_correction_v9",
        "architecture": architecture,
        "current_selector": str(current_path),
        "current_selector_sha256": sha256(current_path),
        "input_mean": mean,
        "input_scale": scale,
        "correction_scale": correction_scale,
        "field_blend": best_blend,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "training": {
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "group_count": int(len(group_ids)),
            "surface_transition_count": int(surface_count),
            "train_groups": int(len(train_groups)),
            "calibration_groups": int(len(calibration_groups)),
            "internal_test_groups": int(len(test_groups)),
            "direct_data": (
                None if direct_path is None else str(direct_path)
            ),
            "direct_data_sha256": (
                None if direct_path is None else sha256(direct_path)
            ),
            "direct_repeat": int(args.direct_repeat),
            "direct_group_count": int(
                0 if direct_splits is None else sum(
                    len(indices) for indices in direct_splits.values()
                )
            ),
            "best_epoch": int(best_epoch),
            "trace": trace,
        },
        "baseline": {
            "calibration": baseline_calibration,
            "internal_test": baseline_test,
        },
        "selected": {
            "field_blend": best_blend.tolist(),
            "calibration": metrics(
                selected_calibration,
                target[calibration_indices],
            ),
            "internal_test": metrics(
                selected_test,
                target[surface_test_indices],
            ),
        },
        "source_contract": {
            "generated_setups": 0,
            "generated_images": 0,
            "system_validation_used_for_training": False,
            "protected_validation_used_for_training": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    if direct_splits is not None:
        direct_report = {}
        for name in ("calibration", "internal_test"):
            indices = direct_splits[name]
            correction = predict_correction(
                torch,
                model,
                values[indices],
                mean,
                scale,
                correction_scale,
                device,
                batch_size * 2,
            )
            selected = current[indices] + correction * best_blend[None, :]
            direct_report[name] = {
                "baseline": metrics(current[indices], target[indices]),
                "selected": metrics(selected, target[indices]),
            }
        report["direct_request"] = direct_report
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "baseline_calibration": baseline_calibration,
                "selected_calibration": report["selected"]["calibration"],
                "baseline_internal_test": baseline_test,
                "selected_internal_test": report["selected"][
                    "internal_test"
                ],
                "best_epoch": best_epoch,
                "field_blend": best_blend.tolist(),
                "seconds": report["seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
