#!/usr/bin/env python3
"""Train a sensor-frame scorer above frozen measurement and forward modules."""

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
    STATUS_INDEX,
    inverse_candidate_features,
    read_json,
    read_jsonl,
)
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.inverse_runtime import state_mapping
from control_rebuild_v3.models import inverse_ranker_model
from control_rebuild_v3.train_forward import configure, iter_batches
from control_rebuild_v3.train_inverse import (
    feature_statistics,
    inverse_metrics,
    predict_corrections,
)
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from control_rebuild_v3.visual_inverse import (
    MeasurementModuleV3,
    legacy_candidates_to_sensor,
    sensor_to_base_legacy,
    true_grid_sensor_states,
)
from measurement_rebuild_v3.common import iter_jsonl
from measurement_rebuild_v4.runtime import MeasurementCalibratorRuntimeV4
from control_rebuild_v4.assess_validation import THRESHOLDS
from control_rebuild_v4.selection_gates import (
    gate_aware_selection_key,
    gate_margin_summary,
)
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    inverse_context,
    matching_mask,
    minimum_motion_index,
    raw_state_array,
)

DEFAULT_CONTROL_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_MEASUREMENT_DATA = REPO_ROOT.parent / "VLM_data/measurement_rebuild_v3"
DEFAULT_V3_CONTROL_RUN = REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed"
DEFAULT_MEASUREMENT_RUN = REPO_ROOT.parent / "VLM_runs/measurement_rebuild_v3_one_seed"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-data", type=Path, default=DEFAULT_CONTROL_DATA)
    parser.add_argument(
        "--measurement-data", type=Path, default=DEFAULT_MEASUREMENT_DATA
    )
    parser.add_argument("--v3-control-run", type=Path, default=DEFAULT_V3_CONTROL_RUN)
    parser.add_argument("--measurement-run", type=Path, default=DEFAULT_MEASUREMENT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-groups", type=int)
    parser.add_argument("--max-val-groups", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--measurement-calibrator",
        type=Path,
        help="Optional v4 calibrator applied to cached v3 measurements.",
    )
    parser.add_argument(
        "--forward-artifact",
        type=Path,
        help="Optional v4 forward artifact; defaults to the frozen v3 forward.",
    )
    parser.add_argument(
        "--inverse-initialization",
        type=Path,
        help="Optional inverse-ranker checkpoint used for initialization.",
    )
    parser.add_argument(
        "--artifact-name",
        default="visual_sensor_scorer_v4.pt",
        help="Output checkpoint filename inside --output-dir.",
    )
    return parser.parse_args()


def measured_state_cache(
    split: str,
    data_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    parameters: Mapping[str, Mapping[str, Any]],
    measurement: MeasurementModuleV3,
    output_dir: Path,
) -> dict[tuple[str, str], np.ndarray]:
    cache_path = output_dir / f"measurement_predictions_{split}.npz"
    expected_keys = [
        (str(row["state_id"]), str(condition))
        for row in rows
        for condition in conditions
    ]
    if cache_path.is_file():
        loaded = np.load(cache_path, allow_pickle=False)
        keys = json.loads(str(loaded["keys_json"].item()))
        if keys == [list(key) for key in expected_keys]:
            values = np.asarray(loaded["predictions"], dtype=np.float32)
            return {key: values[index] for index, key in enumerate(expected_keys)}
    predicted = measurement.measure_dataset_states(
        data_dir, rows, conditions, parameters, batch_size=16
    )
    values = np.asarray([predicted[key] for key in expected_keys], dtype=np.float32)
    np.savez(
        cache_path,
        keys_json=json.dumps([list(key) for key in expected_keys]),
        predictions=values,
    )
    return predicted


def build_visual_arrays(
    grids: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    measured: Mapping[tuple[str, str], np.ndarray],
    conditions: Sequence[str],
    forward: Any,
    forward_batch: int = 256,
) -> dict[str, Any]:
    by_group: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in state_rows:
        by_group.setdefault(str(row["group_id"]), {})[str(row["role"])] = row

    unique_setups: list[Mapping[str, Any]] = []
    unique_current_sensor: list[np.ndarray] = []
    unique_rows: list[dict[str, Any]] = []
    candidate_position: dict[tuple[str, str], int] = {}
    for grid in grids:
        group_id = str(grid["group_id"])
        current_row = by_group[group_id]["current"]
        for condition in conditions:
            position = len(unique_rows)
            candidate_position[(group_id, condition)] = position
            sensor = np.asarray(
                measured[(str(current_row["state_id"]), condition)],
                dtype=np.float32,
            )
            legacy = sensor_to_base_legacy(sensor, [grid["setup"]])[0]
            unique_setups.append(grid["setup"])
            unique_current_sensor.append(sensor)
            unique_rows.append(
                {
                    "group_id": f"{group_id}:{condition}",
                    "setup": grid["setup"],
                    "current_beam_state": state_mapping(legacy),
                }
            )
    candidate_parts = []
    for start in range(0, len(unique_rows), forward_batch):
        rows_out = unique_rows[start : start + forward_batch]
        setups_out = unique_setups[start : start + forward_batch]
        legacy = forward.predict_states(rows_out)
        candidate_parts.append(legacy_candidates_to_sensor(legacy, setups_out))
    candidate_states = np.concatenate(candidate_parts)

    contexts, desired, positives, statuses = [], [], [], []
    selected, candidate_indices, setups = [], [], []
    current_context, desired_context = [], []
    condition_indices, group_ids = [], []
    for grid in grids:
        group_id = str(grid["group_id"])
        roles = by_group[group_id]
        current_row = roles["current"]
        true_candidates = true_grid_sensor_states(grid)
        for target_role in ("target_00", "target_01", "target_02"):
            target_row = roles[target_role]
            true_desired = raw_state_array(target_row["target_state"])
            positive = matching_mask(true_candidates, true_desired)
            matches = np.flatnonzero(positive).tolist()
            status = (
                "infeasible_within_limits"
                if not matches
                else "unique" if len(matches) == 1 else "ambiguous"
            )
            selected_index = -1 if not matches else minimum_motion_index(matches)
            for condition_index, condition in enumerate(conditions):
                current_sensor = np.asarray(
                    measured[(str(current_row["state_id"]), condition)],
                    dtype=np.float32,
                )
                desired_sensor = np.asarray(
                    measured[(str(target_row["state_id"]), condition)],
                    dtype=np.float32,
                )
                current_legacy = sensor_to_base_legacy(current_sensor, [grid["setup"]])[
                    0
                ]
                desired_legacy = sensor_to_base_legacy(desired_sensor, [grid["setup"]])[
                    0
                ]
                contexts.append(
                    inverse_context(
                        grid["setup"],
                        state_mapping(current_legacy),
                        state_mapping(desired_legacy),
                    )
                )
                desired.append(desired_sensor)
                positives.append(positive)
                statuses.append(STATUS_INDEX[status])
                selected.append(selected_index)
                candidate_indices.append(candidate_position[(group_id, condition)])
                setups.append(grid["setup"])
                current_context.append(current_legacy)
                desired_context.append(desired_legacy)
                condition_indices.append(condition_index)
                group_ids.append(group_id)
    return {
        "contexts": np.asarray(contexts, dtype=np.float32),
        "desired": np.asarray(desired, dtype=np.float32),
        "positives": np.asarray(positives, dtype=np.bool_),
        "statuses": np.asarray(statuses, dtype=np.int64),
        "selected": np.asarray(selected, dtype=np.int64),
        "candidate_indices": np.asarray(candidate_indices, dtype=np.int64),
        "candidate_states": candidate_states.astype(np.float32),
        "condition_indices": np.asarray(condition_indices, dtype=np.int64),
        "conditions": list(conditions),
        "group_ids": group_ids,
        "setups": setups,
        "current_context": np.asarray(current_context, dtype=np.float32),
        "desired_context": np.asarray(desired_context, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    control_data = args.control_data.resolve()
    measurement_data = args.measurement_data.resolve()
    v3_control_run = args.v3_control_run.resolve()
    measurement_run = args.measurement_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    measurement_config = read_json(measurement_data / "config.json")
    conditions = list(measurement_config["conditions"])
    parameters = measurement_config["condition_parameters"]
    seed = int(config["seed"])
    torch, device = configure(seed, args.device)
    started = time.perf_counter()

    measurement = MeasurementModuleV3(
        torch, measurement_run / "measurement_v3.pt", device
    )
    forward_path = (
        args.forward_artifact.resolve()
        if args.forward_artifact is not None
        else v3_control_run / "forward_control_v3_calibrated.pt"
    )
    forward_metadata = torch.load(forward_path, map_location="cpu", weights_only=False)
    if forward_metadata.get("model") == "anchored_physics_residual_forward_v4":
        forward, _ = load_forward_runtime_v4(forward_path, torch, device)
    else:
        forward, _ = load_forward_runtime(forward_path, torch, device)
    train_grids = read_jsonl(control_data / "grids/train.jsonl")
    val_grids = read_jsonl(control_data / "grids/val.jsonl")
    if args.max_train_groups is not None:
        train_grids = train_grids[: args.max_train_groups]
    if args.max_val_groups is not None:
        val_grids = val_grids[: args.max_val_groups]
    train_group_set = {str(row["group_id"]) for row in train_grids}
    val_group_set = {str(row["group_id"]) for row in val_grids}
    train_states = [
        row
        for row in iter_jsonl(measurement_data / "states/train.jsonl")
        if str(row["group_id"]) in train_group_set
    ]
    val_states = [
        row
        for row in iter_jsonl(measurement_data / "states/val.jsonl")
        if str(row["group_id"]) in val_group_set
    ]
    measured_train = measured_state_cache(
        "train",
        measurement_data,
        train_states,
        conditions,
        parameters,
        measurement,
        output_dir,
    )
    measured_val = measured_state_cache(
        "val",
        measurement_data,
        val_states,
        conditions,
        parameters,
        measurement,
        output_dir,
    )
    if args.measurement_calibrator is not None:
        calibrator = MeasurementCalibratorRuntimeV4(
            torch, args.measurement_calibrator.resolve(), device
        )
        measured_train = calibrator.calibrate_records(
            train_states,
            measured_train,
            conditions,
            parameters,
        )
        measured_val = calibrator.calibrate_records(
            val_states,
            measured_val,
            conditions,
            parameters,
        )
    train = build_visual_arrays(
        train_grids, train_states, measured_train, conditions, forward
    )
    val = build_visual_arrays(val_grids, val_states, measured_val, conditions, forward)

    context_mean = train["contexts"].mean(axis=0)
    context_scale = train["contexts"].std(axis=0)
    context_scale[context_scale < 1e-7] = 1.0
    ct = ((train["contexts"] - context_mean) / context_scale).astype(np.float32)
    cv = ((val["contexts"] - context_mean) / context_scale).astype(np.float32)
    candidate_mean, candidate_scale, status_mean, status_scale = feature_statistics(
        train["candidate_states"],
        train["candidate_indices"],
        train["desired"],
    )
    sample_candidate, _, sample_status = inverse_candidate_features(
        train["candidate_states"][train["candidate_indices"][:1]],
        train["desired"][:1],
    )
    model = inverse_ranker_model(
        torch,
        ct.shape[1],
        sample_candidate.shape[-1],
        sample_status.shape[-1],
    ).to(device)
    initialization_path = (
        args.inverse_initialization.resolve()
        if args.inverse_initialization is not None
        else v3_control_run / "inverse_control_v3.pt"
    )
    initialization = torch.load(
        initialization_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(initialization["state_dict"])

    training = config["visual_scorer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    status_counts = Counter(train["statuses"].tolist())
    status_weights = torch.as_tensor(
        [
            math.sqrt(len(train["statuses"]) / max(3 * status_counts.get(index, 0), 1))
            for index in range(3)
        ],
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(seed + 401)
    epochs = int(args.epochs or training["epochs"])
    batch_size = int(training["batch_pairs"])
    alpha_candidates = [float(value) for value in training["alpha_candidates"]]
    best_score = float("-inf")
    best_selection_key = None
    best_state = None
    best_alpha = 0.0
    best_epoch = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for index in iter_batches(len(ct), batch_size, rng):
            candidate, cost, status_feature = inverse_candidate_features(
                train["candidate_states"][train["candidate_indices"][index]],
                train["desired"][index],
            )
            candidate = (candidate - candidate_mean) / candidate_scale
            status_feature = (status_feature - status_mean) / status_scale
            correction, status_logits = model(
                torch.as_tensor(ct[index], dtype=torch.float32, device=device),
                torch.as_tensor(candidate, dtype=torch.float32, device=device),
                torch.as_tensor(status_feature, dtype=torch.float32, device=device),
            )
            base = torch.as_tensor(-cost, dtype=torch.float32, device=device)
            scores = base + correction
            positive = torch.as_tensor(
                train["positives"][index],
                dtype=torch.bool,
                device=device,
            )
            feasible = positive.any(dim=1)
            positive_scores = scores[feasible].masked_fill(~positive[feasible], -1e9)
            rank_loss = (
                torch.logsumexp(scores[feasible], dim=1)
                - torch.logsumexp(positive_scores, dim=1)
            ).mean()
            status_loss = torch.nn.functional.cross_entropy(
                status_logits,
                torch.as_tensor(
                    train["statuses"][index],
                    dtype=torch.long,
                    device=device,
                ),
                weight=status_weights,
            )
            loss = (
                rank_loss
                + float(training["status_loss_weight"]) * status_loss
                + float(training["correction_regularization"])
                * correction.square().mean()
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
            val["candidate_states"],
            val["candidate_indices"],
            val["desired"],
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
                val["positives"],
                val["statuses"],
                val["selected"],
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
        score = (
            selected_alpha["target_success_feasible"]
            + 0.05 * selected_alpha["minimum_movement_exact_feasible"]
            + 0.10 * selected_alpha["status_macro_f1"]
        )
        gate_summary = gate_margin_summary(
            {
                "visual_inverse_physical_target_floor": (
                    selected_alpha["target_success_feasible"]
                    - THRESHOLDS["visual_inverse_floor"]
                )
            }
        )
        selection_key = gate_aware_selection_key(
            gate_summary,
            score,
            -float(selected_alpha["alpha"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": running / len(ct),
            "score": score,
            "selected_alpha": selected_alpha,
            "alpha_candidates": alpha_rows,
            "checkpoint_selection_gates": gate_summary,
            "checkpoint_selection_key": list(selection_key),
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best_selection_key is None or selection_key > best_selection_key:
            best_score = score
            best_selection_key = selection_key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_alpha = float(selected_alpha["alpha"])
    if best_state is None:
        raise RuntimeError("visual scorer produced no checkpoint")

    artifact_path = output_dir / args.artifact_name
    torch.save(
        {
            "version": config["version"],
            "seed": seed,
            "model": "visual_sensor_residual_scorer_v4",
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
            "statuses": [
                "unique",
                "ambiguous",
                "infeasible_within_limits",
            ],
            "action_grid": ACTION_GRID,
            "measurement_artifact": str(
                (measurement_run / "measurement_v3.pt").resolve()
            ),
            "measurement_calibrator": (
                None
                if args.measurement_calibrator is None
                else str(args.measurement_calibrator.resolve())
            ),
            "forward_artifact": str(forward_path.resolve()),
            "inverse_initialization": str(initialization_path.resolve()),
            "coordinate_frame": "camera_sensor_array",
            "checkpoint_selection": (
                "validation_only_physical_target_floor_then_margin_then_composite"
            ),
        },
        artifact_path,
    )
    summary = {
        "version": config["version"],
        "device": str(device),
        "artifact": str(artifact_path.resolve()),
        "train_groups": len(train_grids),
        "train_pairs": len(ct),
        "val_groups": len(val_grids),
        "val_pairs": len(cv),
        "conditions": conditions,
        "measurement_calibrator": (
            None
            if args.measurement_calibrator is None
            else str(args.measurement_calibrator.resolve())
        ),
        "forward_artifact": str(forward_path.resolve()),
        "inverse_initialization": str(initialization_path.resolve()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "best_composite_score": best_score,
        "best_selection_key": list(best_selection_key),
        "validation": trace[best_epoch - 1],
        "trace": trace,
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / (Path(args.artifact_name).stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
