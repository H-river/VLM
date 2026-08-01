#!/usr/bin/env python3
"""One-step, paired-Jacobian, regime, and rollout diagnostics for v12."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    Bounds,
    action_vector,
    metrics_vector,
    position_vector,
    tolerance_vector,
)
from continuous_control_v12.evaluation import forward_metrics
from continuous_control_v12.schema import read_jsonl, validate_dataset
from continuous_control_v12.world_model import (
    load_forward_ensemble,
    transition_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "development", "test"), default="test"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-groups", type=int)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite diagnostics: {output_dir}")
    output_dir.mkdir(parents=True)
    validation = validate_dataset(data_dir, config)
    manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(
        data_dir / "transitions" / f"{args.split}.jsonl"
    )
    if args.max_groups is not None:
        groups = sorted({str(row["group_id"]) for row in rows})[
            : args.max_groups
        ]
        selected = set(groups)
        rows = [
            row for row in rows if str(row["group_id"]) in selected
        ]
    bounds = Bounds.from_config(config)
    arrays = transition_arrays(rows, bounds)
    model = load_forward_ensemble(
        args.checkpoint, device_name=args.device
    )
    predictions = []
    uncertainties = []
    prediction_rows = []
    for index, row in enumerate(rows):
        action = action_vector(row["action_mm"])
        result = model.predict(
            row["setup_context"],
            row["positions_mm"],
            row["metrics"],
            action,
        )
        residual = result["next_metric_residual"][0]
        uncertainty = result["uncertainty"][0]
        predictions.append(residual)
        uncertainties.append(uncertainty)
        true = arrays["targets"][index]
        error = residual - true
        record: dict[str, Any] = {
            "transition_id": row["transition_id"],
            "group_id": row["group_id"],
            "regime": manifest["group_regimes"][row["group_id"]],
            "sampling_kind": row["sampling"]["kind"],
            "active_action_axes": int((np.abs(action) > 1e-12).sum()),
            "normalized_action_linf": float(
                np.max(np.abs(action / bounds.action_high))
            ),
        }
        for output_index, field in enumerate(OUTPUT_FIELDS):
            record[f"true_residual_{field}"] = float(true[output_index])
            record[f"predicted_residual_{field}"] = float(
                residual[output_index]
            )
            record[f"error_{field}"] = float(error[output_index])
            record[f"uncertainty_{field}"] = float(
                uncertainty[output_index]
            )
        prediction_rows.append(record)
    prediction_array = np.asarray(predictions)
    uncertainty_array = np.asarray(uncertainties)
    group_regimes = manifest["group_regimes"]
    one_step = forward_metrics(
        prediction_array,
        arrays["targets"],
        group_ids=arrays["group_ids"],
        regimes=[group_regimes[group] for group in arrays["group_ids"]],
        sampling=arrays["sampling"],
        uncertainty=uncertainty_array,
        no_op=arrays["no_op"],
    )

    actions = np.stack([action_vector(row["action_mm"]) for row in rows])
    action_magnitude = np.max(
        np.abs(actions / bounds.action_high[None, :]), axis=1
    )
    absolute_error = np.abs(prediction_array - arrays["targets"])
    sensitivity = np.abs(arrays["targets"]).mean(axis=1)
    error_magnitude = absolute_error.mean(axis=1)
    by_magnitude = {}
    for name, low, high in (
        ("zero", 0.0, 1e-12),
        ("near_zero", 1e-12, 0.15),
        ("small", 0.15, 0.40),
        ("medium", 0.40, 0.75),
        ("large", 0.75, np.inf),
    ):
        mask = (action_magnitude >= low) & (action_magnitude < high)
        by_magnitude[name] = {
            "count": int(mask.sum()),
            "normalized_mae": (
                None if not mask.any() else float(absolute_error[mask].mean())
            ),
            "strict_all_five_accuracy": (
                None
                if not mask.any()
                else float(np.all(absolute_error[mask] <= 1.0, axis=1).mean())
            ),
        }
    by_axis_count = {}
    active_axes = (np.abs(actions) > 1e-12).sum(axis=1)
    for count in range(5):
        mask = active_axes == count
        by_axis_count[str(count)] = {
            "count": int(mask.sum()),
            "normalized_mae": (
                None if not mask.any() else float(absolute_error[mask].mean())
            ),
        }

    pair_groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["sampling"]["kind"] == "paired":
            pair_groups[
                (str(row["group_id"]), str(row["sampling"]["pair_id"]))
            ].append(index)
    pair_records = []
    sign_correct = []
    cosine_values = []
    relative_jacobian_errors = []
    per_output_sign: list[list[bool]] = [[] for _ in OUTPUT_FIELDS]
    for key, indices in pair_groups.items():
        if len(indices) != 2:
            continue
        plus = next(
            (
                index
                for index in indices
                if int(rows[index]["sampling"]["sign"]) == 1
            ),
            None,
        )
        minus = next(
            (
                index
                for index in indices
                if int(rows[index]["sampling"]["sign"]) == -1
            ),
            None,
        )
        if plus is None or minus is None:
            continue
        true_direction = arrays["targets"][plus] - arrays["targets"][minus]
        predicted_direction = (
            prediction_array[plus] - prediction_array[minus]
        )
        informative = np.abs(true_direction) > 1e-4
        correctness = np.sign(predicted_direction[informative]) == np.sign(
            true_direction[informative]
        )
        sign_correct.extend(correctness.tolist())
        for output_index in range(5):
            if informative[output_index]:
                per_output_sign[output_index].append(
                    bool(
                        np.sign(predicted_direction[output_index])
                        == np.sign(true_direction[output_index])
                    )
                )
        denominator = max(float(np.linalg.norm(true_direction)), 1e-8)
        relative_error = float(
            np.linalg.norm(predicted_direction - true_direction)
            / denominator
        )
        cosine = float(
            np.dot(predicted_direction, true_direction)
            / max(
                float(np.linalg.norm(predicted_direction))
                * float(np.linalg.norm(true_direction)),
                1e-8,
            )
        )
        relative_jacobian_errors.append(relative_error)
        cosine_values.append(cosine)
        pair_records.append(
            {
                "group_id": key[0],
                "pair_id": key[1],
                "direction_sign_accuracy": (
                    None
                    if not informative.any()
                    else float(correctness.mean())
                ),
                "relative_directional_jacobian_error": relative_error,
                "direction_cosine_similarity": cosine,
            }
        )
    paired = {
        "pairs": len(pair_records),
        "direction_sign_accuracy": (
            None if not sign_correct else float(np.mean(sign_correct))
        ),
        "per_output_direction_sign_accuracy": {
            field: (
                None
                if not per_output_sign[index]
                else float(np.mean(per_output_sign[index]))
            )
            for index, field in enumerate(OUTPUT_FIELDS)
        },
        "median_relative_directional_jacobian_error": (
            None
            if not relative_jacobian_errors
            else float(np.median(relative_jacobian_errors))
        ),
        "p95_relative_directional_jacobian_error": (
            None
            if not relative_jacobian_errors
            else float(np.quantile(relative_jacobian_errors, 0.95))
        ),
        "mean_direction_cosine_similarity": (
            None if not cosine_values else float(np.mean(cosine_values))
        ),
    }

    trajectories: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["sampling"]["kind"] == "closed_loop_trajectory":
            trajectories[str(row["sampling"]["trajectory_id"])].append(row)
    rollout_errors: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    for trajectory in trajectories.values():
        trajectory.sort(key=lambda row: int(row["sampling"]["step"]))
        for start in range(len(trajectory)):
            predicted_metrics = metrics_vector(trajectory[start]["metrics"])
            positions = position_vector(trajectory[start]["positions_mm"])
            reference_tolerance = tolerance_vector(
                trajectory[start]["metrics"]
            )
            for offset, row in enumerate(trajectory[start:], start=1):
                action = action_vector(row["action_mm"])
                prediction = model.predict(
                    row["setup_context"],
                    positions,
                    predicted_metrics,
                    action,
                )
                predicted_metrics = prediction[
                    "predicted_next_metrics"
                ][0]
                positions = position_vector(row["next_positions_mm"])
                if offset in (1, 3, 5):
                    rollout_errors[offset].append(
                        np.abs(
                            predicted_metrics
                            - metrics_vector(row["next_metrics"])
                        )
                        / reference_tolerance
                    )
    rollout = {}
    for horizon in (1, 3, 5):
        values = np.asarray(rollout_errors.get(horizon, []))
        rollout[str(horizon)] = {
            "count": int(len(values)),
            "normalized_mae": (
                None if not len(values) else float(values.mean())
            ),
            "strict_all_five_accuracy": (
                None
                if not len(values)
                else float(np.all(values <= 1.0, axis=1).mean())
            ),
            "per_output_normalized_mae": {
                field: (
                    None
                    if not len(values)
                    else float(values[:, index].mean())
                )
                for index, field in enumerate(OUTPUT_FIELDS)
            },
        }

    sensitivity_ratio = error_magnitude / np.maximum(sensitivity, 1e-6)
    report = {
        "version": "continuous_forward_diagnostics_v12_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(data_dir),
        "split": args.split,
        "dataset_validation": validation,
        "one_step": one_step,
        "by_action_magnitude": by_magnitude,
        "by_active_action_axes": by_axis_count,
        "paired_action_and_jacobian": paired,
        "rollout_error_by_horizon": rollout,
        "model_error_versus_simulator_sensitivity": {
            "median_error_to_sensitivity_ratio": float(
                np.median(sensitivity_ratio)
            ),
            "p95_error_to_sensitivity_ratio": float(
                np.quantile(sensitivity_ratio, 0.95)
            ),
            "mean_simulator_sensitivity_normalized_delta": float(
                sensitivity.mean()
            ),
            "mean_model_error_normalized_delta": float(
                error_magnitude.mean()
            ),
        },
        "finite_predictions": bool(
            np.isfinite(prediction_array).all()
            and np.isfinite(uncertainty_array).all()
        ),
        "q_star_in_inputs": False,
    }
    (output_dir / "forward_diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "one_step_predictions.csv", prediction_rows)
    write_csv(output_dir / "paired_jacobian.csv", pair_records)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(11, 12))
    for index, field in enumerate(OUTPUT_FIELDS):
        axes.flat[index].scatter(
            arrays["targets"][:, index],
            prediction_array[:, index],
            s=8,
            alpha=0.4,
        )
        low = float(
            min(
                arrays["targets"][:, index].min(),
                prediction_array[:, index].min(),
            )
        )
        high = float(
            max(
                arrays["targets"][:, index].max(),
                prediction_array[:, index].max(),
            )
        )
        axes.flat[index].plot([low, high], [low, high], color="black")
        axes.flat[index].set_title(field)
        axes.flat[index].set_xlabel("true normalized delta")
        axes.flat[index].set_ylabel("predicted normalized delta")
    axes.flat[-1].scatter(
        action_magnitude, error_magnitude, s=8, alpha=0.4
    )
    axes.flat[-1].set_xlabel("normalized action L-infinity")
    axes.flat[-1].set_ylabel("mean absolute normalized error")
    figure.tight_layout()
    figure.savefig(output_dir / "residuals.png", dpi=150)
    plt.close(figure)

    horizons = [1, 3, 5]
    values = [
        np.nan
        if rollout[str(horizon)]["normalized_mae"] is None
        else rollout[str(horizon)]["normalized_mae"]
        for horizon in horizons
    ]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(horizons, values, marker="o")
    axis.set_xlabel("autoregressive horizon")
    axis.set_ylabel("normalized MAE")
    axis.set_xticks(horizons)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_error.png", dpi=150)
    plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
