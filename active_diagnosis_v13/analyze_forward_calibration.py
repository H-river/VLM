#!/usr/bin/env python3
"""Forward-error decomposition and uncertainty calibration on saved H1 traces."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    metrics_vector,
    normalized_error,
    tolerance_vector,
)
from continuous_control_v12.world_model import load_forward_ensemble


VERSION = "active_diagnosis_v13_forward_calibration_forensic_v1"


def _case_index(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[1])


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def _calibration_bins(error: np.ndarray, uncertainty: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for index, indices in enumerate(np.array_split(np.argsort(uncertainty), 5), start=1):
        result.append(
            {
                "quintile": index,
                "records": len(indices),
                "mean_predicted_uncertainty": float(np.mean(uncertainty[indices])),
                "mean_absolute_normalized_error": float(np.mean(np.abs(error[indices]))),
                "rmse_normalized_error": float(np.sqrt(np.mean(np.square(error[indices])))),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "protected"), default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.split == "protected":
        if args.freeze is None or not args.freeze.exists():
            raise ValueError("protected calibration requires frozen decision")
        if not json.loads(args.freeze.read_text(encoding="utf-8")).get("frozen"):
            raise ValueError("protected calibration requires frozen=true")
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    suite = json.loads(
        Path(config["baseline"]["evaluation_suite"]).read_text(encoding="utf-8")
    )
    cases = {str(case["case_id"]): case for case in suite["cases"]}
    episodes = []
    for path in sorted(args.episode_dir.resolve().glob("*__learned_h1.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        index = _case_index(str(episode["case_id"]))
        belongs = index <= 9 if args.split == "development" else index >= 10
        if belongs:
            episodes.append(episode)
    model = load_forward_ensemble(
        Path(config["baseline"]["checkpoint"]), device_name="cpu"
    )
    records = []
    for episode in episodes:
        case = cases[str(episode["case_id"])]
        for step in episode["trace"]:
            prediction = model.predict(
                case["setup_context"],
                step["current_positions_mm"],
                step["current_metrics"],
                [step["effective_action_mm"][field] for field in (
                    "lens_x_delta_mm",
                    "lens_y_delta_mm",
                    "camera_x_delta_mm",
                    "camera_y_delta_mm",
                )],
            )
            predicted = np.asarray(prediction["predicted_next_metrics"][0], dtype=np.float64)
            actual = metrics_vector(step["actual_next_metrics"])
            tolerance = tolerance_vector(step["current_metrics"])
            residual = (actual - predicted) / tolerance
            uncertainty = np.asarray(prediction["uncertainty"][0], dtype=np.float64)
            records.append(
                {
                    "case_id": episode["case_id"],
                    "group_id": episode["group_id"],
                    "stratum": episode["stratum"],
                    "initial_distance_band": episode["initial_distance_band"],
                    "episode_step": int(step["episode_step"]),
                    "normalized_prediction_residual": {
                        field: float(value)
                        for field, value in zip(OUTPUT_FIELDS, residual, strict=True)
                    },
                    "predicted_uncertainty": {
                        field: float(value)
                        for field, value in zip(OUTPUT_FIELDS, uncertainty, strict=True)
                    },
                }
            )
    residuals = np.asarray(
        [
            [row["normalized_prediction_residual"][field] for field in OUTPUT_FIELDS]
            for row in records
        ],
        dtype=np.float64,
    )
    uncertainties = np.asarray(
        [[row["predicted_uncertainty"][field] for field in OUTPUT_FIELDS] for row in records],
        dtype=np.float64,
    )
    per_metric = {}
    for column, field in enumerate(OUTPUT_FIELDS):
        error = residuals[:, column]
        uncertainty = uncertainties[:, column]
        per_metric[field] = {
            "records": len(error),
            "mean_absolute_normalized_error": float(np.mean(np.abs(error))),
            "median_absolute_normalized_error": float(np.median(np.abs(error))),
            "rmse_normalized_error": float(np.sqrt(np.mean(np.square(error)))),
            "mean_predicted_uncertainty": float(np.mean(uncertainty)),
            "coverage": {
                str(z): float(np.mean(np.abs(error) <= z * uncertainty))
                for z in (1, 2, 3)
            },
            "absolute_error_uncertainty_spearman": _finite_spearman(
                np.abs(error), uncertainty
            ),
            "uncertainty_quintiles": _calibration_bins(error, uncertainty),
        }
    flattened_error = np.abs(residuals).reshape(-1)
    flattened_uncertainty = uncertainties.reshape(-1)
    final_failures = []
    contribution = Counter()
    for episode in episodes:
        error = normalized_error(
            episode["final_metrics"], episode["target_metrics"], episode["initial_metrics"]
        )
        failed_metrics = [
            field
            for field, value in zip(OUTPUT_FIELDS, error, strict=True)
            if float(value) > 1.0
        ]
        if failed_metrics:
            contribution.update(failed_metrics)
            final_failures.append(
                {
                    "case_id": episode["case_id"],
                    "stratum": episode["stratum"],
                    "initial_distance_band": episode["initial_distance_band"],
                    "failed_metrics": failed_metrics,
                    "normalized_final_error": {
                        field: float(value)
                        for field, value in zip(OUTPUT_FIELDS, error, strict=True)
                    },
                }
            )
    by_stratum = Counter(row["stratum"] for row in final_failures)
    by_band = Counter(row["initial_distance_band"] for row in final_failures)
    report = {
        "version": VERSION,
        "split": args.split,
        "protected_set_used_for_selection": args.split == "protected",
        "episodes": len(episodes),
        "executed_action_transitions": len(records),
        "per_metric_forward_calibration": per_metric,
        "all_metric_points": {
            "mean_absolute_normalized_error": float(np.mean(flattened_error)),
            "mean_predicted_uncertainty": float(np.mean(flattened_uncertainty)),
            "absolute_error_uncertainty_spearman": _finite_spearman(
                flattened_error, flattened_uncertainty
            ),
            "coverage": {
                str(z): float(
                    np.mean(flattened_error <= z * flattened_uncertainty)
                )
                for z in (1, 2, 3)
            },
        },
        "strict_all_five_failure_decomposition": {
            "failed_episodes": len(final_failures),
            "metric_failure_counts": {
                field: int(contribution[field]) for field in OUTPUT_FIELDS
            },
            "failures_by_stratum": dict(sorted(by_stratum.items())),
            "failures_by_initial_distance_band": dict(sorted(by_band.items())),
            "exact_failures": final_failures,
        },
        "limitations": [
            "Calibration is measured on actions selected and executed by the frozen H1-CEM policy, not an iid action sample.",
            "The artifact is a nominal-gain v12 trace audit and is supporting bottleneck evidence, not Gate A fault evidence.",
        ],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "forward_uncertainty_calibration.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    record_path = output_dir / "forward_uncertainty_records.jsonl"
    with record_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
