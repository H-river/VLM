#!/usr/bin/env python3
"""Group-OOF residual calibration and reranking of saved dev H1 candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    Bounds,
    metrics_vector,
    tolerance_vector,
)

VERSION = "active_diagnosis_v13_oof_candidate_residual_calibration_v1"


def _case_index(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[1])


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_records(episode_dir: Path, bounds: Bounds) -> list[dict[str, Any]]:
    records = []
    for path in sorted(episode_dir.resolve().glob("*__learned_h1.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(episode["case_id"])
        if _case_index(case_id) > 9:
            continue
        trace = {int(row["episode_step"]): row for row in episode["trace"]}
        target = metrics_vector(episode["target_metrics"])
        tolerance = tolerance_vector(metrics_vector(episode["initial_metrics"]))
        for candidate in episode["candidate_ranking_audits"]:
            if candidate["selection"] != "predicted_top":
                continue
            step = int(candidate["episode_step"])
            depth = candidate["counterfactual_depths"][0]
            if int(depth["depth"]) != 1:
                raise ValueError("candidate residual calibration requires H1 depth one")
            predicted = metrics_vector(depth["predicted_metrics"])
            actual = metrics_vector(depth["actual_metrics"])
            current = metrics_vector(trace[step]["current_metrics"])
            action = np.asarray(
                [
                    float(candidate["effective_sequence"][0][field])
                    for field in ACTION_FIELDS
                ],
                dtype=np.float64,
            )
            visible_features = np.concatenate(
                [
                    (predicted - target) / tolerance,
                    (current - target) / tolerance,
                    action / bounds.action_high,
                    np.asarray([step / 4.0], dtype=np.float64),
                ]
            )
            residual = (actual - predicted) / tolerance
            records.append(
                {
                    "case_id": case_id,
                    "group_id": str(episode["group_id"]),
                    "stratum": str(episode["stratum"]),
                    "episode_final_success": bool(episode["success"]),
                    "episode_step": step,
                    "selection_rank": int(candidate["selection_rank"]),
                    "decision_id": f"{case_id}__step_{step}",
                    "visible_features": visible_features.tolist(),
                    "target_metrics": target.tolist(),
                    "tolerance": tolerance.tolist(),
                    "predicted_metrics": predicted.tolist(),
                    "actual_metrics": actual.tolist(),
                    "normalized_residual": residual.tolist(),
                    "actual_terminal_cost": float(candidate["actual_terminal_cost"]),
                    "original_predicted_terminal_cost": float(
                        candidate["predicted_terminal_cost"]
                    ),
                }
            )
    if not records:
        raise ValueError("no development H1 candidate audits found")
    return records


def _oof_predictions(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([row["visible_features"] for row in records], dtype=np.float64)
    residuals = np.asarray([row["normalized_residual"] for row in records], dtype=np.float64)
    groups = np.asarray([row["group_id"] for row in records], dtype=object)
    constant = np.zeros_like(residuals)
    ridge = np.zeros_like(residuals)
    for held_group in sorted(set(groups.tolist())):
        train = groups != held_group
        held = ~train
        if not np.any(train) or not np.any(held):
            raise ValueError("invalid group-held-out residual fold")
        constant[held] = np.mean(residuals[train], axis=0)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features[train], residuals[train])
        ridge[held] = model.predict(features[held])
    return constant, ridge


def _metric_errors(
    actual: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    result = {}
    for name, predicted in predictions.items():
        absolute = np.abs(actual - predicted)
        result[name] = {
            "all_metric_mae_normalized": float(np.mean(absolute)),
            "all_metric_rmse_normalized": float(np.sqrt(np.mean(np.square(actual - predicted)))),
            "per_metric_mae_normalized": {
                field: float(np.mean(absolute[:, index]))
                for index, field in enumerate(OUTPUT_FIELDS)
            },
        }
    return result


def _rerank(
    records: list[dict[str, Any]], corrections: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[row["decision_id"]].append(index)
    decision_rows = []
    for decision_id, indices in sorted(grouped.items()):
        indices.sort(key=lambda index: int(records[index]["selection_rank"]))
        if [records[index]["selection_rank"] for index in indices] != list(
            range(1, len(indices) + 1)
        ):
            raise ValueError(f"candidate ranks are not contiguous: {decision_id}")
        choices = {"original": indices[0]}
        for name, correction in corrections.items():
            corrected_costs = []
            for index in indices:
                row = records[index]
                predicted = np.asarray(row["predicted_metrics"], dtype=np.float64)
                tolerance = np.asarray(row["tolerance"], dtype=np.float64)
                target = np.asarray(row["target_metrics"], dtype=np.float64)
                corrected = predicted + correction[index] * tolerance
                corrected_costs.append(float(np.max(np.abs(corrected - target) / tolerance)))
            choices[name] = indices[int(np.argmin(corrected_costs))]
        actual_best = min(indices, key=lambda index: records[index]["actual_terminal_cost"])
        result = {
            "decision_id": decision_id,
            "case_id": records[indices[0]]["case_id"],
            "group_id": records[indices[0]]["group_id"],
            "stratum": records[indices[0]]["stratum"],
            "episode_final_success": records[indices[0]]["episode_final_success"],
            "episode_step": records[indices[0]]["episode_step"],
            "candidates": len(indices),
            "actual_best_rank": records[actual_best]["selection_rank"],
            "actual_best_cost": records[actual_best]["actual_terminal_cost"],
        }
        for name, index in choices.items():
            result[name] = {
                "selected_rank": records[index]["selection_rank"],
                "actual_terminal_cost": records[index]["actual_terminal_cost"],
                "strict_success": bool(records[index]["actual_terminal_cost"] <= 1.0),
                "selects_actual_best": bool(index == actual_best),
            }
        decision_rows.append(result)
    summary = {"decisions": len(decision_rows)}
    for name in ("original", *corrections):
        summary[name] = {
            "strict_success": float(
                np.mean([bool(row[name]["strict_success"]) for row in decision_rows])
            ),
            "mean_actual_terminal_cost": float(
                np.mean([float(row[name]["actual_terminal_cost"]) for row in decision_rows])
            ),
            "median_actual_terminal_cost": float(
                np.median([float(row[name]["actual_terminal_cost"]) for row in decision_rows])
            ),
            "actual_best_selection_rate": float(
                np.mean([bool(row[name]["selects_actual_best"]) for row in decision_rows])
            ),
        }
    for name in corrections:
        summary[name]["improved_decisions_vs_original"] = sum(
            float(row[name]["actual_terminal_cost"])
            < float(row["original"]["actual_terminal_cost"]) - 1e-12
            for row in decision_rows
        )
        summary[name]["regressed_decisions_vs_original"] = sum(
            float(row[name]["actual_terminal_cost"])
            > float(row["original"]["actual_terminal_cost"]) + 1e-12
            for row in decision_rows
        )
    failed_episode_rows = [row for row in decision_rows if not row["episode_final_success"]]
    summary["exact_failed_episode_decisions"] = {
        "decisions": len(failed_episode_rows),
        "case_ids": sorted({row["case_id"] for row in failed_episode_rows}),
        **{
            name: {
                "strict_success": float(
                    np.mean([bool(row[name]["strict_success"]) for row in failed_episode_rows])
                ),
                "mean_actual_terminal_cost": float(
                    np.mean(
                        [float(row[name]["actual_terminal_cost"]) for row in failed_episode_rows]
                    )
                ),
            }
            for name in ("original", *corrections)
        },
    }
    return summary, decision_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--v12-config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bounds = Bounds.from_config(json.loads(args.v12_config.resolve().read_text()))
    records = _load_records(args.episode_dir, bounds)
    constant, ridge = _oof_predictions(records)
    actual = np.asarray([row["normalized_residual"] for row in records], dtype=np.float64)
    reranking, decisions = _rerank(records, {"constant_oof": constant, "ridge_oof": ridge})
    prediction_rows = []
    for index, row in enumerate(records):
        prediction_rows.append(
            {
                "version": VERSION,
                "case_id": row["case_id"],
                "group_id": row["group_id"],
                "decision_id": row["decision_id"],
                "selection_rank": row["selection_rank"],
                "actual_normalized_residual": row["normalized_residual"],
                "constant_oof_predicted_residual": constant[index].tolist(),
                "ridge_oof_predicted_residual": ridge[index].tolist(),
            }
        )
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "candidate_records": len(records),
        "independent_groups": len({row["group_id"] for row in records}),
        "input_features": (
            "visible predicted target error, visible current target error, effective action, "
            "and episode step; no q_star, hidden gain, actual metrics, or group ID"
        ),
        "group_oof_residual_calibration": _metric_errors(
            actual,
            {
                "uncorrected_zero_residual": np.zeros_like(actual),
                "constant_oof": constant,
                "ridge_oof": ridge,
            },
        ),
        "candidate_reranking": reranking,
        "decision_rows": decisions,
        "interpretation_guard": (
            "Only the ten previously retained predicted-top nominal H1 candidates per "
            "decision are reranked. This is group-OOF offline evidence, not closed-loop control."
        ),
    }
    _write_jsonl(args.predictions, prediction_rows)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "candidate_records": len(records),
                "calibration": report["group_oof_residual_calibration"],
                "reranking": reranking,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
