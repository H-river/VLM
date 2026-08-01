#!/usr/bin/env python3
"""Snapshot classifier predictions and provenance on development probe records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np


VERSION = "active_diagnosis_v13_classifier_prediction_audit_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(root: Path) -> list[dict[str, Any]]:
    result = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                result[str(row["record_id"])] = row
    return list(result.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    model_path = args.model.resolve()
    bundle = joblib.load(model_path)
    rows = sorted(
        (
            row
            for row in _rows(root)
            if str(row["design"]) == args.design
            and float(row["fraction"]) == args.fraction
        ),
        key=lambda row: str(row["record_id"]),
    )
    if len(rows) != 150:
        raise ValueError(f"expected 150 development probe rows, found {len(rows)}")
    if any(int(str(row["case_id"]).rsplit("_", 1)[1]) >= 10 for row in rows):
        raise ValueError("protected case present in classifier audit")
    predictions = []
    for row in rows:
        features = np.asarray(row["policy_record"]["feature_vector"], dtype=np.float64)[
            None, :
        ]
        fold = int(bundle["case_to_fold"][str(row["case_id"])])
        oof = float(bundle["fold_models"][fold].predict(features)[0])
        full = float(bundle["full_model"].predict(features)[0])
        predictions.append(
            {
                "record_id": row["record_id"],
                "case_id": row["case_id"],
                "stratum": row["stratum"],
                "true_gain_evaluator_only": float(row["evaluator_only_true_gain"]),
                "fold": fold,
                "oof_estimated_gain": oof,
                "full_estimated_gain": full,
            }
        )
    continuous_prediction = bundle.get("prediction_semantics") in {
        "posterior_probability_weighted_mean_gain",
        "posterior_mean_if_discrete_prediction_at_least_threshold",
    }
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_version": bundle.get("version"),
        "model_ablation": bundle.get("ablation"),
        "prediction_semantics": bundle.get(
            "prediction_semantics", "discrete_gain_hypothesis"
        ),
        "exact_hypothesis_accuracy_applicable": not continuous_prediction,
        "design": args.design,
        "fraction": args.fraction,
        "records": len(predictions),
        "oof_accuracy": (
            None
            if continuous_prediction
            else float(
                np.mean(
                    [
                        row["oof_estimated_gain"]
                        == row["true_gain_evaluator_only"]
                        for row in predictions
                    ]
                )
            )
        ),
        "oof_mean_absolute_gain_error": float(
            np.mean(
                [
                    abs(row["oof_estimated_gain"] - row["true_gain_evaluator_only"])
                    for row in predictions
                ]
            )
        ),
        "oof_root_mean_squared_gain_error": float(
            np.sqrt(
                np.mean(
                    [
                        (row["oof_estimated_gain"] - row["true_gain_evaluator_only"])
                        ** 2
                        for row in predictions
                    ]
                )
            )
        ),
        "oof_within_0p125_gain": float(
            np.mean(
                [
                    abs(row["oof_estimated_gain"] - row["true_gain_evaluator_only"])
                    <= 0.125
                    for row in predictions
                ]
            )
        ),
        "full_fit_accuracy": (
            None
            if continuous_prediction
            else float(
                np.mean(
                    [
                        row["full_estimated_gain"]
                        == row["true_gain_evaluator_only"]
                        for row in predictions
                    ]
                )
            )
        ),
        "oof_accuracy_by_stratum": (
            None
            if continuous_prediction
            else {
                stratum: float(
                    np.mean(
                        [
                            row["oof_estimated_gain"]
                            == row["true_gain_evaluator_only"]
                            for row in predictions
                            if row["stratum"] == stratum
                        ]
                    )
                )
                for stratum in sorted({str(row["stratum"]) for row in predictions})
            }
        ),
        "predictions": predictions,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "predictions"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
