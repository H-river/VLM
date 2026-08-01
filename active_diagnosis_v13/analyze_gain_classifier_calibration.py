#!/usr/bin/env python3
"""Calibrate the selected probe's group-OOF gain probabilities and errors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np


VERSION = "active_diagnosis_v13_gain_classifier_calibration_v1"


def _rows(root: Path) -> list[dict[str, Any]]:
    by_id = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                by_id[row["record_id"]] = row
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    selected = json.loads(
        (root / "probes" / "selected_probe.json").read_text(encoding="utf-8")
    )
    bundle = joblib.load(Path(selected["classifier_bundle"]))
    rows = [
        row
        for row in _rows(root)
        if str(row["design"]) == str(selected["selected_design"])
        and float(row["fraction"]) == float(selected["selected_fraction"])
    ]
    predictions = []
    for row in rows:
        model = bundle["fold_models"][bundle["case_to_fold"][str(row["case_id"])]]
        X = np.asarray(row["policy_record"]["feature_vector"], dtype=np.float64)[None, :]
        probability = model.predict_proba(X)[0]
        classes = [float(value) for value in model.classes_]
        estimate = float(classes[int(np.argmax(probability))])
        truth = float(row["evaluator_only_true_gain"])
        predictions.append(
            {
                "case_id": row["case_id"],
                "group_id": row["group_id"],
                "stratum": row["stratum"],
                "true_gain_evaluator_only": truth,
                "estimated_gain": estimate,
                "correct": estimate == truth,
                "confidence": float(np.max(probability)),
                "absolute_gain_error": abs(estimate - truth),
                "class_probabilities": {
                    f"{gain:g}": float(value)
                    for gain, value in zip(classes, probability, strict=True)
                },
            }
        )
    confidence = np.asarray([row["confidence"] for row in predictions])
    correct = np.asarray([row["correct"] for row in predictions], dtype=np.float64)
    bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence >= low) & (
            confidence <= high if index == len(edges) - 2 else confidence < high
        )
        if not mask.any():
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidence[mask].mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "records": int(mask.sum()),
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )
    wrong = [row for row in predictions if not row["correct"]]
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only_group_out_of_fold",
        "protected_set_used": False,
        "selected_probe": selected,
        "records": len(predictions),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "expected_calibration_error_10_bins": ece,
        "confidence_bins": bins,
        "mean_absolute_gain_error": float(
            np.mean([row["absolute_gain_error"] for row in predictions])
        ),
        "wrong_predictions": len(wrong),
        "adjacent_class_error_fraction_among_errors": float(
            np.mean([row["absolute_gain_error"] == 0.25 for row in wrong])
        ),
        "large_error_count": int(
            sum(row["absolute_gain_error"] > 0.25 for row in wrong)
        ),
        "exact_misclassifications": wrong,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "gain_classifier_calibration.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    with (output_dir / "oof_gain_predictions.jsonl").open("w") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "exact_misclassifications"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
