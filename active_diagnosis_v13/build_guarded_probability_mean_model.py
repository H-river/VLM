#!/usr/bin/env python3
"""Build a visible discrete-threshold guard around a posterior-mean estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib

from active_diagnosis_v13.gain_estimators import (
    LowerBoundGuardedProbabilityMeanClassifier,
)


VERSION = "active_diagnosis_v13_guarded_probability_mean_gain_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--discrete-threshold", type=float, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_model.resolve()
    bundle = joblib.load(source)
    threshold = float(args.discrete_threshold)
    classes = [float(value) for value in bundle["full_model"].classes_]
    if threshold not in classes:
        raise ValueError(
            f"discrete threshold {threshold:g} must be one of model classes {classes}"
        )
    output_bundle = {
        **bundle,
        "version": VERSION,
        "ablation": f"{bundle.get('ablation', 'classifier')}_guarded_probability_mean",
        "source_classifier_bundle": str(source),
        "source_classifier_bundle_sha256": _sha256(source),
        "prediction_semantics": "posterior_mean_if_discrete_prediction_at_least_threshold",
        "discrete_threshold": threshold,
        "fold_models": [
            LowerBoundGuardedProbabilityMeanClassifier(model, threshold)
            for model in bundle["fold_models"]
        ],
        "full_model": LowerBoundGuardedProbabilityMeanClassifier(
            bundle["full_model"], threshold
        ),
    }
    output = args.output_model.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = output.with_suffix(f".joblib.tmp.{os.getpid()}")
    joblib.dump(output_bundle, temporary_model)
    os.replace(temporary_model, output)
    manifest = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "role": "visible_low_gain_guard_control_ablation_no_protected_reselection",
        "source_classifier_bundle": str(source),
        "source_classifier_bundle_sha256": _sha256(source),
        "classifier_bundle": str(output),
        "classifier_bundle_sha256": _sha256(output),
        "prediction_semantics": "posterior_mean_if_discrete_prediction_at_least_threshold",
        "discrete_threshold": threshold,
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
