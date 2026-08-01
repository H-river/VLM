#!/usr/bin/env python3
"""Wrap a discrete gain model with its continuous posterior-mean estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib

from active_diagnosis_v13.gain_estimators import ProbabilityMeanClassifier


VERSION = "active_diagnosis_v13_probability_mean_gain_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_model.resolve()
    bundle = joblib.load(source)
    output_bundle = {
        **bundle,
        "version": VERSION,
        "ablation": f"{bundle.get('ablation', 'classifier')}_probability_mean",
        "source_classifier_bundle": str(source),
        "source_classifier_bundle_sha256": _sha256(source),
        "prediction_semantics": "posterior_probability_weighted_mean_gain",
        "fold_models": [ProbabilityMeanClassifier(model) for model in bundle["fold_models"]],
        "full_model": ProbabilityMeanClassifier(bundle["full_model"]),
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
        "role": "continuous_estimator_control_ablation_not_observability_selection",
        "source_classifier_bundle": str(source),
        "source_classifier_bundle_sha256": _sha256(source),
        "classifier_bundle": str(output),
        "classifier_bundle_sha256": _sha256(output),
        "prediction_semantics": "posterior_probability_weighted_mean_gain",
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
