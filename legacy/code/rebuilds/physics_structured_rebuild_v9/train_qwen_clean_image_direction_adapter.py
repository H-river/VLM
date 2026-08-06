#!/usr/bin/env python3
"""Train a conservative direction correction gate on clean natural requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v9.runtime import (
    DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_FEATURES = DEFAULT_RUN / "qwen_direction_adapter_features.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_clean_image_direction_adapter"
THRESHOLDS = np.asarray([0.6, 0.7, 0.8, 0.8, 0.7], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=96,
        min_samples_leaf=10,
        max_features=0.8,
        class_weight="balanced",
        n_jobs=2,
        random_state=seed,
    )


def augmented(features: np.ndarray, base: np.ndarray) -> np.ndarray:
    one_hot = np.eye(3, dtype=np.float32)[base].reshape(len(base), -1)
    return np.concatenate([features, one_hot], axis=1).astype(np.float32)


def corrected(
    probabilities: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    learned = probabilities.argmax(axis=2)
    confidence = probabilities.max(axis=2)
    return np.where(
        confidence >= THRESHOLDS[None, :],
        learned,
        base,
    ).astype(np.int8)


def direction_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict:
    correct = predicted == target
    exact = np.all(correct, axis=1)
    return {
        "count": len(exact),
        "all_five_exact_count": int(exact.sum()),
        "all_five_exact": float(exact.mean()),
        "changed_count": None,
        "per_field_accuracy": {
            field: float(correct[:, index].mean())
            for index, field in enumerate(DIRECTION_FIELDS)
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_path = output_dir / "direction_image_adapter.pkl"
    summary_path = output_dir / "direction_image_adapter_summary.json"
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.features.resolve()
    arrays = np.load(feature_path, allow_pickle=False)
    clean = np.asarray(arrays["perturbations"] == "clean", dtype=np.bool_)
    if int(clean.sum()) != 500:
        raise ValueError("expected 500 clean direction cases")
    base = np.asarray(arrays["image_base_labels"][clean], dtype=np.int8)
    target = np.asarray(arrays["labels"][clean], dtype=np.int8)
    features = augmented(
        np.asarray(arrays["image_features"][clean], dtype=np.float32),
        base,
    )
    probabilities = np.zeros((len(target), len(DIRECTION_FIELDS), 3), dtype=np.float32)
    for field in range(len(DIRECTION_FIELDS)):
        labels = target[:, field]
        folds = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(args.seed) + field,
        )
        for fold_index, (train, validation) in enumerate(
            folds.split(features, labels)
        ):
            classifier = model(
                int(args.seed) + 101 * fold_index + 17 * field
            )
            classifier.fit(features[train], labels[train])
            classes = np.asarray(classifier.classes_, dtype=np.int64)
            probability = classifier.predict_proba(features[validation])
            probabilities[
                validation[:, None],
                field,
                classes[None, :],
            ] = probability
    oof = corrected(probabilities, base)
    baseline_metrics = direction_metrics(base, target)
    selected_metrics = direction_metrics(oof, target)
    selected_metrics["changed_count"] = int(np.sum(oof != base))
    if (
        selected_metrics["all_five_exact_count"]
        <= baseline_metrics["all_five_exact_count"]
    ):
        raise ValueError("direction adapter does not improve out of fold")

    models = []
    for field in range(len(DIRECTION_FIELDS)):
        classifier = model(int(args.seed) + 17 * field)
        classifier.fit(features, target[:, field])
        models.append(classifier)
    base_path = DEFAULT_EXTRA_FORWARD_DIRECTION_IMAGE_HYBRID.resolve()
    artifact = {
        "version": "qwen_clean_image_direction_adapter_v9_one_seed",
        "model": "qwen_distribution_direction_correction_gate_v9",
        "route_scope": "predict_direction_from_image_v1",
        "models": models,
        "feature_mode": (
            "engineered_46_plus_dual_forward_prediction_5"
            "_plus_base_direction_one_hot_15"
        ),
        "confidence_threshold": {
            field: float(THRESHOLDS[index])
            for index, field in enumerate(DIRECTION_FIELDS)
        },
        "base_direction_artifact": str(base_path),
        "base_direction_artifact_sha256": sha256(base_path),
        "training_feature_cache": str(feature_path),
        "training_feature_cache_sha256": sha256(feature_path),
        "held_out_validation_used": False,
        "held_out_test_used": False,
    }
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "training_count": len(target),
        "confidence_threshold": artifact["confidence_threshold"],
        "five_fold_oof_baseline": baseline_metrics,
        "five_fold_oof_selected": selected_metrics,
        "source_contract": {
            "training_feature_cache": str(feature_path),
            "training_feature_cache_sha256": sha256(feature_path),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
        "held_out_validation_used": False,
        "held_out_test_used": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
