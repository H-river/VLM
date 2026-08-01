#!/usr/bin/env python3
"""Train a clean-image forward adapter on the matching natural-request subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v9.train_qwen_forward_adapter import (
    BASE_FORWARD,
    metric,
    select_blends,
)
from specialist_rebuild_v2.common import STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_FEATURES = DEFAULT_RUN / "qwen_forward_adapter_features.npz"
DEFAULT_SOURCE = (
    REPO_ROOT.parent
    / "VLM_data/qwen_orchestration/v1/private/source_cases/train.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_clean_image_forward_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def model() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_path = output_dir / "forward_image_adapter.pkl"
    summary_path = output_dir / "forward_image_adapter_summary.json"
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = args.features.resolve()
    arrays = np.load(feature_path, allow_pickle=False)
    source_path = args.source.resolve()
    source_by_group = {
        str(row["group_id"]): row for row in read_jsonl(source_path)
    }
    clean = np.asarray(
        [
            source_by_group[str(group_id)]["perturbation_family"] == "clean"
            for group_id in arrays["group_ids"]
        ],
        dtype=np.bool_,
    )
    if int(clean.sum()) != 500:
        raise ValueError("expected 500 clean image-adaptation cases")
    features = np.asarray(arrays["image_features"][clean], dtype=np.float32)
    target = np.asarray(
        arrays["image_residual_target"][clean],
        dtype=np.float32,
    )
    base = np.asarray(
        arrays["image_base_prediction"][clean],
        dtype=np.float32,
    )
    truth = np.asarray(
        arrays["image_truth_change"][clean],
        dtype=np.float32,
    )
    input_tolerance = np.asarray(
        arrays["image_input_tolerance"][clean],
        dtype=np.float32,
    )
    scoring_tolerance = np.asarray(
        arrays["image_scoring_tolerance"][clean],
        dtype=np.float32,
    )
    folds = KFold(
        n_splits=5,
        shuffle=True,
        random_state=int(args.seed),
    )
    oof = cross_val_predict(
        model(),
        features,
        target,
        cv=folds,
        n_jobs=1,
        method="predict",
    ).astype(np.float32)
    blend, selected_metrics = select_blends(
        base,
        oof,
        truth,
        input_tolerance,
        scoring_tolerance,
    )
    _, baseline_metrics = metric(
        base,
        np.zeros_like(oof),
        np.zeros(len(STATE_FIELDS), dtype=np.float32),
        truth,
        input_tolerance,
        scoring_tolerance,
    )
    if (
        selected_metrics["strict_all_five_count"]
        <= baseline_metrics["strict_all_five_count"]
    ):
        raise ValueError("clean-image adapter does not improve out of fold")

    fitted = model()
    fitted.fit(features, target)
    base_path = BASE_FORWARD["image"].resolve()
    artifact = {
        "version": "qwen_clean_image_forward_adapter_v9_one_seed",
        "model": "qwen_distribution_residual_forward_adapter_v9",
        "route_scope": "image",
        "estimator": "ridge_multioutput",
        "models": fitted,
        "field_blend": {
            field: float(blend[index])
            for index, field in enumerate(STATE_FIELDS)
        },
        "base_forward_artifact": str(base_path),
        "base_forward_artifact_sha256": sha256(base_path),
        "feature_mode": "engineered_46_plus_base_prediction_5",
        "training_feature_cache": str(feature_path),
        "training_feature_cache_sha256": sha256(feature_path),
        "training_subset": "clean_image_only",
        "held_out_validation_used": False,
        "held_out_test_used": False,
    }
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "training_count": int(clean.sum()),
        "field_blend": artifact["field_blend"],
        "five_fold_oof_baseline": baseline_metrics,
        "five_fold_oof_selected": selected_metrics,
        "source_contract": {
            "training_feature_cache": str(feature_path),
            "training_feature_cache_sha256": sha256(feature_path),
            "perturbation_source": str(source_path),
            "perturbation_source_sha256": sha256(source_path),
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
