#!/usr/bin/env python3
"""Train a small dual-forward inverse selector without validation leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_RUN / "inverse_selector_training_features.npz",
    )
    parser.add_argument(
        "--primary-forward",
        type=Path,
        default=(
            DEFAULT_RUN / "forward_residual_calibrated/forward_state.pkl"
        ),
    )
    parser.add_argument(
        "--secondary-forward",
        type=Path,
        default=(
            DEFAULT_RUN
            / "forward_extra_trees_calibrated/forward_state.pkl"
        ),
    )
    parser.add_argument(
        "--inverse-artifact",
        type=Path,
        default=(
            REPO_ROOT.parent
            / "VLM_runs/tabm_transformer_rebuild_v8_one_seed"
            / "transformer/inverse.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "inverse_selector_leakage_free_hgb_v9.pkl",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=80,
        max_leaf_nodes=5,
        min_samples_leaf=20,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    feature_path = args.features.resolve()
    arrays = np.load(feature_path, allow_pickle=False)
    features = np.asarray(arrays["features"], dtype=np.float32)
    feature_names = [str(value) for value in arrays["feature_names"].tolist()]
    primary_success = np.asarray(arrays["primary_success"], dtype=np.bool_)
    secondary_success = np.asarray(arrays["secondary_success"], dtype=np.bool_)
    strata = np.asarray(arrays["strata"], dtype=np.int8)
    exclusive = primary_success ^ secondary_success
    exclusive_features = features[exclusive]
    exclusive_labels = secondary_success[exclusive].astype(np.int8)
    exclusive_strata = strata[exclusive]
    if len(exclusive_labels) < 500 or len(np.unique(exclusive_labels)) != 2:
        raise ValueError("insufficient exclusive-success training examples")

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(args.seed),
    )
    oof = cross_val_predict(
        classifier(int(args.seed)),
        exclusive_features,
        exclusive_labels,
        cv=folds,
        method="predict",
        n_jobs=1,
    ).astype(np.int8)
    oof_correct = oof == exclusive_labels
    subgroup_accuracy = {
        str(index): float(oof_correct[exclusive_strata == index].mean())
        for index in range(4)
    }
    selected_success_count = int(
        np.sum(primary_success & secondary_success) + oof_correct.sum()
    )
    primary_success_count = int(primary_success.sum())
    secondary_success_count = int(secondary_success.sum())
    if selected_success_count <= max(
        primary_success_count,
        secondary_success_count,
    ):
        raise ValueError("selector does not beat both enumerators out of fold")

    final_classifier = classifier(int(args.seed))
    final_classifier.fit(exclusive_features, exclusive_labels)
    primary_path = args.primary_forward.resolve()
    secondary_path = args.secondary_forward.resolve()
    inverse_path = args.inverse_artifact.resolve()
    artifact = {
        "version": "inverse_selector_leakage_free_hgb_v9_one_seed",
        "model": "v8_ranker_dual_forward_hgb_selector_v9",
        "classifier": final_classifier,
        "feature_names": feature_names,
        "primary_forward_artifact": str(primary_path),
        "primary_forward_artifact_sha256": sha256(primary_path),
        "secondary_forward_artifact": str(secondary_path),
        "secondary_forward_artifact_sha256": sha256(secondary_path),
        "inverse_artifact": str(inverse_path),
        "inverse_artifact_sha256": sha256(inverse_path),
        "training": {
            "sample_count": int(len(features)),
            "exclusive_success_count": int(len(exclusive_labels)),
            "exclusive_secondary_label_count": int(exclusive_labels.sum()),
            "five_fold_oof_exclusive_accuracy": float(oof_correct.mean()),
            "five_fold_oof_exclusive_accuracy_by_stratum": subgroup_accuracy,
            "primary_success_count": primary_success_count,
            "secondary_success_count": secondary_success_count,
            "oof_selected_success_count": selected_success_count,
            "oof_selected_success_rate": selected_success_count / len(features),
            "architecture_fixed_before_validation": {
                "max_iter": 80,
                "max_leaf_nodes": 5,
                "min_samples_leaf": 20,
                "learning_rate": 0.05,
                "l2_regularization": 2.0,
            },
        },
        "source_contract": {
            "training_features": str(feature_path),
            "training_features_sha256": sha256(feature_path),
            "protected_validation_files_opened": [],
            "system_validation_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "training": artifact["training"],
                "source_contract": artifact["source_contract"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
