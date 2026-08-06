#!/usr/bin/env python3
"""Train route-specific residual adapters on natural-request train cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specialist_rebuild_v2.common import STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_FEATURES = DEFAULT_RUN / "qwen_forward_adapter_features.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_forward_adapter"
BASE_FORWARD = {
    "state": DEFAULT_RUN
    / "forward_dual_source_calibrated/forward_state.pkl",
    "image": DEFAULT_RUN
    / "forward_dual_source_calibrated/forward_image.pkl",
}
BLENDS = np.asarray(
    [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25],
    dtype=np.float32,
)


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


def state_model(seed: int, trees: int = 128) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=trees,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=2,
        random_state=seed,
    )


def image_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def fit_predict_fold(
    route: str,
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    seed: int,
) -> np.ndarray:
    if route == "state":
        model = state_model(seed)
        model.fit(features[train], target[train])
        return np.asarray(model.predict(features[validation]), dtype=np.float32)
    output = np.empty((len(validation), len(STATE_FIELDS)), dtype=np.float32)
    for field in range(len(STATE_FIELDS)):
        model = image_model(seed + 17 * field)
        model.fit(features[train], target[train, field])
        output[:, field] = model.predict(features[validation])
    return output


def metric(
    base: np.ndarray,
    residual: np.ndarray,
    blend: np.ndarray,
    truth: np.ndarray,
    input_tolerance: np.ndarray,
    scoring_tolerance: np.ndarray,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    prediction = base + residual * blend[None, :]
    error = (
        np.abs(prediction * input_tolerance - truth) / scoring_tolerance
    )
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    key = (
        int(exact.sum()),
        int(passed.sum()),
        -float(error.mean()),
        -float(np.abs(blend).sum()),
    )
    return key, {
        "count": len(exact),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_scoring_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": {
            field: float(passed[:, index].mean())
            for index, field in enumerate(STATE_FIELDS)
        },
    }


def select_blends(
    base: np.ndarray,
    residual: np.ndarray,
    truth: np.ndarray,
    input_tolerance: np.ndarray,
    scoring_tolerance: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.zeros(len(STATE_FIELDS), dtype=np.float32)
    for _ in range(4):
        changed = False
        for field in range(len(STATE_FIELDS)):
            best = None
            for blend in BLENDS:
                proposal = selected.copy()
                proposal[field] = blend
                key, _ = metric(
                    base,
                    residual,
                    proposal,
                    truth,
                    input_tolerance,
                    scoring_tolerance,
                )
                candidate = (key, -float(blend), float(blend))
                if best is None or candidate > best:
                    best = candidate
            assert best is not None
            if selected[field] != best[2]:
                selected[field] = best[2]
                changed = True
        if not changed:
            break
    _, result = metric(
        base,
        residual,
        selected,
        truth,
        input_tolerance,
        scoring_tolerance,
    )
    return selected, result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "forward_adapter_summary.json"
    artifacts = {
        route: output_dir / f"forward_{route}_adapter.pkl"
        for route in ("state", "image")
    }
    if summary_path.exists() or any(path.exists() for path in artifacts.values()):
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.features.resolve()
    arrays = np.load(feature_path, allow_pickle=False)
    folds = list(
        KFold(
            n_splits=5,
            shuffle=True,
            random_state=int(args.seed),
        ).split(np.arange(len(arrays["group_ids"])))
    )
    route_summaries = {}
    for route in ("state", "image"):
        features = np.asarray(
            arrays[f"{route}_features"],
            dtype=np.float32,
        )
        target = np.asarray(
            arrays[f"{route}_residual_target"],
            dtype=np.float32,
        )
        base = np.asarray(
            arrays[f"{route}_base_prediction"],
            dtype=np.float32,
        )
        truth = np.asarray(
            arrays[f"{route}_truth_change"],
            dtype=np.float32,
        )
        input_tolerance = np.asarray(
            arrays[f"{route}_input_tolerance"],
            dtype=np.float32,
        )
        scoring_tolerance = np.asarray(
            arrays[f"{route}_scoring_tolerance"],
            dtype=np.float32,
        )
        oof = np.empty_like(target)
        for fold_index, (train, validation) in enumerate(folds):
            oof[validation] = fit_predict_fold(
                route,
                features,
                target,
                train,
                validation,
                int(args.seed) + 101 * fold_index,
            )
        selected_blend, selected_metrics = select_blends(
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
            raise ValueError(f"{route} adapter does not improve out of fold")

        if route == "state":
            fitted: Any = state_model(int(args.seed))
            fitted.fit(features, target)
            estimator = "random_forest_multioutput"
        else:
            fitted = []
            for field in range(len(STATE_FIELDS)):
                model = image_model(int(args.seed) + 17 * field)
                model.fit(features, target[:, field])
                fitted.append(model)
            estimator = "five_hist_gradient_boosting_heads"
        base_path = BASE_FORWARD[route].resolve()
        artifact = {
            "version": f"qwen_forward_{route}_adapter_v9_one_seed",
            "model": "qwen_distribution_residual_forward_adapter_v9",
            "route_scope": route,
            "estimator": estimator,
            "models": fitted,
            "field_blend": {
                field: float(selected_blend[index])
                for index, field in enumerate(STATE_FIELDS)
            },
            "base_forward_artifact": str(base_path),
            "base_forward_artifact_sha256": sha256(base_path),
            "feature_mode": "engineered_46_plus_base_prediction_5",
            "training_feature_cache": str(feature_path),
            "training_feature_cache_sha256": sha256(feature_path),
            "held_out_validation_used": False,
            "held_out_test_used": False,
        }
        artifact_path = artifacts[route]
        with artifact_path.open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        route_summaries[route] = {
            "artifact": str(artifact_path),
            "artifact_sha256": sha256(artifact_path),
            "estimator": estimator,
            "field_blend": artifact["field_blend"],
            "five_fold_oof_baseline": baseline_metrics,
            "five_fold_oof_selected": selected_metrics,
        }
        print(
            json.dumps(
                {"route": route, **route_summaries[route]},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "version": "qwen_forward_adapter_v9_one_seed",
        "seed": int(args.seed),
        "routes": route_summaries,
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
