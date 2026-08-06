#!/usr/bin/env python3
"""Train a group-balanced physical-success inverse action ranker."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID, MOVEMENT, residual_components
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    normalized_per_request,
    ranker_features,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.inverse_success_runtime import (
    apply_gate,
    sha256,
    sigmoid,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    split_groups,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "combined_natural_inverse_forward_cache.npz"
DEFAULT_BASE = (
    DEFAULT_RUN / "combined_natural_inverse_ranker_adaptation/inverse.pt"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "inverse_success_ranker_v9.pkl"
QUALITY_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
MODEL_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
CANDIDATE_PROBABILITIES = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
PROBABILITY_ADVANTAGES = (0.0, 0.05, 0.1, 0.2, 0.3)
BASE_PROBABILITIES = (1.0, 0.8, 0.6, 0.4)
MODEL_SCORE_ADVANTAGES = (0.0, 0.2, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--estimators", type=int, default=500)
    return parser.parse_args()


def physical_quality(
    states: np.ndarray,
    desired: np.ndarray,
) -> np.ndarray:
    signed = residual_components(
        np.asarray(states, dtype=np.float32),
        np.asarray(desired, dtype=np.float32)[:, None, :],
    )
    return np.maximum.reduce(
        [
            np.hypot(signed[..., 0], signed[..., 1]),
            np.abs(signed[..., 2]),
            np.abs(signed[..., 3]),
            np.abs(signed[..., 4]),
        ]
    ).astype(np.float32)


def group_balanced_weights(
    labels: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    output = np.empty(labels.shape, dtype=np.float32)
    for index, row in enumerate(labels):
        positive = int(row.sum())
        negative = int(len(row) - positive)
        output[index, row] = 0.5 / max(positive, 1)
        output[index, ~row] = 0.5 / max(negative, 1)
    boundary = 1.0 / (0.25 + np.abs(quality - 1.0))
    boundary /= boundary.mean(axis=1, keepdims=True)
    return output * np.sqrt(boundary).astype(np.float32)


def source_outputs(
    classifier: Any,
    regressor: Any,
    features: np.ndarray,
    base_scores: np.ndarray,
    base_indices: np.ndarray,
    quality_weight: float,
    model_weight: float,
) -> dict[str, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    raw = np.asarray(
        classifier.predict(flat, raw_score=True),
        dtype=np.float32,
    ).reshape(features.shape[:2])
    predicted_quality = np.asarray(
        regressor.predict(flat),
        dtype=np.float32,
    ).reshape(features.shape[:2])
    model_score = (
        normalized_per_request(raw)
        + float(quality_weight)
        * normalized_per_request(-predicted_quality)
    )
    blended = (
        normalized_per_request(base_scores)
        + float(model_weight) * normalized_per_request(model_score)
    )
    candidate = (
        blended - 1e-7 * MOVEMENT[None, :]
    ).argmax(axis=1)
    return {
        "base": base_indices,
        "candidate": candidate,
        "probability": sigmoid(raw),
        "model_score": model_score,
    }


def count_success(
    positives: np.ndarray,
    indices: np.ndarray,
) -> int:
    return int(positives[np.arange(len(indices)), indices].sum())


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMClassifier, LGBMRegressor

    cache_path = args.cache.resolve()
    with np.load(cache_path, allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        contexts = np.asarray(cache["contexts"], dtype=np.float32)
        desired = np.asarray(cache["desired"], dtype=np.float32)
        positives = np.asarray(cache["positives"], dtype=np.bool_)
        states = {
            "primary": np.asarray(cache["primary_states"], dtype=np.float32),
            "secondary": np.asarray(
                cache["secondary_states"],
                dtype=np.float32,
            ),
        }
        true_states = np.asarray(
            cache["true_candidate_states"],
            dtype=np.float32,
        )
    quality = physical_quality(true_states, desired)
    if not np.array_equal(quality <= 1.0 + 1e-6, positives):
        raise ValueError("continuous physical quality differs from success mask")
    training, calibration = split_groups(group_ids, int(args.seed))
    torch, device = configure(int(args.seed), args.device)
    base_path = args.base_artifact.resolve()
    base, _ = load_inverse_runtime_v8(base_path, torch, device)

    arrays = {}
    for name, candidate_states in states.items():
        result = base.score_feature_arrays(
            contexts,
            desired,
            candidate_states,
            batch_size=128,
        )
        scores = np.asarray(result["scores"], dtype=np.float32)
        arrays[name] = {
            "base_scores": scores,
            "base_indices": np.asarray(
                result["selected_indices"],
                dtype=np.int64,
            ),
            "features": ranker_features(
                contexts,
                desired,
                candidate_states,
                scores,
                include_action_basis=True,
            ),
        }

    train_features = np.concatenate(
        [
            arrays[name]["features"][training].reshape(
                -1,
                arrays[name]["features"].shape[-1],
            )
            for name in arrays
        ],
        axis=0,
    )
    train_labels = np.concatenate(
        [positives[training].reshape(-1) for _ in arrays]
    ).astype(np.int8)
    base_weights = group_balanced_weights(
        positives[training],
        quality[training],
    )
    train_weights = np.concatenate(
        [base_weights.reshape(-1) for _ in arrays]
    )
    train_quality = np.concatenate(
        [np.log1p(quality[training]).reshape(-1) for _ in arrays]
    ).astype(np.float32)

    common = {
        "n_estimators": int(args.estimators),
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_child_samples": 30,
        "reg_lambda": 2.0,
        "max_bin": 255,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "n_jobs": 2,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "random_state": int(args.seed),
    }
    classifier = LGBMClassifier(objective="binary", **common)
    classifier.fit(
        train_features,
        train_labels,
        sample_weight=train_weights,
    )
    regressor = LGBMRegressor(objective="huber", **common)
    regressor.fit(
        train_features,
        train_quality,
        sample_weight=train_weights,
    )

    baseline_counts = {
        name: count_success(
            positives[calibration],
            arrays[name]["base_indices"][calibration],
        )
        for name in arrays
    }
    best = None
    search_count = 0
    for quality_weight in QUALITY_WEIGHTS:
        for model_weight in MODEL_WEIGHTS:
            source = {
                name: source_outputs(
                    classifier,
                    regressor,
                    arrays[name]["features"][calibration],
                    arrays[name]["base_scores"][calibration],
                    arrays[name]["base_indices"][calibration],
                    quality_weight,
                    model_weight,
                )
                for name in arrays
            }
            for candidate_probability in CANDIDATE_PROBABILITIES:
                for probability_advantage in PROBABILITY_ADVANTAGES:
                    for base_probability in BASE_PROBABILITIES:
                        for model_advantage in MODEL_SCORE_ADVANTAGES:
                            rule = {
                                "candidate_probability_min": float(
                                    candidate_probability
                                ),
                                "probability_advantage_min": float(
                                    probability_advantage
                                ),
                                "base_probability_max": float(
                                    base_probability
                                ),
                                "model_score_advantage_min": float(
                                    model_advantage
                                ),
                            }
                            counts = {}
                            changed = {}
                            for name, values in source.items():
                                chosen, use = apply_gate(
                                    values["base"],
                                    values["candidate"],
                                    values["probability"],
                                    values["model_score"],
                                    rule,
                                )
                                counts[name] = count_success(
                                    positives[calibration],
                                    chosen,
                                )
                                changed[name] = int(use.sum())
                            key = (
                                min(counts.values()),
                                sum(counts.values()),
                                -sum(changed.values()),
                                -float(model_weight),
                                -float(quality_weight),
                            )
                            candidate = {
                                "key": key,
                                "quality_weight": float(quality_weight),
                                "model_weight": float(model_weight),
                                "gate_rule": rule,
                                "success_count": counts,
                                "changed_count": changed,
                            }
                            if best is None or key > best["key"]:
                                best = candidate
                            search_count += 1
    assert best is not None
    artifact = {
        "version": "inverse_success_ranker_v9_one_seed",
        "model": "group_balanced_inverse_success_v9",
        "classifier": classifier,
        "regressor": regressor,
        "quality_weight": best["quality_weight"],
        "model_weight": best["model_weight"],
        "gate_rule": best["gate_rule"],
        "base_inverse_artifact": str(base_path),
        "base_inverse_artifact_sha256": sha256(base_path),
        "training_cache": str(cache_path),
        "training_cache_sha256": sha256(cache_path),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "training": {
            "group_count": int(len(training)),
            "candidate_row_count": int(len(train_labels)),
            "positive_row_count": int(train_labels.sum()),
            "group_balanced": True,
            "continuous_target": "log1p(max_normalized_physical_violation)",
        },
        "internal_calibration": {
            "group_count": int(len(calibration)),
            "baseline_success_count": baseline_counts,
            "selected": {
                **best,
                "key": list(best["key"]),
            },
            "searched_rule_count": int(search_count),
        },
        "source_contract": {
            "training_cache": str(cache_path),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
        "held_out_validation_used": False,
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
