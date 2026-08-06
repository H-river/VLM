#!/usr/bin/env python3
"""Train a listwise LightGBM re-ranker over frozen inverse-transformer scores."""

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

from control_rebuild_v3.common import ACTION_GRID, MOVEMENT
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    normalized_per_request,
    ranker_features,
    sha256,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
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
DEFAULT_OUTPUT = DEFAULT_RUN / "inverse_lightgbm_reranker_v9.pkl"
SOURCE_WEIGHTS = {
    "primary": 1.0,
    "secondary": 0.75,
    "true": 0.25,
}
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument(
        "--include-action-basis",
        action="store_true",
        help="Append the explicit four-action vector and interaction basis.",
    )
    return parser.parse_args()


def selected_success(
    positives: np.ndarray,
    base_scores: np.ndarray,
    ranker_scores: np.ndarray,
    weight: float,
) -> tuple[int, np.ndarray]:
    scores = (
        normalized_per_request(base_scores)
        + float(weight) * normalized_per_request(ranker_scores)
    )
    selected = (scores - 1e-7 * MOVEMENT[None, :]).argmax(axis=1)
    success = positives[np.arange(len(selected)), selected]
    return int(success.sum()), selected


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMRanker

    cache_path = args.cache.resolve()
    with np.load(cache_path, allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        contexts = np.asarray(cache["contexts"], dtype=np.float32)
        desired = np.asarray(cache["desired"], dtype=np.float32)
        positives = np.asarray(cache["positives"], dtype=np.bool_)
        states = {
            "primary": np.asarray(
                cache["primary_states"],
                dtype=np.float32,
            ),
            "secondary": np.asarray(
                cache["secondary_states"],
                dtype=np.float32,
            ),
            "true": np.asarray(
                cache["true_candidate_states"],
                dtype=np.float32,
            ),
        }
    training_groups, calibration_groups = split_groups(
        group_ids,
        int(args.seed),
    )
    torch, device = configure(int(args.seed), args.device)
    base_path = args.base_artifact.resolve()
    base, _ = load_inverse_runtime_v8(base_path, torch, device)

    source_arrays = {}
    for source, candidate_states in states.items():
        base_result = base.score_feature_arrays(
            contexts,
            desired,
            candidate_states,
            batch_size=128,
        )
        base_scores = np.asarray(base_result["scores"], dtype=np.float32)
        source_arrays[source] = {
            "base_scores": base_scores,
            "features": ranker_features(
                contexts,
                desired,
                candidate_states,
                base_scores,
                include_action_basis=bool(args.include_action_basis),
            ),
        }
        baseline_count, _ = selected_success(
            positives[calibration_groups],
            base_scores[calibration_groups],
            base_scores[calibration_groups],
            0.0,
        )
        print(
            json.dumps(
                {
                    "source": source,
                    "internal_calibration_baseline_success_count": (
                        baseline_count
                    ),
                    "internal_calibration_count": len(calibration_groups),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    train_features = np.concatenate(
        [
            source_arrays[source]["features"][training_groups].reshape(
                -1,
                source_arrays[source]["features"].shape[-1],
            )
            for source in SOURCE_WEIGHTS
        ],
        axis=0,
    )
    train_labels = np.concatenate(
        [
            positives[training_groups].reshape(-1).astype(np.int8)
            for _ in SOURCE_WEIGHTS
        ]
    )
    sample_weight = np.concatenate(
        [
            np.full(
                len(training_groups) * len(ACTION_GRID),
                float(SOURCE_WEIGHTS[source]),
                dtype=np.float32,
            )
            for source in SOURCE_WEIGHTS
        ]
    )
    groups = np.full(
        len(training_groups) * len(SOURCE_WEIGHTS),
        len(ACTION_GRID),
        dtype=np.int32,
    )
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=int(args.estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        reg_lambda=1.0,
        max_bin=255,
        feature_fraction=0.90,
        bagging_fraction=0.90,
        bagging_freq=1,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        random_state=int(args.seed),
    )
    ranker.fit(
        train_features,
        train_labels,
        group=groups,
        sample_weight=sample_weight,
    )

    calibration_scores = {}
    for source in SOURCE_WEIGHTS:
        features = source_arrays[source]["features"][calibration_groups]
        calibration_scores[source] = ranker.predict(
            features.reshape(-1, features.shape[-1])
        ).reshape(len(calibration_groups), len(ACTION_GRID))
    candidates = []
    for weight in BLEND_WEIGHTS:
        counts = {}
        for source in SOURCE_WEIGHTS:
            counts[source], _ = selected_success(
                positives[calibration_groups],
                source_arrays[source]["base_scores"][calibration_groups],
                calibration_scores[source],
                weight,
            )
        candidate = {
            "ranker_weight": float(weight),
            "success_count": counts,
            "key": [
                counts["primary"],
                counts["secondary"],
                counts["true"],
                -float(weight),
            ],
        }
        candidates.append(candidate)
    selected = max(candidates, key=lambda row: tuple(row["key"]))
    artifact = {
        "version": "inverse_lightgbm_reranker_v9_one_seed",
        "model": "lightgbm_inverse_reranker_v9",
        "seed": int(args.seed),
        "ranker": ranker,
        "ranker_weight": float(selected["ranker_weight"]),
        "feature_mode": (
            "inverse_context31_candidate23_status12_transformer_score_rank_gap"
            + (
                "_action4_basis"
                if args.include_action_basis
                else ""
            )
        ),
        "include_action_basis": bool(args.include_action_basis),
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
            "group_count": int(len(training_groups)),
            "expanded_source_group_count": int(len(groups)),
            "candidate_row_count": int(len(train_labels)),
            "positive_row_count": int(train_labels.sum()),
            "source_weights": SOURCE_WEIGHTS,
        },
        "internal_calibration": {
            "group_count": int(len(calibration_groups)),
            "candidates": candidates,
            "selected": selected,
        },
        "hyperparameters": {
            "estimators": int(args.estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_child_samples": int(args.min_child_samples),
            "include_action_basis": bool(args.include_action_basis),
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
