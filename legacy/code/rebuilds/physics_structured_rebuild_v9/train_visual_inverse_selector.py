#!/usr/bin/env python3
"""Train and calibrate a visual inverse expert selector without validation leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v9.runtime import (
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "visual_inverse_selector_cache_v9.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "visual_inverse_expert_selector_v9.pkl"
DEFAULT_SECONDARY = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v4_quickcheck_12h"
    / "forward_physics_residual_v4.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--primary-forward",
        type=Path,
        default=DEFAULT_NATURAL_GRID_FORWARD_STATE,
    )
    parser.add_argument(
        "--secondary-forward",
        type=Path,
        default=DEFAULT_SECONDARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    hashed = np.asarray(
        [
            int(
                hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16],
                16,
            )
            for group_id in group_ids
        ],
        dtype=np.uint64,
    )
    calibration = hashed % 5 == 0
    return np.flatnonzero(~calibration), np.flatnonzero(calibration)


def gamma_bucket(gamma: np.ndarray) -> np.ndarray:
    values = np.asarray(gamma, dtype=np.float32)
    return np.where(values < 0.75, 0, np.where(values <= 1.05, 1, 2))


def threshold_search(
    probability: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    gamma: np.ndarray,
) -> dict[str, Any]:
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([np.inf], dtype=np.float64),
                np.asarray(probability, dtype=np.float64),
            ]
        )
    )
    primary_total = int(primary.sum())
    buckets = gamma_bucket(gamma)
    primary_by_bucket = {
        int(bucket): int(primary[buckets == bucket].sum())
        for bucket in np.unique(buckets)
    }
    best: tuple[int, int, float] | None = None
    best_report: dict[str, Any] | None = None
    for threshold in candidates:
        choose_secondary = probability >= threshold
        selected = np.where(choose_secondary, secondary, primary)
        selected_total = int(selected.sum())
        selected_by_bucket = {
            int(bucket): int(selected[buckets == bucket].sum())
            for bucket in np.unique(buckets)
        }
        if selected_total < primary_total:
            continue
        if any(
            selected_by_bucket[bucket] < primary_by_bucket[bucket]
            for bucket in primary_by_bucket
        ):
            continue
        key = (
            selected_total,
            -int(choose_secondary.sum()),
            float(threshold),
        )
        if best is None or key > best:
            best = key
            best_report = {
                "threshold": float(threshold),
                "primary_success_count": primary_total,
                "selected_success_count": selected_total,
                "secondary_count": int(choose_secondary.sum()),
                "primary_by_gamma_bucket": primary_by_bucket,
                "selected_by_gamma_bucket": selected_by_bucket,
            }
    if best_report is None:
        raise RuntimeError("no protected visual selector threshold exists")
    return best_report


def metrics(
    probability: np.ndarray,
    threshold: float,
    primary: np.ndarray,
    secondary: np.ndarray,
) -> dict[str, Any]:
    choose_secondary = probability >= threshold
    selected = np.where(choose_secondary, secondary, primary)
    return {
        "count": int(len(primary)),
        "primary_success_count": int(primary.sum()),
        "secondary_success_count": int(secondary.sum()),
        "oracle_union_success_count": int((primary | secondary).sum()),
        "selected_success_count": int(selected.sum()),
        "selected_success_rate": float(selected.mean()),
        "secondary_count": int(choose_secondary.sum()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        train_features = np.asarray(
            cache["train_features"],
            dtype=np.float32,
        )
        train_primary = np.asarray(
            cache["train_primary_success"],
            dtype=np.bool_,
        )
        train_secondary = np.asarray(
            cache["train_secondary_success"],
            dtype=np.bool_,
        )
        train_group_ids = np.asarray(cache["train_group_ids"], dtype=np.str_)
        train_gamma = np.asarray(cache["train_gamma"], dtype=np.float32)
        val_features = np.asarray(cache["val_features"], dtype=np.float32)
        val_primary = np.asarray(
            cache["val_primary_success"],
            dtype=np.bool_,
        )
        val_secondary = np.asarray(
            cache["val_secondary_success"],
            dtype=np.bool_,
        )
    training_indices, calibration_indices = stable_split(
        train_group_ids,
        int(args.seed),
    )
    exclusive = train_primary ^ train_secondary
    selected_train = training_indices[exclusive[training_indices]]
    labels = train_secondary[selected_train].astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("training split lacks both exclusive expert classes")
    weights = (len(labels) / (2.0 * counts))[labels]
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=240,
        max_leaf_nodes=31,
        min_samples_leaf=24,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=int(args.seed),
    )
    classifier.fit(
        train_features[selected_train],
        labels,
        sample_weight=weights,
    )
    calibration_probability = classifier.predict_proba(
        train_features[calibration_indices]
    )[:, 1]
    calibration = threshold_search(
        calibration_probability,
        train_primary[calibration_indices],
        train_secondary[calibration_indices],
        train_gamma[calibration_indices],
    )
    threshold = float(calibration["threshold"])
    validation_probability = classifier.predict_proba(val_features)[:, 1]
    validation = metrics(
        validation_probability,
        threshold,
        val_primary,
        val_secondary,
    )
    artifact = {
        "version": "visual_inverse_expert_selector_v9_one_seed",
        "model": "hgb_visual_inverse_expert_selector_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": threshold,
        "primary_forward": str(args.primary_forward.resolve()),
        "primary_forward_sha256": sha256(args.primary_forward.resolve()),
        "secondary_forward": str(args.secondary_forward.resolve()),
        "secondary_forward_sha256": sha256(args.secondary_forward.resolve()),
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
            "request_count": int(len(training_indices)),
            "exclusive_count": int(len(selected_train)),
            "primary_only_count": int(counts[0]),
            "secondary_only_count": int(counts[1]),
        },
        "calibration": calibration,
        "validation": validation,
        "promotion_passed": (
            validation["selected_success_count"]
            >= validation["primary_success_count"]
        ),
        "source_contract": {
            "validation_used_for_fit_or_threshold": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
