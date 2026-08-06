#!/usr/bin/env python3
"""Train an inverse-enumerator selector on disjoint natural training grids."""

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
    DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "grouped_inverse_selector_cache_v9.npz"
DEFAULT_GROUPED = DEFAULT_RUN / "grouped_forward_tree_protected_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "grouped_inverse_selector_v9.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--primary-forward",
        type=Path,
        default=DEFAULT_NATURAL_GRID_FORWARD_STATE,
    )
    parser.add_argument("--grouped-forward", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument(
        "--secondary-kind",
        choices=("grouped_tree", "full_basis"),
        default="grouped_tree",
    )
    parser.add_argument(
        "--inverse-artifact",
        type=Path,
        default=DEFAULT_COMBINED_NATURAL_INVERSE_V9,
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


def split(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.asarray(
        [
            int(
                hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16],
                16,
            )
            % 5
            == 0
            for group_id in group_ids
        ],
        dtype=np.bool_,
    )
    return np.flatnonzero(~calibration), np.flatnonzero(calibration)


def choose_threshold(
    probability: np.ndarray,
    primary: np.ndarray,
    grouped: np.ndarray,
) -> dict[str, Any]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([np.inf], dtype=np.float64),
                probability.astype(np.float64),
            ]
        )
    )
    best = None
    best_report = None
    for threshold in thresholds:
        choose = probability >= threshold
        selected = np.where(choose, grouped, primary)
        key = (
            int(selected.sum()),
            -int(choose.sum()),
            float(threshold),
        )
        if best is None or key > best:
            best = key
            best_report = {
                "threshold": float(threshold),
                "primary_success_count": int(primary.sum()),
                "grouped_success_count": int(grouped.sum()),
                "oracle_union_success_count": int((primary | grouped).sum()),
                "selected_success_count": int(selected.sum()),
                "grouped_count": int(choose.sum()),
            }
    assert best_report is not None
    return best_report


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        features = np.asarray(cache["features"], dtype=np.float32)
        feature_names = [
            str(value) for value in cache["feature_names"].tolist()
        ]
        primary = np.asarray(cache["primary_success"], dtype=np.bool_)
        grouped = np.asarray(cache["grouped_success"], dtype=np.bool_)
    training, calibration = split(group_ids, int(args.seed))
    exclusive = primary ^ grouped
    selected_training = training[exclusive[training]]
    labels = grouped[selected_training].astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("inverse selector lacks both exclusive classes")
    weights = (len(labels) / (2.0 * counts))[labels]
    classifier = HistGradientBoostingClassifier(
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        learning_rate=0.04,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=20,
        random_state=int(args.seed),
    )
    classifier.fit(
        features[selected_training],
        labels,
        sample_weight=weights,
    )
    probability = classifier.predict_proba(features[calibration])[:, 1]
    threshold = choose_threshold(
        probability,
        primary[calibration],
        grouped[calibration],
    )
    artifact = {
        "version": "grouped_inverse_selector_v9_one_seed",
        "model": "grouped_forward_inverse_hgb_selector_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": float(threshold["threshold"]),
        "feature_names": feature_names,
        "primary_forward_artifact": str(args.primary_forward.resolve()),
        "primary_forward_artifact_sha256": sha256(
            args.primary_forward.resolve()
        ),
        "secondary_forward_artifact": str(args.grouped_forward.resolve()),
        "secondary_forward_artifact_sha256": sha256(
            args.grouped_forward.resolve()
        ),
        "secondary_forward_kind": str(args.secondary_kind),
        "inverse_artifact": str(args.inverse_artifact.resolve()),
        "inverse_artifact_sha256": sha256(args.inverse_artifact.resolve()),
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
            "exclusive_count": int(len(labels)),
            "primary_only_count": int(counts[0]),
            "grouped_only_count": int(counts[1]),
        },
        "calibration": threshold,
        "source_contract": {
            "system_validation_used": False,
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
