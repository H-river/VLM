#!/usr/bin/env python3
"""Train a protected action-level selector for the grouped forward candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import tolerance_from_current
from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from physics_structured_rebuild_v9.calibrate_forward_system import (
    action_high_mask,
    specialist_counts,
)
from physics_structured_rebuild_v9.calibrate_system_direction import read_jsonl
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_selector_runtime import (
    grouped_selector_features,
)
from physics_structured_rebuild_v9.grouped_forward_tree_runtime import (
    load_grouped_forward_tree_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "combined_natural_inverse_forward_cache.npz"
DEFAULT_CANDIDATE = DEFAULT_RUN / "grouped_forward_tree_protected_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "grouped_forward_selector_v9.pkl"
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--current-forward",
        type=Path,
        default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
    )
    parser.add_argument("--candidate-forward", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_contexts(
    contexts: np.ndarray,
    group_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index, context in enumerate(contexts):
        setup = {
            field: float(context[position])
            for position, field in enumerate(SETUP_FIELDS)
        }
        current_values = np.asarray(context[12:17], dtype=np.float64).copy()
        current_values[-1] = math.expm1(float(current_values[-1]))
        rows.append(
            {
                "group_id": str(group_ids[index]),
                "setup": setup,
                "current_beam_state": {
                    field: float(current_values[position])
                    for position, field in enumerate(STATE_FIELDS)
                },
            }
        )
    return rows


def group_split(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
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


def threshold_search(
    probability: np.ndarray,
    current_success: np.ndarray,
    candidate_success: np.ndarray,
    protected: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
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
        selected = np.where(choose, candidate_success, current_success)
        protected_reports = []
        valid = True
        for block_probability, block_current, block_candidate, high in protected:
            block_choose = block_probability >= threshold
            block_selected = np.where(
                block_choose,
                block_candidate,
                block_current,
            )
            current_counts = (
                int(block_current.sum()),
                int(block_current[high].sum()),
            )
            selected_counts = (
                int(block_selected.sum()),
                int(block_selected[high].sum()),
            )
            if any(
                selected_value < current_value
                for selected_value, current_value in zip(
                    selected_counts,
                    current_counts,
                    strict=True,
                )
            ):
                valid = False
                break
            protected_reports.append(
                {
                    "current": list(current_counts),
                    "selected": list(selected_counts),
                    "candidate_count": int(block_choose.sum()),
                }
            )
        if not valid:
            continue
        key = (
            int(selected.sum()),
            -int(choose.sum()),
            float(threshold),
        )
        if best is None or key > best:
            best = key
            best_report = {
                "threshold": float(threshold),
                "current_success_count": int(current_success.sum()),
                "selected_success_count": int(selected.sum()),
                "candidate_count": int(choose.sum()),
                "protected": protected_reports,
            }
    if best_report is None:
        raise RuntimeError("no protected grouped selector threshold exists")
    return best_report


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    started = time.perf_counter()
    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        contexts = np.asarray(cache["contexts"][:, :17], dtype=np.float32)
        truth_states = np.asarray(
            cache["true_candidate_states"],
            dtype=np.float32,
        )
    rows = rows_from_contexts(contexts, group_ids)
    current_state = contexts[:, 12:17].copy()
    current_state[:, -1] = np.expm1(current_state[:, -1])
    tolerance = np.stack(
        [tolerance_from_current(values) for values in current_state]
    ).astype(np.float32)
    target = (
        truth_states - current_state[:, None, :]
    ) / tolerance[:, None, :]
    torch, device = configure(int(args.seed), args.device)
    current_runtime, _ = load_forward_selector_ensemble_runtime_v9(
        args.current_forward.resolve(),
        torch,
        device,
    )
    candidate_runtime, _ = load_grouped_forward_tree_runtime_v9(
        args.candidate_forward.resolve(),
        torch,
        device,
    )
    current_prediction = current_runtime.predict_changes(rows)
    candidate_prediction = candidate_runtime.predict_changes(rows)
    features = grouped_selector_features(
        rows,
        current_prediction,
        candidate_prediction,
    )
    current_success = np.all(
        np.abs(current_prediction - target) <= 1.0,
        axis=2,
    )
    candidate_success = np.all(
        np.abs(candidate_prediction - target) <= 1.0,
        axis=2,
    )
    training_groups, calibration_groups = group_split(
        group_ids,
        int(args.seed),
    )
    exclusive = current_success ^ candidate_success
    training_mask = np.zeros(len(group_ids), dtype=np.bool_)
    training_mask[training_groups] = True
    selected = exclusive & training_mask[:, None]
    labels = candidate_success[selected].astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("grouped selector lacks both exclusive classes")
    weights = (len(labels) / (2.0 * counts))[labels]
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=int(args.seed),
    )
    classifier.fit(
        features[selected],
        labels,
        sample_weight=weights,
    )
    protected = []
    protected_names = []
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        block_rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_current_prediction = current_runtime.predict_changes(block_rows)
        block_candidate_prediction = candidate_runtime.predict_changes(
            block_rows
        )
        block_features = grouped_selector_features(
            block_rows,
            block_current_prediction,
            block_candidate_prediction,
        )
        block_probability = classifier.predict_proba(
            block_features.reshape(-1, block_features.shape[-1])
        )[:, 1]
        block_target = arrays.normalized_changes
        block_current_success = np.all(
            np.abs(
                block_current_prediction.reshape(-1, 5) - block_target
            )
            <= 1.0,
            axis=1,
        )
        block_candidate_success = np.all(
            np.abs(
                block_candidate_prediction.reshape(-1, 5) - block_target
            )
            <= 1.0,
            axis=1,
        )
        protected.append(
            (
                block_probability,
                block_current_success,
                block_candidate_success,
                action_high_mask(len(block_rows)),
            )
        )
        protected_names.append(name)
    calibration_features = features[calibration_groups].reshape(
        -1,
        features.shape[-1],
    )
    calibration_probability = classifier.predict_proba(
        calibration_features
    )[:, 1]
    calibration_current = current_success[calibration_groups].reshape(-1)
    calibration_candidate = candidate_success[calibration_groups].reshape(-1)
    threshold = threshold_search(
        calibration_probability,
        calibration_current,
        calibration_candidate,
        protected,
    )
    threshold["protected"] = {
        name: value
        for name, value in zip(
            protected_names,
            threshold["protected"],
            strict=True,
        )
    }
    artifact = {
        "version": "grouped_forward_selector_v9_one_seed",
        "model": "hgb_grouped_forward_selector_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": float(threshold["threshold"]),
        "current_forward": str(args.current_forward.resolve()),
        "current_forward_sha256": sha256(args.current_forward.resolve()),
        "candidate_forward": str(args.candidate_forward.resolve()),
        "candidate_forward_sha256": sha256(
            args.candidate_forward.resolve()
        ),
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
            "exclusive_transition_count": int(len(labels)),
            "current_only_count": int(counts[0]),
            "candidate_only_count": int(counts[1]),
        },
        "calibration": threshold,
        "calibration_oracle_union_success_count": int(
            (
                calibration_current | calibration_candidate
            ).sum()
        ),
        "seconds": time.perf_counter() - started,
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
