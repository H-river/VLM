#!/usr/bin/env python3
"""Fit an even-group calibrator for the complementary full-basis expert."""

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

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9,
)
from physics_structured_rebuild_v9.train_forward_selector_extension import (
    extension_features,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import sha256
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    fixed_action_grid,
    read_jsonl,
)

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_CANDIDATE = DEFAULT_RUN / "full_basis_forward_surface_v9.pt"
DEFAULT_OUTPUT = DEFAULT_RUN / "full_basis_forward_selector_v9.pkl"
DEFAULT_OLD = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current", type=Path, default=DEFAULT_FORWARD_SELECTOR_ENSEMBLE_V9
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation", type=Path, default=DEFAULT_DIFFICULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def counts(success: np.ndarray, high: np.ndarray, mask: np.ndarray) -> list[int]:
    return [
        int(success[mask].sum()),
        int(success[mask & high].sum()),
    ]


def threshold_search(
    blocks: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    probabilities = np.concatenate(
        [
            values["probability"][values["calibration"]]
            for values in blocks.values()
        ]
    )
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([np.inf], dtype=np.float64),
                np.quantile(probabilities, np.linspace(0.0, 1.0, 401)),
            ]
        )
    )
    candidates = []
    for threshold in thresholds:
        margins = []
        chosen = 0
        reports = {}
        for name, values in blocks.items():
            mask = values["calibration"]
            choose = values["probability"] >= threshold
            selected = np.where(
                choose, values["candidate_success"], values["current_success"]
            )
            current_counts = counts(
                values["current_success"], values["high"], mask
            )
            selected_counts = counts(selected, values["high"], mask)
            block_margins = [
                selected_counts[index] - current_counts[index]
                for index in range(2)
            ]
            margins.extend(block_margins)
            chosen += int((choose & mask).sum())
            reports[name] = {
                "current": current_counts,
                "selected": selected_counts,
                "margins": block_margins,
            }
        nonregression = all(margin >= 0 for margin in margins)
        key = (
            int(nonregression),
            min(margins),
            sum(margins),
            -chosen,
            float(threshold),
        )
        candidates.append((key, float(threshold), reports, chosen))
    best = max(candidates, key=lambda row: row[0])
    return {
        "threshold": best[1],
        "calibration": best[2],
        "candidate_count": best[3],
    }


def split_report(
    values: dict[str, np.ndarray],
    threshold: float,
    split: str,
) -> dict[str, Any]:
    mask = values[split]
    choose = values["probability"] >= threshold
    selected = np.where(
        choose, values["candidate_success"], values["current_success"]
    )
    current_counts = counts(values["current_success"], values["high"], mask)
    candidate_counts = counts(
        values["candidate_success"], values["high"], mask
    )
    selected_counts = counts(selected, values["high"], mask)
    return {
        "transition_count": int(mask.sum()),
        "high_complexity_count": int((mask & values["high"]).sum()),
        "current": current_counts,
        "candidate": candidate_counts,
        "selected": selected_counts,
        "margins": [
            selected_counts[index] - current_counts[index]
            for index in range(2)
        ],
        "candidate_count": int((choose & mask).sum()),
        "candidate_only_captured": int(
            (
                choose
                & mask
                & values["candidate_success"]
                & ~values["current_success"]
            ).sum()
        ),
        "current_only_sacrificed": int(
            (
                choose
                & mask
                & values["current_success"]
                & ~values["candidate_success"]
            ).sum()
        ),
        "oracle_union_count": int(
            (
                mask
                & (values["candidate_success"] | values["current_success"])
            ).sum()
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from lightgbm import LGBMClassifier

    started = time.perf_counter()
    torch, device = configure(int(args.seed), args.device)
    current_path = args.current.resolve()
    candidate_path = args.candidate.resolve()
    current, _ = load_forward_selector_ensemble_runtime_v9(
        current_path, torch, device
    )
    candidate, _ = load_full_basis_forward_surface_runtime_v9(
        candidate_path, torch, device
    )
    blocks: dict[str, dict[str, np.ndarray]] = {}
    training_features = []
    training_labels = []
    action_complexity = np.asarray(
        [
            sum(abs(float(action[field])) > 0.0 for field in ACTION_FIELDS)
            for action in fixed_action_grid()
        ],
        dtype=np.int64,
    )
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        prior = current.base.predict_changes(rows).reshape(-1, 5)
        current_prediction = current.predict_changes(rows).reshape(-1, 5)
        candidate_prediction = candidate.predict_changes(rows).reshape(-1, 5)
        physical = np.concatenate([arrays.features, prior], axis=1)
        features = extension_features(
            physical, prior, current_prediction, candidate_prediction
        )
        target = arrays.normalized_changes
        current_success = np.all(
            np.abs(current_prediction - target) <= 1.0, axis=1
        )
        candidate_success = np.all(
            np.abs(candidate_prediction - target) <= 1.0, axis=1
        )
        group_index = np.repeat(np.arange(arrays.group_count), 81)
        training = group_index % 4 == 0
        calibration = group_index % 4 == 2
        confirmation = group_index % 2 == 1
        high = np.tile(action_complexity >= 3, arrays.group_count)
        exclusive_train = training & (current_success ^ candidate_success)
        training_features.append(features[exclusive_train])
        training_labels.append(
            candidate_success[exclusive_train].astype(np.int64)
        )
        blocks[name] = {
            "features": features,
            "current_success": current_success,
            "candidate_success": candidate_success,
            "training": training,
            "calibration": calibration,
            "confirmation": confirmation,
            "high": high,
        }
    fit_features = np.concatenate(training_features)
    fit_labels = np.concatenate(training_labels)
    class_counts = np.bincount(fit_labels, minlength=2).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("protected selector lacks both exclusive classes")
    sample_weight = (len(fit_labels) / (2.0 * class_counts))[fit_labels]
    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=35,
        reg_lambda=4.0,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        random_state=int(args.seed),
    )
    classifier.fit(
        fit_features, fit_labels, sample_weight=sample_weight
    )
    for values in blocks.values():
        values["probability"] = classifier.predict_proba(
            values["features"]
        )[:, 1]
    selection = threshold_search(blocks)
    threshold = float(selection["threshold"])
    reports = {
        name: {
            "calibration": split_report(values, threshold, "calibration"),
            "confirmation": split_report(values, threshold, "confirmation"),
        }
        for name, values in blocks.items()
    }
    confirmation_margins = [
        margin
        for block in reports.values()
        for margin in block["confirmation"]["margins"]
    ]
    confirmation_passed = bool(
        all(margin >= 0 for margin in confirmation_margins)
        and sum(confirmation_margins[::2]) > 0
    )
    artifact = {
        "version": "full_basis_forward_selector_v9_one_seed",
        "model": "full_basis_forward_selector_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": threshold,
        "current_selector": str(current_path),
        "current_selector_sha256": sha256(current_path),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": sha256(candidate_path),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "threshold": threshold,
        "training": {
            "exclusive_transition_count": int(len(fit_labels)),
            "current_only_count": int(class_counts[0]),
            "candidate_only_count": int(class_counts[1]),
            "protected_group_contract": (
                "group_index_mod_4_zero_fit; mod_4_two_threshold; odd_confirmation"
            ),
        },
        "selection": selection,
        "blocks": reports,
        "confirmation_passed": confirmation_passed,
        "source_contract": {
            "protected_validation_used_as_split_calibrator": True,
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
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
