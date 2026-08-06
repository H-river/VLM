#!/usr/bin/env python3
"""Fit a split-protected whole-vector gate for the clean direction model."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    load_boundary_direction_correction_runtime_v9,
)
from physics_structured_rebuild_v9.full_basis_forward_surface import (
    load_full_basis_forward_surface_runtime_v9,
)
from physics_structured_rebuild_v9.train_forward_tree_residual import sha256
from specialist_rebuild_v2.common import read_jsonl

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_RUN / "boundary_direction_correction_state_v9.pkl",
    )
    parser.add_argument(
        "--forward",
        type=Path,
        default=DEFAULT_RUN / "full_basis_forward_surface_v9.pt",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_RUN / "clean_full_direction_v9.pkl",
    )
    parser.add_argument(
        "--old-validation",
        type=Path,
        default=REPO_ROOT.parent
        / "VLM_data/specialist_rebuild_v2/grids/val.jsonl",
    )
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=REPO_ROOT.parent
        / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def gate_features(
    physical: np.ndarray,
    current: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    candidate = probabilities.argmax(axis=2)
    ordered = np.sort(probabilities, axis=2)
    return np.concatenate(
        [
            physical,
            np.eye(3, dtype=np.float32)[current].reshape(len(current), -1),
            probabilities.reshape(len(current), -1),
            ordered[:, :, -1],
            ordered[:, :, -1] - ordered[:, :, -2],
            (candidate != current).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def choose_threshold(blocks: dict[str, dict[str, np.ndarray]]) -> float:
    values = np.concatenate(
        [block["probability"][block["calibration"]] for block in blocks.values()]
    )
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([np.inf]),
                np.quantile(values, np.linspace(0.0, 1.0, 401)),
            ]
        )
    )
    candidates = []
    for threshold in thresholds:
        margins = []
        selected_count = 0
        for block in blocks.values():
            mask = block["calibration"]
            choose = block["probability"] >= threshold
            selected = np.where(
                choose, block["candidate_success"], block["current_success"]
            )
            margins.append(
                int(selected[mask].sum())
                - int(block["current_success"][mask].sum())
            )
            selected_count += int((choose & mask).sum())
        key = (
            int(all(margin >= 0 for margin in margins)),
            min(margins),
            sum(margins),
            -selected_count,
            float(threshold),
        )
        candidates.append((key, float(threshold)))
    return max(candidates, key=lambda row: row[0])[1]


def report_split(
    block: dict[str, np.ndarray],
    threshold: float,
    split: str,
) -> dict[str, Any]:
    mask = block[split]
    choose = block["probability"] >= threshold
    selected = np.where(
        choose, block["candidate_success"], block["current_success"]
    )
    current_count = int(block["current_success"][mask].sum())
    selected_count = int(selected[mask].sum())
    return {
        "count": int(mask.sum()),
        "current_count": current_count,
        "candidate_count": int(block["candidate_success"][mask].sum()),
        "selected_count": selected_count,
        "margin": selected_count - current_count,
        "gate_count": int((choose & mask).sum()),
        "candidate_only_captured": int(
            (
                choose
                & mask
                & block["candidate_success"]
                & ~block["current_success"]
            ).sum()
        ),
        "current_only_sacrificed": int(
            (
                choose
                & mask
                & block["current_success"]
                & ~block["candidate_success"]
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

    torch, device = configure(int(args.seed), args.device)
    current_path = args.current.resolve()
    forward_path = args.forward.resolve()
    candidate_path = args.candidate.resolve()
    current, _ = load_boundary_direction_correction_runtime_v9(
        current_path, torch, device
    )
    forward, _ = load_full_basis_forward_surface_runtime_v9(
        forward_path, torch, device
    )
    with candidate_path.open("rb") as stream:
        direction = pickle.load(stream)
    models = list(direction["models"])
    blocks = {}
    fit_features = []
    fit_labels = []
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        _, _, _, current_grid = current.base_grid(rows)
        current_labels = current_grid.reshape(-1, 5)
        prior, correction = forward.predict_correction(rows)
        full_prediction = (
            prior
            + correction * forward.field_blend[None, None, :]
        )
        physical = np.concatenate(
            [
                arrays.features,
                prior.reshape(-1, 5),
                full_prediction.reshape(-1, 5),
            ],
            axis=1,
        ).astype(np.float32)
        probabilities = np.stack(
            [model.predict_proba(physical) for model in models], axis=1
        ).astype(np.float32)
        candidate_labels = probabilities.argmax(axis=2)
        features = gate_features(physical, current_labels, probabilities)
        current_success = np.all(current_labels == arrays.labels, axis=1)
        candidate_success = np.all(candidate_labels == arrays.labels, axis=1)
        group_index = np.repeat(np.arange(arrays.group_count), 81)
        training = group_index % 4 == 0
        calibration = group_index % 4 == 2
        confirmation = group_index % 2 == 1
        exclusive = training & (current_success ^ candidate_success)
        fit_features.append(features[exclusive])
        fit_labels.append(candidate_success[exclusive].astype(np.int64))
        blocks[name] = {
            "features": features,
            "current_success": current_success,
            "candidate_success": candidate_success,
            "calibration": calibration,
            "confirmation": confirmation,
        }
    train_x = np.concatenate(fit_features)
    train_y = np.concatenate(fit_labels)
    class_count = np.bincount(train_y, minlength=2).astype(np.float64)
    weight = (len(train_y) / (2.0 * class_count))[train_y]
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
    classifier.fit(train_x, train_y, sample_weight=weight)
    for block in blocks.values():
        block["probability"] = classifier.predict_proba(
            block["features"]
        )[:, 1]
    threshold = choose_threshold(blocks)
    reports = {
        name: {
            "calibration": report_split(block, threshold, "calibration"),
            "confirmation": report_split(block, threshold, "confirmation"),
        }
        for name, block in blocks.items()
    }
    margins = [
        values["confirmation"]["margin"] for values in reports.values()
    ]
    passed = all(margin >= 0 for margin in margins) and sum(margins) > 0
    artifact = {
        "version": "clean_full_direction_gate_v9_one_seed",
        "model": "clean_full_direction_whole_vector_gate_v9",
        "seed": int(args.seed),
        "classifier": classifier,
        "threshold": float(threshold),
        "current_artifact": str(current_path),
        "current_artifact_sha256": sha256(current_path),
        "forward_artifact": str(forward_path),
        "forward_artifact_sha256": sha256(forward_path),
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
        "threshold": float(threshold),
        "training": {
            "exclusive_count": int(len(train_y)),
            "current_only_count": int(class_count[0]),
            "candidate_only_count": int(class_count[1]),
        },
        "blocks": reports,
        "confirmation_passed": bool(passed),
        "source_contract": {
            "protected_validation_used_as_split_calibrator": True,
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
