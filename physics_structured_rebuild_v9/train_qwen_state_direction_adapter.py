#!/usr/bin/env python3
"""Train a calibrated state-direction correction on rounded Qwen requests."""

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
    DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID,
)
from specialist_rebuild_v2.common import DIRECTION_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_FEATURES = DEFAULT_RUN / "qwen_direction_adapter_features.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "qwen_state_direction_adapter_v9.pkl"
CONFIDENCE = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MARGINS = (0.0, 0.1, 0.2, 0.3)
MODES = ("all", "rare_only", "base_no_change_to_rare", "to_no_change")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
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


def augmented(features: np.ndarray, base: np.ndarray) -> np.ndarray:
    one_hot = np.eye(3, dtype=np.float32)[base].reshape(len(base), -1)
    return np.concatenate([features, one_hot], axis=1).astype(np.float32)


def gate_mask(
    probability: np.ndarray,
    base: np.ndarray,
    choice: dict[str, Any],
) -> np.ndarray:
    ordered = np.sort(probability, axis=1)
    learned = probability.argmax(axis=1)
    mask = (
        (ordered[:, -1] >= float(choice["confidence"]))
        & (
            ordered[:, -1] - ordered[:, -2]
            >= float(choice["margin"])
        )
        & (learned != base)
    )
    mode = str(choice["mode"])
    if mode == "rare_only":
        mask &= learned != 1
    elif mode == "base_no_change_to_rare":
        mask &= (base == 1) & (learned != 1)
    elif mode == "to_no_change":
        mask &= learned == 1
    elif mode != "all":
        raise ValueError("unknown direction adapter mode")
    return mask


def apply_selection(
    base: np.ndarray,
    probabilities: np.ndarray,
    selection: list[dict[str, Any] | None],
) -> np.ndarray:
    output = np.asarray(base, dtype=np.int8).copy()
    for field, choice in enumerate(selection):
        if choice is None:
            continue
        probability = probabilities[:, field]
        learned = probability.argmax(axis=1).astype(np.int8)
        mask = gate_mask(probability, output[:, field], choice)
        output[mask, field] = learned[mask]
    return output


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    correct = predicted == target
    exact = np.all(correct, axis=1)
    return {
        "count": int(len(exact)),
        "all_five_exact_count": int(exact.sum()),
        "all_five_exact": float(exact.mean()),
        "per_field_correct": correct.sum(axis=0).astype(int).tolist(),
    }


def select_gates(
    probabilities: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
) -> tuple[list[dict[str, Any] | None], list[dict[str, Any]]]:
    choices: list[dict[str, Any] | None] = [None]
    choices.extend(
        {
            "mode": mode,
            "confidence": confidence,
            "margin": margin,
        }
        for mode in MODES
        for confidence in CONFIDENCE
        for margin in MARGINS
    )
    selection: list[dict[str, Any] | None] = [None] * 5
    trace = []
    for pass_index in range(4):
        changed = False
        for field in range(5):
            best = None
            for choice in choices:
                proposal = list(selection)
                proposal[field] = choice
                predicted = apply_selection(base, probabilities, proposal)
                correct = predicted == target
                exact = np.all(correct, axis=1)
                changed_fields = int(np.sum(predicted != base))
                key = (
                    int(exact.sum()),
                    int(correct.sum()),
                    int(correct[:, field].sum()),
                    -changed_fields,
                    choice is None,
                )
                if best is None or key > best[0]:
                    best = (key, choice)
            assert best is not None
            if selection[field] != best[1]:
                selection[field] = best[1]
                changed = True
        selected = apply_selection(base, probabilities, selection)
        trace.append(
            {
                "pass": pass_index + 1,
                "selection": selection,
                "metrics": metrics(selected, target),
                "changed_field_count": int(np.sum(selected != base)),
            }
        )
        if not changed:
            break
    return selection, trace


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    with np.load(args.features.resolve(), allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        features = np.asarray(cache["state_features"], dtype=np.float32)
        base = np.asarray(cache["state_base_labels"], dtype=np.int8)
        target = np.asarray(cache["labels"], dtype=np.int8)
    model_features = augmented(features, base)
    training, calibration = split(group_ids, int(args.seed))
    models = []
    for field in range(5):
        labels = target[training, field].astype(np.int64)
        counts = np.bincount(labels, minlength=3).astype(np.float64)
        weights_by_class = np.sqrt(
            len(labels) / (3.0 * np.maximum(counts, 1.0))
        )
        model = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=4.0,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=int(args.seed) + 17 * field,
        )
        model.fit(
            model_features[training],
            labels,
            sample_weight=weights_by_class[labels],
        )
        models.append(model)
    probabilities = np.zeros((len(calibration), 5, 3), dtype=np.float32)
    rows = np.arange(len(calibration))
    for field, model in enumerate(models):
        classes = np.asarray(model.classes_, dtype=np.int64)
        probabilities[
            rows[:, None],
            field,
            classes[None, :],
        ] = model.predict_proba(model_features[calibration])
    selection, trace = select_gates(
        probabilities,
        base[calibration],
        target[calibration],
    )
    selected = apply_selection(
        base[calibration],
        probabilities,
        selection,
    )
    base_path = DEFAULT_EXTRA_FORWARD_DIRECTION_STATE_HYBRID.resolve()
    artifact = {
        "version": "qwen_state_direction_adapter_v9_one_seed",
        "model": "qwen_state_direction_hgb_correction_gate_v9",
        "seed": int(args.seed),
        "models": models,
        "selection": selection,
        "base_direction_artifact": str(base_path),
        "base_direction_artifact_sha256": sha256(base_path),
        "feature_mode": (
            "engineered_46_plus_dual_forward_prediction_5"
            "_plus_base_direction_one_hot_15"
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
        "training_count": int(len(training)),
        "calibration_count": int(len(calibration)),
        "training_class_counts": {
            field: np.bincount(
                target[training, index],
                minlength=3,
            ).astype(int).tolist()
            for index, field in enumerate(DIRECTION_FIELDS)
        },
        "selection": selection,
        "calibration": {
            "baseline": metrics(base[calibration], target[calibration]),
            "selected": metrics(selected, target[calibration]),
            "changed_field_count": int(
                np.sum(selected != base[calibration])
            ),
        },
        "trace": trace,
        "source_contract": {
            "system_validation_used_for_training": False,
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
