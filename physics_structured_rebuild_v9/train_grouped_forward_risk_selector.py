#!/usr/bin/env python3
"""Train a protected failure-risk selector for two forward surfaces."""

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
)
from physics_structured_rebuild_v9.calibrate_system_direction import read_jsonl
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_risk_selector_runtime import (
    grouped_risk_features,
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
DEFAULT_OUTPUT = DEFAULT_RUN / "grouped_forward_risk_selector_v9.pkl"
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
        current_values[-1] = np.expm1(current_values[-1])
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


def score_arrays(
    models: dict[str, Any],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    candidate_probability = models[
        "candidate_success_classifier"
    ].predict_proba(flat)[:, 1]
    current_probability = models[
        "current_success_classifier"
    ].predict_proba(flat)[:, 1]
    probability_advantage = candidate_probability - current_probability
    error_advantage = (
        models["current_error_regressor"].predict(flat)
        - models["candidate_error_regressor"].predict(flat)
    )
    return {
        "probability_advantage": probability_advantage.astype(np.float32),
        "error_advantage": error_advantage.astype(np.float32),
    }


def candidate_thresholds(score: np.ndarray) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, 801)
    return np.unique(
        np.concatenate(
            [
                np.asarray([np.inf], dtype=np.float64),
                np.quantile(score.astype(np.float64), quantiles),
            ]
        )
    )


def threshold_search(
    score: np.ndarray,
    current_success: np.ndarray,
    candidate_success: np.ndarray,
    protected: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any] | None:
    best = None
    best_report = None
    for threshold in candidate_thresholds(score):
        choose = score >= threshold
        selected = np.where(choose, candidate_success, current_success)
        protected_reports = []
        valid = True
        for block_score, block_current, block_candidate, high in protected:
            block_choose = block_score >= threshold
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
            int((selected & ~current_success).sum()),
            -int((~selected & current_success).sum()),
            -int(choose.sum()),
            float(threshold),
        )
        if best is None or key > best:
            best = key
            best_report = {
                "threshold": float(threshold),
                "current_success_count": int(current_success.sum()),
                "candidate_success_count": int(candidate_success.sum()),
                "oracle_union_success_count": int(
                    (current_success | candidate_success).sum()
                ),
                "selected_success_count": int(selected.sum()),
                "candidate_count": int(choose.sum()),
                "protected": protected_reports,
            }
    return best_report


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    started = time.perf_counter()
    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        contexts = np.asarray(cache["contexts"][:, :17], dtype=np.float32)
        truth_states = np.asarray(cache["true_candidate_states"], dtype=np.float32)
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
    features = grouped_risk_features(rows, current_prediction, candidate_prediction)
    current_abs_error = np.abs(current_prediction - target)
    candidate_abs_error = np.abs(candidate_prediction - target)
    current_error = current_abs_error.max(axis=2)
    candidate_error = candidate_abs_error.max(axis=2)
    current_success = current_error <= 1.0
    candidate_success = candidate_error <= 1.0
    training_groups, calibration_groups = group_split(group_ids, int(args.seed))
    training_features = features[training_groups].reshape(-1, features.shape[-1])
    current_training_success = current_success[training_groups].reshape(-1)
    candidate_training_success = candidate_success[training_groups].reshape(-1)
    current_training_error = np.log1p(
        current_error[training_groups].reshape(-1)
    )
    candidate_training_error = np.log1p(
        candidate_error[training_groups].reshape(-1)
    )

    classifier_parameters = {
        "learning_rate": 0.045,
        "max_iter": 180,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 2.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": int(args.seed),
    }
    regressor_parameters = dict(classifier_parameters)
    models = {
        "current_success_classifier": HistGradientBoostingClassifier(
            **classifier_parameters
        ),
        "candidate_success_classifier": HistGradientBoostingClassifier(
            **{**classifier_parameters, "random_state": int(args.seed) + 1}
        ),
        "current_error_regressor": HistGradientBoostingRegressor(
            loss="squared_error",
            **{**regressor_parameters, "random_state": int(args.seed) + 2},
        ),
        "candidate_error_regressor": HistGradientBoostingRegressor(
            loss="squared_error",
            **{**regressor_parameters, "random_state": int(args.seed) + 3},
        ),
    }
    training_targets = {
        "current_success_classifier": current_training_success,
        "candidate_success_classifier": candidate_training_success,
        "current_error_regressor": current_training_error,
        "candidate_error_regressor": candidate_training_error,
    }
    for name, model in models.items():
        model.fit(training_features, training_targets[name])
        print(json.dumps({"trained": name}, sort_keys=True), flush=True)

    training_scores = score_arrays(models, features[training_groups])
    probability_scale = {
        "mean": float(training_scores["probability_advantage"].mean()),
        "std": float(
            max(training_scores["probability_advantage"].std(), 1e-6)
        ),
    }
    error_scale = {
        "mean": float(training_scores["error_advantage"].mean()),
        "std": float(max(training_scores["error_advantage"].std(), 1e-6)),
    }

    def all_scores(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = dict(raw)
        probability_z = (
            raw["probability_advantage"] - probability_scale["mean"]
        ) / probability_scale["std"]
        error_z = (
            raw["error_advantage"] - error_scale["mean"]
        ) / error_scale["std"]
        for weight in (0.25, 0.5, 1.0, 2.0):
            values[f"hybrid_{weight:g}"] = (
                probability_z + weight * error_z
            ).astype(np.float32)
        return values

    calibration_scores = all_scores(
        score_arrays(models, features[calibration_groups])
    )
    calibration_current = current_success[calibration_groups].reshape(-1)
    calibration_candidate = candidate_success[calibration_groups].reshape(-1)
    protected_raw = []
    protected_names = []
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        block_rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_current_prediction = current_runtime.predict_changes(block_rows)
        block_candidate_prediction = candidate_runtime.predict_changes(block_rows)
        block_features = grouped_risk_features(
            block_rows,
            block_current_prediction,
            block_candidate_prediction,
        )
        block_target = arrays.normalized_changes
        block_current = np.all(
            np.abs(block_current_prediction.reshape(-1, 5) - block_target)
            <= 1.0,
            axis=1,
        )
        block_candidate = np.all(
            np.abs(block_candidate_prediction.reshape(-1, 5) - block_target)
            <= 1.0,
            axis=1,
        )
        protected_raw.append(
            (
                all_scores(score_arrays(models, block_features)),
                block_current,
                block_candidate,
                action_high_mask(len(block_rows)),
            )
        )
        protected_names.append(name)

    searches = {}
    for score_name, score in calibration_scores.items():
        protected = [
            (raw[score_name], current, candidate, high)
            for raw, current, candidate, high in protected_raw
        ]
        result = threshold_search(
            score,
            calibration_current,
            calibration_candidate,
            protected,
        )
        if result is not None:
            result["protected"] = {
                name: value
                for name, value in zip(
                    protected_names,
                    result["protected"],
                    strict=True,
                )
            }
            searches[score_name] = result
    if not searches:
        raise RuntimeError("no protected risk selector threshold exists")
    score_kind, selected = max(
        searches.items(),
        key=lambda item: (
            item[1]["selected_success_count"],
            -item[1]["candidate_count"],
            item[0] == "probability_advantage",
        ),
    )
    hybrid_weight = (
        float(score_kind.split("_", 1)[1])
        if score_kind.startswith("hybrid_")
        else 0.0
    )
    artifact = {
        "version": "grouped_forward_risk_selector_v9_one_seed",
        "model": "hgb_grouped_forward_risk_selector_v9",
        "seed": int(args.seed),
        **models,
        "score_kind": score_kind,
        "hybrid_weight": hybrid_weight,
        "threshold": float(selected["threshold"]),
        "probability_scale": probability_scale,
        "error_scale": error_scale,
        "current_forward": str(args.current_forward.resolve()),
        "current_forward_sha256": sha256(args.current_forward.resolve()),
        "candidate_forward": str(args.candidate_forward.resolve()),
        "candidate_forward_sha256": sha256(args.candidate_forward.resolve()),
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
            "transition_count": int(len(training_features)),
            "current_success_count": int(current_training_success.sum()),
            "candidate_success_count": int(candidate_training_success.sum()),
        },
        "calibration": {
            "group_count": int(len(calibration_groups)),
            "score_kind": score_kind,
            **selected,
        },
        "score_searches": searches,
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
