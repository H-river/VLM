#!/usr/bin/env python3
"""Train a system-validation-free selector over two calibrated forward experts."""

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

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_TRAINING = DEFAULT_RUN / "qwen_combined_forward_training_features.npz"
DEFAULT_CURRENT = DEFAULT_RUN / "forward_natural_grid_extra_calibrated"
DEFAULT_ALTERNATIVE = DEFAULT_RUN / "forward_novel_natural_extra_calibrated"
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "forward_natural_expert_selector_v9.pkl"
ROUTES = {
    "state": "predict_forward_from_state_v1",
    "image": "predict_forward_from_image_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument(
        "--alternative-dir",
        type=Path,
        default=DEFAULT_ALTERNATIVE,
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
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


def model_predictions(models: list[Any], features: np.ndarray) -> np.ndarray:
    return np.stack(
        [model.predict(features) for model in models],
        axis=1,
    ).astype(np.float32)


def calibrated_prediction(
    artifact_path: Path,
    features: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    with artifact_path.open("rb") as stream:
        artifact = pickle.load(stream)
    with Path(str(artifact["tree_artifact"])).open("rb") as stream:
        primary = pickle.load(stream)
    primary_residual = model_predictions(list(primary["models"]), features)
    secondary_residual = None
    if artifact.get("secondary_tree_artifact") is not None:
        with Path(str(artifact["secondary_tree_artifact"])).open(
            "rb"
        ) as stream:
            secondary = pickle.load(stream)
        secondary_residual = model_predictions(
            list(secondary["models"]),
            features,
        )
    prediction = np.empty_like(prior)
    declared_sources = artifact.get("field_tree_source")
    for field_index, field in enumerate(STATE_FIELDS):
        source = (
            "primary"
            if declared_sources is None
            else str(declared_sources[field])
        )
        residual = (
            primary_residual[:, field_index]
            if source == "primary"
            else secondary_residual[:, field_index]
        )
        prediction[:, field_index] = (
            prior[:, field_index]
            + float(artifact["field_blend"][field]) * residual
        )
    return prediction


def selector_features(
    prior: np.ndarray,
    current: np.ndarray,
    alternative: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            prior,
            current,
            alternative,
            np.abs(current - alternative),
        ],
        axis=1,
    ).astype(np.float32)


def normalized_success(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    return np.all(np.abs(prediction - target) <= 1.0, axis=1)


def raw_success(
    prediction: np.ndarray,
    input_tolerance: np.ndarray,
    truth: np.ndarray,
    scoring_tolerance: np.ndarray,
) -> np.ndarray:
    return np.all(
        np.abs(prediction * input_tolerance - truth) <= scoring_tolerance,
        axis=1,
    )


def split_groups(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(
        [
            int(
                hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[
                    :16
                ],
                16,
            )
            for group_id in group_ids
        ],
        dtype=np.uint64,
    )
    order = np.argsort(keys)
    calibration_groups = order[:400]
    training_groups = order[400:]
    return training_groups, calibration_groups


def row_indices(groups: np.ndarray) -> np.ndarray:
    return (
        groups[:, None] * 81 + np.arange(81, dtype=np.int64)[None, :]
    ).reshape(-1)


def action_high_mask(group_count: int) -> np.ndarray:
    per_action = np.asarray(
        [
            sum(
                abs(float(action[field])) > 0.0
                for field in ACTION_FIELDS
            )
            >= 3
            for action in ACTION_GRID
        ],
        dtype=np.bool_,
    )
    return np.tile(per_action, group_count)


def choose_threshold(
    probability: np.ndarray,
    current_success: np.ndarray,
    alternative_success: np.ndarray,
) -> dict[str, Any]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0, np.inf]),
                np.quantile(probability, np.linspace(0.0, 1.0, 201)),
            ]
        )
    )
    best = None
    for threshold in thresholds:
        choose_alternative = probability >= float(threshold)
        success = np.where(
            choose_alternative,
            alternative_success,
            current_success,
        )
        candidate = {
            "threshold": float(threshold),
            "success_count": int(success.sum()),
            "alternative_count": int(choose_alternative.sum()),
        }
        key = (candidate["success_count"], -candidate["alternative_count"])
        if best is None or key > best[0]:
            best = (key, candidate)
    assert best is not None
    return best[1]


def protected_counts(
    success: np.ndarray,
    high: np.ndarray,
) -> list[int]:
    return [
        int(success.sum()),
        int(success[high].sum()),
    ]


def system_predictions(
    candidate_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cache_path = candidate_dir / "forward_system_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {key: np.asarray(cache[key]) for key in cache.files}
    predictions = {}
    for route_key, route_name in ROUTES.items():
        artifact_path = candidate_dir / f"forward_{route_key}.pkl"
        with artifact_path.open("rb") as stream:
            artifact = pickle.load(stream)
        prediction = np.empty_like(arrays["prior"])
        mask = arrays["routes"] == route_name
        for field_index, field in enumerate(STATE_FIELDS):
            source = str(artifact["field_tree_source"][field])
            residual = (
                arrays["residual"][:, field_index]
                if source == "primary"
                else arrays["secondary_residual"][:, field_index]
            )
            prediction[mask, field_index] = (
                arrays["prior"][mask, field_index]
                + float(artifact["field_blend"][field]) * residual[mask]
            )
        predictions[route_key] = prediction
    return predictions, arrays


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    training_path = args.training_cache.resolve()
    with np.load(training_path, allow_pickle=False) as cache:
        features = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        target = np.asarray(cache["grid_target_normalized"], dtype=np.float32)
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    training_groups, calibration_groups = split_groups(
        group_ids,
        int(args.seed),
    )
    train_indices = row_indices(training_groups)
    calibration_indices = row_indices(calibration_groups)

    current_dir = args.current_dir.resolve()
    alternative_dir = args.alternative_dir.resolve()
    route_models = {}
    route_reports = {}
    for route_index, route_key in enumerate(ROUTES):
        current_prediction = calibrated_prediction(
            current_dir / f"forward_{route_key}.pkl",
            features,
            prior,
        )
        alternative_prediction = calibrated_prediction(
            alternative_dir / f"forward_{route_key}.pkl",
            features,
            prior,
        )
        current_success = normalized_success(current_prediction, target)
        alternative_success = normalized_success(
            alternative_prediction,
            target,
        )
        disagreement = current_success ^ alternative_success
        selected_train = train_indices[disagreement[train_indices]]
        labels = alternative_success[selected_train].astype(np.int64)
        if len(np.unique(labels)) != 2:
            raise ValueError("selector training lacks both exclusive classes")
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        sample_weight = (len(labels) / (2.0 * counts))[labels]
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=160,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=int(args.seed) + route_index,
        )
        all_selector_features = selector_features(
            prior,
            current_prediction,
            alternative_prediction,
        )
        classifier.fit(
            all_selector_features[selected_train],
            labels,
            sample_weight=sample_weight,
        )
        probability = classifier.predict_proba(
            all_selector_features[calibration_indices]
        )[:, 1]
        threshold = choose_threshold(
            probability,
            current_success[calibration_indices],
            alternative_success[calibration_indices],
        )
        route_models[route_key] = {
            "classifier": classifier,
            "threshold": float(threshold["threshold"]),
        }
        route_reports[route_key] = {
            "exclusive_training_count": int(len(selected_train)),
            "exclusive_training_class_count": {
                "current_only": int(counts[0]),
                "alternative_only": int(counts[1]),
            },
            "calibration_current_success_count": int(
                current_success[calibration_indices].sum()
            ),
            "calibration_alternative_success_count": int(
                alternative_success[calibration_indices].sum()
            ),
            "calibration_selected": threshold,
        }
        print(
            json.dumps(
                {"route": route_key, **route_reports[route_key]},
                sort_keys=True,
            ),
            flush=True,
        )

    torch, device = configure(int(args.seed), args.device)
    forward_path = args.forward_artifact.resolve()
    forward, _ = load_forward_direction_runtime_v7(
        forward_path,
        torch,
        device,
    )
    protected = {}
    for block, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_prior = stream_forward_changes(path, forward)
        block_features = np.concatenate(
            [arrays.features, block_prior],
            axis=1,
        ).astype(np.float32)
        high = action_high_mask(arrays.group_count)
        protected[block] = {}
        for route_key in ROUTES:
            current_prediction = calibrated_prediction(
                current_dir / f"forward_{route_key}.pkl",
                block_features,
                block_prior,
            )
            alternative_prediction = calibrated_prediction(
                alternative_dir / f"forward_{route_key}.pkl",
                block_features,
                block_prior,
            )
            current_success = normalized_success(
                current_prediction,
                arrays.normalized_changes,
            )
            alternative_success = normalized_success(
                alternative_prediction,
                arrays.normalized_changes,
            )
            selector_input = selector_features(
                block_prior,
                current_prediction,
                alternative_prediction,
            )
            probability = route_models[route_key][
                "classifier"
            ].predict_proba(selector_input)[:, 1]
            choose_alternative = probability >= route_models[route_key][
                "threshold"
            ]
            selected_success = np.where(
                choose_alternative,
                alternative_success,
                current_success,
            )
            protected[block][route_key] = {
                "current": protected_counts(current_success, high),
                "alternative": protected_counts(alternative_success, high),
                "selected": protected_counts(selected_success, high),
                "alternative_count": int(choose_alternative.sum()),
            }

    current_system, current_arrays = system_predictions(current_dir)
    alternative_system, alternative_arrays = system_predictions(
        alternative_dir
    )
    for key in (
        "prior",
        "truth_change",
        "input_tolerance",
        "scoring_tolerance",
        "routes",
    ):
        if not np.array_equal(current_arrays[key], alternative_arrays[key]):
            raise ValueError(f"system candidate caches differ for {key}")
    system = {}
    selected_success_parts = []
    protected_non_regression = True
    for route_key, route_name in ROUTES.items():
        mask = current_arrays["routes"] == route_name
        current_prediction = current_system[route_key][mask]
        alternative_prediction = alternative_system[route_key][mask]
        current_success = raw_success(
            current_prediction,
            current_arrays["input_tolerance"][mask],
            current_arrays["truth_change"][mask],
            current_arrays["scoring_tolerance"][mask],
        )
        alternative_success = raw_success(
            alternative_prediction,
            current_arrays["input_tolerance"][mask],
            current_arrays["truth_change"][mask],
            current_arrays["scoring_tolerance"][mask],
        )
        selector_input = selector_features(
            current_arrays["prior"][mask],
            current_prediction,
            alternative_prediction,
        )
        probability = route_models[route_key][
            "classifier"
        ].predict_proba(selector_input)[:, 1]
        choose_alternative = probability >= route_models[route_key][
            "threshold"
        ]
        selected_success = np.where(
            choose_alternative,
            alternative_success,
            current_success,
        )
        selected_success_parts.append(selected_success)
        system[route_key] = {
            "count": int(mask.sum()),
            "current_success_count": int(current_success.sum()),
            "alternative_success_count": int(alternative_success.sum()),
            "selected_success_count": int(selected_success.sum()),
            "oracle_union_success_count": int(
                (current_success | alternative_success).sum()
            ),
            "alternative_count": int(choose_alternative.sum()),
        }
        for block in protected.values():
            values = block[route_key]
            protected_non_regression &= all(
                selected >= current
                for selected, current in zip(
                    values["selected"],
                    values["current"],
                    strict=True,
                )
            )

    artifact = {
        "version": "forward_natural_expert_selector_v9_one_seed",
        "model": "route_specific_hgb_forward_expert_selector_v9",
        "seed": int(args.seed),
        "feature_mode": (
            "v7_prior_5_current_prediction_5_alternative_prediction_5"
            "_absolute_disagreement_5"
        ),
        "routes": route_models,
        "current_dir": str(current_dir),
        "alternative_dir": str(alternative_dir),
        "current_state_sha256": sha256(current_dir / "forward_state.pkl"),
        "current_image_sha256": sha256(current_dir / "forward_image.pkl"),
        "alternative_state_sha256": sha256(
            alternative_dir / "forward_state.pkl"
        ),
        "alternative_image_sha256": sha256(
            alternative_dir / "forward_image.pkl"
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
            "cache": str(training_path),
            "cache_sha256": sha256(training_path),
            "training_group_count": int(len(training_groups)),
            "calibration_group_count": int(len(calibration_groups)),
            "routes": route_reports,
        },
        "protected_validation": protected,
        "protected_non_regression": bool(protected_non_regression),
        "system_validation": {
            "routes": system,
            "selected_success_count": int(
                sum(values.sum() for values in selected_success_parts)
            ),
            "selected_success_rate": float(
                sum(values.sum() for values in selected_success_parts) / 300
            ),
        },
        "source_contract": {
            "natural_training_cache": str(training_path),
            "old_validation": str(args.old_validation.resolve()),
            "difficult_validation": str(args.difficult_validation.resolve()),
            "system_validation_caches": [
                str(current_dir / "forward_system_cache.npz"),
                str(alternative_dir / "forward_system_cache.npz"),
            ],
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
