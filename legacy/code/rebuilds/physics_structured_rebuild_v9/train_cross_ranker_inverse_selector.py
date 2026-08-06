#!/usr/bin/env python3
"""Train a leakage-free selector over base-primary and adapted-secondary inverse calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.inverse_runtime_v8 import load_inverse_runtime_v8

DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
DEFAULT_BASE = (
    REPO_ROOT.parent
    / "VLM_runs/tabm_transformer_rebuild_v8_one_seed/transformer/inverse.pt"
)
DEFAULT_ADAPTED = DEFAULT_RUN / "qwen_inverse_ranker_adaptation/inverse.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=DEFAULT_RUN / "qwen_inverse_forward_cache.npz",
    )
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapted-artifact", type=Path, default=DEFAULT_ADAPTED)
    parser.add_argument(
        "--base-system-cache",
        type=Path,
        default=DEFAULT_RUN / "inverse_enumerator_ensemble_search.npz",
    )
    parser.add_argument(
        "--adapted-system-cache",
        type=Path,
        default=DEFAULT_RUN / "qwen_adapted_inverse_system_state_features.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "cross_ranker_inverse_selector_v9.pkl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_RUN / "cross_ranker_inverse_selector_validation.json",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_values(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return matrix[np.arange(len(indices)), indices]


def margins(scores: np.ndarray) -> np.ndarray:
    ordered = np.partition(scores, kth=-2, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def action_digits(indices: np.ndarray) -> np.ndarray:
    """Decode the fixed 3^4 action-grid index into four ternary coordinates."""
    remaining = np.asarray(indices, dtype=np.int64).copy()
    digits = np.empty((len(remaining), 4), dtype=np.float32)
    for column, divisor in enumerate((27, 9, 3, 1)):
        digits[:, column] = remaining // divisor
        remaining %= divisor
    return digits


FEATURE_NAMES = (
    "base_selected_score_advantage",
    "base_selected_base_cost_advantage",
    "base_score_margin_advantage",
    "adapted_selected_score_advantage",
    "adapted_selected_base_cost_advantage",
    "adapted_score_margin_advantage",
    "primary_action_0",
    "primary_action_1",
    "primary_action_2",
    "primary_action_3",
    "secondary_action_0",
    "secondary_action_1",
    "secondary_action_2",
    "secondary_action_3",
    "action_distance_0",
    "action_distance_1",
    "action_distance_2",
    "action_distance_3",
)


def selector_features(
    base_primary: dict[str, Any],
    base_secondary: dict[str, Any],
    adapted_primary: dict[str, Any],
    adapted_secondary: dict[str, Any],
) -> np.ndarray:
    primary_index = np.asarray(base_primary["selected_indices"], dtype=np.int64)
    secondary_index = np.asarray(
        adapted_secondary["selected_indices"],
        dtype=np.int64,
    )
    primary_digits = action_digits(primary_index)
    secondary_digits = action_digits(secondary_index)
    diagnostics = []
    for primary, secondary in (
        (base_primary, base_secondary),
        (adapted_primary, adapted_secondary),
    ):
        p_index = np.asarray(primary["selected_indices"], dtype=np.int64)
        s_index = np.asarray(secondary["selected_indices"], dtype=np.int64)
        p_scores = np.asarray(primary["scores"], dtype=np.float32)
        s_scores = np.asarray(secondary["scores"], dtype=np.float32)
        p_costs = np.asarray(primary["base_costs"], dtype=np.float32)
        s_costs = np.asarray(secondary["base_costs"], dtype=np.float32)
        diagnostics.extend(
            [
                selected_values(s_scores, s_index)
                - selected_values(p_scores, p_index),
                selected_values(p_costs, p_index)
                - selected_values(s_costs, s_index),
                margins(s_scores) - margins(p_scores),
            ]
        )
    return np.column_stack(
        [
            *diagnostics,
            primary_digits,
            secondary_digits,
            np.abs(primary_digits - secondary_digits),
        ]
    ).astype(np.float32)


def cached_system_features(base: Any, adapted: Any) -> np.ndarray:
    primary_index = np.asarray(base["primary_index"], dtype=np.int64)
    secondary_index = np.asarray(adapted["secondary_index"], dtype=np.int64)
    primary_digits = action_digits(primary_index)
    secondary_digits = action_digits(secondary_index)
    diagnostics = np.column_stack(
        [
            np.asarray(base["feature_selected_score_advantage"]),
            np.asarray(base["feature_selected_base_cost_advantage"]),
            np.asarray(base["feature_score_margin_advantage"]),
            np.asarray(adapted["feature_selected_score_advantage"]),
            np.asarray(adapted["feature_selected_base_cost_advantage"]),
            np.asarray(adapted["feature_score_margin_advantage"]),
        ]
    )
    return np.column_stack(
        [
            diagnostics,
            primary_digits,
            secondary_digits,
            np.abs(primary_digits - secondary_digits),
        ]
    ).astype(np.float32)


def model_candidates(seed: int) -> dict[str, Any]:
    return {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=5,
            min_samples_leaf=15,
            learning_rate=0.05,
            l2_regularization=2.0,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output, report_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite output: {path}")

    cache_path = args.training_cache.resolve()
    arrays = np.load(cache_path, allow_pickle=False)
    contexts = np.asarray(arrays["contexts"], dtype=np.float32)
    desired = np.asarray(arrays["desired"], dtype=np.float32)
    positives = np.asarray(arrays["positives"], dtype=np.bool_)
    primary_states = np.asarray(arrays["primary_states"], dtype=np.float32)
    secondary_states = np.asarray(arrays["secondary_states"], dtype=np.float32)

    torch, device = configure(int(args.seed), args.device)
    base, _ = load_inverse_runtime_v8(args.base_artifact.resolve(), torch, device)
    adapted, _ = load_inverse_runtime_v8(
        args.adapted_artifact.resolve(),
        torch,
        device,
    )
    base_primary = base.score_feature_arrays(contexts, desired, primary_states)
    base_secondary = base.score_feature_arrays(contexts, desired, secondary_states)
    adapted_primary = adapted.score_feature_arrays(
        contexts,
        desired,
        primary_states,
    )
    adapted_secondary = adapted.score_feature_arrays(
        contexts,
        desired,
        secondary_states,
    )
    primary_index = np.asarray(
        base_primary["selected_indices"],
        dtype=np.int64,
    )
    secondary_index = np.asarray(
        adapted_secondary["selected_indices"],
        dtype=np.int64,
    )
    features = selector_features(
        base_primary,
        base_secondary,
        adapted_primary,
        adapted_secondary,
    )
    row = np.arange(len(features))
    primary_success = positives[row, primary_index]
    secondary_success = positives[row, secondary_index]
    exclusive = primary_success ^ secondary_success
    exclusive_features = features[exclusive]
    exclusive_labels = secondary_success[exclusive].astype(np.int8)
    print(
        json.dumps(
            {
                "natural_primary_success_count": int(primary_success.sum()),
                "natural_secondary_success_count": int(secondary_success.sum()),
                "natural_exclusive_count": int(exclusive.sum()),
                "natural_primary_only_count": int(
                    np.sum(primary_success & ~secondary_success)
                ),
                "natural_secondary_only_count": int(
                    np.sum(~primary_success & secondary_success)
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if len(exclusive_labels) < 30 or len(np.unique(exclusive_labels)) != 2:
        raise ValueError("insufficient exclusive natural training cases")

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(args.seed),
    )
    candidates = []
    for name, model in model_candidates(int(args.seed)).items():
        oof = cross_val_predict(
            model,
            exclusive_features,
            exclusive_labels,
            cv=folds,
            method="predict",
            n_jobs=1,
        ).astype(np.int8)
        correct = oof == exclusive_labels
        selected_success = int(
            np.sum(primary_success & secondary_success) + correct.sum()
        )
        candidates.append(
            {
                "name": name,
                "oof_exclusive_accuracy": float(correct.mean()),
                "oof_selected_success_count": selected_success,
                "oof_selected_success_rate": selected_success / len(features),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["oof_selected_success_count"],
            item["oof_exclusive_accuracy"],
            item["name"],
        ),
        reverse=True,
    )
    selected_name = str(candidates[0]["name"])
    classifier = clone(model_candidates(int(args.seed))[selected_name])
    classifier.fit(exclusive_features, exclusive_labels)

    base_system_path = args.base_system_cache.resolve()
    adapted_system_path = args.adapted_system_cache.resolve()
    base_system = np.load(base_system_path, allow_pickle=False)
    adapted_system = np.load(adapted_system_path, allow_pickle=False)
    system_features = cached_system_features(base_system, adapted_system)
    choose_secondary = classifier.predict(system_features).astype(np.int8) == 1
    system_primary_success = np.asarray(
        base_system["primary_success"],
        dtype=np.bool_,
    )
    system_secondary_success = np.asarray(
        adapted_system["secondary_success"],
        dtype=np.bool_,
    )
    selected_system_success = np.where(
        choose_secondary,
        system_secondary_success,
        system_primary_success,
    )

    base_path = args.base_artifact.resolve()
    adapted_path = args.adapted_artifact.resolve()
    artifact = {
        "version": "cross_ranker_inverse_selector_v9_one_seed",
        "model": selected_name,
        "classifier": classifier,
        "feature_names": FEATURE_NAMES,
        "base_inverse_artifact": str(base_path),
        "base_inverse_artifact_sha256": sha256(base_path),
        "adapted_inverse_artifact": str(adapted_path),
        "adapted_inverse_artifact_sha256": sha256(adapted_path),
        "training": {
            "count": int(len(features)),
            "primary_success_count": int(primary_success.sum()),
            "secondary_success_count": int(secondary_success.sum()),
            "exclusive_count": int(exclusive.sum()),
            "primary_only_count": int(np.sum(primary_success & ~secondary_success)),
            "secondary_only_count": int(np.sum(~primary_success & secondary_success)),
            "candidates": candidates,
            "selected_model": selected_name,
        },
        "source_contract": {
            "training_cache": str(cache_path),
            "training_cache_sha256": sha256(cache_path),
            "system_validation_used_for_model_selection": False,
            "held_out_test_used": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "version": "cross_ranker_inverse_selector_validation_v9_one_seed",
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "training": artifact["training"],
        "system_state_inverse": {
            "count": int(len(selected_system_success)),
            "base_primary_success_count": int(system_primary_success.sum()),
            "adapted_secondary_success_count": int(system_secondary_success.sum()),
            "oracle_union_success_count": int(
                np.sum(system_primary_success | system_secondary_success)
            ),
            "selected_success_count": int(selected_system_success.sum()),
            "selected_success_rate": float(selected_system_success.mean()),
            "secondary_count": int(choose_secondary.sum()),
        },
        "source_contract": {
            **artifact["source_contract"],
            "base_system_cache": str(base_system_path),
            "adapted_system_cache": str(adapted_system_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
