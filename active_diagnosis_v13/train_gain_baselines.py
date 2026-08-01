#!/usr/bin/env python3
"""Train group-disjoint linear and small-MLP gain classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from active_diagnosis_v13.contracts import require_frozen_branch_a_refinement

VERSION = "active_diagnosis_v13_gain_baselines_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(root: Path) -> list[dict[str, Any]]:
    result = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                result[row["record_id"]] = row
    return list(result.values())


def _model(
    name: str,
    seed: int,
    retained_feature_indices: list[int] | None = None,
    full_feature_count: int | None = None,
) -> Pipeline:
    if name == "linear":
        estimator = LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=seed
        )
    elif name == "small_mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=1200,
            random_state=seed,
        )
    else:
        raise ValueError(name)
    steps = []
    if retained_feature_indices is not None:
        if full_feature_count is None:
            raise ValueError("full feature count is required with retained indices")
        if len(retained_feature_indices) != full_feature_count:
            steps.append(
                (
                    "select",
                    ColumnTransformer(
                        [("retained", "passthrough", retained_feature_indices)],
                        remainder="drop",
                    ),
                )
            )
    steps.extend((("scale", StandardScaler()), ("classifier", estimator)))
    return Pipeline(steps)


def _bootstrap_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    seed: int,
    samples: int = 4000,
) -> dict[str, float]:
    unique = np.asarray(sorted(set(groups.tolist())))
    grouped = {
        group: np.flatnonzero(groups == group) for group in unique
    }
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = unique[rng.integers(0, len(unique), len(unique))]
        indices = np.concatenate([grouped[group] for group in chosen])
        estimates.append(float(np.mean(prediction[indices] == truth[indices])))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(np.mean(prediction == truth)),
        "low": float(low),
        "high": float(high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080103)
    parser.add_argument(
        "--retained-feature-model",
        type=Path,
        help="Optional audited bundle whose retained feature indices define the reduced view.",
    )
    args = parser.parse_args()
    gate_dir = args.gate_dir.resolve()
    selection = json.loads(
        (gate_dir / "probes" / "selected_probe.json").read_text()
    )
    rows = [
        row
        for row in _rows(gate_dir)
        if row["design"] == selection["selected_design"]
        and float(row["fraction"]) == float(selection["selected_fraction"])
    ]
    X_full = np.asarray(
        [row["policy_record"]["feature_vector"] for row in rows], dtype=np.float64
    )
    full_feature_names = list(rows[0]["policy_record"]["feature_names"])
    retained_indices = list(range(X_full.shape[1]))
    feature_filter = None
    if args.retained_feature_model is not None:
        retained_path = args.retained_feature_model.resolve()
        require_frozen_branch_a_refinement(gate_dir, retained_path)
        retained_bundle = joblib.load(retained_path)
        retained_indices = list(map(int, retained_bundle["retained_feature_indices"]))
        if not retained_indices or min(retained_indices) < 0 or max(retained_indices) >= X_full.shape[1]:
            raise ValueError("retained feature indices are empty or out of range")
        if len(set(retained_indices)) != len(retained_indices):
            raise ValueError("retained feature indices contain duplicates")
        if retained_bundle.get("selected_design") not in (None, selection["selected_design"]):
            raise ValueError("retained feature model design differs from selected probe")
        retained_fraction = retained_bundle.get("selected_fraction")
        if retained_fraction is not None and float(retained_fraction) != float(
            selection["selected_fraction"]
        ):
            raise ValueError("retained feature model fraction differs from selected probe")
        feature_filter = {
            "source_model": str(retained_path),
            "source_model_sha256": _sha256(retained_path),
            "ablation": retained_bundle.get("ablation"),
            "retained_feature_indices": retained_indices,
            "retained_feature_count": len(retained_indices),
        }
    X = X_full
    retained_feature_names = [full_feature_names[index] for index in retained_indices]
    y = np.asarray(
        [f"{float(row['evaluator_only_true_gain']):g}" for row in rows],
        dtype=np.str_,
    )
    groups = np.asarray([str(row["group_id"]) for row in rows], dtype=np.str_)
    labels = sorted(set(y.tolist()), key=float)
    splitter = GroupKFold(n_splits=5)
    reports = {}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, name in enumerate(("linear", "small_mlp")):
        predictions = np.empty_like(y)
        fold_models = []
        case_to_fold: dict[str, int] = {}
        for fold, (train, validation) in enumerate(splitter.split(X, y, groups)):
            model = _model(
                name,
                args.seed + offset * 100 + fold,
                retained_indices,
                X_full.shape[1],
            )
            model.fit(X[train], y[train])
            predictions[validation] = model.predict(X[validation])
            fold_models.append(model)
            for index in validation:
                case_to_fold[str(rows[index]["case_id"])] = fold
        full = _model(
            name,
            args.seed + offset * 1000,
            retained_indices,
            X_full.shape[1],
        )
        full.fit(X, y)
        model_path = output_dir / f"{name}.joblib"
        joblib.dump(
            {
                "version": VERSION,
                "model_name": name,
                "model": full,
                "full_model": full,
                "fold_models": fold_models,
                "case_to_fold": case_to_fold,
                "feature_names": full_feature_names,
                "retained_feature_names": retained_feature_names,
                "retained_feature_indices": retained_indices,
                "feature_filter": feature_filter,
                "gain_classes": labels,
                "selected_design": selection["selected_design"],
                "selected_fraction": selection["selected_fraction"],
            },
            model_path,
        )
        reports[name] = {
            "groups": len(set(groups.tolist())),
            "records": len(rows),
            "group_out_of_fold_accuracy_95": _bootstrap_accuracy(
                y, predictions, groups, args.seed + offset
            ),
            "macro_f1": float(f1_score(y, predictions, labels=labels, average="macro")),
            "per_gain_accuracy": {
                label: float(np.mean(predictions[y == label] == y[y == label]))
                for label in labels
            },
            "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
                y, predictions, labels=labels
            ).tolist(),
            "labels": labels,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
        }
    report = {
        "version": VERSION,
        "selection_split": "development_only",
        "protected_set_used": False,
        "selected_probe": selection,
        "features_identical_to_qwen_visible_numeric_features": True,
        "feature_filter": feature_filter,
        "full_feature_count": X_full.shape[1],
        "retained_feature_count": len(retained_indices),
        "models": reports,
    }
    temporary = output_dir / f"baseline_report.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "baseline_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
