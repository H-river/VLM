#!/usr/bin/env python3
"""Group-out-of-fold feature ablations for the selected development probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "active_diagnosis_v13_probe_feature_ablations_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(mask: np.ndarray, seed: int) -> Pipeline:
    indices = np.flatnonzero(mask).tolist()
    return Pipeline(
        [
            (
                "select",
                ColumnTransformer(
                    [("retained", "passthrough", indices)], remainder="drop"
                ),
            ),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=3000, random_state=seed
                ),
            ),
        ]
    )


def _rows(root: Path) -> list[dict[str, Any]]:
    by_id = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                by_id[row["record_id"]] = row
    return list(by_id.values())


def _bootstrap_accuracy(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray, seed: int
) -> dict[str, float]:
    unique = np.asarray(sorted(set(groups.tolist())))
    grouped = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(4000):
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
    parser.add_argument("--seed", type=int, default=2026080105)
    parser.add_argument("--design")
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    selection = json.loads(
        (root / "probes" / "selected_probe.json").read_text(encoding="utf-8")
    )
    design = str(args.design or selection["selected_design"])
    fraction = float(
        selection["selected_fraction"] if args.fraction is None else args.fraction
    )
    if (args.design is None) != (args.fraction is None):
        raise ValueError("--design and --fraction must be provided together")
    rows = [
        row
        for row in _rows(root)
        if str(row["design"]) == design and float(row["fraction"]) == fraction
    ]
    if len(rows) != 150:
        raise ValueError(f"expected 150 probe rows, found {len(rows)}")
    names = list(rows[0]["policy_record"]["feature_names"])
    X = np.asarray(
        [row["policy_record"]["feature_vector"] for row in rows], dtype=np.float64
    )
    y = np.asarray(
        [f"{float(row['evaluator_only_true_gain']):g}" for row in rows], dtype=np.str_
    )
    groups = np.asarray([str(row["group_id"]) for row in rows], dtype=np.str_)
    masks = {
        "full": np.ones(len(names), dtype=bool),
        "no_uncertainty": np.asarray(
            [
                "uncertainty_normalized_residual" not in name
                and "ensemble_uncertainty" not in name
                for name in names
            ]
        ),
        "no_residual_history": np.asarray(
            [
                "prediction_residual" not in name
                and "uncertainty_normalized_residual" not in name
                and "repeat_consistency" not in name
                for name in names
            ]
        ),
        "no_gain_belief": np.asarray(
            ["gain_ratio_projection" not in name for name in names]
        ),
        "no_candidate_boundary_features": np.ones(len(names), dtype=bool),
    }
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "selected_probe": {
            **selection,
            "selected_design": design,
            "selected_fraction": fraction,
            "analysis_override_from_preregistered_selection": bool(
                args.design is not None
            ),
        },
        "feature_count": len(names),
        "ablations": {},
    }
    splitter = GroupKFold(n_splits=5)
    for offset, (name, mask) in enumerate(masks.items()):
        removed = [feature for feature, keep in zip(names, mask, strict=True) if not keep]
        if name == "no_candidate_boundary_features" and not removed:
            report["ablations"][name] = {
                "status": "not_applicable",
                "reason": "selected probe representation contains no candidate-boundary features",
                "removed_features": [],
            }
            continue
        prediction = np.empty_like(y)
        fold_models = []
        case_to_fold: dict[str, int] = {}
        for fold, (train, validation) in enumerate(splitter.split(X, y, groups)):
            model = _model(mask, args.seed + offset * 100 + fold)
            model.fit(X[train], y[train])
            prediction[validation] = model.predict(X[validation])
            fold_models.append(model)
            for index in validation:
                case_to_fold[str(rows[index]["case_id"])] = fold
        full_model = _model(mask, args.seed + offset * 1000).fit(X, y)
        model_dir = args.output_dir.resolve() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(
            {
                "version": VERSION,
                "ablation": name,
                "feature_names": names,
                "retained_feature_indices": np.flatnonzero(mask).tolist(),
                "fold_models": fold_models,
                "case_to_fold": case_to_fold,
                "full_model": full_model,
            },
            model_path,
        )
        report["ablations"][name] = {
            "status": "completed",
            "retained_feature_count": int(mask.sum()),
            "removed_feature_count": int((~mask).sum()),
            "removed_features": removed,
            "group_out_of_fold_accuracy_95": _bootstrap_accuracy(
                y, prediction, groups, args.seed + offset
            ),
            "macro_f1": float(f1_score(y, prediction, average="macro")),
            "per_gain_accuracy": {
                label: float(np.mean(prediction[y == label] == y[y == label]))
                for label in sorted(set(y.tolist()), key=float)
            },
            "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
                y, prediction, labels=sorted(set(y.tolist()), key=float)
            ).tolist(),
            "labels": sorted(set(y.tolist()), key=float),
            "classifier_bundle": str(model_path),
            "classifier_bundle_sha256": _sha256(model_path),
        }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "probe_feature_ablation_report.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
