#!/usr/bin/env python3
"""Calibrate boundary-only tree corrections over frozen v7 forward changes."""

from __future__ import annotations

import argparse
import hashlib
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

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from control_rebuild_v5.forward_runtime import ZERO_ACTION_INDEX
from direction_rebuild_v4.data import (
    labels_from_normalized_change,
    load_grid_arrays,
)
from joint_forward_direction_v6.train import geometric_score
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from joint_forward_direction_v7.train import direction_metric_bundle

DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_TREE = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed/direction_tree_v4.pkl"
)
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed/hybrid_direction"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--tree-artifact", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def tree_probabilities(
    models: list[Any],
    features: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(features), 5, 3), dtype=np.float32)
    for field, model in enumerate(models):
        probability = model.predict_proba(features)
        output[:, field, np.asarray(model.classes_, dtype=np.int64)] = (
            probability
        )
    return output


def selection_values(
    old_metrics: dict[str, Any],
    difficult_metrics: dict[str, Any],
) -> list[float]:
    return [
        old_metrics["overall"]["joint_exact"],
        difficult_metrics["overall"]["joint_exact"],
        old_metrics["by_action_complexity"]["three_or_four"]["joint_exact"],
        difficult_metrics["by_action_complexity"]["three_or_four"][
            "joint_exact"
        ],
    ]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "hybrid_direction_summary.json"
    artifact_path = output_dir / "hybrid_direction.pkl"
    if summary_path.exists() or artifact_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(int(args.seed), args.device)
    runtime, _ = load_forward_direction_runtime_v7(
        args.forward_artifact.resolve(),
        torch,
        device,
    )
    old_rows = read_jsonl(args.old_validation.resolve())
    difficult_rows = read_jsonl(args.difficult_validation.resolve())
    old_arrays = load_grid_arrays(
        args.old_validation.resolve(),
        include_legacy_features=False,
    )
    difficult_arrays = load_grid_arrays(
        args.difficult_validation.resolve(),
        include_legacy_features=False,
    )
    old_changes = runtime.predict_changes(old_rows).reshape(-1, 5)
    difficult_changes = runtime.predict_changes(difficult_rows).reshape(-1, 5)
    with args.tree_artifact.resolve().open("rb") as stream:
        tree_artifact = pickle.load(stream)
    if (
        tree_artifact.get("model")
        != "balanced_five_head_hist_gradient_boosting_direction_v4"
    ):
        raise ValueError("unexpected direction-tree artifact")
    models = list(tree_artifact["models"])
    old_tree = tree_probabilities(models, old_arrays.features)
    difficult_tree = tree_probabilities(models, difficult_arrays.features)
    rows = []
    for boundary_limit in (0.0, 0.25, 0.50, 0.75, 1.0, 2.0):
        for confidence_limit in (0.50, 0.60, 0.70, 0.80, 0.90):
            for margin_limit in (0.0, 0.10, 0.20, 0.30, 0.40):
                metrics = {}
                apply_counts = {}
                for (
                    name,
                    arrays,
                    changes,
                    tree_probability,
                ) in (
                    (
                        "old_iid",
                        old_arrays,
                        old_changes,
                        old_tree,
                    ),
                    (
                        "difficult",
                        difficult_arrays,
                        difficult_changes,
                        difficult_tree,
                    ),
                ):
                    threshold = labels_from_normalized_change(changes)
                    ordered = np.sort(tree_probability, axis=-1)
                    tree_class = tree_probability.argmax(axis=-1)
                    distance = np.abs(np.abs(changes) - 1.0)
                    apply = (
                        (distance <= float(boundary_limit))
                        & (
                            ordered[..., -1]
                            >= float(confidence_limit)
                        )
                        & (
                            ordered[..., -1] - ordered[..., -2]
                            >= float(margin_limit)
                        )
                        & (tree_class != threshold)
                    )
                    predicted = np.where(
                        apply,
                        tree_class,
                        threshold,
                    ).astype(np.int64)
                    zero = (
                        np.arange(len(predicted)) % len(ACTION_GRID)
                        == ZERO_ACTION_INDEX
                    )
                    predicted[zero] = 1
                    metrics[name] = direction_metric_bundle(
                        arrays,
                        predicted,
                    )
                    apply_counts[name] = int(apply.sum())
                values = selection_values(
                    metrics["old_iid"],
                    metrics["difficult"],
                )
                rows.append(
                    {
                        "boundary_limit": float(boundary_limit),
                        "confidence_limit": float(confidence_limit),
                        "margin_limit": float(margin_limit),
                        "selection_score": geometric_score(values),
                        "selection_mean": float(np.mean(values)),
                        "apply_counts": apply_counts,
                        "metrics": metrics,
                    }
                )
    selected = max(
        rows,
        key=lambda row: (
            row["selection_score"],
            row["selection_mean"],
            -row["apply_counts"]["old_iid"]
            - row["apply_counts"]["difficult"],
        ),
    )
    baseline = next(
        row
        for row in rows
        if row["boundary_limit"] == 0.0
        and row["confidence_limit"] == 0.5
        and row["margin_limit"] == 0.0
    )
    artifact = {
        "version": "hybrid_boundary_direction_v9_one_seed",
        "model": "frozen_v7_forward_plus_boundary_tree_v4",
        "forward_artifact": str(args.forward_artifact.resolve()),
        "forward_artifact_sha256": sha256(args.forward_artifact.resolve()),
        "tree_artifact": str(args.tree_artifact.resolve()),
        "tree_artifact_sha256": sha256(args.tree_artifact.resolve()),
        "calibration": {
            name: selected[name]
            for name in (
                "boundary_limit",
                "confidence_limit",
                "margin_limit",
            )
        },
        "held_out_test_used": False,
    }
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "calibration": artifact["calibration"],
        "validation": {
            "thresholded_v7_baseline": baseline["metrics"],
            "hybrid_boundary_v9": selected["metrics"],
        },
        "selection": {
            "baseline_score": baseline["selection_score"],
            "selected_score": selected["selection_score"],
            "baseline_mean": baseline["selection_mean"],
            "selected_mean": selected["selection_mean"],
            "apply_counts": selected["apply_counts"],
            "candidate_count": len(rows),
        },
        "source_contract": {
            "old_validation": str(args.old_validation.resolve()),
            "difficult_validation": str(
                args.difficult_validation.resolve()
            ),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "summary": str(summary_path),
                "calibration": artifact["calibration"],
                "selection": summary["selection"],
                "validation": {
                    split: selected["metrics"][split]["overall"][
                        "joint_exact"
                    ]
                    for split in ("old_iid", "difficult")
                },
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
