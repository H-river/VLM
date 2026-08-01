#!/usr/bin/env python3
"""Calibrate action-complexity-specific mixtures of two frozen forward trees."""

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
from direction_rebuild_v4.data import load_grid_arrays
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.calibrate_forward_system import (
    action_high_mask,
    specialist_counts,
    tree_residuals,
)
from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7.pt"
)
DEFAULT_PRIMARY = (
    DEFAULT_RUN / "forward_tree_residual/forward_tree_residual_v9.pkl"
)
DEFAULT_SECONDARY = (
    DEFAULT_RUN
    / "forward_extra_trees_residual/forward_extra_trees_residual_v9.pkl"
)
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--primary-tree", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--secondary-tree", type=Path, default=DEFAULT_SECONDARY)
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN / "forward_complexity_mix",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--top-pairs", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_complexity(group_count: int) -> np.ndarray:
    per_action = np.asarray(
        [
            sum(
                abs(float(action[field])) > 0.0
                for field in ACTION_FIELDS
            )
            for action in ACTION_GRID
        ],
        dtype=np.int8,
    )
    return np.tile(per_action, group_count)


def top_weight_pairs(
    prior: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    target: np.ndarray,
    top_count: int,
) -> list[tuple[float, float]]:
    weights = np.round(np.arange(-0.50, 1.5001, 0.05), 2)
    candidates = []
    for primary_weight in weights:
        base = prior + float(primary_weight) * primary
        predictions = (
            base[None, :]
            + weights[:, None].astype(np.float32) * secondary[None, :]
        )
        errors = np.abs(predictions - target[None, :])
        counts = np.sum(errors <= 1.0, axis=1)
        maes = np.mean(errors, axis=1)
        for index, secondary_weight in enumerate(weights):
            candidates.append(
                (
                    int(counts[index]),
                    -float(maes[index]),
                    -abs(float(primary_weight))
                    - abs(float(secondary_weight)),
                    float(primary_weight),
                    float(secondary_weight),
                )
            )
    candidates.sort(reverse=True)
    output = []
    for candidate in candidates:
        pair = (candidate[3], candidate[4])
        if pair not in output:
            output.append(pair)
        if len(output) >= top_count:
            break
    return output


def prediction_from_selection(
    prior: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    complexities: np.ndarray,
    selection: dict[tuple[int, int], tuple[float, float]],
) -> np.ndarray:
    prediction = prior.copy()
    for complexity in range(1, 5):
        mask = complexities == complexity
        for field in range(len(STATE_FIELDS)):
            primary_weight, secondary_weight = selection[(complexity, field)]
            prediction[mask, field] = (
                prior[mask, field]
                + primary_weight * primary[mask, field]
                + secondary_weight * secondary[mask, field]
            )
    prediction[complexities == 0] = 0.0
    return prediction


def score(
    old_prediction: np.ndarray,
    difficult_prediction: np.ndarray,
    old_target: np.ndarray,
    difficult_target: np.ndarray,
    old_high: np.ndarray,
    difficult_high: np.ndarray,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    counts = specialist_counts(
        old_prediction,
        difficult_prediction,
        old_target,
        difficult_target,
        old_high,
        difficult_high,
    )
    old_pass = np.abs(old_prediction - old_target) <= 1.0
    difficult_pass = np.abs(difficult_prediction - difficult_target) <= 1.0
    mae = float(
        np.mean(
            np.concatenate(
                [
                    np.abs(old_prediction - old_target).reshape(-1),
                    np.abs(
                        difficult_prediction - difficult_target
                    ).reshape(-1),
                ]
            )
        )
    )
    key = (
        counts[0] + counts[1],
        min(counts[0], counts[1]),
        counts[2] + counts[3],
        int(old_pass.sum() + difficult_pass.sum()),
        -mae,
    )
    return key, {
        "old_exact_count": counts[0],
        "difficult_exact_count": counts[1],
        "old_high_exact_count": counts[2],
        "difficult_high_exact_count": counts[3],
        "old_per_field_pass": {
            field: int(old_pass[:, index].sum())
            for index, field in enumerate(STATE_FIELDS)
        },
        "difficult_per_field_pass": {
            field: int(difficult_pass[:, index].sum())
            for index, field in enumerate(STATE_FIELDS)
        },
        "combined_mae_in_tolerance_units": mae,
    }


def current_start(path: Path) -> dict[tuple[int, int], tuple[float, float]]:
    with path.open("rb") as stream:
        artifact = pickle.load(stream)
    sources = artifact["field_tree_source"]
    blends = artifact["field_blend"]
    selection = {}
    for complexity in range(1, 5):
        for field_index, field in enumerate(STATE_FIELDS):
            blend = float(blends[field])
            selection[(complexity, field_index)] = (
                (blend, 0.0)
                if sources[field] == "primary"
                else (0.0, blend)
            )
    return selection


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    artifact_path = output_dir / "forward_complexity_mix.pkl"
    summary_path = output_dir / "forward_complexity_mix_summary.json"
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, device = configure(int(args.seed), args.device)
    forward_path = args.forward_artifact.resolve()
    primary_path = args.primary_tree.resolve()
    secondary_path = args.secondary_tree.resolve()
    base, _ = load_forward_direction_runtime_v7(forward_path, torch, device)
    with primary_path.open("rb") as stream:
        primary_tree = pickle.load(stream)
    with secondary_path.open("rb") as stream:
        secondary_tree = pickle.load(stream)
    primary_models = list(primary_tree["models"])
    secondary_models = list(secondary_tree["models"])

    old_path = args.old_validation.resolve()
    difficult_path = args.difficult_validation.resolve()
    old = load_grid_arrays(old_path, include_legacy_features=False)
    difficult = load_grid_arrays(
        difficult_path,
        include_legacy_features=False,
    )
    from physics_structured_rebuild_v9.calibrate_system_direction import (
        read_jsonl,
    )

    old_prior = base.predict_changes(read_jsonl(old_path)).reshape(-1, 5)
    difficult_prior = base.predict_changes(
        read_jsonl(difficult_path)
    ).reshape(-1, 5)
    old_primary = tree_residuals(
        primary_models,
        old.features,
        old_prior,
    )
    difficult_primary = tree_residuals(
        primary_models,
        difficult.features,
        difficult_prior,
    )
    old_secondary = tree_residuals(
        secondary_models,
        old.features,
        old_prior,
    )
    difficult_secondary = tree_residuals(
        secondary_models,
        difficult.features,
        difficult_prior,
    )
    old_complexity = action_complexity(old.group_count)
    difficult_complexity = action_complexity(difficult.group_count)
    old_high = action_high_mask(old.group_count)
    difficult_high = action_high_mask(difficult.group_count)

    candidate_pairs: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for complexity in range(1, 5):
        old_mask = old_complexity == complexity
        difficult_mask = difficult_complexity == complexity
        for field in range(len(STATE_FIELDS)):
            candidate_pairs[(complexity, field)] = top_weight_pairs(
                np.concatenate(
                    [
                        old_prior[old_mask, field],
                        difficult_prior[difficult_mask, field],
                    ]
                ),
                np.concatenate(
                    [
                        old_primary[old_mask, field],
                        difficult_primary[difficult_mask, field],
                    ]
                ),
                np.concatenate(
                    [
                        old_secondary[old_mask, field],
                        difficult_secondary[difficult_mask, field],
                    ]
                ),
                np.concatenate(
                    [
                        old.normalized_changes[old_mask, field],
                        difficult.normalized_changes[difficult_mask, field],
                    ]
                ),
                int(args.top_pairs),
            )

    independent = {
        key: values[0] for key, values in candidate_pairs.items()
    }
    zero = {key: (0.0, 0.0) for key in candidate_pairs}
    state_start = current_start(
        DEFAULT_RUN / "forward_dual_source_calibrated/forward_state.pkl"
    )
    image_start = current_start(
        DEFAULT_RUN / "forward_dual_source_calibrated/forward_image.pkl"
    )
    for key, pairs in candidate_pairs.items():
        for pair in (zero[key], state_start[key], image_start[key]):
            if pair not in pairs:
                pairs.append(pair)

    traces = []
    finals = []
    for start_name, start_selection in (
        ("independent", independent),
        ("zero", zero),
        ("current_state", state_start),
        ("current_image", image_start),
    ):
        selection = dict(start_selection)
        old_prediction = prediction_from_selection(
            old_prior,
            old_primary,
            old_secondary,
            old_complexity,
            selection,
        )
        difficult_prediction = prediction_from_selection(
            difficult_prior,
            difficult_primary,
            difficult_secondary,
            difficult_complexity,
            selection,
        )
        pass_trace = []
        for pass_index in range(3):
            changed = False
            for complexity, field in sorted(candidate_pairs):
                key = (complexity, field)
                current_pair = selection[key]
                best_pair = current_pair
                best_key = None
                old_mask = old_complexity == complexity
                difficult_mask = difficult_complexity == complexity
                for pair in candidate_pairs[key]:
                    proposal_old = old_prediction.copy()
                    proposal_difficult = difficult_prediction.copy()
                    proposal_old[old_mask, field] = (
                        old_prior[old_mask, field]
                        + pair[0] * old_primary[old_mask, field]
                        + pair[1] * old_secondary[old_mask, field]
                    )
                    proposal_difficult[difficult_mask, field] = (
                        difficult_prior[difficult_mask, field]
                        + pair[0] * difficult_primary[difficult_mask, field]
                        + pair[1]
                        * difficult_secondary[difficult_mask, field]
                    )
                    candidate_key, _ = score(
                        proposal_old,
                        proposal_difficult,
                        old.normalized_changes,
                        difficult.normalized_changes,
                        old_high,
                        difficult_high,
                    )
                    if best_key is None or candidate_key > best_key:
                        best_key = candidate_key
                        best_pair = pair
                if best_pair != current_pair:
                    selection[key] = best_pair
                    changed = True
                    old_prediction[old_mask, field] = (
                        old_prior[old_mask, field]
                        + best_pair[0] * old_primary[old_mask, field]
                        + best_pair[1] * old_secondary[old_mask, field]
                    )
                    difficult_prediction[difficult_mask, field] = (
                        difficult_prior[difficult_mask, field]
                        + best_pair[0]
                        * difficult_primary[difficult_mask, field]
                        + best_pair[1]
                        * difficult_secondary[difficult_mask, field]
                    )
            key_value, metrics = score(
                old_prediction,
                difficult_prediction,
                old.normalized_changes,
                difficult.normalized_changes,
                old_high,
                difficult_high,
            )
            pass_trace.append(
                {
                    "pass": pass_index + 1,
                    "changed": changed,
                    "key": list(key_value),
                    "metrics": metrics,
                }
            )
            if not changed:
                break
        finals.append((key_value, selection, metrics, start_name))
        traces.append({"start": start_name, "passes": pass_trace})
        print(
            json.dumps(
                {
                    "start": start_name,
                    "key": list(key_value),
                    "metrics": metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    best_key, best_selection, best_metrics, best_start = max(
        finals,
        key=lambda item: item[0],
    )
    complexity_blends = {
        str(complexity): {
            field: {
                "primary": float(best_selection[(complexity, field_index)][0]),
                "secondary": float(
                    best_selection[(complexity, field_index)][1]
                ),
            }
            for field_index, field in enumerate(STATE_FIELDS)
        }
        for complexity in range(1, 5)
    }
    artifact = {
        "version": "forward_complexity_mix_v9_one_seed",
        "model": "frozen_v7_plus_residual_tree_v9",
        "route_scope": "state_and_image_forward_candidate",
        "forward_artifact": str(forward_path),
        "forward_artifact_sha256": sha256(forward_path),
        "tree_artifact": str(primary_path),
        "tree_artifact_sha256": sha256(primary_path),
        "secondary_tree_artifact": str(secondary_path),
        "secondary_tree_artifact_sha256": sha256(secondary_path),
        "complexity_field_blend": complexity_blends,
        "field_blend": {field: 0.0 for field in STATE_FIELDS},
        "held_out_test_used": False,
    }
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "selected_start": best_start,
        "selected_key": list(best_key),
        "selected_metrics": best_metrics,
        "complexity_field_blend": complexity_blends,
        "search_traces": traces,
        "seconds": time.perf_counter() - started,
        "source_contract": {
            "calibration_files": [str(old_path), str(difficult_path)],
            "system_validation_files_opened": [],
            "held_out_test_files_opened": [],
            "primary_tree": str(primary_path),
            "secondary_tree": str(secondary_path),
        },
        "held_out_test_used": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
