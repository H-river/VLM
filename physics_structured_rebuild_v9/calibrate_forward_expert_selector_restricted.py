#!/usr/bin/env python3
"""Calibrate a protected state-only forward expert selector."""

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
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    DEFAULT_ALTERNATIVE,
    DEFAULT_CURRENT,
    DEFAULT_DIFFICULT,
    DEFAULT_FORWARD,
    DEFAULT_OLD,
    DEFAULT_OUTPUT,
    DEFAULT_TRAINING,
    ROUTES,
    action_high_mask,
    calibrated_prediction,
    normalized_success,
    protected_counts,
    raw_success,
    row_indices,
    selector_features,
    sha256,
    split_groups,
    system_predictions,
)

DEFAULT_SOURCE = DEFAULT_OUTPUT
DEFAULT_RESTRICTED = (
    DEFAULT_OUTPUT.parent / "forward_natural_expert_selector_restricted_v9.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_RESTRICTED)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def best_protected_threshold(
    probability: np.ndarray,
    current_success: np.ndarray,
    alternative_success: np.ndarray,
    protected: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> dict[str, Any]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0, np.inf]),
                np.quantile(probability, np.linspace(0.0, 1.0, 1001)),
            ]
        )
    )
    best = None
    valid_count = 0
    for threshold in thresholds:
        protected_ok = True
        for block_probability, block_current, block_alternative, high in (
            protected
        ):
            selected = np.where(
                block_probability >= threshold,
                block_alternative,
                block_current,
            )
            protected_ok &= int(selected.sum()) >= int(block_current.sum())
            protected_ok &= int(selected[high].sum()) >= int(
                block_current[high].sum()
            )
        if not protected_ok:
            continue
        valid_count += 1
        selected = np.where(
            probability >= threshold,
            alternative_success,
            current_success,
        )
        alternative_count = int((probability >= threshold).sum())
        candidate = {
            "threshold": float(threshold),
            "calibration_success_count": int(selected.sum()),
            "calibration_alternative_count": alternative_count,
        }
        key = (candidate["calibration_success_count"], -alternative_count)
        if best is None or key > best[0]:
            best = (key, candidate)
    if best is None:
        raise RuntimeError("no protected selector threshold was found")
    best[1]["valid_threshold_count"] = valid_count
    return best[1]


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    with source_path.open("rb") as stream:
        source = pickle.load(stream)
    if source.get("model") != "route_specific_hgb_forward_expert_selector_v9":
        raise ValueError("unexpected source forward selector")

    training_path = args.training_cache.resolve()
    with np.load(training_path, allow_pickle=False) as cache:
        features = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        target = np.asarray(
            cache["grid_target_normalized"],
            dtype=np.float32,
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    _, calibration_groups = split_groups(group_ids, int(args.seed))
    calibration_indices = row_indices(calibration_groups)

    current_dir = args.current_dir.resolve()
    alternative_dir = args.alternative_dir.resolve()
    route_training = {}
    route_natural = {}
    for route_key in ROUTES:
        current = calibrated_prediction(
            current_dir / f"forward_{route_key}.pkl",
            features,
            prior,
        )
        alternative = calibrated_prediction(
            alternative_dir / f"forward_{route_key}.pkl",
            features,
            prior,
        )
        route_training[route_key] = (current, alternative)
        current_success = normalized_success(current, target)
        alternative_success = normalized_success(alternative, target)
        route_natural[route_key] = {
            "current_success": current_success,
            "alternative_success": alternative_success,
            "probability": source["routes"][route_key][
                "classifier"
            ].predict_proba(
                selector_features(
                    prior[calibration_indices],
                    current[calibration_indices],
                    alternative[calibration_indices],
                )
            )[:, 1],
        }

    torch, device = configure(int(args.seed), args.device)
    forward, _ = load_forward_direction_runtime_v7(
        args.forward_artifact.resolve(),
        torch,
        device,
    )
    protected = {}
    state_constraint_blocks = []
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
            current = calibrated_prediction(
                current_dir / f"forward_{route_key}.pkl",
                block_features,
                block_prior,
            )
            alternative = calibrated_prediction(
                alternative_dir / f"forward_{route_key}.pkl",
                block_features,
                block_prior,
            )
            current_success = normalized_success(
                current,
                arrays.normalized_changes,
            )
            alternative_success = normalized_success(
                alternative,
                arrays.normalized_changes,
            )
            probability = source["routes"][route_key][
                "classifier"
            ].predict_proba(
                selector_features(block_prior, current, alternative)
            )[:, 1]
            protected[block][route_key] = {
                "current_success": current_success,
                "alternative_success": alternative_success,
                "probability": probability,
                "high": high,
            }
            if route_key == "state":
                state_constraint_blocks.append(
                    (
                        probability,
                        current_success,
                        alternative_success,
                        high,
                    )
                )

    state_natural = route_natural["state"]
    state_threshold = best_protected_threshold(
        state_natural["probability"],
        state_natural["current_success"][calibration_indices],
        state_natural["alternative_success"][calibration_indices],
        state_constraint_blocks,
    )
    routes = {}
    protected_report = {}
    for block, block_values in protected.items():
        protected_report[block] = {}
        for route_key, values in block_values.items():
            if route_key == "state":
                choose_alternative = (
                    values["probability"] >= state_threshold["threshold"]
                )
            else:
                choose_alternative = np.zeros(
                    len(values["probability"]),
                    dtype=np.bool_,
                )
            selected = np.where(
                choose_alternative,
                values["alternative_success"],
                values["current_success"],
            )
            protected_report[block][route_key] = {
                "current": protected_counts(
                    values["current_success"],
                    values["high"],
                ),
                "selected": protected_counts(selected, values["high"]),
                "alternative_count": int(choose_alternative.sum()),
            }

    for route_key in ROUTES:
        routes[route_key] = {
            "classifier": source["routes"][route_key]["classifier"],
            "threshold": (
                float(state_threshold["threshold"])
                if route_key == "state"
                else float("inf")
            ),
            "enabled": route_key == "state",
        }
    artifact = {
        **source,
        "version": "forward_natural_expert_selector_restricted_v9_one_seed",
        "model": "route_restricted_hgb_forward_expert_selector_v9",
        "routes": routes,
        "source_selector": str(source_path),
        "source_selector_sha256": sha256(source_path),
        "selection_contract": (
            "state threshold maximizes natural calibration success subject "
            "to no regression on old and difficult protected validation; "
            "image route remains on retained current expert"
        ),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)

    current_system, current_arrays = system_predictions(current_dir)
    alternative_system, alternative_arrays = system_predictions(
        alternative_dir
    )
    for key in ("prior", "truth_change", "routes"):
        if not np.array_equal(current_arrays[key], alternative_arrays[key]):
            raise ValueError(f"system candidate caches differ for {key}")
    system = {}
    total = 0
    for route_key, route_name in ROUTES.items():
        mask = current_arrays["routes"] == route_name
        current = current_system[route_key][mask]
        alternative = alternative_system[route_key][mask]
        current_success = raw_success(
            current,
            current_arrays["input_tolerance"][mask],
            current_arrays["truth_change"][mask],
            current_arrays["scoring_tolerance"][mask],
        )
        alternative_success = raw_success(
            alternative,
            current_arrays["input_tolerance"][mask],
            current_arrays["truth_change"][mask],
            current_arrays["scoring_tolerance"][mask],
        )
        if route_key == "state":
            probability = routes[route_key]["classifier"].predict_proba(
                selector_features(
                    current_arrays["prior"][mask],
                    current,
                    alternative,
                )
            )[:, 1]
            choose_alternative = probability >= routes[route_key]["threshold"]
        else:
            choose_alternative = np.zeros(mask.sum(), dtype=np.bool_)
        selected = np.where(
            choose_alternative,
            alternative_success,
            current_success,
        )
        total += int(selected.sum())
        system[route_key] = {
            "current_success_count": int(current_success.sum()),
            "selected_success_count": int(selected.sum()),
            "alternative_count": int(choose_alternative.sum()),
        }
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "state_threshold": state_threshold,
        "protected_validation": protected_report,
        "protected_non_regression": all(
            route["selected"][index] >= route["current"][index]
            for block in protected_report.values()
            for route in block.values()
            for index in (0, 1)
        ),
        "system_validation": {
            "routes": system,
            "selected_success_count": total,
            "selected_success_rate": total / 300,
        },
        "source_contract": {
            "training_cache": str(training_path),
            "training_cache_sha256": sha256(training_path),
            "source_selector": str(source_path),
            "system_validation_used_for_threshold_selection": False,
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
