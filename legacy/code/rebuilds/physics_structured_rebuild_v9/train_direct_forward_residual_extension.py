#!/usr/bin/env python3
"""Fit a protected direct-request residual on the accepted forward ensemble."""

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

from control_rebuild_v3.common import tolerance_from_current
from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    action_index,
)
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    DEFAULT_DIFFICULT,
    DEFAULT_FORWARD,
    DEFAULT_OLD,
    action_high_mask,
    normalized_success,
    protected_counts,
    raw_success,
    sha256,
)
from physics_structured_rebuild_v9.train_forward_selector_extension import (
    current_system_prediction,
    ensemble_prediction,
    load_ensemble,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from specialist_rebuild_v2.common import STATE_FIELDS, forward_feature, read_jsonl

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_GRID_CACHE = DEFAULT_RUN / "qwen_forward_grid_training_features.npz"
DEFAULT_DIRECT_DATA = (
    REPO_ROOT.parent
    / "VLM_data/physics_structured_rebuild_v9/qwen_adaptation/train.jsonl"
)
DEFAULT_QWEN = REPO_ROOT.parent / "VLM_data/qwen_orchestration/v1"
DEFAULT_CURRENT = DEFAULT_RUN / "forward_selector_ensemble_v9.pkl"
DEFAULT_OUTPUT = DEFAULT_RUN / "direct_forward_residual_extension_v9.pkl"
STATE_ROUTE = "predict_forward_from_state_v1"
BLENDS = np.asarray(
    [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-cache", type=Path, default=DEFAULT_GRID_CACHE)
    parser.add_argument("--direct-data", type=Path, default=DEFAULT_DIRECT_DATA)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--current-selector", type=Path, default=DEFAULT_CURRENT)
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


def correction_features(
    physical_features: np.ndarray,
    current_prediction: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [physical_features, current_prediction],
        axis=1,
    ).astype(np.float32)


def predict_corrections(
    models: dict[str, Any],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "extra_trees": np.asarray(
            models["extra_trees"].predict(features),
            dtype=np.float32,
        ),
        "hgb": np.stack(
            [
                np.asarray(model.predict(features), dtype=np.float32)
                for model in models["hgb"]
            ],
            axis=1,
        ),
    }


def selected_prediction(
    current: np.ndarray,
    corrections: dict[str, np.ndarray],
    selection: list[tuple[str, float]],
) -> np.ndarray:
    output = np.asarray(current, dtype=np.float32).copy()
    for field, (source, blend) in enumerate(selection):
        if source != "none":
            output[:, field] += (
                corrections[source][:, field] * float(blend)
            )
    return output


def metric(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    error = np.abs(prediction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=1)
    return {
        "count": int(len(exact)),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "per_field_tolerance_pass": [
            float(passed[:, field].mean()) for field in range(5)
        ],
        "mae_in_tolerance_units": float(error.mean()),
    }


def search_selection(
    calibration_current: np.ndarray,
    calibration_target: np.ndarray,
    calibration_corrections: dict[str, np.ndarray],
    protected: list[
        tuple[
            np.ndarray,
            np.ndarray,
            dict[str, np.ndarray],
            np.ndarray,
        ]
    ],
) -> tuple[list[tuple[str, float]], list[dict[str, Any]]]:
    choices = [("none", 0.0)] + [
        (source, float(blend))
        for source in ("extra_trees", "hgb")
        for blend in BLENDS[1:]
    ]
    protected_baseline = [
        protected_counts(
            normalized_success(current, target),
            high,
        )
        for current, target, _, high in protected
    ]

    def key(selection: list[tuple[str, float]]) -> tuple[Any, ...]:
        for (
            block_current,
            block_target,
            block_corrections,
            high,
        ), baseline in zip(protected, protected_baseline, strict=True):
            block_prediction = selected_prediction(
                block_current,
                block_corrections,
                selection,
            )
            observed = protected_counts(
                normalized_success(block_prediction, block_target),
                high,
            )
            if any(
                value < base
                for value, base in zip(observed, baseline, strict=True)
            ):
                return (False, 0, 0, float("-inf"), float("-inf"))
        prediction = selected_prediction(
            calibration_current,
            calibration_corrections,
            selection,
        )
        error = np.abs(prediction - calibration_target)
        passed = error <= 1.0
        exact = np.all(passed, axis=1)
        return (
            True,
            int(exact.sum()),
            int(passed.sum()),
            -float(error.mean()),
            -sum(abs(blend) for _, blend in selection),
        )

    starts: list[list[tuple[str, float]]] = [
        [("none", 0.0)] * 5,
    ]
    for source in ("extra_trees", "hgb"):
        for blend in (0.25, 0.5, 1.0):
            starts.append([(source, blend)] * 5)
    valid_starts = [
        start for start in starts if bool(key(start)[0])
    ] or [[("none", 0.0)] * 5]
    finals = []
    traces = []
    for start_index, initial in enumerate(valid_starts):
        selected = list(initial)
        passes = []
        for pass_index in range(4):
            changed = False
            for field in range(5):
                best = None
                for choice in choices:
                    proposal = list(selected)
                    proposal[field] = choice
                    proposal_key = key(proposal)
                    if not bool(proposal_key[0]):
                        continue
                    candidate = (
                        proposal_key,
                        choice[0] == "none",
                        -abs(choice[1]),
                        choice,
                    )
                    if best is None or candidate > best:
                        best = candidate
                if best is None:
                    continue
                if selected[field] != best[3]:
                    selected[field] = best[3]
                    changed = True
            passes.append(
                {
                    "pass": pass_index + 1,
                    "selection": [
                        {"source": source, "blend": blend}
                        for source, blend in selected
                    ],
                    "calibration": metric(
                        selected_prediction(
                            calibration_current,
                            calibration_corrections,
                            selected,
                        ),
                        calibration_target,
                    ),
                }
            )
            if not changed:
                break
        finals.append(list(selected))
        traces.append({"start": start_index, "passes": passes})
    return max(finals, key=key), traces


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
    )

    direct_rows = read_jsonl(args.direct_data.resolve())
    if len(direct_rows) != 1000:
        raise ValueError("expected 1,000 direct forward training rows")
    group_ids = np.asarray(
        [str(row["group_id"]) for row in direct_rows],
        dtype=np.str_,
    )
    current_path = args.current_selector.resolve()
    current_artifact = load_ensemble(current_path)
    torch, device = configure(int(args.seed), args.device)
    current_runtime, _ = load_forward_selector_ensemble_runtime_v9(
        current_path,
        torch,
        device,
    )
    runtime_rows = [
        {
            "group_id": str(row["group_id"]),
            "setup": row["setup"],
            "current_beam_state": row["current_beam_state"],
        }
        for row in direct_rows
    ]
    current_grid = current_runtime.predict_changes(runtime_rows)
    current = np.stack(
        [
            current_grid[index, action_index(row["action"])]
            for index, row in enumerate(direct_rows)
        ]
    ).astype(np.float32)
    engineered = np.asarray(
        [
            forward_feature(
                row["setup"],
                row["current_beam_state"],
                row["action"],
            )
            for row in direct_rows
        ],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            np.asarray(
                [
                    float(row["truth_change"][field])
                    for field in STATE_FIELDS
                ],
                dtype=np.float32,
            )
            / tolerance_from_current(row["current_beam_state"])
            for row in direct_rows
        ],
        dtype=np.float32,
    )
    model_features = correction_features(engineered, current)
    residual_target = target - current
    training, calibration = split(group_ids, int(args.seed))
    models: dict[str, Any] = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=320,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=2,
            random_state=int(args.seed),
        ),
        "hgb": [],
    }
    models["extra_trees"].fit(
        model_features[training],
        residual_target[training],
    )
    for field in range(5):
        model = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=4.0,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=int(args.seed) + 17 * field,
        )
        model.fit(
            model_features[training],
            residual_target[training, field],
        )
        models["hgb"].append(model)
    calibration_corrections = predict_corrections(
        models,
        model_features[calibration],
    )

    base, _ = load_forward_direction_runtime_v7(
        args.forward_artifact.resolve(),
        torch,
        device,
    )
    protected = []
    protected_names = []
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_prior = stream_forward_changes(path, base)
        block_features = np.concatenate(
            [arrays.features, block_prior],
            axis=1,
        ).astype(np.float32)
        block_current = ensemble_prediction(
            current_artifact,
            block_features,
            block_prior,
        )
        block_corrections = predict_corrections(
            models,
            correction_features(arrays.features, block_current),
        )
        protected.append(
            (
                block_current,
                arrays.normalized_changes,
                block_corrections,
                action_high_mask(arrays.group_count),
            )
        )
        protected_names.append(name)
    selection, traces = search_selection(
        current[calibration],
        target[calibration],
        calibration_corrections,
        protected,
    )
    protected_report = {}
    for name, (
        block_current,
        block_target,
        block_corrections,
        high,
    ) in zip(protected_names, protected, strict=True):
        baseline_success = normalized_success(block_current, block_target)
        selected_success = normalized_success(
            selected_prediction(
                block_current,
                block_corrections,
                selection,
            ),
            block_target,
        )
        protected_report[name] = {
            "current": protected_counts(baseline_success, high),
            "selected": protected_counts(selected_success, high),
        }

    system_current, system_arrays = current_system_prediction(
        current_artifact
    )
    system_mask = system_arrays["routes"] == STATE_ROUTE
    canonical = [
        row
        for row in read_jsonl(
            args.qwen_data.resolve() / "canonical/val.jsonl"
        )
        if row["target_decision"].get("route_name") == STATE_ROUTE
    ]
    engineered = np.asarray(
        [
            forward_feature(
                row["target_decision"]["arguments"]["setup"],
                row["target_decision"]["arguments"]["current_beam_state"],
                row["target_decision"]["arguments"]["action"],
            )
            for row in canonical
        ],
        dtype=np.float32,
    )
    system_corrections = predict_corrections(
        models,
        correction_features(engineered, system_current),
    )
    system_selected = selected_prediction(
        system_current,
        system_corrections,
        selection,
    )
    success_arguments = (
        system_arrays["input_tolerance"][system_mask],
        system_arrays["truth_change"][system_mask],
        system_arrays["scoring_tolerance"][system_mask],
    )
    system_current_success = raw_success(system_current, *success_arguments)
    system_selected_success = raw_success(system_selected, *success_arguments)

    artifact = {
        "version": "rounded_direct_forward_residual_extension_v9_one_seed",
        "model": "protected_rounded_direct_forward_residual_extension_v9",
        "seed": int(args.seed),
        "models": models,
        "selection": [
            {"source": source, "blend": float(blend)}
            for source, blend in selection
        ],
        "current_selector": str(current_path),
        "current_selector_sha256": sha256(current_path),
        "forward_artifact": str(args.forward_artifact.resolve()),
        "forward_artifact_sha256": sha256(args.forward_artifact.resolve()),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    baseline_calibration = metric(current[calibration], target[calibration])
    selected_calibration = metric(
        selected_prediction(
            current[calibration],
            calibration_corrections,
            selection,
        ),
        target[calibration],
    )
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "training_count": int(len(training)),
        "calibration_count": int(len(calibration)),
        "selection": artifact["selection"],
        "calibration": {
            "current": baseline_calibration,
            "selected": selected_calibration,
        },
        "protected": protected_report,
        "protected_non_regression": all(
            selected >= current_count
            for block in protected_report.values()
            for selected, current_count in zip(
                block["selected"],
                block["current"],
                strict=True,
            )
        ),
        "system_validation": {
            "current_success_count": int(system_current_success.sum()),
            "selected_success_count": int(system_selected_success.sum()),
            "current_only_count": int(
                (system_current_success & ~system_selected_success).sum()
            ),
            "selected_only_count": int(
                (~system_current_success & system_selected_success).sum()
            ),
        },
        "search_traces": traces,
        "source_contract": {
            "rounded_direct_training_data": str(args.direct_data.resolve()),
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
