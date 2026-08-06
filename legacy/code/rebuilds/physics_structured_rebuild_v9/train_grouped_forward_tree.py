#!/usr/bin/env python3
"""Fit protected per-field trees for complete fixed-grid forward surfaces."""

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

from control_rebuild_v3.common import action_basis, tolerance_from_current
from control_rebuild_v3.train_forward import configure
from direction_rebuild_v4.data import load_grid_arrays
from physics_structured_rebuild_v9.calibrate_forward_system import (
    action_high_mask,
    specialist_counts,
)
from physics_structured_rebuild_v9.calibrate_system_direction import read_jsonl
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.grouped_forward_surface import (
    grouped_context_features,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)
from physics_structured_rebuild_v9.train_grouped_forward_surface import (
    metric,
    stable_split,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "combined_natural_inverse_forward_cache.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "grouped_forward_tree_v9.pkl"
DEFAULT_OLD = (
    REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2/grids/val.jsonl"
)
DEFAULT_DIFFICULT = (
    REPO_ROOT.parent
    / "VLM_data/control_rebuild_v4_quickcheck/grids/val.jsonl"
)
BLENDS = np.asarray(
    [0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--base-forward",
        type=Path,
        default=DEFAULT_NATURAL_GRID_FORWARD_STATE,
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_features_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    contexts = np.asarray(
        [
            [
                *[float(row["setup"][field]) for field in SETUP_FIELDS],
                *[
                    (
                        np.log1p(
                            max(float(row["current_beam_state"][field]), 0.0)
                        )
                        if field == "peak_intensity"
                        else float(row["current_beam_state"][field])
                    )
                    for field in STATE_FIELDS
                ],
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    return grouped_context_features(contexts)


def correction_from_models(
    models: list[Any],
    features: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    coefficients = np.stack(
        [
            np.asarray(model.predict(features), dtype=np.float32)
            for model in models
        ],
        axis=2,
    )
    return np.einsum("ab,gbf->gaf", basis, coefficients).astype(
        np.float32
    )


def selected_prediction(
    prior: np.ndarray,
    correction: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    return prior + correction * BLENDS[selected][None, None, :]


def coordinate_search(
    target: np.ndarray,
    prior: np.ndarray,
    correction: np.ndarray,
    old_target: np.ndarray,
    old_prior: np.ndarray,
    old_correction: np.ndarray,
    difficult_target: np.ndarray,
    difficult_prior: np.ndarray,
    difficult_correction: np.ndarray,
    old_high: np.ndarray,
    difficult_high: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    baseline_counts = specialist_counts(
        old_prior.reshape(-1, 5),
        difficult_prior.reshape(-1, 5),
        old_target.reshape(-1, 5),
        difficult_target.reshape(-1, 5),
        old_high,
        difficult_high,
    )
    starts = [
        np.full(5, value, dtype=np.int64)
        for value in (0, 2, 4, 5)
    ]
    finals = []
    traces = []
    for start_index, initial in enumerate(starts):
        selected = initial.copy()
        route_trace = []
        for pass_index in range(5):
            changed = False
            for field in range(5):
                best = None
                for candidate in range(len(BLENDS)):
                    proposal = selected.copy()
                    proposal[field] = candidate
                    old_prediction = selected_prediction(
                        old_prior,
                        old_correction,
                        proposal,
                    )
                    difficult_prediction = selected_prediction(
                        difficult_prior,
                        difficult_correction,
                        proposal,
                    )
                    protected = specialist_counts(
                        old_prediction.reshape(-1, 5),
                        difficult_prediction.reshape(-1, 5),
                        old_target.reshape(-1, 5),
                        difficult_target.reshape(-1, 5),
                        old_high,
                        difficult_high,
                    )
                    if any(
                        observed < baseline
                        for observed, baseline in zip(
                            protected,
                            baseline_counts,
                            strict=True,
                        )
                    ):
                        continue
                    metrics = metric(
                        target,
                        prior,
                        correction,
                        0.0,
                    )
                    prediction = selected_prediction(
                        prior,
                        correction,
                        proposal,
                    )
                    error = np.abs(prediction - target)
                    exact = np.all(error <= 1.0, axis=2)
                    key = (
                        int(exact.sum()),
                        int((error <= 1.0).sum()),
                        -float(error.mean()),
                        sum(protected),
                        -float(np.abs(BLENDS[proposal]).sum()),
                    )
                    if best is None or key > best[0]:
                        best = (key, candidate, metrics)
                if best is None:
                    continue
                if int(selected[field]) != int(best[1]):
                    selected[field] = int(best[1])
                    changed = True
            prediction = selected_prediction(prior, correction, selected)
            error = np.abs(prediction - target)
            route_trace.append(
                {
                    "pass": pass_index + 1,
                    "field_blend": BLENDS[selected].tolist(),
                    "strict_all_five_count": int(
                        np.all(error <= 1.0, axis=2).sum()
                    ),
                    "mae_in_tolerance_units": float(error.mean()),
                }
            )
            if not changed:
                break
        finals.append(selected.copy())
        traces.append({"start": start_index, "passes": route_trace})

    def final_key(selected: np.ndarray) -> tuple[Any, ...]:
        protected = specialist_counts(
            selected_prediction(
                old_prior,
                old_correction,
                selected,
            ).reshape(-1, 5),
            selected_prediction(
                difficult_prior,
                difficult_correction,
                selected,
            ).reshape(-1, 5),
            old_target.reshape(-1, 5),
            difficult_target.reshape(-1, 5),
            old_high,
            difficult_high,
        )
        if any(
            observed < baseline
            for observed, baseline in zip(
                protected,
                baseline_counts,
                strict=True,
            )
        ):
            return (False, 0, 0, float("-inf"), float("-inf"))
        prediction = selected_prediction(prior, correction, selected)
        error = np.abs(prediction - target)
        return (
            True,
            int(np.all(error <= 1.0, axis=2).sum()),
            int((error <= 1.0).sum()),
            -float(error.mean()),
            -float(np.abs(BLENDS[selected]).sum()),
        )

    return max(finals, key=final_key), traces


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import ExtraTreesRegressor

    started = time.perf_counter()
    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        contexts = np.asarray(cache["contexts"][:, :17], dtype=np.float32)
        truth_states = np.asarray(
            cache["true_candidate_states"],
            dtype=np.float32,
        )
        base_states = np.asarray(cache["primary_states"], dtype=np.float32)
    current = contexts[:, 12:17].copy()
    current[:, -1] = np.expm1(current[:, -1])
    tolerance = np.stack(
        [tolerance_from_current(values) for values in current]
    ).astype(np.float32)
    target = (truth_states - current[:, None, :]) / tolerance[:, None, :]
    prior = (base_states - current[:, None, :]) / tolerance[:, None, :]
    basis = np.asarray(action_basis(), dtype=np.float32)
    pseudoinverse = np.linalg.pinv(basis).astype(np.float32)
    base_coefficients = np.einsum(
        "ba,gaf->gbf",
        pseudoinverse,
        prior,
    )
    residual_coefficients = np.einsum(
        "ba,gaf->gbf",
        pseudoinverse,
        target - prior,
    )
    features = np.concatenate(
        [
            grouped_context_features(contexts),
            base_coefficients.reshape(len(contexts), -1),
        ],
        axis=1,
    ).astype(np.float32)
    training, validation = stable_split(group_ids, int(args.seed))
    models = []
    for field in range(5):
        model = ExtraTreesRegressor(
            n_estimators=int(args.trees),
            min_samples_leaf=int(args.min_samples_leaf),
            max_features=float(args.max_features),
            n_jobs=2,
            random_state=int(args.seed) + 17 * field,
        )
        model.fit(features[training], residual_coefficients[training, :, field])
        models.append(model)
        print(
            json.dumps(
                {"trained_field": field + 1, "field_count": 5},
                sort_keys=True,
            ),
            flush=True,
        )
    correction = correction_from_models(
        models,
        features[validation],
        basis,
    )
    torch, device = configure(int(args.seed), args.device)
    base_runtime, _ = load_residual_forward_runtime_v9(
        args.base_forward.resolve(),
        torch,
        device,
    )
    protected = {}
    protected_values = {}
    for name, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        rows = read_jsonl(path)
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_prior = base_runtime.predict_changes(rows)
        block_coefficients = np.einsum(
            "ba,gaf->gbf",
            pseudoinverse,
            block_prior,
        )
        block_features = np.concatenate(
            [
                context_features_from_rows(rows),
                block_coefficients.reshape(len(rows), -1),
            ],
            axis=1,
        ).astype(np.float32)
        block_correction = correction_from_models(
            models,
            block_features,
            basis,
        )
        block_target = arrays.normalized_changes.reshape(
            len(rows),
            81,
            5,
        )
        protected_values[name] = (
            block_target,
            block_prior,
            block_correction,
        )
    old_target, old_prior, old_correction = protected_values["old_iid"]
    (
        difficult_target,
        difficult_prior,
        difficult_correction,
    ) = protected_values["difficult"]
    old_high = action_high_mask(len(old_target))
    difficult_high = action_high_mask(len(difficult_target))
    selected, traces = coordinate_search(
        target[validation],
        prior[validation],
        correction,
        old_target,
        old_prior,
        old_correction,
        difficult_target,
        difficult_prior,
        difficult_correction,
        old_high,
        difficult_high,
    )
    baseline_counts = specialist_counts(
        old_prior.reshape(-1, 5),
        difficult_prior.reshape(-1, 5),
        old_target.reshape(-1, 5),
        difficult_target.reshape(-1, 5),
        old_high,
        difficult_high,
    )
    selected_counts = specialist_counts(
        selected_prediction(
            old_prior,
            old_correction,
            selected,
        ).reshape(-1, 5),
        selected_prediction(
            difficult_prior,
            difficult_correction,
            selected,
        ).reshape(-1, 5),
        old_target.reshape(-1, 5),
        difficult_target.reshape(-1, 5),
        old_high,
        difficult_high,
    )
    artifact = {
        "version": "grouped_forward_tree_v9_one_seed",
        "model": "grouped_action_basis_extra_trees_v9",
        "seed": int(args.seed),
        "models": models,
        "field_blend": BLENDS[selected].astype(np.float32),
        "base_forward": str(args.base_forward.resolve()),
        "base_forward_sha256": sha256(args.base_forward.resolve()),
        "training_cache": str(args.cache.resolve()),
        "training_cache_sha256": sha256(args.cache.resolve()),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    selected_correction = correction * BLENDS[selected][None, None, :]
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "training_count": int(len(training)),
        "calibration_count": int(len(validation)),
        "field_blend": BLENDS[selected].tolist(),
        "calibration_baseline": metric(
            target[validation],
            prior[validation],
            np.zeros_like(correction),
            0.0,
        ),
        "calibration_selected": metric(
            target[validation],
            prior[validation],
            selected_correction,
            1.0,
        ),
        "protected": {
            "baseline_counts": list(baseline_counts),
            "selected_counts": list(selected_counts),
            "non_regression": all(
                selected_value >= baseline_value
                for selected_value, baseline_value in zip(
                    selected_counts,
                    baseline_counts,
                    strict=True,
                )
            ),
        },
        "search_traces": traces,
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
