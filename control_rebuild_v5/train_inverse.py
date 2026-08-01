#!/usr/bin/env python3
"""Train candidate-feasibility and status trees for numerical inverse v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    ACTION_GRID,
    inverse_candidate_features,
    read_jsonl,
    residual_cost,
)
from control_rebuild_v3.train_inverse import inverse_metrics
from control_rebuild_v4.inverse_data import derived_inverse_pairs
from control_rebuild_v4.inverse_runtime import load_inverse_runtime_v4
from control_rebuild_v4.train_inverse import prepare_requests
from control_rebuild_v5.forward_runtime import load_forward_runtime_v5
from control_rebuild_v5.inverse_runtime import InverseTreeRuntimeV5
from specialist_rebuild_v2.common import raw_state_array

DEFAULT_OLD_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_DIFFICULT_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v4_quickcheck"
)
DEFAULT_ADDITIONAL_DATA = (
    REPO_ROOT.parent / "VLM_data/control_rebuild_v5_numerical"
)
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v5_one_seed"
DEFAULT_ERROR_BANK = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v4_one_seed/measurement_error_bank_v4.npz"
)
DEFAULT_V4_INVERSE = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v4_quickcheck_12h/inverse_control_v4.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument(
        "--difficult-data",
        type=Path,
        default=DEFAULT_DIFFICULT_DATA,
    )
    parser.add_argument(
        "--additional-data",
        type=Path,
        default=DEFAULT_ADDITIONAL_DATA,
    )
    parser.add_argument("--include-additional-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--forward-artifact",
        type=Path,
        default=DEFAULT_OUTPUT / "forward_tree_v5.pkl",
    )
    parser.add_argument("--error-bank", type=Path, default=DEFAULT_ERROR_BANK)
    parser.add_argument("--v4-inverse", type=Path, default=DEFAULT_V4_INVERSE)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--hard-negatives", type=int, default=16)
    parser.add_argument("--spread-negatives", type=int, default=8)
    parser.add_argument("--candidate-max-iter", type=int, default=220)
    parser.add_argument("--correction-max-iter", type=int, default=220)
    parser.add_argument("--status-max-iter", type=int, default=180)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_truth(pairs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            -1
            if pair["selected_index"] is None
            else int(pair["selected_index"])
            for pair in pairs
        ],
        dtype=np.int64,
    )


def status_training_matrix(
    arrays: Mapping[str, Any],
    chunk_size: int = 1024,
) -> np.ndarray:
    parts = []
    for start in range(0, len(arrays["contexts"]), chunk_size):
        stop = min(start + chunk_size, len(arrays["contexts"]))
        _, _, status = inverse_candidate_features(
            arrays["candidate_bank"][
                arrays["candidate_indices"][start:stop]
            ],
            arrays["desired_observed"][start:stop],
        )
        parts.append(
            np.concatenate(
                [arrays["contexts"][start:stop], status],
                axis=1,
            )
        )
    return np.concatenate(parts).astype(np.float32)


def candidate_training_matrix(
    arrays: Mapping[str, Any],
    true_candidate_states: np.ndarray,
    hard_negatives: int,
    spread_negatives: int,
    seed: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_parts = []
    label_parts = []
    weight_parts = []
    correction_parts = []
    request_count = 0
    positive_count = 0
    negative_count = 0
    rng = np.random.default_rng(seed)
    for start in range(0, len(arrays["contexts"]), chunk_size):
        stop = min(start + chunk_size, len(arrays["contexts"]))
        candidate, cost, _ = inverse_candidate_features(
            arrays["candidate_bank"][
                arrays["candidate_indices"][start:stop]
            ],
            arrays["desired_observed"][start:stop],
        )
        positives = arrays["positives"][start:stop]
        true_cost = residual_cost(
            true_candidate_states[arrays["group_indices"][start:stop]],
            arrays["desired_true"][start:stop, None, :],
        )
        for local in range(stop - start):
            positive_indices = np.flatnonzero(positives[local])
            if not len(positive_indices):
                continue
            masked = cost[local].copy()
            masked[positive_indices] = np.inf
            hard = np.argsort(masked)[:hard_negatives]
            remaining = np.flatnonzero(
                (~positives[local])
                & (~np.isin(np.arange(len(ACTION_GRID)), hard))
            )
            if len(remaining) and spread_negatives:
                spread = rng.choice(
                    remaining,
                    size=min(spread_negatives, len(remaining)),
                    replace=False,
                )
            else:
                spread = np.empty(0, dtype=np.int64)
            negative_indices = np.unique(
                np.concatenate([hard, spread]).astype(np.int64)
            )
            selected = np.concatenate(
                [positive_indices, negative_indices]
            )
            labels = np.concatenate(
                [
                    np.ones(len(positive_indices), dtype=np.int64),
                    np.zeros(len(negative_indices), dtype=np.int64),
                ]
            )
            context = np.repeat(
                arrays["contexts"][start + local][None, :],
                len(selected),
                axis=0,
            )
            feature_parts.append(
                np.concatenate(
                    [context, candidate[local, selected]],
                    axis=1,
                )
            )
            label_parts.append(labels)
            weights = np.concatenate(
                [
                    np.full(
                        len(positive_indices),
                        0.5 / len(positive_indices),
                        dtype=np.float32,
                    ),
                    np.full(
                        len(negative_indices),
                        0.5 / len(negative_indices),
                        dtype=np.float32,
                    ),
                ]
            )
            weight_parts.append(weights)
            correction_parts.append(
                np.clip(
                    true_cost[local, selected] - cost[local, selected],
                    -12.0,
                    12.0,
                ).astype(np.float32)
            )
            request_count += 1
            positive_count += len(positive_indices)
            negative_count += len(negative_indices)
    features = np.concatenate(feature_parts).astype(np.float32)
    labels = np.concatenate(label_parts)
    weights = np.concatenate(weight_parts)
    corrections = np.concatenate(correction_parts)
    weights *= len(weights) / weights.sum()
    return (
        features,
        labels,
        weights,
        corrections,
        {
            "reachable_requests": request_count,
            "candidate_examples": len(labels),
            "positive_examples": positive_count,
            "negative_examples": negative_count,
            "hard_negatives_per_request": hard_negatives,
            "spread_negatives_per_request": spread_negatives,
        },
    )


def predict_block(
    candidate_model: Any,
    correction_model: Any,
    status_model: Any,
    arrays: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    alpha_candidates: Sequence[float],
    beta_candidates: Sequence[float],
    chunk_size: int = 512,
) -> dict[str, Any]:
    logit_parts = []
    cost_parts = []
    correction_parts = []
    status_probability_parts = []
    for start in range(0, len(arrays["contexts"]), chunk_size):
        stop = min(start + chunk_size, len(arrays["contexts"]))
        candidate, cost, status = inverse_candidate_features(
            arrays["candidate_bank"][
                arrays["candidate_indices"][start:stop]
            ],
            arrays["desired_observed"][start:stop],
        )
        repeated_context = np.repeat(
            arrays["contexts"][start:stop, None, :],
            len(ACTION_GRID),
            axis=1,
        )
        candidate_input = np.concatenate(
            [repeated_context, candidate],
            axis=-1,
        ).reshape(-1, repeated_context.shape[-1] + candidate.shape[-1])
        probability = candidate_model.predict_proba(candidate_input)[:, 1]
        probability = probability.reshape(stop - start, len(ACTION_GRID))
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logit_parts.append(np.log(clipped) - np.log1p(-clipped))
        correction_parts.append(
            correction_model.predict(candidate_input).reshape(
                stop - start,
                len(ACTION_GRID),
            )
        )
        cost_parts.append(cost)
        status_input = np.concatenate(
            [arrays["contexts"][start:stop], status],
            axis=1,
        )
        status_probability_parts.append(
            status_model.predict_proba(status_input)
        )
    logits = np.concatenate(logit_parts)
    costs = np.concatenate(cost_parts)
    corrections = np.concatenate(correction_parts)
    status_probability = np.concatenate(status_probability_parts)
    status_logits = np.log(np.clip(status_probability, 1e-8, 1.0))
    candidates = []
    truth = selected_truth(pairs)
    for beta in beta_candidates:
        estimated_cost = np.maximum(
            costs + float(beta) * corrections,
            0.0,
        )
        for alpha in alpha_candidates:
            candidates.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    **inverse_metrics(
                        -estimated_cost + float(alpha) * logits,
                        status_logits,
                        arrays["positives"],
                        arrays["statuses"],
                        truth,
                    ),
                }
            )
    return {"calibration_candidates": candidates}


def reference_v4(
    runtime: Any,
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    setups = [
        rows[int(index)]["setup"] for index in arrays["group_indices"]
    ]
    candidate_states = arrays["candidate_bank"][arrays["candidate_indices"]]
    output = runtime.score_requests(
        setups,
        arrays["observed_current"],
        arrays["desired_observed"],
        candidate_states,
    )
    return inverse_metrics(
        output["scores"],
        output["status_logits"],
        arrays["positives"],
        arrays["statuses"],
        selected_truth(pairs),
    )


def main() -> None:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old_train_rows = read_jsonl(
        args.old_data.resolve() / "grids/train.jsonl"
    )
    old_val_rows = read_jsonl(args.old_data.resolve() / "grids/val.jsonl")
    difficult_train_rows = read_jsonl(
        args.difficult_data.resolve() / "grids/train.jsonl"
    )
    difficult_val_rows = read_jsonl(
        args.difficult_data.resolve() / "grids/val.jsonl"
    )
    additional_rows: list[dict[str, Any]] = []
    if args.include_additional_data:
        additional_rows = read_jsonl(
            args.additional_data.resolve() / "grids/train.jsonl"
        )
    old_train_pairs = read_jsonl(
        args.old_data.resolve() / "inverse/train.jsonl"
    )
    old_val_pairs = read_jsonl(args.old_data.resolve() / "inverse/val.jsonl")
    difficult_train_pairs = derived_inverse_pairs(difficult_train_rows)
    difficult_val_pairs = derived_inverse_pairs(difficult_val_rows)
    additional_pairs = derived_inverse_pairs(additional_rows)
    train_rows = [
        *old_train_rows,
        *difficult_train_rows,
        *additional_rows,
    ]
    train_pairs = [
        *old_train_pairs,
        *difficult_train_pairs,
        *additional_pairs,
    ]

    loaded_bank = np.load(args.error_bank.resolve(), allow_pickle=False)
    error_bank = np.asarray(loaded_bank["normalized_errors"], dtype=np.float32)
    error_conditions = np.asarray(
        loaded_bank["condition_indices"],
        dtype=np.int64,
    )
    forward, _ = load_forward_runtime_v5(args.forward_artifact.resolve())
    train = prepare_requests(
        train_rows,
        train_pairs,
        forward,
        error_bank,
        error_conditions,
        seed=int(args.seed) + 701,
        noisy_fraction=0.5,
    )
    validation = {
        "iid_clean": (
            old_val_rows,
            old_val_pairs,
            prepare_requests(
                old_val_rows,
                old_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 702,
                noisy_fraction=0.0,
            ),
        ),
        "iid_measurement_augmented": (
            old_val_rows,
            old_val_pairs,
            prepare_requests(
                old_val_rows,
                old_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 702,
                noisy_fraction=1.0,
            ),
        ),
        "difficult_clean": (
            difficult_val_rows,
            difficult_val_pairs,
            prepare_requests(
                difficult_val_rows,
                difficult_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 703,
                noisy_fraction=0.0,
            ),
        ),
        "difficult_measurement_augmented": (
            difficult_val_rows,
            difficult_val_pairs,
            prepare_requests(
                difficult_val_rows,
                difficult_val_pairs,
                forward,
                error_bank,
                error_conditions,
                seed=int(args.seed) + 703,
                noisy_fraction=1.0,
            ),
        ),
    }

    true_candidate_states = np.asarray(
        [
            [
                raw_state_array(candidate["next_state"])
                for candidate in row["candidates"]
            ]
            for row in train_rows
        ],
        dtype=np.float32,
    )
    (
        candidate_features,
        candidate_labels,
        candidate_weights,
        correction_targets,
        sample_report,
    ) = (
        candidate_training_matrix(
            train,
            true_candidate_states,
            hard_negatives=int(args.hard_negatives),
            spread_negatives=int(args.spread_negatives),
            seed=int(args.seed) + 704,
        )
    )
    candidate_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.07,
        max_iter=int(args.candidate_max_iter),
        max_leaf_nodes=127,
        min_samples_leaf=50,
        l2_regularization=0.10,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=20,
        tol=1e-6,
        random_state=int(args.seed) + 705,
    )
    candidate_started = time.perf_counter()
    candidate_model.fit(
        candidate_features,
        candidate_labels,
        sample_weight=candidate_weights,
    )
    candidate_seconds = time.perf_counter() - candidate_started
    correction_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.07,
        max_iter=int(args.correction_max_iter),
        max_leaf_nodes=127,
        min_samples_leaf=50,
        l2_regularization=0.10,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=20,
        tol=1e-6,
        random_state=int(args.seed) + 707,
    )
    correction_started = time.perf_counter()
    correction_model.fit(
        candidate_features,
        correction_targets,
        sample_weight=candidate_weights,
    )
    correction_seconds = time.perf_counter() - correction_started

    status_features = status_training_matrix(train)
    status_counts = Counter(int(value) for value in train["statuses"])
    status_weights = np.asarray(
        [
            len(train["statuses"])
            / max(3 * status_counts[int(value)], 1)
            for value in train["statuses"]
        ],
        dtype=np.float32,
    )
    status_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.07,
        max_iter=int(args.status_max_iter),
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=0.10,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=20,
        tol=1e-6,
        random_state=int(args.seed) + 706,
    )
    status_started = time.perf_counter()
    status_model.fit(
        status_features,
        train["statuses"],
        sample_weight=status_weights,
    )
    status_seconds = time.perf_counter() - status_started

    alpha_candidates = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    beta_candidates = (0.0, 0.25, 0.50, 0.75, 1.0)
    blocks = {
        name: predict_block(
            candidate_model,
            correction_model,
            status_model,
            arrays,
            pairs,
            alpha_candidates,
            beta_candidates,
        )
        for name, (_, pairs, arrays) in validation.items()
    }
    candidate_rows = []
    for beta in beta_candidates:
        for alpha in alpha_candidates:
            metrics = {
                name: next(
                    row
                    for row in block["calibration_candidates"]
                    if row["alpha"] == alpha and row["beta"] == beta
                )
                for name, block in blocks.items()
            }
            clean = [
                metrics["iid_clean"]["target_success_feasible"],
                metrics["difficult_clean"]["target_success_feasible"],
            ]
            all_target = [
                row["target_success_feasible"] for row in metrics.values()
            ]
            noisy = [
                metrics["iid_measurement_augmented"][
                    "target_success_feasible"
                ],
                metrics["difficult_measurement_augmented"][
                    "target_success_feasible"
                ],
            ]
            selection_key = (
                min(clean),
                float(np.mean(all_target)),
                min(noisy),
                float(
                    np.mean(
                        [
                            row["minimum_movement_exact_feasible"]
                            for row in metrics.values()
                        ]
                    )
                ),
                -float(alpha),
                -float(beta),
            )
            candidate_rows.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "selection_key": list(selection_key),
                    "metrics": metrics,
                }
            )
    selected_alpha = max(
        candidate_rows,
        key=lambda row: tuple(row["selection_key"]),
    )

    artifact = {
        "version": "control_rebuild_v5_one_seed",
        "model": "candidate_feasibility_inverse_tree_v5",
        "seed": int(args.seed),
        "candidate_model": candidate_model,
        "cost_correction_model": correction_model,
        "status_model": status_model,
        "candidate_logit_alpha": selected_alpha["alpha"],
        "cost_correction_beta": selected_alpha["beta"],
        "candidate_input_dim": int(candidate_features.shape[1]),
        "status_input_dim": int(status_features.shape[1]),
        "statuses": [
            "unique",
            "ambiguous",
            "infeasible_within_limits",
        ],
        "action_grid": ACTION_GRID,
        "forward_artifact": str(args.forward_artifact.resolve()),
        "measurement_error_bank": str(args.error_bank.resolve()),
        "measurement_augmented_fraction": 0.5,
        "held_out_test_used": False,
    }
    runtime = InverseTreeRuntimeV5(artifact)
    artifact_path = output_dir / "inverse_tree_v5.pkl"
    with artifact_path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v4_inverse, _ = load_inverse_runtime_v4(
        args.v4_inverse.resolve(),
        torch,
        device,
    )
    v4_reference = {
        name: reference_v4(v4_inverse, rows, pairs, arrays)
        for name, (rows, pairs, arrays) in validation.items()
    }
    runtime_validation = {}
    for name, (rows, pairs, arrays) in validation.items():
        setups = [
            rows[int(index)]["setup"] for index in arrays["group_indices"]
        ]
        output = runtime.score_requests(
            setups,
            arrays["observed_current"],
            arrays["desired_observed"],
            arrays["candidate_bank"][arrays["candidate_indices"]],
        )
        runtime_validation[name] = inverse_metrics(
            output["scores"],
            output["status_logits"],
            arrays["positives"],
            arrays["statuses"],
            selected_truth(pairs),
        )

    summary = {
        "version": artifact["version"],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "forward_artifact": str(args.forward_artifact.resolve()),
        "seed": int(args.seed),
        "training": {
            "old_pairs": len(old_train_pairs),
            "difficult_pairs": len(difficult_train_pairs),
            "additional_pairs": len(additional_pairs),
            "total_unique_pairs": len(train_pairs),
            "measurement_augmented_pairs": int(train["noisy"].sum()),
            "candidate_sample": sample_report,
            "candidate_model_iterations": int(candidate_model.n_iter_),
            "candidate_model_seconds": candidate_seconds,
            "cost_correction_iterations": int(correction_model.n_iter_),
            "cost_correction_seconds": correction_seconds,
            "cost_correction_target_mae": float(
                np.mean(
                    np.abs(
                        correction_model.predict(candidate_features)
                        - correction_targets
                    )
                )
            ),
            "status_class_counts": dict(sorted(status_counts.items())),
            "status_model_iterations": int(status_model.n_iter_),
            "status_model_seconds": status_seconds,
        },
        "selection": {
            "selected_alpha": selected_alpha["alpha"],
            "selected_beta": selected_alpha["beta"],
            "selected_metrics": selected_alpha["metrics"],
            "calibration_candidates": candidate_rows,
        },
        "validation": {
            "forward_only": {
                name: next(
                    row
                    for row in block["calibration_candidates"]
                    if row["alpha"] == 0.0 and row["beta"] == 0.0
                )
                for name, block in blocks.items()
            },
            "inverse_v4_with_forward_v5": v4_reference,
            "inverse_tree_v5": runtime_validation,
        },
        "source_contract": {
            "old_train": str(args.old_data.resolve() / "inverse/train.jsonl"),
            "old_val": str(args.old_data.resolve() / "inverse/val.jsonl"),
            "difficult_train": str(
                args.difficult_data.resolve() / "grids/train.jsonl"
            ),
            "difficult_val": str(
                args.difficult_data.resolve() / "grids/val.jsonl"
            ),
            "additional_train": (
                str(args.additional_data.resolve() / "grids/train.jsonl")
                if args.include_additional_data
                else None
            ),
            "held_out_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "inverse_tree_v5_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "summary": str(summary_path),
                "selected_alpha": selected_alpha["alpha"],
                "selected_beta": selected_alpha["beta"],
                "validation": runtime_validation,
                "seconds": summary["seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
