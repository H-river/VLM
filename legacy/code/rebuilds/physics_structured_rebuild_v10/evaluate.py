"""Grouped evaluation and bootstrap utilities shared by v10 experiments."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from physics_structured_rebuild_v10.contracts import (
    STATE_FIELDS,
    action_cardinalities,
    stable_seed,
)


def mean_ci(
    values: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        size = min(200, draws - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def paired_difference_ci(
    candidate: np.ndarray,
    baseline: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, float | int]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    result = mean_ci(difference, seed, draws)
    result["candidate_count"] = int(np.asarray(candidate).sum())
    result["baseline_count"] = int(np.asarray(baseline).sum())
    return result


def forward_metrics(
    rows: Sequence[Mapping[str, Any]],
    prediction: np.ndarray,
    target: np.ndarray,
    log_variance: np.ndarray | None,
    bootstrap_seed: int,
    draws: int = 2000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    residual = np.asarray(prediction - target, dtype=np.float64)
    passed = np.abs(residual) <= 1.0
    strict = np.all(passed, axis=2)
    natural_indices = np.asarray(
        [int(row["natural_requested_action_index"]) for row in rows],
        dtype=np.int64,
    )
    group_indices = np.arange(len(rows))
    natural_strict = strict[group_indices, natural_indices]
    natural_fields = passed[group_indices, natural_indices]
    cardinalities = action_cardinalities()
    regime_values: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        regime_values[str(row["regime"])].append(float(natural_strict[index]))
    by_regime = {
        regime: mean_ci(
            np.asarray(values),
            stable_seed(bootstrap_seed, "regime", regime),
            draws,
        )
        for regime, values in sorted(regime_values.items())
    }
    by_cardinality = {}
    for cardinality in range(5):
        mask = cardinalities == cardinality
        group_values = strict[:, mask].mean(axis=1)
        by_cardinality[str(cardinality)] = mean_ci(
            group_values,
            stable_seed(bootstrap_seed, "cardinality", cardinality),
            draws,
        )
    flattened = np.abs(residual).reshape(-1, 5)
    metrics: dict[str, Any] = {
        "natural_requested_action": {
            "strict_all_five": mean_ci(
                natural_strict.astype(np.float64),
                stable_seed(bootstrap_seed, "natural"),
                draws,
            ),
            "strict_count": int(natural_strict.sum()),
            "per_field_tolerance_accuracy": {
                field: float(natural_fields[:, index].mean())
                for index, field in enumerate(STATE_FIELDS)
            },
        },
        "full_81_action_surface": {
            "strict_all_five": mean_ci(
                strict.mean(axis=1),
                stable_seed(bootstrap_seed, "surface"),
                draws,
            ),
            "strict_count": int(strict.sum()),
            "transition_count": int(strict.size),
            "per_field_tolerance_accuracy": {
                field: float(passed[:, :, index].mean())
                for index, field in enumerate(STATE_FIELDS)
            },
        },
        "normalized_absolute_residual_quantiles": {
            field: {
                "p50": float(np.quantile(flattened[:, index], 0.50)),
                "p90": float(np.quantile(flattened[:, index], 0.90)),
                "p95": float(np.quantile(flattened[:, index], 0.95)),
                "p99": float(np.quantile(flattened[:, index], 0.99)),
            }
            for index, field in enumerate(STATE_FIELDS)
        },
        "by_regime_natural_requested_action": by_regime,
        "by_action_cardinality_full_surface": by_cardinality,
    }
    if log_variance is not None:
        standard_deviation = np.exp(
            0.5 * np.asarray(log_variance, dtype=np.float64)
        )
        standardized = np.abs(residual) / np.maximum(standard_deviation, 1e-6)
        metrics["uncertainty"] = {
            "mean_predicted_standard_deviation": {
                field: float(standard_deviation[:, :, index].mean())
                for index, field in enumerate(STATE_FIELDS)
            },
            "coverage_1sigma": {
                field: float((standardized[:, :, index] <= 1.0).mean())
                for index, field in enumerate(STATE_FIELDS)
            },
            "coverage_2sigma": {
                field: float((standardized[:, :, index] <= 2.0).mean())
                for index, field in enumerate(STATE_FIELDS)
            },
        }
    vectors = {
        "natural_strict": natural_strict.astype(np.float64),
        "natural_fields": natural_fields.astype(np.float64),
        "surface_group_success": strict.mean(axis=1),
        "strict_matrix": strict.astype(np.float64),
    }
    return metrics, vectors


def request_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    text = "\n".join(
        f"{row['context_hash']}:{row['natural_requested_action_index']}"
        for row in rows
    )
    return hashlib.sha256(text.encode()).hexdigest()

