"""Group-level v12 forward/controller metrics and bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from continuous_control_v12.contracts import OUTPUT_FIELDS


def forward_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    group_ids: Sequence[str],
    regimes: Sequence[str] | None = None,
    sampling: Sequence[str] | None = None,
    uncertainty: np.ndarray | None = None,
    no_op: np.ndarray | None = None,
) -> dict[str, Any]:
    error = np.abs(
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(target, dtype=np.float64)
    )
    strict = np.all(error <= 1.0, axis=1)

    def block(mask: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(mask.sum()),
            "normalized_mae": float(error[mask].mean()),
            "strict_all_five_accuracy": float(strict[mask].mean()),
        }

    result: dict[str, Any] = {
        "groups": len(set(map(str, group_ids))),
        "transitions": len(error),
        "normalized_mae": float(error.mean()),
        "per_output_normalized_mae": {
            field: float(error[:, index].mean())
            for index, field in enumerate(OUTPUT_FIELDS)
        },
        "per_output_tolerance_accuracy": {
            field: float((error[:, index] <= 1.0).mean())
            for index, field in enumerate(OUTPUT_FIELDS)
        },
        "strict_all_five_accuracy": float(strict.mean()),
    }
    if regimes is not None:
        values = np.asarray(regimes, dtype=np.str_)
        result["by_regime"] = {
            name: block(values == name) for name in sorted(set(values.tolist()))
        }
    if sampling is not None:
        values = np.asarray(sampling, dtype=np.str_)
        result["by_sampling_kind"] = {
            name: block(values == name) for name in sorted(set(values.tolist()))
        }
    if no_op is not None and np.asarray(no_op).any():
        mask = np.asarray(no_op, dtype=bool)
        result["no_op_consistency"] = {
            "count": int(mask.sum()),
            "max_abs_normalized_residual": float(
                np.abs(np.asarray(prediction)[mask]).max()
            ),
            "passes_one_tolerance": bool(
                np.all(np.abs(np.asarray(prediction)[mask]) <= 1.0)
            ),
        }
    if uncertainty is not None:
        sigma = np.maximum(np.asarray(uncertainty, dtype=np.float64), 1e-6)
        result["uncertainty_calibration"] = {
            "one_sigma_coverage": float((error <= sigma).mean()),
            "two_sigma_coverage": float((error <= 2.0 * sigma).mean()),
            "mean_predicted_sigma": float(sigma.mean()),
        }
    return result


def group_bootstrap_ci(
    values: Sequence[float],
    group_ids: Sequence[str],
    *,
    seed: int,
    samples: int = 1000,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    """Bootstrap independent groups; correlated rows remain together."""

    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for group_id, value in zip(group_ids, values, strict=True):
        grouped[str(group_id)].append(float(value))
    groups = sorted(grouped)
    rng = np.random.default_rng(seed)
    observed = float(
        statistic(
            np.concatenate(
                [np.asarray(grouped[group], dtype=np.float64) for group in groups]
            )
        )
    )
    estimates = []
    for _ in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate(
            [np.asarray(grouped[groups[index]], dtype=np.float64) for index in selected]
        )
        estimates.append(float(statistic(rows)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": observed, "low": float(low), "high": float(high)}


def controller_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_regime: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_mode: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_category[str(episode["target_category"])].append(episode)
        by_regime[str(episode.get("regime", "unspecified"))].append(episode)
        by_mode[str(episode.get("mode", "unspecified"))].append(episode)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        regrets = [
            float(row["oracle_regret"])
            for row in rows
            if row.get("oracle_regret") is not None
            and np.isfinite(float(row["oracle_regret"]))
        ]
        successful_steps = [
            float(row["steps"]) for row in rows if bool(row["success"])
        ]
        return {
            "episodes": len(rows),
            "reachable_conditioned_success": float(
                np.mean(
                    [
                        bool(row["success"])
                        for row in rows
                        if row["target_category"]
                        in {"one_step_reachable", "multi_step_reachable"}
                    ]
                    or [False]
                )
            ),
            "strict_all_five_success": float(
                np.mean([bool(row["success"]) for row in rows])
            ),
            "oracle_regret": None if not regrets else float(np.mean(regrets)),
            "final_normalized_distance": float(
                np.mean([float(row["final_normalized_distance"]) for row in rows])
            ),
            "steps_to_success": (
                None
                if not successful_steps
                else float(np.mean(successful_steps))
            ),
            "cumulative_actuator_movement_mm": float(
                np.mean(
                    [
                        float(row["cumulative_actuator_movement_mm"])
                        for row in rows
                    ]
                )
            ),
            "clipping_boundary_failure_rate": float(
                np.mean(
                    [
                        bool(row.get("clipping_or_boundary_failure", False))
                        for row in rows
                    ]
                )
            ),
        }

    return {
        "overall": summarize(episodes),
        "by_target_category": {
            category: summarize(rows)
            for category, rows in sorted(by_category.items())
        },
        "by_regime": {
            regime: summarize(rows) for regime, rows in sorted(by_regime.items())
        },
        "by_mode": {
            mode: summarize(rows) for mode, rows in sorted(by_mode.items())
        },
    }
