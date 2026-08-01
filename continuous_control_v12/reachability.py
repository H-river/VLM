"""Estimated continuous and exact legacy-grid reachability diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from continuous_control_v12.contracts import (
    LEGACY_ACTIONS,
    OUTPUT_FIELDS,
    Bounds,
    action_dict,
    apply_action,
    legacy_action_array,
    metrics_vector,
    normalized_distance,
    position_dict,
    position_vector,
)
from continuous_control_v12.simulator import (
    build_optical_setup,
    is_corrected_semantics,
    simulate_state,
)
from control_rebuild_v5.simulator_grid import simulate_fixed_action_grid


def classify_estimated_reachability(
    distances: np.ndarray | list[float],
    *,
    reachable_below: float = 0.8,
    infeasible_above: float = 1.2,
    agreement_tolerance: float = 0.05,
) -> tuple[str, bool]:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("oracle distances must be a finite non-empty vector")
    repeated_agreement = bool(
        len(values) >= 3
        and np.all(values > infeasible_above)
        and float(values.max() - values.min()) <= agreement_tolerance
    )
    if float(values.min()) < reachable_below:
        return "clearly_reachable", repeated_agreement
    if float(values.min()) > infeasible_above and repeated_agreement:
        return "candidate_infeasible", repeated_agreement
    return "ambiguous_boundary", repeated_agreement


def estimate_continuous_oracle(
    *,
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    current_positions_mm: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    target_metrics: Mapping[str, Any],
    base_config_path: str,
    bounds: Bounds,
    seed: int,
    multistarts: int,
    max_iterations: int,
    reachable_below: float = 0.8,
    infeasible_above: float = 1.2,
    agreement_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Multi-start bounded search; never claims a proof of infeasibility."""

    if multistarts < 1 or max_iterations < 1:
        raise ValueError("oracle multistarts/iterations must be positive")
    current = position_vector(current_positions_mm)
    engine = qmc.Sobol(d=4, scramble=True, seed=int(seed % (2**32)))
    exponent = max(0, int(np.ceil(np.log2(max(multistarts - 1, 1)))))
    starts = [current]
    if multistarts > 1:
        candidates = qmc.scale(
            engine.random_base2(exponent),
            bounds.position_low,
            bounds.position_high,
        )
        starts.extend(candidates[: multistarts - 1])
    evaluations = 0
    cache: dict[tuple[float, ...], tuple[float, dict[str, Any]]] = {}

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        clipped = np.clip(values, bounds.position_low, bounds.position_high)
        key = tuple(np.round(clipped, 10).tolist())
        cached = cache.get(key)
        if cached is not None:
            return cached[0]
        capture = simulate_state(
            setup_context,
            position_dict(clipped),
            simulator_fixed,
            base_config_path,
            bounds,
        )
        value = normalized_distance(
            capture["metrics"], target_metrics, current_metrics
        )
        evaluations += 1
        cache[key] = (value, capture)
        return value

    results = []
    for start_index, start in enumerate(starts):
        result = minimize(
            objective,
            x0=np.asarray(start, dtype=np.float64),
            method="Powell",
            bounds=list(zip(bounds.position_low, bounds.position_high, strict=True)),
            options={
                "maxiter": int(max_iterations),
                "xtol": 1e-5,
                "ftol": 1e-5,
            },
        )
        position = np.clip(result.x, bounds.position_low, bounds.position_high)
        distance = objective(position)
        results.append(
            {
                "start_index": start_index,
                "distance": float(distance),
                "position_mm": position_dict(position),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
            }
        )
    ordered = sorted(results, key=lambda item: item["distance"])
    best = ordered[0]
    distances = np.asarray([item["distance"] for item in results], dtype=np.float64)
    label, repeated_agreement = classify_estimated_reachability(
        distances,
        reachable_below=reachable_below,
        infeasible_above=infeasible_above,
        agreement_tolerance=agreement_tolerance,
    )
    return {
        "kind": "estimated_continuous_oracle",
        "label": label,
        "best_distance": float(best["distance"]),
        "best_position_mm": best["position_mm"],
        "best_reachable_profile": cache[
            tuple(
                np.round(position_vector(best["position_mm"]), 10).tolist()
            )
        ][1]["metrics"],
        "multistarts": int(multistarts),
        "max_iterations": int(max_iterations),
        "evaluations": evaluations,
        "repeated_optimizer_agreement": repeated_agreement,
        "optimizer_runs": results,
        "claim": "estimated_reachability_not_physical_infeasibility_proof",
    }


def discrete_81_oracle(
    *,
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    current_positions_mm: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    target_metrics: Mapping[str, Any],
    base_config_path: str,
    bounds: Bounds,
) -> dict[str, Any]:
    """Evaluate the canonical grid in its exact historical order."""

    corrected = is_corrected_semantics(simulator_fixed)
    surface = None
    if not corrected:
        setup = build_optical_setup(
            setup_context,
            current_positions_mm,
            simulator_fixed,
            base_config_path,
        )
        surface = simulate_fixed_action_grid(setup)
    actions = legacy_action_array()
    current = position_vector(current_positions_mm)
    legal = np.ones(len(actions), dtype=bool)
    distances = np.full(len(actions), np.inf, dtype=np.float64)
    states: list[dict[str, float] | None] = [None] * len(actions)
    for index, action in enumerate(actions):
        try:
            next_positions = apply_action(current, action, bounds)
        except ValueError:
            legal[index] = False
            continue
        if corrected:
            capture = simulate_state(
                setup_context,
                position_dict(next_positions),
                simulator_fixed,
                base_config_path,
                bounds,
            )
            state = {
                field: float(capture["metrics"][field])
                for field in OUTPUT_FIELDS
            }
        else:
            if surface is None:
                raise AssertionError("legacy action surface was not constructed")
            state = {
                field: float(surface[index]["state"][field])
                for field in OUTPUT_FIELDS
            }
        states[index] = state
        distances[index] = normalized_distance(
            state, target_metrics, current_metrics
        )
    if not legal.any():
        raise ValueError("no legal legacy actions at current position")
    best_index = min(
        np.flatnonzero(legal),
        key=lambda index: (
            float(distances[index]),
            float(np.abs(actions[index]).sum()),
            int(index),
        ),
    )
    successful = legal & (distances <= 1.0)
    return {
        "kind": "discrete_81_oracle",
        "successful_legacy_actions": int(successful.sum()),
        "best_legacy_action_index": int(best_index),
        "best_legacy_action": action_dict(actions[best_index]),
        "best_distance": float(distances[best_index]),
        "best_reachable_profile": states[best_index],
        "legal_action_count": int(legal.sum()),
        "canonical_action_count": len(LEGACY_ACTIONS),
    }


def oracle_gap(
    continuous: Mapping[str, Any],
    discrete: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "continuous_distance": float(continuous["best_distance"]),
        "discrete_81_distance": float(discrete["best_distance"]),
        "continuous_minus_discrete": float(
            continuous["best_distance"] - discrete["best_distance"]
        ),
        "discrete_minus_continuous": float(
            discrete["best_distance"] - continuous["best_distance"]
        ),
    }
