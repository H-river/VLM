"""Bounded continuous CEM model-predictive control with replanning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from continuous_control_v12.cem_diversity import (
    mean_pairwise_normalized_distance,
    minimum_pairwise_normalized_distance,
    select_diverse_elite_indices,
)
from continuous_control_v12.cem_feasible_proposal import (
    sample_truncated_normal_by_resampling,
)
from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    POSITION_FIELDS,
    Bounds,
    action_dict,
    apply_action,
    assert_no_q_star,
    metrics_dict,
    metrics_vector,
    normalized_distance,
    position_dict,
    position_vector,
    project_action,
    tolerance_vector,
)
from continuous_control_v12.simulator import simulate_state
from continuous_control_v12.world_model import ForwardEnsemble

Predictor = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]],
]


def allowed_dof_mask(allowed_dofs: Sequence[str]) -> np.ndarray:
    names = ("lens_x", "lens_y", "camera_x", "camera_y")
    unknown = set(allowed_dofs) - set(names)
    if unknown:
        raise ValueError(f"unknown allowed DOFs: {sorted(unknown)}")
    return np.asarray([name in set(allowed_dofs) for name in names], dtype=bool)


class CEMMPC:
    """Plan action sequences, execute the first action, then replan."""

    def __init__(
        self,
        *,
        bounds: Bounds,
        predictor: Predictor,
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        self.bounds = bounds
        self.predictor = predictor
        self.config = dict(config)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # Candidate-audit sampling must never perturb the CEM proposal stream.
        self.audit_rng = np.random.default_rng(
            np.uint64(self.seed) ^ np.uint64(0xA5A5A5A55A5A5A5A)
        )

    def plan(
        self,
        *,
        positions_mm: Mapping[str, Any] | Sequence[float],
        current_metrics: Mapping[str, Any] | Sequence[float],
        target_metrics: Mapping[str, Any] | Sequence[float],
        allowed_dofs: Sequence[str],
        tolerance_reference: Mapping[str, Any] | Sequence[float] | None = None,
    ) -> dict[str, Any]:
        observation = {
            "positions_mm": positions_mm,
            "current_metrics": current_metrics,
            "target_metrics": target_metrics,
            "allowed_dofs": list(allowed_dofs),
        }
        assert_no_q_star(observation)
        positions = position_vector(positions_mm)
        current = metrics_vector(current_metrics)
        target = metrics_vector(target_metrics)
        reference = (
            current
            if tolerance_reference is None
            else metrics_vector(tolerance_reference)
        )
        if normalized_distance(current, target, reference) <= 1.0:
            return {
                "selected_action": action_dict(np.zeros(4)),
                "selected_requested_action": action_dict(np.zeros(4)),
                "selected_effective_action": action_dict(np.zeros(4)),
                "predicted_next_metrics": metrics_dict(current),
                "predicted_normalized_target_distance": normalized_distance(
                    current, target, reference
                ),
                "uncertainty": [0.0] * 5,
                "current_best_reachable_profile": metrics_dict(current),
                "termination_reason": "reached",
                "predicted_terminal_metrics": metrics_dict(current),
                "predicted_terminal_distance": normalized_distance(
                    current, target, reference
                ),
                "planned_sequence": [action_dict(np.zeros(4))],
                "planned_requested_sequence": [action_dict(np.zeros(4))],
                "planned_effective_sequence": [action_dict(np.zeros(4))],
                "predicted_rollout_metrics": [metrics_dict(current)],
                "rollout_backend_calls": 0,
                "candidate_sequences_evaluated": 0,
                "iteration_history": [],
                "candidate_audit": [],
            }
        horizon = int(self.config["horizon"])
        population = int(self.config["population"])
        elites = int(self.config["elites"])
        iterations = int(self.config["cem_iterations"])
        if not (1 <= elites < population):
            raise ValueError("CEM elites must be between 1 and population-1")
        elite_minimum_distance = float(
            self.config.get("elite_min_normalized_distance", 0.0)
        )
        if elite_minimum_distance < 0:
            raise ValueError("elite minimum normalized distance must be nonnegative")
        feasible_resample_attempts = int(
            self.config.get("feasible_proposal_resample_attempts", 0)
        )
        if feasible_resample_attempts < 0:
            raise ValueError("feasible proposal resample attempts must be nonnegative")
        if feasible_resample_attempts > 0 and horizon != 1:
            raise ValueError("feasible proposal resampling currently requires horizon one")
        dof_mask = allowed_dof_mask(allowed_dofs)
        mean = np.zeros((horizon, 4), dtype=np.float64)
        std = np.broadcast_to(
            self.bounds.action_high[None, :] * 0.75, (horizon, 4)
        ).copy()
        mean[:, ~dof_mask] = 0.0
        std[:, ~dof_mask] = 0.0
        best: dict[str, Any] | None = None
        rollout_backend_calls = 0
        iteration_history: list[dict[str, Any]] = []
        final_population: dict[str, np.ndarray] | None = None
        for iteration in range(iterations):
            feasible_low = np.broadcast_to(
                np.maximum(
                    self.bounds.action_low,
                    self.bounds.position_low - positions,
                )[None, :],
                (horizon, 4),
            ).copy()
            feasible_high = np.broadcast_to(
                np.minimum(
                    self.bounds.action_high,
                    self.bounds.position_high - positions,
                )[None, :],
                (horizon, 4),
            ).copy()
            feasible_low[:, ~dof_mask] = 0.0
            feasible_high[:, ~dof_mask] = 0.0
            if feasible_resample_attempts == 0:
                requested_sequences = self.rng.normal(
                    mean[None, :, :], std[None, :, :], size=(population, horizon, 4)
                )
                initial_feasible_invalid = (
                    (requested_sequences < feasible_low[None, :, :])
                    | (requested_sequences > feasible_high[None, :, :])
                )
                proposal_diagnostic: dict[str, float | int] = {
                    "initially_out_of_feasible_bounds_fraction": float(
                        np.mean(initial_feasible_invalid)
                    ),
                    "remaining_out_of_feasible_bounds_before_clip_fraction": float(
                        np.mean(initial_feasible_invalid)
                    ),
                    "resample_attempts_requested": 0,
                    "resample_attempts_used": 0,
                    "scalar_draws_resampled": 0,
                }
                requested_sequences = np.clip(
                    requested_sequences,
                    self.bounds.action_low[None, None, :],
                    self.bounds.action_high[None, None, :],
                )
            else:
                requested_sequences, proposal_diagnostic = (
                    sample_truncated_normal_by_resampling(
                        self.rng,
                        mean,
                        std,
                        population=population,
                        lower=feasible_low,
                        upper=feasible_high,
                        resample_attempts=feasible_resample_attempts,
                    )
                )
            requested_sequences[:, :, ~dof_mask] = 0.0
            requested_sequences[0] = 0.0
            injected = 1
            for axis in range(4):
                if not dof_mask[axis]:
                    continue
                for sign in (-1.0, 1.0):
                    if injected >= population:
                        break
                    requested_sequences[injected] = 0.0
                    requested_sequences[injected, 0, axis] = (
                        feasible_low[0, axis]
                        if feasible_resample_attempts > 0 and sign < 0
                        else feasible_high[0, axis]
                        if feasible_resample_attempts > 0
                        else sign * self.bounds.action_high[axis]
                    )
                    injected += 1
            sequences = requested_sequences.copy()
            scores = np.zeros(population, dtype=np.float64)
            terminal_metrics = np.empty((population, 5), dtype=np.float64)
            terminal_positions = np.empty((population, 4), dtype=np.float64)
            first_next_metrics = np.empty((population, 5), dtype=np.float64)
            predicted_rollout_metrics = np.empty(
                (population, horizon, 5), dtype=np.float64
            )
            total_uncertainty = np.zeros((population, 5), dtype=np.float64)
            limit_projection = np.zeros(population, dtype=bool)
            best_profiles = np.empty((population, 5), dtype=np.float64)
            best_distances = np.full(population, np.inf, dtype=np.float64)
            for candidate in range(population):
                candidate_positions = positions.copy()
                candidate_metrics = current.copy()
                movement = 0.0
                boundary_cost = 0.0
                uncertainty_cost = 0.0
                intermediate = 0.0
                for step in range(horizon):
                    requested = sequences[candidate, step]
                    action = project_action(
                        candidate_positions, requested, self.bounds
                    )
                    if not np.allclose(action, requested, atol=1e-12, rtol=0.0):
                        limit_projection[candidate] = True
                    sequences[candidate, step] = action
                    next_positions = apply_action(
                        candidate_positions, action, self.bounds
                    )
                    next_metrics, uncertainty, auxiliary = self.predictor(
                        candidate_positions, candidate_metrics, action
                    )
                    rollout_backend_calls += 1
                    predicted_rollout_metrics[candidate, step] = next_metrics
                    if step == 0:
                        first_next_metrics[candidate] = next_metrics
                    distance = normalized_distance(
                        next_metrics, target, reference
                    )
                    if distance < best_distances[candidate]:
                        best_distances[candidate] = distance
                        best_profiles[candidate] = next_metrics
                    intermediate += distance / horizon
                    movement += float(np.abs(action / self.bounds.action_high).sum())
                    uncertainty_cost += float(np.mean(uncertainty))
                    total_uncertainty[candidate] += uncertainty
                    boundary_cost += float(
                        auxiliary.get("clipping_probability", np.asarray([0.0]))[0]
                    )
                    boundary_cost += float(
                        auxiliary.get("boundary_probability", np.asarray([0.0]))[0]
                    )
                    candidate_positions = next_positions
                    candidate_metrics = next_metrics
                terminal_metrics[candidate] = candidate_metrics
                terminal_positions[candidate] = candidate_positions
                normalized = np.abs(candidate_metrics - target) / tolerance_vector(
                    reference
                )
                terminal_max = float(normalized.max())
                terminal_mean = float(normalized.mean())
                scores[candidate] = (
                    terminal_max
                    + float(self.config["mean_error_weight"]) * terminal_mean
                    + 0.10 * intermediate
                    + float(self.config["movement_weight"]) * movement
                    + float(self.config["boundary_penalty"]) * boundary_cost
                    + float(self.config["uncertainty_weight"]) * uncertainty_cost
                    + float(self.config["limit_penalty"])
                    * float(limit_projection[candidate])
                )
            if elite_minimum_distance == 0.0:
                # Preserve the frozen/default ordering path byte-for-byte.
                elite_indices = np.argsort(scores)[:elites]
            else:
                elite_indices = select_diverse_elite_indices(
                    scores,
                    sequences,
                    elites=elites,
                    action_scale=self.bounds.action_high,
                    minimum_normalized_distance=elite_minimum_distance,
                )
            elite_sequences = sequences[elite_indices]
            elite_mean_distance = mean_pairwise_normalized_distance(
                sequences, elite_indices, self.bounds.action_high
            )
            elite_minimum_observed_distance = minimum_pairwise_normalized_distance(
                sequences, elite_indices, self.bounds.action_high
            )
            mean = elite_sequences.mean(axis=0)
            std = np.maximum(
                elite_sequences.std(axis=0), self.bounds.action_high * 0.03
            )
            mean[:, ~dof_mask] = 0.0
            std[:, ~dof_mask] = 0.0
            best_index = int(elite_indices[0])
            iteration_history.append(
                {
                    "iteration": iteration + 1,
                    "best_score": float(scores[best_index]),
                    "best_terminal_distance": normalized_distance(
                        terminal_metrics[best_index], target, reference
                    ),
                    "median_terminal_distance": float(
                        np.median(
                            [
                                normalized_distance(row, target, reference)
                                for row in terminal_metrics
                            ]
                        )
                    ),
                    "best_first_step_distance": normalized_distance(
                        first_next_metrics[best_index], target, reference
                    ),
                    "elite_score_mean": float(scores[elite_indices].mean()),
                    "elite_mean_pairwise_normalized_action_distance": float(
                        elite_mean_distance
                    ),
                    "elite_minimum_pairwise_normalized_action_distance": float(
                        elite_minimum_observed_distance
                    ),
                    "elite_minimum_normalized_distance_constraint": float(
                        elite_minimum_distance
                    ),
                    "diversity_constraint_satisfied": bool(
                        elite_minimum_distance == 0.0
                        or elite_minimum_observed_distance + 1e-12
                        >= elite_minimum_distance
                    ),
                    "proposal_initially_out_of_feasible_bounds_fraction": float(
                        proposal_diagnostic[
                            "initially_out_of_feasible_bounds_fraction"
                        ]
                    ),
                    "proposal_remaining_out_of_feasible_bounds_before_clip_fraction": float(
                        proposal_diagnostic[
                            "remaining_out_of_feasible_bounds_before_clip_fraction"
                        ]
                    ),
                    "proposal_resample_attempts_used": int(
                        proposal_diagnostic["resample_attempts_used"]
                    ),
                    "population_limit_projection_rate": float(
                        np.mean(limit_projection)
                    ),
                    "unique_effective_sequence_fraction": float(
                        len(np.unique(sequences.reshape(population, -1), axis=0))
                        / population
                    ),
                }
            )
            candidate_best = {
                "score": float(scores[best_index]),
                "requested_sequence": requested_sequences[best_index].copy(),
                "sequence": sequences[best_index].copy(),
                "first_next_metrics": first_next_metrics[best_index].copy(),
                "predicted_rollout_metrics": predicted_rollout_metrics[
                    best_index
                ].copy(),
                "terminal_metrics": terminal_metrics[best_index].copy(),
                "terminal_positions": terminal_positions[best_index].copy(),
                "uncertainty": (
                    total_uncertainty[best_index] / max(horizon, 1)
                ).copy(),
                "best_profile": best_profiles[best_index].copy(),
                "best_distance": float(best_distances[best_index]),
                "terminal_distance": normalized_distance(
                    terminal_metrics[best_index], target, reference
                ),
                "actuator_limited": bool(limit_projection[best_index]),
            }
            if best is None or candidate_best["score"] < best["score"]:
                best = candidate_best
            if iteration == iterations - 1:
                final_population = {
                    "requested_sequences": requested_sequences.copy(),
                    "effective_sequences": sequences.copy(),
                    "predicted_rollout_metrics": (
                        predicted_rollout_metrics.copy()
                    ),
                    "terminal_metrics": terminal_metrics.copy(),
                    "scores": scores.copy(),
                }
        if best is None:
            raise RuntimeError("CEM did not evaluate any candidate")
        candidate_audit: list[dict[str, Any]] = []
        top_k = int(self.config.get("candidate_audit_top_k", 0))
        reference_k = int(
            self.config.get("candidate_audit_reference_k", 0)
        )
        if final_population is not None and (top_k > 0 or reference_k > 0):
            ranked = np.argsort(final_population["scores"])
            top_indices = ranked[: min(top_k, population)]
            remaining = np.setdiff1d(
                np.arange(population, dtype=np.int64),
                top_indices,
                assume_unique=True,
            )
            if reference_k > len(remaining):
                raise ValueError(
                    "candidate audit needs population >= top_k + reference_k"
                )
            reference_indices = (
                np.asarray([], dtype=np.int64)
                if reference_k == 0
                else np.sort(
                    self.audit_rng.choice(
                        remaining,
                        size=reference_k,
                        replace=False,
                    )
                )
            )
            initial_distance = normalized_distance(
                current, target, reference
            )
            for selection, indices in (
                ("predicted_top", top_indices),
                ("deterministic_reference", reference_indices),
            ):
                for rank, candidate_index in enumerate(indices):
                    terminal = final_population["terminal_metrics"][
                        candidate_index
                    ]
                    terminal_distance = normalized_distance(
                        terminal, target, reference
                    )
                    candidate_audit.append(
                        {
                            "selection": selection,
                            "selection_rank": rank + 1,
                            "candidate_index": int(candidate_index),
                            "predicted_score": float(
                                final_population["scores"][candidate_index]
                            ),
                            "predicted_terminal_target_cost": float(
                                terminal_distance
                            ),
                            "predicted_terminal_improvement": float(
                                initial_distance - terminal_distance
                            ),
                            "requested_sequence": [
                                action_dict(action)
                                for action in final_population[
                                    "requested_sequences"
                                ][candidate_index]
                            ],
                            "effective_sequence": [
                                action_dict(action)
                                for action in final_population[
                                    "effective_sequences"
                                ][candidate_index]
                            ],
                            "predicted_rollout_metrics": [
                                metrics_dict(row)
                                for row in final_population[
                                    "predicted_rollout_metrics"
                                ][candidate_index]
                            ],
                        }
                    )
        if best["actuator_limited"]:
            reason = "actuator_limited"
        elif best["terminal_distance"] <= 1.0:
            reason = "planning"
        elif best["terminal_distance"] > 1.2 and float(
            np.mean(best["uncertainty"])
        ) < 0.5:
            reason = "likely_unreachable"
        else:
            reason = "best_effort"
        return {
            "selected_action": action_dict(best["sequence"][0]),
            "selected_requested_action": action_dict(
                best["requested_sequence"][0]
            ),
            "selected_effective_action": action_dict(best["sequence"][0]),
            "predicted_next_metrics": metrics_dict(best["first_next_metrics"]),
            "predicted_normalized_target_distance": normalized_distance(
                best["first_next_metrics"], target, reference
            ),
            "uncertainty": best["uncertainty"].tolist(),
            "current_best_reachable_profile": metrics_dict(best["best_profile"]),
            "termination_reason": reason,
            "predicted_terminal_metrics": metrics_dict(best["terminal_metrics"]),
            "predicted_terminal_distance": float(best["terminal_distance"]),
            "planned_sequence": [
                action_dict(action) for action in best["sequence"]
            ],
            "planned_requested_sequence": [
                action_dict(action) for action in best["requested_sequence"]
            ],
            "planned_effective_sequence": [
                action_dict(action) for action in best["sequence"]
            ],
            "predicted_rollout_metrics": [
                metrics_dict(row) for row in best["predicted_rollout_metrics"]
            ],
            "rollout_backend_calls": int(rollout_backend_calls),
            "candidate_sequences_evaluated": int(population * iterations),
            "iteration_history": iteration_history,
            "candidate_audit": candidate_audit,
        }


def learned_predictor(
    model: ForwardEnsemble,
    setup_context: Mapping[str, Any],
) -> Predictor:
    if model.image_conditioning:
        raise ValueError(
            "image-conditioned one-step ablation is not enabled for imagined "
            "multi-step rollouts because v12 does not predict future images"
        )
    def predict(
        positions: np.ndarray,
        metrics: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        result = model.predict(
            setup_context, positions, metrics, action[None, :]
        )
        auxiliary = result["auxiliary_predictions"]
        return (
            result["predicted_next_metrics"][0],
            result["uncertainty"][0],
            {
                "clipping_probability": np.asarray(
                    [auxiliary["clipping_fraction"][0]]
                ),
                "boundary_probability": np.asarray(
                    [auxiliary["camera_boundary_probability"][0]]
                ),
                "actuator_limit_probability": np.asarray(
                    [auxiliary["actuator_limit_probability"][0]]
                ),
            },
        )

    return predict


def simulator_predictor(
    *,
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    base_config_path: str,
    bounds: Bounds,
) -> Predictor:
    def predict(
        positions: np.ndarray,
        metrics: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        next_positions = apply_action(positions, action, bounds)
        capture = simulate_state(
            setup_context,
            position_dict(next_positions),
            simulator_fixed,
            base_config_path,
            bounds,
        )
        auxiliary = capture["auxiliary"]
        return (
            metrics_vector(capture["metrics"]),
            np.zeros(5, dtype=np.float64),
            {
                "clipping_probability": np.asarray(
                    [float(auxiliary["clipping_fraction"] or 0.0)]
                ),
                "boundary_probability": np.asarray(
                    [float(bool(auxiliary["camera_boundary_indicator"]))]
                ),
                "actuator_limit_probability": np.asarray(
                    [float(bool(auxiliary["actuator_limit_indicator"]))]
                ),
            },
        )

    return predict


def run_closed_loop(
    *,
    planner: CEMMPC,
    setup_context: Mapping[str, Any],
    simulator_fixed: Mapping[str, Any],
    initial_positions_mm: Mapping[str, Any],
    initial_metrics: Mapping[str, Any],
    target_metrics: Mapping[str, Any],
    allowed_dofs: Sequence[str],
    base_config_path: str,
    bounds: Bounds,
    max_steps: int,
) -> dict[str, Any]:
    """Execute one action at a time against the simulator and replan."""

    positions = position_vector(initial_positions_mm)
    metrics = metrics_vector(initial_metrics)
    target = metrics_vector(target_metrics)
    reference = metrics.copy()
    trace = []
    cumulative = 0.0
    final_reason = "best_effort"
    for step in range(max_steps):
        before = normalized_distance(metrics, target, reference)
        plan = planner.plan(
            positions_mm=positions,
            current_metrics=metrics,
            target_metrics=target,
            allowed_dofs=allowed_dofs,
            tolerance_reference=reference,
        )
        if plan["termination_reason"] == "reached":
            final_reason = "reached"
            break
        requested_action = np.asarray(
            [
                plan.get(
                    "selected_requested_action", plan["selected_action"]
                )[field]
                for field in ACTION_FIELDS
            ],
            dtype=np.float64,
        )
        effective_action = project_action(
            positions, requested_action, bounds
        )
        next_positions = apply_action(positions, effective_action, bounds)
        observed = simulate_state(
            setup_context,
            position_dict(next_positions),
            simulator_fixed,
            base_config_path,
            bounds,
        )
        next_metrics = metrics_vector(observed["metrics"])
        after = normalized_distance(next_metrics, target, reference)
        predicted_after = float(
            plan["predicted_normalized_target_distance"]
        )
        predicted_improvement = float(before - predicted_after)
        actual_improvement = float(before - after)
        projected_action = project_action(
            positions, effective_action, bounds
        )
        illegal_action = bool(
            np.any(~np.isfinite(requested_action))
            or np.any(~np.isfinite(effective_action))
            or not np.allclose(
                effective_action,
                projected_action,
                atol=1e-12,
                rtol=0.0,
            )
        )
        cumulative += float(np.abs(effective_action).sum())
        trace.append(
            {
                "step": step,
                "before_normalized_distance": before,
                "predicted_after_normalized_distance": predicted_after,
                "after_normalized_distance": after,
                "target_cost_definition": (
                    "max_absolute_error_in_initial_state_tolerances"
                ),
                "predicted_target_cost": predicted_after,
                "actual_simulator_target_cost": after,
                "predicted_versus_actual_target_cost_gap": float(
                    predicted_after - after
                ),
                "predicted_improvement": predicted_improvement,
                "actual_simulator_improvement": actual_improvement,
                "predicted_versus_actual_distance_gap": float(
                    predicted_after - after
                ),
                "planner_exploitation_event": bool(
                    predicted_improvement > 0.0 and actual_improvement < 0.0
                ),
                "illegal_action": illegal_action,
                "requested_action_mm": action_dict(requested_action),
                "effective_action_mm": action_dict(effective_action),
                "action_mm": action_dict(effective_action),
                "positions_mm": position_dict(next_positions),
                "observed_metrics": metrics_dict(next_metrics),
                "planner": plan,
                "auxiliary": observed["auxiliary"],
            }
        )
        positions = next_positions
        metrics = next_metrics
        if not bool(observed["auxiliary"]["simulator_valid"]):
            final_reason = "invalid_simulation"
            break
        if after <= 1.0:
            final_reason = "reached"
            break
        final_reason = str(plan["termination_reason"])
    return {
        "status": final_reason,
        "success": bool(normalized_distance(metrics, target, reference) <= 1.0),
        "steps": len(trace),
        "initial_normalized_distance": normalized_distance(
            initial_metrics, target, reference
        ),
        "final_normalized_distance": normalized_distance(
            metrics, target, reference
        ),
        "final_positions_mm": position_dict(positions),
        "final_metrics": metrics_dict(metrics),
        "current_best_reachable_profile": (
            trace[-1]["planner"]["current_best_reachable_profile"]
            if trace
            else metrics_dict(metrics)
        ),
        "cumulative_actuator_movement_mm": cumulative,
        "planner_exploitation_events": int(
            sum(bool(row["planner_exploitation_event"]) for row in trace)
        ),
        "planner_predicted_improvement_steps": int(
            sum(float(row["predicted_improvement"]) > 0.0 for row in trace)
        ),
        "illegal_actions": int(sum(bool(row["illegal_action"]) for row in trace)),
        "trace": trace,
    }
