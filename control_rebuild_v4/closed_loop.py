"""Bounded closed-loop control with replaceable actuator and observer callbacks."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import residual_cost
from control_rebuild_v4.inverse_data import state_mapping
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    matching_mask,
    raw_state_array,
)

Executor = Callable[
    [Mapping[str, Any], Mapping[str, float]],
    tuple[dict[str, Any], np.ndarray],
]
Observer = Callable[[Mapping[str, Any], np.ndarray], np.ndarray]


def target_reached(state: np.ndarray, desired: np.ndarray) -> bool:
    return bool(
        matching_mask(
            np.asarray(state, dtype=np.float32)[None, :],
            np.asarray(desired, dtype=np.float32),
        )[0]
    )


def target_cost(state: np.ndarray, desired: np.ndarray) -> float:
    return float(
        residual_cost(
            np.asarray(state, dtype=np.float32)[None, :],
            np.asarray(desired, dtype=np.float32)[None, :],
        )[0]
    )


class NumericalClosedLoopControllerV4:
    """Re-plan after each observation instead of issuing an open-loop sequence."""

    def __init__(self, forward: Any, inverse: Any) -> None:
        self.forward = forward
        self.inverse = inverse

    def propose(
        self,
        setup: Mapping[str, Any],
        current_observed: np.ndarray,
        desired_observed: np.ndarray,
        request_id: str,
    ) -> dict[str, Any]:
        row = {
            "group_id": request_id,
            "setup": setup,
            "current_beam_state": state_mapping(current_observed),
        }
        candidates = self.forward.predict_states([row])
        result = self.inverse.score_requests(
            [setup],
            np.asarray(current_observed, dtype=np.float32)[None, :],
            np.asarray(desired_observed, dtype=np.float32)[None, :],
            candidates,
        )
        return {
            "selected_index": int(result["selected_indices"][0]),
            "selected_action": result["selected_actions"][0],
            "predicted_status": result["predicted_statuses"][0],
            "predicted_candidate_states": candidates[0],
            "scores": result["scores"][0],
        }

    def run(
        self,
        setup: Mapping[str, Any],
        current_true: np.ndarray,
        desired_true: np.ndarray,
        executor: Executor,
        observer: Observer | None = None,
        desired_observed: np.ndarray | None = None,
        request_id: str = "closed_loop",
        max_steps: int = 3,
    ) -> dict[str, Any]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        observe = (
            (lambda _setup, state: np.asarray(state, dtype=np.float32))
            if observer is None
            else observer
        )
        setup_out = copy.deepcopy(dict(setup))
        true_state = np.asarray(current_true, dtype=np.float32).copy()
        observed_state = np.asarray(observe(setup_out, true_state), dtype=np.float32)
        desired_truth = np.asarray(desired_true, dtype=np.float32)
        desired_input = (
            desired_truth
            if desired_observed is None
            else np.asarray(desired_observed, dtype=np.float32)
        )
        initial_reached = target_reached(true_state, desired_truth)
        trace = []
        stop_reason = "target_reached" if initial_reached else "step_limit"
        if not initial_reached:
            for step in range(1, max_steps + 1):
                before_cost = target_cost(true_state, desired_truth)
                proposal = self.propose(
                    setup_out,
                    observed_state,
                    desired_input,
                    f"{request_id}:step:{step}",
                )
                setup_out, true_state = executor(setup_out, proposal["selected_action"])
                true_state = np.asarray(true_state, dtype=np.float32)
                observed_state = np.asarray(
                    observe(setup_out, true_state), dtype=np.float32
                )
                reached = target_reached(true_state, desired_truth)
                after_cost = target_cost(true_state, desired_truth)
                zero_action = all(
                    float(proposal["selected_action"][field]) == 0.0
                    for field in ACTION_FIELDS
                )
                trace.append(
                    {
                        "step": step,
                        "selected_index": proposal["selected_index"],
                        "selected_action": proposal["selected_action"],
                        "predicted_status": proposal["predicted_status"],
                        "true_cost_before": before_cost,
                        "true_cost_after": after_cost,
                        "true_target_reached": reached,
                        "zero_action": zero_action,
                    }
                )
                if reached:
                    stop_reason = "target_reached"
                    break
                if zero_action:
                    stop_reason = "zero_action_stall"
                    break
        return {
            "request_id": request_id,
            "initial_target_reached": initial_reached,
            "final_target_reached": target_reached(true_state, desired_truth),
            "executed_steps": len(trace),
            "stop_reason": stop_reason,
            "trace": trace,
            "final_setup": setup_out,
            "final_true_state": state_mapping(true_state),
            "final_observed_state": state_mapping(observed_state),
            "desired_true": state_mapping(desired_truth),
        }


class OpticalSimulatorExecutor:
    """Apply one action to setup offsets and return the new simulator state."""

    def __init__(self, repo_root: Path) -> None:
        from optical_sim.src.experiment_generator import (
            load_yaml as load_sim_yaml,
        )

        self.repo_root = repo_root.resolve()
        self.base = load_sim_yaml(
            self.repo_root / "optical_sim/configs/base_config.yaml"
        )

    def __call__(
        self,
        setup: Mapping[str, Any],
        action: Mapping[str, float],
    ) -> tuple[dict[str, Any], np.ndarray]:
        from optics_understanding_sft.build_dataset import simulator_result
        from optics_understanding_sft.direction_inverse_v1.build_inverse import (
            config_from_visible,
        )

        next_setup = copy.deepcopy(dict(setup))
        next_setup["lens_x_offset_mm"] = float(next_setup["lens_x_offset_mm"]) + float(
            action["lens_x_delta_mm"]
        )
        next_setup["lens_y_offset_mm"] = float(next_setup["lens_y_offset_mm"]) + float(
            action["lens_y_delta_mm"]
        )
        next_setup["camera_x_offset_mm"] = float(
            next_setup["camera_x_offset_mm"]
        ) + float(action["camera_x_delta_mm"])
        next_setup["camera_y_offset_mm"] = float(
            next_setup["camera_y_offset_mm"]
        ) + float(action["camera_y_delta_mm"])
        visible = {
            **next_setup,
            "sensor_resolution_px": [1024, 1024],
        }
        simulator_config = config_from_visible(visible, self.base)
        result = simulator_result(simulator_config)
        return next_setup, raw_state_array(result["state"])
