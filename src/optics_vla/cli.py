"""Small, CPU-compatible validation commands for the canonical stack."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from optics_vla.common.config import ControllerConfig, load_controller_config


def _emit(component: str, status: str, **details: Any) -> dict[str, Any]:
    result = {"component": component, "status": status, **details}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def _runtime(
    config: ControllerConfig,
) -> tuple[
    dict[str, Any],
    Any,
    dict[str, Any],
    dict[str, float],
    str,
    dict[str, float],
]:
    from continuous_control_v12.contracts import Bounds
    from continuous_control_v12.simulator import default_simulator_fixed, sample_group_setup

    v12 = json.loads(config.resolve("simulator.config").read_text(encoding="utf-8"))
    bounds = Bounds.from_config(v12)
    bounds.validate()
    base = str(config.resolve("simulator.base_config"))
    fixed = default_simulator_fixed(
        base,
        grid_size=128,
        grid_extent_mm=6.25,
        sensor_resolution=[64, 64],
        semantics=v12["simulator"]["semantics"],
    )
    context, positions = sample_group_setup(
        "ordinary", "optics_vla_smoke", 2026081501, fixed, bounds
    )
    return v12, bounds, fixed, positions, base, context


def sanity(config: ControllerConfig) -> dict[str, Any]:
    required = (
        "continuous_control_v12",
        "optical_sim",
        "optics_sft",
        "qwen_vl_supervisor_v1",
        "specialist_rebuild_v2",
    )
    loaded = []
    for name in required:
        importlib.import_module(name)
        loaded.append(name)
    return _emit(
        "environment",
        "PASS",
        python=sys.version.split()[0],
        imports=loaded,
        controller_config=str(config.source),
    )


def config_check(config: ControllerConfig) -> dict[str, Any]:
    required_paths = {
        key: config.resolve(key).is_file()
        for key in ("simulator.config", "simulator.base_config")
    }
    if not all(required_paths.values()):
        return _emit("controller_config", "FAIL", required_paths=required_paths)
    return _emit(
        "controller_config",
        "PASS",
        initial_control_steps=config.initial_control_steps,
        maximum_horizon=config.maximum_horizon,
        minimum_last_step_improvement=config.minimum_last_step_improvement,
        checkpoint_available=config.checkpoint_available(),
        checkpoint_hash_matches=config.verify_checkpoint(),
    )


def simulator_smoke(config: ControllerConfig) -> dict[str, Any]:
    from continuous_control_v12.contracts import OUTPUT_FIELDS
    from continuous_control_v12.simulator import simulate_state

    _, bounds, fixed, positions, base, context = _runtime(config)
    capture = simulate_state(context, positions, fixed, base, bounds)
    metrics = capture["metrics"]
    passed = bool(
        capture["auxiliary"]["simulator_valid"]
        and tuple(metrics) == tuple(OUTPUT_FIELDS)
        and all(np.isfinite(float(metrics[field])) for field in OUTPUT_FIELDS)
    )
    return _emit(
        "simulator",
        "PASS" if passed else "FAIL",
        semantics=fixed["simulator_semantics_version"],
        state_fields=list(metrics),
        grid_size=fixed["grid_size"],
        sensor_resolution=fixed["sensor_resolution_px"],
    )


def h1_smoke(config: ControllerConfig) -> dict[str, Any]:
    checkpoint = config.resolve("dynamics.checkpoint")
    if not checkpoint.is_file():
        return _emit(
            "learned_h1",
            "BLOCKED",
            reason="configured checkpoint is unavailable",
            checkpoint=str(checkpoint),
        )
    if not config.verify_checkpoint():
        return _emit(
            "learned_h1",
            "FAIL",
            reason="configured checkpoint SHA-256 does not match",
            checkpoint=str(checkpoint),
        )
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return _emit(
            "learned_h1",
            "BLOCKED",
            reason="PyTorch is not installed in this environment",
        )
    from continuous_control_v12.simulator import simulate_state
    from continuous_control_v12.world_model import load_forward_ensemble

    _, bounds, fixed, positions, base, context = _runtime(config)
    current = simulate_state(context, positions, fixed, base, bounds)["metrics"]
    model = load_forward_ensemble(checkpoint, device_name="cpu")
    prediction = model.predict(context, positions, current, np.zeros((1, 4)))
    predicted = np.asarray(prediction["predicted_next_metrics"], dtype=np.float64)
    passed = bool(
        predicted.shape == (1, 5)
        and np.isfinite(predicted).all()
        and np.allclose(predicted[0], np.asarray(list(current.values())), atol=1e-12)
    )
    return _emit(
        "learned_h1",
        "PASS" if passed else "FAIL",
        device="cpu",
        ensemble_members=len(model.members),
        zero_action_preserved=passed,
    )


def cem_smoke(config: ControllerConfig) -> dict[str, Any]:
    from continuous_control_v12.contracts import apply_action, position_dict
    from continuous_control_v12.mpc import CEMMPC, run_closed_loop, simulator_predictor
    from continuous_control_v12.simulator import simulate_state

    _, bounds, fixed, positions, base, context = _runtime(config)
    current = simulate_state(context, positions, fixed, base, bounds)
    target_positions = apply_action(positions, np.asarray([0.05, 0.0, 0.0, 0.0]), bounds)
    target = simulate_state(
        context, position_dict(target_positions), fixed, base, bounds
    )
    planner_config = {
        "horizon": int(config.raw["dynamics"]["model_horizon"]),
        **{
            key: config.raw["controller"][key]
            for key in (
                "population",
                "elites",
                "cem_iterations",
                "mean_error_weight",
                "movement_weight",
                "limit_penalty",
                "boundary_penalty",
                "uncertainty_weight",
            )
        },
    }
    planner = CEMMPC(
        bounds=bounds,
        predictor=simulator_predictor(
            setup_context=context,
            simulator_fixed=fixed,
            base_config_path=base,
            bounds=bounds,
        ),
        config=planner_config,
        seed=2026081502,
    )
    episode = run_closed_loop(
        planner=planner,
        setup_context=context,
        simulator_fixed=fixed,
        initial_positions_mm=positions,
        initial_metrics=current["metrics"],
        target_metrics=target["metrics"],
        allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
        base_config_path=base,
        bounds=bounds,
        max_steps=1,
    )
    passed = bool(
        episode["success"]
        and episode["final_normalized_distance"] < episode["initial_normalized_distance"]
        and episode["illegal_actions"] == 0
    )
    return _emit(
        "cem_controller",
        "PASS" if passed else "FAIL",
        seed=2026081502,
        population=planner_config["population"],
        iterations=planner_config["cem_iterations"],
        planned_horizon=planner_config["horizon"],
        executed_steps=len(episode["trace"]),
        success=bool(episode["success"]),
    )


def qwen_contract() -> dict[str, Any]:
    from qwen_vl_supervisor_v1.closed_loop_adapter import (
        SupervisorDecisionError,
        parse_supervisor_decision,
    )

    valid = (
        {
            "diagnosis": "nominal",
            "measurement_policy": "standard",
            "supervisor_action": "execute",
        },
        {
            "diagnosis": "sensor_saturation",
            "measurement_policy": "lower_exposure_reacquire",
            "supervisor_action": "reacquire",
        },
        {
            "diagnosis": "secondary_reflection",
            "measurement_policy": "primary_spot",
            "supervisor_action": "switch_measurement",
        },
    )
    accepted = [parse_supervisor_decision(payload).to_dict() for payload in valid]
    rejected_actuator = False
    try:
        parse_supervisor_decision({**valid[0], "lens_x_delta_mm": 0.05})
    except SupervisorDecisionError:
        rejected_actuator = True
    return _emit(
        "qwen_contract",
        "PASS" if len(accepted) == 3 and rejected_actuator else "FAIL",
        accepted_canonical_combinations=len(accepted),
        rejected_continuous_actuator_output=rejected_actuator,
        model_loaded=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-config", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "sanity",
        "config-check",
        "simulator-smoke",
        "h1-smoke",
        "cem-smoke",
        "qwen-contract",
    ):
        subparsers.add_parser(name)
    all_smoke = subparsers.add_parser("all-smoke")
    all_smoke.add_argument("--include-h1", action="store_true")
    args = parser.parse_args(argv)
    config = load_controller_config(args.controller_config)
    commands = {
        "sanity": lambda: sanity(config),
        "config-check": lambda: config_check(config),
        "simulator-smoke": lambda: simulator_smoke(config),
        "h1-smoke": lambda: h1_smoke(config),
        "cem-smoke": lambda: cem_smoke(config),
        "qwen-contract": qwen_contract,
    }
    if args.command != "all-smoke":
        result = commands[args.command]()
        return 0 if result["status"] == "PASS" else 2
    selected = [sanity(config), config_check(config), simulator_smoke(config), cem_smoke(config), qwen_contract()]
    if args.include_h1:
        selected.append(h1_smoke(config))
    statuses = [item["status"] for item in selected]
    return 0 if all(status == "PASS" for status in statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
