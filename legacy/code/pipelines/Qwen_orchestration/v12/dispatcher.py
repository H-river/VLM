"""Strict v12 decision validation and deterministic route dispatch."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from Qwen_orchestration.runtime.errors import ContractError, SpecialistError
from Qwen_orchestration.v12.adapter import V12Adapter


PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = PACKAGE_ROOT / "orchestration_decision_v12.schema.json"
REGISTRY_PATH = PACKAGE_ROOT / "model_registry.yaml"
REPO_ROOT = PACKAGE_ROOT.parents[1]


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ContractError("v12 registry must be an object")
    return registry


def _reject_nonfinite(value: Any, path: str = "<root>") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, f"{path}[{index}]")


def validate_v12_decision(
    decision: Mapping[str, Any],
    available_images: Mapping[str, str | Path],
) -> dict[str, Path]:
    if not isinstance(decision, Mapping):
        raise ContractError("decision must be an object")
    errors = sorted(
        _validator().iter_errors(decision), key=lambda error: list(error.absolute_path)
    )
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.absolute_path)) or "<root>"
        raise ContractError(f"schema violation at {path}: {error.message}")
    _reject_nonfinite(decision)
    if decision["status"] != "ready":
        return {}
    route_name = str(decision["route_name"])
    route = _registry()["routes"].get(route_name)
    if route is None:
        raise ContractError(f"v12 route is not registered: {route_name}")
    if decision["task_type"] != route["task_type"]:
        raise ContractError("task_type does not match the v12 route registry")
    actual_args = set(decision["arguments"])
    required_args = set(route["required_argument_groups"])
    if actual_args != required_args:
        raise ContractError(
            f"{route_name} argument groups differ: missing={sorted(required_args-actual_args)}, "
            f"extra={sorted(actual_args-required_args)}"
        )
    actual_roles = set(decision["image_roles"])
    required_roles = set(route["required_image_roles"])
    if actual_roles != required_roles:
        raise ContractError(
            f"{route_name} image roles differ: missing={sorted(required_roles-actual_roles)}, "
            f"extra={sorted(actual_roles-required_roles)}"
        )
    references = list(decision["image_roles"].values())
    if len(references) != len(set(references)):
        raise ContractError("one image reference cannot fill multiple image roles")
    resolved: dict[str, Path] = {}
    for role, reference in decision["image_roles"].items():
        if reference not in available_images:
            raise ContractError(f"image reference is unavailable: {reference}")
        path = Path(available_images[reference]).resolve()
        if not path.is_file():
            raise ContractError(f"image path does not exist: {path}")
        resolved[role] = path
    return resolved


class V12OrchestrationRuntime:
    def __init__(self, adapter: V12Adapter | None = None) -> None:
        self.adapter = adapter or V12Adapter()

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
        *,
        run_id: str = "v12_runtime",
        execution_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        images = validate_v12_decision(decision, available_images or {})
        status = str(decision["status"])
        if status != "ready":
            return {
                "status": status,
                "reason": decision["reason"],
                "missing_fields": list(decision["missing_fields"]),
                "executed": False,
            }
        route = str(decision["route_name"])
        args = decision["arguments"]
        try:
            result = self._run_ready(
                route,
                args,
                images,
                run_id=run_id,
                execution_context=execution_context or {},
            )
        except ContractError:
            raise
        except Exception as error:
            raise SpecialistError(
                f"v12 specialist {route} failed: {type(error).__name__}: {error}"
            ) from error
        return {
            "status": "ready",
            "task_type": decision["task_type"],
            "route_name": route,
            "executed": True,
            "executed_backend": "v12",
            "result": result,
        }

    def _measure_v12_image(
        self,
        *,
        role: str,
        path: Path,
        args: Mapping[str, Any],
        run_id: str,
        route: str,
    ) -> dict[str, Any]:
        setup = self.adapter.canonicalize_setup(args["setup_context"])
        position = self.adapter.canonicalize_position(args["actuator_position"])
        return self.adapter.measure(
            image_path=path,
            calibration=args["image_calibration"],
            run_id=f"{run_id}:{role}",
            route=route,
            task="measurement",
            actuator_position_mm=position,
            setup_context=setup,
        )

    def _run_ready(
        self,
        route: str,
        args: Mapping[str, Any],
        images: Mapping[str, Path],
        *,
        run_id: str,
        execution_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        if route == "measure_beam_profile_v12":
            return self.adapter.measure(
                image_path=images["beam"],
                calibration=args["image_calibration"],
                run_id=run_id,
                route=route,
            )
        if route in {
            "predict_direction_from_state_v12",
            "predict_forward_from_state_v12",
        }:
            method = (
                self.adapter.direction
                if route.startswith("predict_direction")
                else self.adapter.forward
            )
            return method(
                setup_context=args["setup_context"],
                actuator_position=args["actuator_position"],
                current_beam_state=args["current_beam_state"],
                continuous_action=args["continuous_action"],
                run_id=run_id,
                route=route,
            )
        if route in {
            "predict_direction_from_image_v12",
            "predict_forward_from_image_v12",
        }:
            measured = self._measure_v12_image(
                role="current",
                path=images["current_beam"],
                args=args,
                run_id=run_id,
                route=route,
            )
            method = (
                self.adapter.direction
                if route.startswith("predict_direction")
                else self.adapter.forward
            )
            downstream = method(
                setup_context=args["setup_context"],
                actuator_position=args["actuator_position"],
                current_beam_state=measured["beam_state"],
                continuous_action=args["continuous_action"],
                run_id=run_id,
                route=route,
            )
            return {**downstream, "measurement": measured}
        if route in {
            "inverse_control_from_states_v12_h1",
            "inverse_control_from_images_v12_h1",
        }:
            if route.endswith("images_v12_h1"):
                current_measurement = self._measure_v12_image(
                    role="current",
                    path=images["current_beam"],
                    args=args,
                    run_id=run_id,
                    route=route,
                )
                target_measurement = self._measure_v12_image(
                    role="target",
                    path=images["target_beam"],
                    args=args,
                    run_id=run_id,
                    route=route,
                )
                current = current_measurement["beam_state"]
                target = target_measurement["beam_state"]
            else:
                current = args["current_beam_state"]
                target = args["target_beam_state"]
                current_measurement = target_measurement = None
            if "simulator_fixed" not in execution_context:
                raise ContractError("inverse v12 execution requires simulator_fixed context")
            base = execution_context.get(
                "base_config_path",
                REPO_ROOT / self.adapter.v12_config["simulator"]["base_config"],
            )
            result = self.adapter.inverse(
                setup_context=args["setup_context"],
                actuator_position=args["actuator_position"],
                current_beam_state=current,
                target_beam_state=target,
                simulator_fixed=execution_context["simulator_fixed"],
                base_config_path=Path(base),
                run_id=run_id,
                route=route,
                planner_seed=execution_context.get("planner_seed"),
            )
            if current_measurement is not None:
                result["current_measurement"] = current_measurement
                result["target_measurement"] = target_measurement
            return result
        raise ContractError(f"unregistered v12 route: {route}")
