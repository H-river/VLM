"""Strict orchestration decision validation and deterministic dispatch."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from .errors import ContractError, SpecialistError
from .specialists import run_specialist


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PACKAGE_ROOT / "schemas/orchestration_decision.schema.json"
REGISTRY_PATH = PACKAGE_ROOT / "configs/model_registry.yaml"


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    with SCHEMA_PATH.open(encoding="utf-8") as stream:
        schema = json.load(stream)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    with REGISTRY_PATH.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ContractError("model registry must contain an object")
    return value


def _path_text(error: Any) -> str:
    return ".".join(str(item) for item in error.absolute_path) or "<root>"


def _reject_non_finite(value: Any, path: str = "<root>") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def validate_decision(
    decision: Mapping[str, Any],
    available_images: Mapping[str, str | Path],
    *,
    require_enabled: bool = True,
) -> dict[str, str | Path]:
    """Validate a Qwen decision and return role-to-path image bindings."""
    if not isinstance(decision, Mapping):
        raise ContractError("decision must be an object")
    errors = sorted(
        _schema_validator().iter_errors(decision),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        raise ContractError(f"schema violation at {_path_text(first)}: {first.message}")
    _reject_non_finite(decision)

    status = decision["status"]
    if status != "ready":
        return {}

    registry = _registry()
    route_name = decision["route_name"]
    route = registry["routes"].get(route_name)
    if route is None:
        raise ContractError(f"route is not registered: {route_name}")
    if require_enabled and not route["enabled"]:
        raise ContractError(f"route is disabled: {route_name}")
    if decision["task_type"] != route["task_type"]:
        raise ContractError(
            f"task_type {decision['task_type']} does not match route {route_name}"
        )

    actual_arguments = set(decision["arguments"])
    required_arguments = set(route["required_argument_groups"])
    if actual_arguments != required_arguments:
        raise ContractError(
            f"{route_name} argument groups differ: "
            f"missing={sorted(required_arguments - actual_arguments)}, "
            f"extra={sorted(actual_arguments - required_arguments)}"
        )
    forbidden = set(route["forbidden_argument_groups"])
    if actual_arguments & forbidden:
        raise ContractError(
            f"{route_name} contains forbidden arguments: "
            f"{sorted(actual_arguments & forbidden)}"
        )

    actual_roles = set(decision["image_roles"])
    required_roles = set(route["required_image_roles"])
    if actual_roles != required_roles:
        raise ContractError(
            f"{route_name} image roles differ: "
            f"missing={sorted(required_roles - actual_roles)}, "
            f"extra={sorted(actual_roles - required_roles)}"
        )
    references = list(decision["image_roles"].values())
    if len(references) != len(set(references)):
        raise ContractError("one image reference cannot fill multiple image roles")
    resolved: dict[str, str | Path] = {}
    for role, reference in decision["image_roles"].items():
        if reference not in available_images:
            raise ContractError(f"image reference is unavailable: {reference}")
        path = Path(available_images[reference])
        if not path.is_file():
            raise ContractError(f"image path does not exist for {reference}: {path}")
        resolved[role] = path
    return resolved


class OrchestrationRuntime:
    """Execute validated decisions against the immutable specialist registry."""

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        images = available_images or {}
        resolved = validate_decision(decision, images)
        status = decision["status"]
        if status == "needs_clarification":
            return {
                "status": status,
                "missing_fields": list(decision["missing_fields"]),
                "clarification_question": decision["clarification_question"],
                "executed": False,
            }
        if status == "unsupported":
            return {
                "status": status,
                "supported_task_types": list(_registry()["task_types"]),
                "executed": False,
            }
        try:
            result = run_specialist(
                str(decision["route_name"]), decision["arguments"], resolved
            )
        except ContractError:
            raise
        except Exception as error:
            raise SpecialistError(
                f"specialist {decision['route_name']} failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        return {
            "status": "ready",
            "task_type": decision["task_type"],
            "route_name": decision["route_name"],
            "executed": True,
            "result": result,
        }

