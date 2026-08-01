#!/usr/bin/env python3
"""Validate static Qwen orchestration contracts without changing any artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a mapping")
    return value


def non_null_enum(schema: dict[str, Any], property_name: str) -> set[str]:
    values = schema["properties"][property_name]["enum"]
    return {value for value in values if isinstance(value, str)}


def main() -> None:
    decision = read_json(ROOT / "schemas/orchestration_decision.schema.json")
    route_decision = read_json(ROOT / "schemas/orchestration_route.schema.json")
    training_record = read_json(ROOT / "schemas/training_record.schema.json")
    manifest = read_json(ROOT / "freeze/baseline_manifest.json")
    registry = read_yaml(ROOT / "configs/model_registry.yaml")
    stage1 = read_yaml(ROOT / "configs/qwen25vl_3b_orchestrator_stage1_v1.yaml")
    stage2 = read_yaml(ROOT / "configs/qwen25vl_3b_orchestrator_stage2_v1.yaml")

    assert training_record["$schema"].endswith("2020-12/schema")
    assert manifest["freeze_id"] == registry["freeze_id"]

    route_names = set(registry["routes"])
    task_types = set(registry["task_types"])
    assert route_names == non_null_enum(decision, "route_name")
    assert route_names == non_null_enum(route_decision, "route_name")
    assert task_types == non_null_enum(decision, "task_type")
    assert task_types == non_null_enum(route_decision, "task_type")

    argument_groups = set(decision["properties"]["arguments"]["properties"])
    image_roles = set(decision["properties"]["image_roles"]["properties"])
    frozen_paths = {entry["path"] for entry in manifest["files"]}

    for name, route in registry["routes"].items():
        assert route["task_type"] in task_types, name
        assert route["implementation_status"] == "ready", name
        assert route["enabled"] is True, name
        required = set(route["required_argument_groups"])
        forbidden = set(route["forbidden_argument_groups"])
        assert required <= argument_groups, name
        assert forbidden <= argument_groups, name
        assert not required & forbidden, name
        assert set(route["required_image_roles"]) <= image_roles, name
        for artifact in route["specialist_artifacts"]:
            assert artifact in frozen_paths, f"{name}: artifact is not frozen: {artifact}"
            assert (REPO / artifact).is_file(), f"{name}: missing artifact: {artifact}"

    defaults = registry["deterministic_defaults"]
    grid = defaults["inverse_action_grid"]
    assert math.prod(len(grid[field]) for field in (
        "lens_x_delta_mm", "lens_y_delta_mm", "camera_x_delta_mm", "camera_y_delta_mm"
    )) == 81
    tolerance = defaults["inverse_matching_tolerance"]
    assert tolerance == {
        "centroid_vector_px": 0.5,
        "width_each_px": 1.0,
        "peak_relative": 0.02,
    }

    for config, expected_accumulation, expected_steps, expected_warmup in (
        (stage1, 8, 1250, 100),
        (stage2, 4, 2500, 125),
    ):
        training = config["training"]
        assert training["per_device_train_batch_size"] == 1
        assert training["gradient_accumulation_steps"] == expected_accumulation
        assert training["max_steps"] == expected_steps
        assert round(training["max_steps"] * training["warmup_ratio"]) == expected_warmup
        assert training["completion_only_loss"] is True
        assert training["decision_span_weight"] == 1.0

    print(
        "Scaffold validation passed: "
        f"{len(route_names)} routes, {len(task_types)} task types, "
        f"{len(frozen_paths)} frozen files."
    )


if __name__ == "__main__":
    main()
