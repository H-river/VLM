"""Execute frozen Qwen route decisions against candidate v4 specialists."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, residual_cost
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.inverse_runtime import load_inverse_runtime_v4
from control_rebuild_v4.visual_runtime import VisualInversePipelineV4
from direction_rebuild_v4.direction_runtime import load_direction_runtime_v4
from Qwen_orchestration.runtime.dispatcher import _registry, validate_decision
from Qwen_orchestration.runtime.errors import ContractError, SpecialistError
from Qwen_orchestration.runtime.specialists import run_specialist
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    STATE_FIELDS,
    change_and_directions,
    raw_state_array,
)

ROUTE_BACKENDS = {
    "measure_beam_profile_v1": "guarded_measurement_v4",
    "predict_direction_from_state_v1": "frozen_direction_v1",
    "predict_direction_from_image_v1": (
        "guarded_measurement_v4_then_frozen_direction_v1"
    ),
    "predict_forward_from_state_v1": "forward_v4",
    "predict_forward_from_image_v1": "guarded_measurement_v4_then_forward_v4",
    "select_inverse_action_from_states_v1": "forward_v4_then_inverse_v4",
    "select_inverse_action_from_images_v1": "modular_visual_inverse_v4",
}
ROUTE_ARTIFACTS = {
    "measure_beam_profile_v1": (
        "measurement_v3",
        "measurement_calibrator_v4",
    ),
    "predict_direction_from_state_v1": ("direction_v1",),
    "predict_direction_from_image_v1": (
        "measurement_v3",
        "measurement_calibrator_v4",
        "direction_v1",
    ),
    "predict_forward_from_state_v1": ("forward_v4",),
    "predict_forward_from_image_v1": (
        "measurement_v3",
        "measurement_calibrator_v4",
        "forward_v4",
    ),
    "select_inverse_action_from_states_v1": ("forward_v4", "inverse_v4"),
    "select_inverse_action_from_images_v1": (
        "measurement_v3",
        "measurement_calibrator_v4",
        "forward_v4",
        "inverse_v4",
        "visual_scorer_v4",
    ),
}
DIRECTION_V4_ROUTE_BACKENDS = {
    **ROUTE_BACKENDS,
    "predict_direction_from_state_v1": "balanced_direction_v4",
    "predict_direction_from_image_v1": (
        "guarded_measurement_v4_then_balanced_direction_v4"
    ),
}
DIRECTION_V4_ROUTE_ARTIFACTS = {
    **ROUTE_ARTIFACTS,
    "predict_direction_from_state_v1": ("forward_v4", "direction_v4"),
    "predict_direction_from_image_v1": (
        "measurement_v3",
        "measurement_calibrator_v4",
        "forward_v4",
        "direction_v4",
    ),
}


def _strict_state(values: Mapping[str, Any], label: str) -> dict[str, float]:
    if set(values) != set(STATE_FIELDS):
        raise ContractError(f"{label} fields differ from the five-state contract")
    output = {field: float(values[field]) for field in STATE_FIELDS}
    if not all(np.isfinite(value) for value in output.values()):
        raise ContractError(f"{label} contains a non-finite value")
    return output


def _action_index(values: Mapping[str, Any]) -> int:
    if set(values) != set(ACTION_FIELDS):
        raise ContractError("action fields differ from the four-action contract")
    action = {field: float(values[field]) for field in ACTION_FIELDS}
    for index, candidate in enumerate(ACTION_GRID):
        if all(action[field] == float(candidate[field]) for field in ACTION_FIELDS):
            return index
    raise ContractError(
        "v4 forward specialist accepts the registered 81-action grid only"
    )


class OrchestratedSpecialistRuntimeV4:
    """Preserve Qwen decisions while replacing eligible v1 specialists."""

    def __init__(
        self,
        torch: Any,
        measurement_artifact: Path,
        measurement_calibrator_artifact: Path,
        forward_artifact: Path,
        inverse_artifact: Path,
        visual_scorer_artifact: Path,
        device: Any,
        direction_artifact: Path | None = None,
    ) -> None:
        self.forward, _ = load_forward_runtime_v4(forward_artifact, torch, device)
        self.inverse, _ = load_inverse_runtime_v4(inverse_artifact, torch, device)
        self.direction = (
            None
            if direction_artifact is None
            else load_direction_runtime_v4(
                direction_artifact,
                torch,
                device,
                forward_runtime=self.forward,
            )[0]
        )
        self.visual = VisualInversePipelineV4(
            torch,
            measurement_artifact,
            measurement_calibrator_artifact,
            forward_artifact,
            inverse_artifact,
            visual_scorer_artifact,
            device,
        )

    @staticmethod
    def _verified_entry(
        entry: Mapping[str, Any],
        label: str,
    ) -> Path:
        path = Path(str(entry["path"])).resolve()
        if not path.is_file():
            raise ContractError(f"candidate manifest artifact is missing: {path}")
        if path.stat().st_size != int(entry["size"]):
            raise ContractError(f"candidate manifest size mismatch: {label}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != str(entry["sha256"]):
            raise ContractError(f"candidate manifest digest mismatch: {label}")
        return path

    @classmethod
    def from_manifest(
        cls,
        torch: Any,
        manifest_path: Path,
        device: Any,
    ) -> OrchestratedSpecialistRuntimeV4:
        path = manifest_path.resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest_version = manifest.get("manifest_version")
        if manifest_version == "qwen_to_specialists_v4_candidate_manifest":
            expected_backends = ROUTE_BACKENDS
            expected_route_artifacts = ROUTE_ARTIFACTS
            direction_key = "direction_v1"
        elif (
            manifest_version
            == "qwen_to_specialists_v4_direction_candidate_manifest"
        ):
            expected_backends = DIRECTION_V4_ROUTE_BACKENDS
            expected_route_artifacts = DIRECTION_V4_ROUTE_ARTIFACTS
            direction_key = "direction_v4"
        else:
            raise ContractError("candidate overlay manifest version is invalid")
        if (
            manifest.get("complete") is not True
            or manifest.get("action_grid_size") != len(ACTION_GRID)
            or manifest.get("simulator_at_inference") is not False
            or manifest.get("seed") != 20260726
        ):
            raise ContractError("candidate overlay manifest contract is invalid")
        if set(manifest.get("routes", {})) != set(expected_backends):
            raise ContractError("candidate overlay manifest route set is invalid")
        if {
            route: value.get("backend") for route, value in manifest["routes"].items()
        } != expected_backends:
            raise ContractError("candidate overlay manifest backend map is invalid")
        if {
            route: tuple(value.get("artifacts", []))
            for route, value in manifest["routes"].items()
        } != expected_route_artifacts:
            raise ContractError("candidate overlay route artifact map is invalid")
        artifact_keys = set(manifest.get("artifacts", {}))
        if any(
            not set(route["artifacts"]).issubset(artifact_keys)
            for route in manifest["routes"].values()
        ):
            raise ContractError("candidate route refers to an unpinned artifact")
        policy = manifest.get("direct_measurement_policy", {})
        if (
            float(policy.get("learned_gamma_min", math.nan))
            != VisualInversePipelineV4.LEARNED_GAMMA_MIN
            or float(policy.get("learned_gamma_max", math.nan))
            != VisualInversePipelineV4.LEARNED_GAMMA_MAX
        ):
            raise ContractError("candidate measurement policy differs from runtime")
        cls._verified_entry(manifest["frozen_qwen"]["registry"], "frozen_qwen.registry")
        cls._verified_entry(
            manifest["frozen_qwen"]["decision_schema"],
            "frozen_qwen.decision_schema",
        )
        artifacts = {
            key: cls._verified_entry(manifest["artifacts"][key], key)
            for key in (
                "measurement_v3",
                "measurement_calibrator_v4",
                "forward_v4",
                "inverse_v4",
                "visual_scorer_v4",
                direction_key,
            )
        }
        return cls(
            torch,
            artifacts["measurement_v3"],
            artifacts["measurement_calibrator_v4"],
            artifacts["forward_v4"],
            artifacts["inverse_v4"],
            artifacts["visual_scorer_v4"],
            device,
            direction_artifact=(
                artifacts["direction_v4"]
                if direction_key == "direction_v4"
                else None
            ),
        )

    def _forward_from_state(
        self,
        setup: Mapping[str, Any],
        current: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        current_out = _strict_state(current, "current_beam_state")
        action_position = _action_index(action)
        row = {
            "group_id": "orchestrated_forward_v4",
            "setup": dict(setup),
            "current_beam_state": current_out,
        }
        states = self.forward.predict_states([row])[0]
        predicted = state_mapping(states[action_position])
        change, directions = change_and_directions(current_out, predicted)
        return {
            "change": change,
            "predicted_beam_state": predicted,
            "directions": directions,
            "model_version": "control_rebuild_v4_one_seed",
            "simulator_at_inference": False,
        }

    def _inverse_from_states(
        self,
        setup: Mapping[str, Any],
        current: Mapping[str, Any],
        desired: Mapping[str, Any],
    ) -> dict[str, Any]:
        current_out = _strict_state(current, "current_beam_state")
        desired_out = _strict_state(desired, "desired_beam_state")
        row = {
            "group_id": "orchestrated_inverse_v4",
            "setup": dict(setup),
            "current_beam_state": current_out,
        }
        candidates = self.forward.predict_states([row])
        result = self.inverse.score_requests(
            [setup],
            raw_state_array(current_out)[None, :],
            raw_state_array(desired_out)[None, :],
            candidates,
        )
        selected = int(result["selected_indices"][0])
        return {
            "predicted_status": result["predicted_statuses"][0],
            "selected_index": selected,
            "selected_action": result["selected_actions"][0],
            "predicted_beam_state": state_mapping(candidates[0, selected]),
            "best_predicted_normalized_residual": float(
                residual_cost(
                    candidates[0, selected][None, :],
                    raw_state_array(desired_out)[None, :],
                )[0]
            ),
            "action_grid_size": len(ACTION_GRID),
            "model_version": "control_rebuild_v4_one_seed",
            "simulator_at_inference": False,
        }

    def _inverse_from_images(
        self,
        setup: Mapping[str, Any],
        current_image: Path,
        desired_image: Path,
        calibration: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = self.visual.predict_from_images(
            setup,
            current_image,
            desired_image,
            calibration,
            request_id="orchestrated_visual_inverse_v4",
        )
        selected = int(result["selected_indices"][0])
        return {
            "predicted_status": result["predicted_statuses"][0],
            "selected_index": selected,
            "selected_action": result["selected_actions"][0],
            "measured_current_beam_state": state_mapping(
                result["current_image_measurement"]["beam_state"]
            ),
            "measured_desired_beam_state": state_mapping(
                result["desired_image_measurement"]["beam_state"]
            ),
            "current_measurement_source": result["current_image_measurement"][
                "measurement_source"
            ],
            "desired_measurement_source": result["desired_image_measurement"][
                "measurement_source"
            ],
            "predicted_beam_state": state_mapping(
                result["candidate_sensor_states"][0, selected]
            ),
            "action_grid_size": len(ACTION_GRID),
            "coordinate_frame": "camera_sensor_array",
            "model_version": "control_rebuild_v4_one_seed",
            "simulator_at_inference": False,
        }

    def execute_ready(
        self,
        route_name: str,
        arguments: Mapping[str, Any],
        images: Mapping[str, Path],
    ) -> dict[str, Any]:
        if route_name == "measure_beam_profile_v1":
            measured = self.visual.measure_image(
                images["beam"], arguments["image_calibration"]
            )
            return {
                "beam_state": state_mapping(measured["beam_state"]),
                "coordinate_frame": "camera_sensor_array",
                "measurement_source": measured["measurement_source"],
                "model_version": measured["model_version"],
                "simulator_at_inference": False,
            }
        if route_name == "predict_direction_from_state_v1":
            if getattr(self, "direction", None) is not None:
                return self.direction.predict_one(
                    arguments["setup"],
                    arguments["current_beam_state"],
                    arguments["action"],
                )
            return run_specialist(route_name, arguments, images)
        if route_name == "predict_direction_from_image_v1":
            measured = self.visual.measure_image(
                images["current_beam"], arguments["image_calibration"]
            )
            legacy = sensor_to_base_legacy(
                measured["beam_state"], [arguments["setup"]]
            )[0]
            if getattr(self, "direction", None) is not None:
                result = self.direction.predict_one(
                    arguments["setup"],
                    state_mapping(legacy),
                    arguments["action"],
                )
            else:
                result = run_specialist(
                    "predict_direction_from_state_v1",
                    {
                        "setup": arguments["setup"],
                        "current_beam_state": state_mapping(legacy),
                        "action": arguments["action"],
                    },
                    {},
                )
            return {
                **result,
                "measured_current_beam_state": state_mapping(measured["beam_state"]),
                "measurement_source": measured["measurement_source"],
            }
        if route_name == "predict_forward_from_state_v1":
            return self._forward_from_state(
                arguments["setup"],
                arguments["current_beam_state"],
                arguments["action"],
            )
        if route_name == "predict_forward_from_image_v1":
            measured = self.visual.measure_image(
                images["current_beam"], arguments["image_calibration"]
            )
            legacy = sensor_to_base_legacy(
                measured["beam_state"], [arguments["setup"]]
            )[0]
            result = self._forward_from_state(
                arguments["setup"],
                state_mapping(legacy),
                arguments["action"],
            )
            return {
                **result,
                "measured_current_beam_state": state_mapping(measured["beam_state"]),
                "measurement_source": measured["measurement_source"],
            }
        if route_name == "select_inverse_action_from_states_v1":
            return self._inverse_from_states(
                arguments["setup"],
                arguments["current_beam_state"],
                arguments["desired_beam_state"],
            )
        if route_name == "select_inverse_action_from_images_v1":
            return self._inverse_from_images(
                arguments["setup"],
                images["current_beam"],
                images["desired_beam"],
                arguments["image_calibration"],
            )
        raise ContractError(f"unregistered specialist route: {route_name}")

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        resolved = validate_decision(decision, available_images or {})
        status = str(decision["status"])
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
            result = self.execute_ready(
                str(decision["route_name"]),
                decision["arguments"],
                {role: Path(path) for role, path in resolved.items()},
            )
        except ContractError:
            raise
        except Exception as error:
            raise SpecialistError(
                f"v4 specialist {decision['route_name']} failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        return {
            "status": "ready",
            "task_type": decision["task_type"],
            "route_name": decision["route_name"],
            "executed": True,
            "specialist_generation": "v4_candidate_overlay",
            "result": result,
        }
