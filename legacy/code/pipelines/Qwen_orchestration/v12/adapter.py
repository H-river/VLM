"""Deterministic adapter shared by every v12 orchestration route."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    SETUP_CONTEXT_FIELDS,
    Bounds,
    action_dict,
    apply_action,
    metrics_dict,
    metrics_vector,
    position_dict,
    stable_hash,
    stable_seed,
    tolerance_vector,
    validate_action,
    validate_positions,
)
from continuous_control_v12.mpc import CEMMPC, learned_predictor, run_closed_loop
from continuous_control_v12.world_model import load_forward_ensemble
from Qwen_orchestration.runtime.errors import ContractError, SpecialistError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_CONFIG = Path(__file__).with_name("runtime_config.json")
ACTION_VALUE_FIELDS = ("lens_x", "lens_y", "camera_x", "camera_y")
FEATURE_ORDER = (
    *SETUP_CONTEXT_FIELDS,
    *POSITION_FIELDS,
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "log1p_peak_intensity",
    "lens_x_action_over_bound",
    "lens_y_action_over_bound",
    "camera_x_action_over_bound",
    "camera_y_action_over_bound",
    "lens_x_action_squared",
    "lens_y_action_squared",
    "camera_x_action_squared",
    "camera_y_action_squared",
    "lens_x_action_x_lens_y_action",
    "lens_x_action_x_camera_x_action",
    "lens_x_action_x_camera_y_action",
    "lens_y_action_x_camera_x_action",
    "lens_y_action_x_camera_y_action",
    "camera_x_action_x_camera_y_action",
    "lens_x_position_x_action",
    "lens_y_position_x_action",
    "camera_x_position_x_action",
    "camera_y_position_x_action",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _verified_path(entry: Mapping[str, Any], label: str) -> Path:
    path = _resolve(str(entry["path"] if "path" in entry else entry[f"{label}_path"]))
    if not path.is_file():
        raise SpecialistError(f"{label} is missing: {path}")
    expected = str(entry.get("sha256", entry.get(f"{label}_sha256", "")))
    actual = _sha256(path)
    if actual != expected:
        raise SpecialistError(
            f"{label} hash mismatch: expected={expected}, actual={actual}, path={path}"
        )
    return path


def _strict_mapping(
    value: Any, fields: Sequence[str], name: str
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    expected, actual = set(fields), set(value)
    if expected != actual:
        raise ContractError(
            f"{name} fields differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    result: dict[str, float] = {}
    for field in fields:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractError(f"{name}.{field} must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ContractError(f"{name}.{field} must be finite")
        result[field] = number
    return result


def _unit_scale(unit: Any, name: str) -> float:
    if not isinstance(unit, str) or not unit.strip():
        raise ContractError(f"{name}.unit is required; silent unit guessing is forbidden")
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"canonical", "mm"}:
        return 1.0
    if normalized == "um":
        return 1e-3
    raise ContractError(f"{name}.unit is unsupported: {unit!r}")


def canonical_four_vector(
    quantity: Any, *, name: str, output_fields: Sequence[str]
) -> dict[str, float]:
    if not isinstance(quantity, Mapping) or set(quantity) != {"values", "unit"}:
        raise ContractError(f"{name} must contain exactly values and unit")
    values = _strict_mapping(quantity["values"], ACTION_VALUE_FIELDS, f"{name}.values")
    scale = _unit_scale(quantity["unit"], name)
    return {
        output_fields[index]: float(values[field] * scale)
        for index, field in enumerate(ACTION_VALUE_FIELDS)
    }


class V12Adapter:
    """Validate, canonicalize, infer, decode, measure, and plan for v12."""

    def __init__(
        self,
        runtime_config_path: Path = DEFAULT_RUNTIME_CONFIG,
        *,
        device_name: str = "cpu",
    ) -> None:
        self.runtime_config_path = runtime_config_path.resolve()
        self.runtime_config = json.loads(
            self.runtime_config_path.read_text(encoding="utf-8")
        )
        self.v12_config_path = _verified_path(
            self.runtime_config["v12_config"], "v12_config"
        )
        self.checkpoint_path = _verified_path(
            self.runtime_config["v12_checkpoint"], "v12_checkpoint"
        )
        self.v12_config = json.loads(self.v12_config_path.read_text(encoding="utf-8"))
        self.bounds = Bounds.from_config(self.v12_config)
        self.bounds.validate()
        self.model = load_forward_ensemble(
            self.checkpoint_path, device_name=device_name
        )
        self._validate_checkpoint_contract()
        self.checkpoint_hash = str(
            self.runtime_config["v12_checkpoint"]["sha256"]
        )
        self.model_config_hash = stable_hash(
            {
                "model_config": self.model_config,
                "input_dim": len(FEATURE_ORDER),
                "feature_order": FEATURE_ORDER,
            }
        )
        self.normalization_config_hash = stable_hash(
            {
                "feature_order": FEATURE_ORDER,
                "feature_mean": self.model.mean.tolist(),
                "feature_scale": self.model.scale.tolist(),
                "target": "metric_delta_divided_by_current_state_tolerance",
            }
        )
        self.planner_config = self._planner_config()
        self.planner_config_hash = stable_hash(self.planner_config)
        self._measurement = None

    @property
    def model_config(self) -> dict[str, Any]:
        import torch

        artifact = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        return dict(artifact["model_config"])

    def _validate_checkpoint_contract(self) -> None:
        if tuple(OUTPUT_FIELDS) != (
            "centroid_x_px",
            "centroid_y_px",
            "sigma_x_px",
            "sigma_y_px",
            "peak_intensity",
        ):
            raise SpecialistError("runtime output order differs from the locked v12 order")
        if len(FEATURE_ORDER) != 35 or self.model.mean.shape != (35,) or self.model.scale.shape != (35,):
            raise SpecialistError("v12 checkpoint feature/normalization dimension mismatch")
        if len(self.model.members) != 3:
            raise SpecialistError("v12 checkpoint must contain three ensemble members")
        if self.model.image_conditioning:
            raise SpecialistError("image-conditioned v12 ablation is not deployable here")
        if not (
            np.array_equal(self.model.bounds.action_low, self.bounds.action_low)
            and np.array_equal(self.model.bounds.action_high, self.bounds.action_high)
            and np.array_equal(self.model.bounds.position_low, self.bounds.position_low)
            and np.array_equal(self.model.bounds.position_high, self.bounds.position_high)
        ):
            raise SpecialistError("v12 checkpoint/config bounds mismatch")

    def _planner_config(self) -> dict[str, Any]:
        locked = dict(self.v12_config["mpc"])
        declared = dict(self.runtime_config["planner"])
        locked.update(
            {
                "horizon": 1,
                "population": int(declared["population"]),
                "elites": int(declared["elites"]),
                "cem_iterations": int(declared["cem_iterations"]),
            }
        )
        return {
            **locked,
            "backend": "learned_h1_cem",
            "max_control_steps": int(declared["max_control_steps"]),
            "termination_max_normalized_error": float(
                declared["termination_max_normalized_error"]
            ),
            "seed_namespace": str(declared["seed_namespace"]),
        }

    def canonicalize_setup(self, value: Any) -> dict[str, float]:
        return _strict_mapping(value, SETUP_CONTEXT_FIELDS, "setup_context")

    def canonicalize_position(self, value: Any) -> dict[str, float]:
        output = canonical_four_vector(
            value, name="actuator_position", output_fields=POSITION_FIELDS
        )
        try:
            validate_positions(output, self.bounds)
        except ValueError as error:
            raise ContractError(str(error)) from error
        return output

    def canonicalize_action(
        self, value: Any, position: Mapping[str, Any]
    ) -> dict[str, float]:
        output = canonical_four_vector(
            value, name="continuous_action", output_fields=ACTION_FIELDS
        )
        try:
            validate_action(output, self.bounds)
            apply_action(position, output, self.bounds)
        except ValueError as error:
            raise ContractError(str(error)) from error
        return output

    def canonicalize_state(self, value: Any, name: str) -> dict[str, float]:
        output = _strict_mapping(value, OUTPUT_FIELDS, name)
        if output["sigma_x_px"] < 0.0 or output["sigma_y_px"] < 0.0:
            raise ContractError(f"{name} widths must be non-negative")
        if output["peak_intensity"] < 0.0:
            raise ContractError(f"{name}.peak_intensity must be non-negative")
        return output

    def _audit(
        self,
        *,
        run_id: str,
        task: str,
        route: str,
        started: float,
        action: Mapping[str, float] | None,
        inverse: bool = False,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "orchestrator_task": task,
            "selected_route": route,
            "executed_backend": "v12",
            "canonical_action_vector": (
                None if action is None else [float(action[field]) for field in ACTION_FIELDS]
            ),
            "canonical_action_fields": list(ACTION_FIELDS),
            "canonical_units": "mm",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_hash,
            "model_config_sha256": self.model_config_hash,
            "normalization_config_sha256": self.normalization_config_hash,
            "planner_config_sha256": self.planner_config_hash if inverse else None,
            "validation_result": "passed",
            "inference_latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    def forward(
        self,
        *,
        setup_context: Any,
        actuator_position: Any,
        current_beam_state: Any,
        continuous_action: Any,
        run_id: str,
        route: str,
        task: str = "forward_prediction_v12",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        setup = self.canonicalize_setup(setup_context)
        position = self.canonicalize_position(actuator_position)
        current = self.canonicalize_state(current_beam_state, "current_beam_state")
        action = self.canonicalize_action(continuous_action, position)
        result = self.model.predict(
            setup,
            position,
            current,
            np.asarray([[action[field] for field in ACTION_FIELDS]], dtype=np.float64),
        )
        normalized = np.asarray(result["next_metric_residual"][0], dtype=np.float64)
        physical = normalized * tolerance_vector(current)
        predicted = metrics_vector(current) + physical
        member = np.asarray(result["member_metric_residuals"][:, 0, :], dtype=np.float64)
        auxiliary = result["auxiliary_predictions"]
        return {
            "mean_normalized_metric_delta": normalized.tolist(),
            "decoded_physical_metric_delta": metrics_dict(physical),
            "predicted_next_beam_state": metrics_dict(predicted),
            "per_metric_uncertainty": np.asarray(result["uncertainty"][0]).tolist(),
            "ensemble_disagreement": member.std(axis=0).tolist(),
            "auxiliary_output": {
                "captured_power": float(auxiliary["captured_power"][0]),
                "clipping_fraction": float(auxiliary["clipping_fraction"][0]),
                "camera_boundary_probability": float(
                    auxiliary["camera_boundary_probability"][0]
                ),
                "actuator_limit_probability": float(
                    auxiliary["actuator_limit_probability"][0]
                ),
                "raw": np.asarray(result["raw_auxiliary_predictions"][0]).tolist(),
            },
            "metric_order": list(OUTPUT_FIELDS),
            "audit": self._audit(
                run_id=run_id,
                task=task,
                route=route,
                started=started,
                action=action,
            ),
        }

    def direction(self, **kwargs: Any) -> dict[str, Any]:
        kwargs["task"] = "direction_prediction_v12"
        forward = self.forward(**kwargs)
        normalized = np.asarray(forward["mean_normalized_metric_delta"], dtype=np.float64)
        labels = [
            "decrease" if value < -1.0 else "increase" if value > 1.0 else "unchanged"
            for value in normalized
        ]
        return {
            **forward,
            "directions": {
                field: labels[index] for index, field in enumerate(OUTPUT_FIELDS)
            },
            "direction_criterion": "normalized delta < -1 decrease; > 1 increase; otherwise unchanged",
        }

    def _measurement_runtime(self) -> Any:
        if self._measurement is not None:
            return self._measurement
        import torch

        from control_rebuild_v3.visual_inverse import MeasurementModuleV3
        from control_rebuild_v4.visual_runtime import VisualInversePipelineV4
        from measurement_rebuild_v4.runtime import MeasurementCalibratorRuntimeV4

        measurement = self.runtime_config["measurement"]
        model_path = _verified_path(
            {
                "path": measurement["model_path"],
                "sha256": measurement["model_sha256"],
            },
            "measurement_model",
        )
        calibrator_path = _verified_path(
            {
                "path": measurement["calibrator_path"],
                "sha256": measurement["calibrator_sha256"],
            },
            "measurement_calibrator",
        )
        runtime = VisualInversePipelineV4.__new__(VisualInversePipelineV4)
        runtime.measurement = MeasurementModuleV3(
            torch, model_path, torch.device("cpu")
        )
        runtime.measurement_calibrator = MeasurementCalibratorRuntimeV4(
            torch, calibrator_path, torch.device("cpu")
        )
        self._measurement = runtime
        return runtime

    def measure(
        self,
        *,
        image_path: Path,
        calibration: Mapping[str, Any],
        run_id: str,
        route: str,
        task: str = "measurement",
        actuator_position_mm: Mapping[str, float] | None = None,
        setup_context: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = self._measurement_runtime().measure_image(
            image_path.resolve(), calibration
        )
        sensor = self.canonicalize_state(
            {
                field: float(result["beam_state"][index])
                for index, field in enumerate(OUTPUT_FIELDS)
            },
            "measured_sensor_beam_state",
        )
        lab = dict(sensor)
        if actuator_position_mm is not None or setup_context is not None:
            if actuator_position_mm is None or setup_context is None:
                raise ContractError(
                    "image-to-v12 conversion requires both position and setup"
                )
            pitch_mm = float(setup_context["pixel_size_um"]) * 1e-3
            if pitch_mm <= 0.0:
                raise ContractError("setup_context.pixel_size_um must be positive")
            lab["centroid_x_px"] += float(actuator_position_mm["camera_x_mm"]) / pitch_mm
            lab["centroid_y_px"] += float(actuator_position_mm["camera_y_mm"]) / pitch_mm
        return {
            "beam_state": lab,
            "sensor_beam_state": sensor,
            "coordinate_frame": (
                "lab_frame_legacy_pseudo_pixels"
                if actuator_position_mm is not None
                else "camera_sensor_array"
            ),
            "measurement_source": result["measurement_source"],
            "measurement_model_version": result["model_version"],
            "audit": self._audit(
                run_id=run_id,
                task=task,
                route=route,
                started=started,
                action=None,
            ),
        }

    def inverse(
        self,
        *,
        setup_context: Any,
        actuator_position: Any,
        current_beam_state: Any,
        target_beam_state: Any,
        simulator_fixed: Mapping[str, Any],
        base_config_path: Path,
        run_id: str,
        route: str,
        planner_seed: int | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        setup = self.canonicalize_setup(setup_context)
        position = self.canonicalize_position(actuator_position)
        current = self.canonicalize_state(current_beam_state, "current_beam_state")
        target = self.canonicalize_state(target_beam_state, "target_beam_state")
        seed = int(
            planner_seed
            if planner_seed is not None
            else stable_seed(
                self.v12_config["seed"], run_id, self.planner_config["seed_namespace"]
            )
        )
        cem_config = {
            key: value
            for key, value in self.planner_config.items()
            if key
            not in {
                "backend",
                "max_control_steps",
                "termination_max_normalized_error",
                "seed_namespace",
            }
        }
        planner = CEMMPC(
            bounds=self.bounds,
            predictor=learned_predictor(self.model, setup),
            config=cem_config,
            seed=seed,
        )
        episode = run_closed_loop(
            planner=planner,
            setup_context=setup,
            simulator_fixed=simulator_fixed,
            initial_positions_mm=position,
            initial_metrics=current,
            target_metrics=target,
            allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
            base_config_path=str(base_config_path.resolve()),
            bounds=self.bounds,
            max_steps=int(self.planner_config["max_control_steps"]),
        )
        first_action = (
            None if not episode["trace"] else episode["trace"][0]["action_mm"]
        )
        return {
            "planner_backend": "learned_h1_cem",
            "planner_seed": seed,
            "planner_config": dict(self.planner_config),
            "episode": episode,
            "audit": self._audit(
                run_id=run_id,
                task="inverse_control_v12",
                route=route,
                started=started,
                action=first_action,
                inverse=True,
            ),
        }
