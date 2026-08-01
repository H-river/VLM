"""Modular image measurement and sensor-frame inverse-control runtime v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from control_rebuild_v3.inverse_runtime import state_mapping
from control_rebuild_v3.visual_inverse import (
    MeasurementModuleV3,
    legacy_candidates_to_sensor,
    sensor_to_base_legacy,
)
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from control_rebuild_v4.inverse_runtime import (
    load_inverse_runtime_v4,
    load_visual_scorer_runtime_v4,
)
from measurement_rebuild_v3.common import (
    analytic_measurement,
    measurement_tolerance,
    transform_vector,
)
from measurement_rebuild_v4.runtime import MeasurementCalibratorRuntimeV4


class VisualInversePipelineV4:
    """Compose measurement, calibration, physics, and two action scorers."""

    LEARNED_GAMMA_MIN = 0.75
    LEARNED_GAMMA_MAX = 1.05

    def __init__(
        self,
        torch: Any,
        measurement_artifact: Path,
        measurement_calibrator_artifact: Path,
        forward_artifact: Path,
        inverse_artifact: Path,
        visual_scorer_artifact: Path,
        device: Any,
    ) -> None:
        self.measurement = MeasurementModuleV3(torch, measurement_artifact, device)
        self.measurement_calibrator = MeasurementCalibratorRuntimeV4(
            torch, measurement_calibrator_artifact, device
        )
        self.forward, self.forward_artifact = load_forward_runtime_v4(
            forward_artifact, torch, device
        )
        self.inverse, self.inverse_artifact = load_inverse_runtime_v4(
            inverse_artifact, torch, device
        )
        self.visual_scorer, self.visual_artifact = load_visual_scorer_runtime_v4(
            visual_scorer_artifact, torch, device
        )

    @staticmethod
    def _image_arrays(
        image_path: Path,
        calibration: Mapping[str, Any],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
        dict[str, Any],
    ]:
        path = image_path.resolve()
        if not path.is_file():
            raise ValueError(f"beam image does not exist: {path}")
        with Image.open(path) as image:
            if len(image.getbands()) != 1:
                image = image.convert("L")
            raw = np.asarray(image)
        if raw.ndim != 2:
            raise ValueError("beam image must be a two-dimensional grayscale image")
        observed = np.asarray(raw, dtype=np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            observed /= float(np.iinfo(raw.dtype).max)
        elif float(observed.max(initial=0.0)) > 1.0:
            raise ValueError("floating-point beam images must be normalized to [0, 1]")
        observed = np.clip(observed, 0.0, 1.0)
        gamma = float(calibration.get("gamma", 1.0))
        if gamma <= 0.0:
            raise ValueError("image calibration gamma must be positive")
        high = float(calibration["linear_intensity_high"])
        low = float(calibration.get("linear_intensity_low", 0.0))
        if not high > low:
            raise ValueError("linear_intensity_high must exceed linear_intensity_low")
        source_resolution = calibration["source_sensor_resolution_px"]
        if len(source_resolution) != 2:
            raise ValueError(
                "source_sensor_resolution_px must contain width and height"
            )
        linearized = np.power(observed, 1.0 / gamma).astype(np.float32)
        valid = np.ones_like(linearized, dtype=np.float32)
        height, width = observed.shape
        full_calibration = {
            "linear_intensity_high": high,
            "linear_intensity_low": low,
            "source_sensor_resolution_px": [
                int(source_resolution[0]),
                int(source_resolution[1]),
            ],
            "stored_resolution_px": [width, height],
            "stored_bit_depth": int(calibration.get("stored_bit_depth", 16)),
            "stored_transfer": str(
                calibration.get(
                    "stored_transfer",
                    "linear" if gamma == 1.0 else "gamma_encoded",
                )
            ),
            "coordinate_frame": str(
                calibration.get("coordinate_frame", "camera_sensor_array")
            ),
        }
        transform = {
            "exposure": 1.0,
            "gamma": gamma,
            "noise_std": 0.0,
            "blur_sigma_px": 0.0,
            "saturation_level": 1.0,
            "crop_left_px": 0,
            "crop_right_px": 0,
            "crop_top_px": 0,
            "crop_bottom_px": 0,
        }
        return observed, linearized, valid, full_calibration, transform

    def measure_image(
        self,
        image_path: Path,
        calibration: Mapping[str, Any],
    ) -> dict[str, Any]:
        (
            observed,
            linearized,
            valid,
            full_calibration,
            transform,
        ) = self._image_arrays(image_path, calibration)
        baseline, analytic = analytic_measurement(
            linearized,
            valid,
            float(full_calibration["linear_intensity_high"])
            - float(full_calibration["linear_intensity_low"]),
            tuple(full_calibration["source_sensor_resolution_px"]),
        )
        baseline[4] += float(full_calibration["linear_intensity_low"])
        gamma = float(transform["gamma"])
        if not self.LEARNED_GAMMA_MIN <= gamma <= self.LEARNED_GAMMA_MAX:
            return {
                "beam_state": baseline,
                "v3_beam_state": None,
                "analytic_baseline": baseline,
                "assumed_condition": None,
                "measurement_source": "calibrated_analytic_moments",
                "selection_reason": (
                    f"gamma {gamma:g} is outside learned range "
                    f"[{self.LEARNED_GAMMA_MIN:g}, "
                    f"{self.LEARNED_GAMMA_MAX:g}]"
                ),
                "coordinate_frame": "camera_sensor_array",
                "model_version": "calibrated_analytic_moments_v1",
                "simulator_at_inference": False,
            }
        with self.measurement.torch.inference_mode():
            correction = (
                self.measurement.model(
                    self.measurement.torch.as_tensor(
                        observed[None, None, ...],
                        dtype=self.measurement.torch.float32,
                        device=self.measurement.device,
                    ),
                    self.measurement.torch.as_tensor(
                        linearized[None, None, ...],
                        dtype=self.measurement.torch.float32,
                        device=self.measurement.device,
                    ),
                    self.measurement.torch.as_tensor(
                        valid[None, None, ...],
                        dtype=self.measurement.torch.float32,
                        device=self.measurement.device,
                    ),
                    self.measurement.torch.as_tensor(
                        transform_vector(full_calibration, transform)[None, :],
                        dtype=self.measurement.torch.float32,
                        device=self.measurement.device,
                    ),
                    self.measurement.torch.as_tensor(
                        analytic[None, :],
                        dtype=self.measurement.torch.float32,
                        device=self.measurement.device,
                    ),
                )
                .float()
                .cpu()
                .numpy()[0]
            )
        v3_prediction = (
            baseline + correction * measurement_tolerance(baseline)
        ).astype(np.float32)
        condition = "clean" if abs(gamma - 1.0) < 1e-6 else "gamma_shift"
        calibrated = self.measurement_calibrator.calibrate_one(
            v3_prediction,
            full_calibration,
            transform,
            condition,
        )
        return {
            "beam_state": calibrated,
            "v3_beam_state": v3_prediction,
            "analytic_baseline": baseline,
            "assumed_condition": condition,
            "measurement_source": "measurement_v3_plus_calibrator_v4",
            "selection_reason": (
                f"gamma {gamma:g} is inside learned range "
                f"[{self.LEARNED_GAMMA_MIN:g}, {self.LEARNED_GAMMA_MAX:g}]"
            ),
            "coordinate_frame": "camera_sensor_array",
            "model_version": "measurement_rebuild_v4_one_seed",
            "simulator_at_inference": False,
        }

    def measure_dataset_states(
        self,
        data_dir: Path,
        rows: Sequence[Mapping[str, Any]],
        conditions: Sequence[str],
        parameters: Mapping[str, Mapping[str, Any]],
        batch_size: int = 16,
    ) -> dict[tuple[str, str], np.ndarray]:
        raw = self.measurement.measure_dataset_states(
            data_dir,
            rows,
            conditions,
            parameters,
            batch_size=batch_size,
        )
        return self.measurement_calibrator.calibrate_records(
            rows, raw, conditions, parameters
        )

    def predict_from_states(
        self,
        setups: Sequence[Mapping[str, Any]],
        current_sensor: np.ndarray,
        desired_sensor: np.ndarray,
        group_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        current_sensor_out = np.asarray(current_sensor, dtype=np.float32)
        desired_sensor_out = np.asarray(desired_sensor, dtype=np.float32)
        if current_sensor_out.ndim == 1:
            current_sensor_out = current_sensor_out[None, :]
        if desired_sensor_out.ndim == 1:
            desired_sensor_out = desired_sensor_out[None, :]
        if not (len(setups) == len(current_sensor_out) == len(desired_sensor_out)):
            raise ValueError("setups, current states, and desired states must align")
        current_legacy = sensor_to_base_legacy(current_sensor_out, setups)
        desired_legacy = sensor_to_base_legacy(desired_sensor_out, setups)
        ids = (
            [f"visual_v4_{index:06d}" for index in range(len(setups))]
            if group_ids is None
            else list(group_ids)
        )
        rows = [
            {
                "group_id": ids[index],
                "setup": setups[index],
                "current_beam_state": state_mapping(current_legacy[index]),
            }
            for index in range(len(setups))
        ]
        candidate_legacy = self.forward.predict_states(rows)
        candidate_sensor = legacy_candidates_to_sensor(candidate_legacy, setups)
        numerical = self.inverse.score_requests(
            setups,
            current_legacy,
            desired_legacy,
            candidate_legacy,
        )
        visual = self.visual_scorer.score_requests(
            setups,
            current_legacy,
            desired_legacy,
            candidate_sensor,
            feature_desired=desired_sensor_out,
        )
        return {
            **visual,
            "decision_source": "visual_sensor_residual_scorer_v4",
            "numerical_selected_indices": numerical["selected_indices"],
            "numerical_predicted_statuses": numerical["predicted_statuses"],
            "current_sensor_measurement": current_sensor_out,
            "desired_sensor_measurement": desired_sensor_out,
            "current_base_legacy": current_legacy,
            "desired_base_legacy": desired_legacy,
            "candidate_legacy_states": candidate_legacy,
            "candidate_sensor_states": candidate_sensor,
        }

    def predict_from_images(
        self,
        setup: Mapping[str, Any],
        current_image: Path,
        desired_image: Path,
        calibration: Mapping[str, Any],
        request_id: str = "visual_image_v4",
    ) -> dict[str, Any]:
        current = self.measure_image(current_image, calibration)
        desired = self.measure_image(desired_image, calibration)
        result = self.predict_from_states(
            [setup],
            current["beam_state"],
            desired["beam_state"],
            group_ids=[request_id],
        )
        result.update(
            {
                "current_image_measurement": current,
                "desired_image_measurement": desired,
                "input_contract": (
                    "setup + current beam image + desired beam image "
                    "+ image calibration"
                ),
            }
        )
        return result
