"""Modular visual inverse control: measure, adapt frame, predict, and score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_VALUES
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.inverse_runtime import load_inverse_runtime, state_mapping
from measurement_rebuild_v3.common import (
    STATE_FIELDS,
    analytic_measurement,
    measurement_tolerance,
    stable_seed,
    transform_vector,
)
from measurement_rebuild_v3.models import measurement_model_v3
from measurement_rebuild_v3.train import (
    apply_transform,
    load_linear_image,
)
from specialist_rebuild_v2.common import raw_state_array


def camera_pitch_mm(setup: Mapping[str, Any]) -> float:
    pitch = float(setup["pixel_size_um"]) / 1000.0
    if pitch <= 0.0:
        raise ValueError("pixel_size_um must be positive")
    return pitch


def sensor_to_base_legacy(
    states: np.ndarray, setups: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Express sensor-array centroids in the base camera's legacy frame."""

    output = np.asarray(states, dtype=np.float32).copy()
    if output.ndim == 1:
        output = output[None, :]
    if len(output) != len(setups):
        raise ValueError("one setup is required per state")
    for index, setup in enumerate(setups):
        pitch = camera_pitch_mm(setup)
        output[index, 0] += float(setup["camera_x_offset_mm"]) / pitch
        output[index, 1] += float(setup["camera_y_offset_mm"]) / pitch
    return output


def legacy_candidates_to_sensor(
    states: np.ndarray, setups: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Convert each action candidate into its resulting sensor-array frame."""

    output = np.asarray(states, dtype=np.float32).copy()
    if output.ndim != 3 or output.shape[1:] != (81, 5):
        raise ValueError("candidate states must have shape [requests, 81, 5]")
    if len(output) != len(setups):
        raise ValueError("one setup is required per request")
    for index, setup in enumerate(setups):
        pitch = camera_pitch_mm(setup)
        output[index, :, 0] -= (
            float(setup["camera_x_offset_mm"]) + ACTION_VALUES[:, 2]
        ) / pitch
        output[index, :, 1] -= (
            float(setup["camera_y_offset_mm"]) + ACTION_VALUES[:, 3]
        ) / pitch
    return output


class MeasurementModuleV3:
    """Load the image measurement model once for repeated pipeline calls."""

    def __init__(
        self, torch: Any, artifact_path: Path, device: Any
    ) -> None:
        self.torch = torch
        self.device = device
        self.artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=False
        )
        contract = self.artifact["input_contract"]
        self.model = measurement_model_v3(
            torch,
            calibration_dim=int(contract["calibration_dim"]),
            analytic_dim=int(contract["analytic_dim"]),
        ).to(device)
        self.model.load_state_dict(self.artifact["state_dict"])
        self.model.eval()

    def measure_dataset_states(
        self,
        data_dir: Path,
        rows: Sequence[Mapping[str, Any]],
        conditions: Sequence[str],
        parameters: Mapping[str, Mapping[str, Any]],
        batch_size: int = 16,
    ) -> dict[tuple[str, str], np.ndarray]:
        """Measure deterministic rendered views keyed by (state_id, condition)."""

        keys: list[tuple[str, str]] = []
        predictions: list[np.ndarray] = []
        prepared: list[tuple[np.ndarray, ...]] = []

        def flush() -> None:
            if not prepared:
                return
            columns = [
                np.stack([item[index] for item in prepared])
                for index in range(7)
            ]
            with self.torch.inference_mode():
                correction = self.model(
                    *[
                        self.torch.as_tensor(
                            columns[index],
                            dtype=self.torch.float32,
                            device=self.device,
                        )
                        for index in range(5)
                    ]
                )
                predicted = (
                    self.torch.as_tensor(
                        columns[5],
                        dtype=self.torch.float32,
                        device=self.device,
                    )
                    + correction
                    * self.torch.as_tensor(
                        columns[6],
                        dtype=self.torch.float32,
                        device=self.device,
                    )
                )
            predictions.extend(predicted.float().cpu().numpy())
            prepared.clear()

        for row in rows:
            base = load_linear_image(data_dir / str(row["base_image"]))
            calibration = row["image_calibration"]
            for condition in conditions:
                transform = parameters[condition]
                observed, linearized, valid = apply_transform(
                    base,
                    transform,
                    stable_seed(row["state_id"], condition, "view"),
                )
                baseline, analytic = analytic_measurement(
                    linearized,
                    valid,
                    float(calibration["linear_intensity_high"]),
                    tuple(calibration["source_sensor_resolution_px"]),
                )
                prepared.append(
                    (
                        observed[None].astype(np.float32),
                        linearized[None].astype(np.float32),
                        valid[None].astype(np.float32),
                        transform_vector(calibration, transform),
                        analytic,
                        baseline,
                        measurement_tolerance(baseline),
                    )
                )
                keys.append((str(row["state_id"]), str(condition)))
                if len(prepared) == batch_size:
                    flush()
        flush()
        return {
            key: predictions[index] for index, key in enumerate(keys)
        }


class VisualInversePipeline:
    """Compose frozen measurement, forward, frame, and inverse modules."""

    def __init__(
        self,
        torch: Any,
        measurement_artifact: Path,
        forward_artifact: Path,
        inverse_artifact: Path,
        device: Any,
    ) -> None:
        self.measurement = MeasurementModuleV3(
            torch, measurement_artifact, device
        )
        self.forward, _ = load_forward_runtime(
            forward_artifact, torch, device
        )
        self.inverse, _ = load_inverse_runtime(
            inverse_artifact, torch, device
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
        current_legacy = sensor_to_base_legacy(current_sensor_out, setups)
        desired_base_legacy = sensor_to_base_legacy(desired_sensor_out, setups)
        ids = (
            [f"visual_{index:06d}" for index in range(len(setups))]
            if group_ids is None
            else list(group_ids)
        )
        forward_rows = [
            {
                "group_id": ids[index],
                "setup": setups[index],
                "current_beam_state": state_mapping(current_legacy[index]),
            }
            for index in range(len(setups))
        ]
        candidate_legacy = self.forward.predict_states(forward_rows)
        candidate_sensor = legacy_candidates_to_sensor(
            candidate_legacy, setups
        )
        result = self.inverse.score_requests(
            setups,
            current_legacy,
            desired_base_legacy,
            candidate_sensor,
            feature_desired=desired_sensor_out,
        )
        result.update(
            {
                "current_sensor_measurement": current_sensor_out,
                "desired_sensor_measurement": desired_sensor_out,
                "current_base_legacy": current_legacy,
                "desired_base_legacy": desired_base_legacy,
                "candidate_legacy_states": candidate_legacy,
                "candidate_sensor_states": candidate_sensor,
            }
        )
        return result


def true_grid_sensor_states(
    grid: Mapping[str, Any],
) -> np.ndarray:
    legacy = np.asarray(
        [
            raw_state_array(candidate["next_state"])
            for candidate in grid["candidates"]
        ],
        dtype=np.float32,
    )
    return legacy_candidates_to_sensor(
        legacy[None, ...], [grid["setup"]]
    )[0]
