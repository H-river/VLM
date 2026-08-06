"""Runtime helpers for the v4 residual measurement calibrator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from measurement_rebuild_v3.common import measurement_tolerance
from measurement_rebuild_v4.models import measurement_calibrator_v4
from measurement_rebuild_v4.train_calibrator import feature_vector


class MeasurementCalibratorRuntimeV4:
    def __init__(self, torch: Any, artifact_path: Path, device: Any) -> None:
        self.torch = torch
        self.device = device
        self.artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=False
        )
        self.model = measurement_calibrator_v4(
            torch, int(self.artifact["input_dim"])
        ).to(device)
        self.model.load_state_dict(self.artifact["state_dict"])
        self.model.eval()
        self.conditions = list(self.artifact["conditions"])

    def calibrate_one(
        self,
        prediction: np.ndarray,
        calibration: Mapping[str, Any],
        transform: Mapping[str, Any],
        condition: str,
    ) -> np.ndarray:
        if condition not in self.conditions:
            raise ValueError(f"unknown calibration condition: {condition}")
        values = np.asarray(prediction, dtype=np.float32)
        features = feature_vector(
            values,
            calibration,
            transform,
            self.conditions.index(condition),
            len(self.conditions),
        )
        normalized = (
            features - np.asarray(self.artifact["feature_mean"], dtype=np.float32)
        ) / np.asarray(self.artifact["feature_scale"], dtype=np.float32)
        self.model.eval()
        with self.torch.inference_mode():
            correction = (
                self.model(
                    self.torch.as_tensor(
                        normalized[None, :],
                        dtype=self.torch.float32,
                        device=self.device,
                    )
                )
                .float()
                .cpu()
                .numpy()[0]
            )
        return (values + correction * measurement_tolerance(values)).astype(np.float32)

    def calibrate_records(
        self,
        rows: Sequence[Mapping[str, Any]],
        predictions: Mapping[tuple[str, str], np.ndarray],
        conditions: Sequence[str],
        parameters: Mapping[str, Mapping[str, Any]],
        batch_size: int = 2048,
    ) -> dict[tuple[str, str], np.ndarray]:
        keys: list[tuple[str, str]] = []
        baseline, features = [], []
        condition_position = {
            condition: index for index, condition in enumerate(self.conditions)
        }
        for row in rows:
            calibration = row["image_calibration"]
            for condition in conditions:
                if condition not in condition_position:
                    raise ValueError(f"unknown calibration condition: {condition}")
                key = (str(row["state_id"]), str(condition))
                prediction = np.asarray(predictions[key], dtype=np.float32)
                keys.append(key)
                baseline.append(prediction)
                features.append(
                    feature_vector(
                        prediction,
                        calibration,
                        parameters[condition],
                        condition_position[condition],
                        len(self.conditions),
                    )
                )
        baseline_out = np.asarray(baseline, dtype=np.float32)
        features_out = np.asarray(features, dtype=np.float32)
        features_out = (
            features_out - np.asarray(self.artifact["feature_mean"], dtype=np.float32)
        ) / np.asarray(self.artifact["feature_scale"], dtype=np.float32)
        output_parts = []
        self.model.eval()
        with self.torch.inference_mode():
            for start in range(0, len(features_out), batch_size):
                stop = min(start + batch_size, len(features_out))
                correction = (
                    self.model(
                        self.torch.as_tensor(
                            features_out[start:stop],
                            dtype=self.torch.float32,
                            device=self.device,
                        )
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                base = baseline_out[start:stop]
                tolerance = np.asarray(
                    [measurement_tolerance(row) for row in base],
                    dtype=np.float32,
                )
                output_parts.append(base + correction * tolerance)
        values = np.concatenate(output_parts)
        return {key: values[index] for index, key in enumerate(keys)}
