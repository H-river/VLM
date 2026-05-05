"""Simple offline baselines for profile2setup v2 evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from profile2setup.evaluation.profile_metrics import FEATURE_NAMES, compute_profile_features_from_path
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.normalization import (
    load_variables_config,
    make_zero_setup_vector,
    normalize_delta_vector,
    normalize_setup_vector,
)


def _records_and_config(train_dataset_or_records, variables_config_path=None):
    if hasattr(train_dataset_or_records, "records"):
        records = train_dataset_or_records.records
        variables_config = train_dataset_or_records.variables_config
    else:
        records = train_dataset_or_records
        variables_config = load_variables_config(variables_config_path or "profile2setup/configs/variables.yaml")
    return list(records), variables_config


def _mean_or_zero(rows) -> np.ndarray:
    if not rows:
        return make_zero_setup_vector()
    return np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)


def _zeros_like_batch(batch) -> torch.Tensor:
    return torch.zeros(
        (batch["current_setup"].shape[0], len(VARIABLE_ORDER)),
        dtype=batch["current_setup"].dtype,
        device=batch["current_setup"].device,
    )


def _repeat_vector(vector: np.ndarray, batch) -> torch.Tensor:
    value = torch.as_tensor(vector, dtype=batch["current_setup"].dtype, device=batch["current_setup"].device)
    return value.unsqueeze(0).repeat(batch["current_setup"].shape[0], 1)


class BaseBaseline:
    """Base interface for model-like offline baselines."""

    def fit(self, train_dataset_or_records):
        return self

    def predict_batch(self, batch):
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


class MeanAbsoluteBaseline(BaseBaseline):
    """Predict the mean normalized target setup for every sample."""

    def __init__(self, variables_config_path=None):
        self.variables_config_path = variables_config_path
        self.mean_target_setup = make_zero_setup_vector()

    def fit(self, train_dataset_or_records):
        records, variables_config = _records_and_config(train_dataset_or_records, self.variables_config_path)
        rows = [
            normalize_setup_vector(record["target_setup"], variables_config)
            for record in records
            if record.get("target_setup") is not None
        ]
        self.mean_target_setup = _mean_or_zero(rows)
        return self

    def predict_batch(self, batch):
        zeros = _zeros_like_batch(batch)
        return {
            "absolute": _repeat_vector(self.mean_target_setup, batch),
            "delta": zeros,
            "change_logits": zeros.clone(),
        }


class MeanDeltaBaseline(BaseBaseline):
    """Predict mean normalized target setup and mean normalized delta."""

    def __init__(self, variables_config_path=None):
        self.variables_config_path = variables_config_path
        self.mean_target_setup = make_zero_setup_vector()
        self.mean_target_delta = make_zero_setup_vector()

    def fit(self, train_dataset_or_records):
        records, variables_config = _records_and_config(train_dataset_or_records, self.variables_config_path)
        setup_rows = []
        delta_rows = []
        for record in records:
            if record.get("target_setup") is not None:
                setup_rows.append(normalize_setup_vector(record["target_setup"], variables_config))
            if record.get("target_delta") is not None:
                delta_rows.append(normalize_delta_vector(record["target_delta"], variables_config))
        self.mean_target_setup = _mean_or_zero(setup_rows)
        self.mean_target_delta = _mean_or_zero(delta_rows)
        return self

    def predict_batch(self, batch):
        zeros = _zeros_like_batch(batch)
        return {
            "absolute": _repeat_vector(self.mean_target_setup, batch),
            "delta": _repeat_vector(self.mean_target_delta, batch),
            "change_logits": zeros,
        }


class ZeroDeltaMeanAbsoluteBaseline(MeanAbsoluteBaseline):
    """Mean absolute setup with zero delta so routing preserves current setup when present."""


def _feature_values(features: dict | None) -> list[float]:
    if not features:
        return [0.0] * len(FEATURE_NAMES)
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def _record_feature_vector(record: dict) -> np.ndarray | None:
    values = []
    any_profile = False
    for key in ("current_profile_path", "target_profile_path"):
        path = record.get(key)
        if path and Path(path).exists():
            try:
                features = compute_profile_features_from_path(path)
            except Exception:
                features = None
            if features is not None:
                any_profile = True
            values.extend(_feature_values(features))
            values.append(1.0 if features is not None else 0.0)
        else:
            values.extend([0.0] * len(FEATURE_NAMES))
            values.append(0.0)
    if not any_profile:
        return None
    return np.asarray(values, dtype=np.float64)


class NearestNeighborProfileBaseline(BaseBaseline):
    """Nearest-neighbor baseline over simple current/target profile features."""

    def __init__(self, variables_config_path=None):
        self.variables_config_path = variables_config_path
        self.fallback = MeanDeltaBaseline(variables_config_path=variables_config_path)
        self.features = None
        self.target_setup = None
        self.target_delta = None

    def fit(self, train_dataset_or_records):
        records, variables_config = _records_and_config(train_dataset_or_records, self.variables_config_path)
        self.fallback.fit(train_dataset_or_records)
        feature_rows = []
        setup_rows = []
        delta_rows = []
        for record in records:
            if record.get("target_setup") is None:
                continue
            feature = _record_feature_vector(record)
            if feature is None:
                continue
            feature_rows.append(feature)
            setup_rows.append(normalize_setup_vector(record["target_setup"], variables_config))
            if record.get("target_delta") is not None:
                delta_rows.append(normalize_delta_vector(record["target_delta"], variables_config))
            else:
                delta_rows.append(make_zero_setup_vector())

        if feature_rows:
            self.features = np.stack(feature_rows, axis=0)
            self.target_setup = np.stack(setup_rows, axis=0).astype(np.float32)
            self.target_delta = np.stack(delta_rows, axis=0).astype(np.float32)
            scale = np.std(self.features, axis=0)
            scale[scale <= 1.0e-12] = 1.0
            self.features = self.features / scale.reshape(1, -1)
            self.feature_scale = scale
        else:
            self.features = None
            self.target_setup = None
            self.target_delta = None
            self.feature_scale = None
        return self

    def predict_batch(self, batch):
        if self.features is None or self.target_setup is None or self.target_delta is None:
            return self.fallback.predict_batch(batch)

        setup_preds = []
        delta_preds = []
        fallback_outputs = self.fallback.predict_batch(batch)
        for idx in range(batch["current_setup"].shape[0]):
            record = {
                "current_profile_path": batch["current_profile_path"][idx],
                "target_profile_path": batch["target_profile_path"][idx],
            }
            feature = _record_feature_vector(record)
            if feature is None:
                setup_preds.append(fallback_outputs["absolute"][idx].detach().cpu().numpy())
                delta_preds.append(fallback_outputs["delta"][idx].detach().cpu().numpy())
                continue
            feature = feature / self.feature_scale
            distances = np.sum(np.square(self.features - feature.reshape(1, -1)), axis=1)
            nearest = int(np.argmin(distances))
            setup_preds.append(self.target_setup[nearest])
            delta_preds.append(self.target_delta[nearest])

        absolute = torch.as_tensor(
            np.stack(setup_preds, axis=0),
            dtype=batch["current_setup"].dtype,
            device=batch["current_setup"].device,
        )
        delta = torch.as_tensor(
            np.stack(delta_preds, axis=0),
            dtype=batch["current_setup"].dtype,
            device=batch["current_setup"].device,
        )
        return {
            "absolute": absolute,
            "delta": delta,
            "change_logits": torch.zeros_like(delta),
        }
