"""Preregistered rule, numeric-MLP, random, and frequency meta baselines."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from specialist_rebuild_v2.common import ACTION_FIELDS, STATE_FIELDS

from .contracts import MetaControllerDecision, parse_meta_input, parse_meta_output


POSITION_FIELDS = ("lens_x_mm", "lens_y_mm", "camera_x_mm", "camera_y_mm")
MEASUREMENT_STATES = ("valid", "requires_recovery", "invalid")
SUPERVISOR_DIAGNOSES = (
    "nominal",
    "sensor_saturation",
    "secondary_reflection",
    "unknown",
)
MEASUREMENT_POLICIES = (
    "standard",
    "lower_exposure_reacquire",
    "primary_spot",
    "unknown",
)


def canonical_configuration(value: Mapping[str, Any] | MetaControllerDecision) -> str:
    decision = parse_meta_output(value)
    return json.dumps(
        decision.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _base_output(
    *,
    decision: str,
    observation_request: str = "reuse_current",
    objective_profile: str = "balanced",
    mask_profile: str = "default",
    directions: Mapping[str, str] | None = None,
    step_scale: str = "default",
    risk_mode: str = "standard",
    confidence: str = "medium",
    reason_codes: Sequence[str] = ("insufficient_history",),
) -> dict[str, Any]:
    output = {
        "schema_version": "qwen_h1_meta_v0",
        "decision": decision,
        "observation_request": observation_request,
        "objective_profile": objective_profile,
        "mask_profile": mask_profile,
        "directional_prior": {
            field: "unknown" for field in ACTION_FIELDS
        }
        if directions is None
        else {field: directions[field] for field in ACTION_FIELDS},
        "step_scale": step_scale,
        "risk_mode": risk_mode,
        "confidence": confidence,
        "reason_codes": list(dict.fromkeys(reason_codes)),
    }
    return parse_meta_output(output).to_dict()


class RuleMetaBaseline:
    """Non-weak deterministic baseline using only runtime-visible evidence."""

    uncertainty_threshold = 5.0
    near_target_threshold = 1.0
    medium_scale_threshold = 2.0

    def predict(self, runtime_input: Mapping[str, Any]) -> dict[str, Any]:
        state = parse_meta_input(runtime_input).to_dict()
        validity = state["measurement_validity"]["state"]
        budget = state["remaining_budget"]
        if validity != "valid":
            return _base_output(
                decision="reobserve",
                observation_request="revalidate",
                risk_mode="conservative",
                confidence="high",
                reason_codes=("measurement_requires_revalidation",),
            )
        if int(budget["control_steps"]) <= 0:
            return _base_output(
                decision="stop",
                risk_mode="conservative",
                confidence="high",
                reason_codes=("budget_exhausted",),
            )

        error = np.asarray(
            [float(state["normalized_signed_error"][field]) for field in STATE_FIELDS],
            dtype=np.float64,
        )
        current = np.asarray(
            [float(state["current_beam_state"][field]) for field in STATE_FIELDS],
            dtype=np.float64,
        )
        target = np.asarray(
            [float(state["target_beam_state"][field]) for field in STATE_FIELDS],
            dtype=np.float64,
        )
        # Infer exactly the runtime normalization represented by the visible
        # signed error. Zero-error dimensions never affect the dot product and
        # use a neutral scale solely to avoid division by zero.
        tolerance = np.ones(len(STATE_FIELDS), dtype=np.float64)
        nonzero_error = np.abs(error) > 1e-12
        tolerance[nonzero_error] = np.abs(
            (current[nonzero_error] - target[nonzero_error]) / error[nonzero_error]
        )
        tolerance[tolerance < 1e-12] = 1.0
        maximum_error = float(np.max(np.abs(error)))
        if maximum_error <= self.near_target_threshold:
            return _base_output(
                decision="run_default_h1",
                step_scale="fine",
                risk_mode="conservative",
                confidence="high",
                reason_codes=("near_target",),
            )

        group_scores = {
            "centroid_priority": float(np.max(np.abs(error[:2]))),
            "width_priority": float(np.max(np.abs(error[2:4]))),
            "intensity_priority": float(abs(error[4])),
        }
        objective = min(
            group_scores,
            key=lambda name: (-group_scores[name], name),
        )
        reason = {
            "centroid_priority": "centroid_error_dominant",
            "width_priority": "width_error_dominant",
            "intensity_priority": "intensity_error_dominant",
        }[objective]

        valid_history = [row for row in state["history"] if row["valid"]]
        directions = {field: "unknown" for field in ACTION_FIELDS}
        evidence = np.zeros(len(ACTION_FIELDS), dtype=np.float64)
        mismatch = False
        for row in valid_history:
            action = np.asarray(
                [float(row["executed_action_mm"][field]) for field in ACTION_FIELDS]
            )
            measured = np.asarray(
                [float(row["measured_beam_delta"][field]) for field in STATE_FIELDS]
            ) / tolerance
            residual = np.asarray(
                [float(row["prediction_residual"][field]) for field in STATE_FIELDS]
            ) / tolerance
            # normalized_signed_error is (target-current)/tolerance, so a
            # measured delta aligned with +error is productive.
            scalar_benefit = float(np.dot(error, measured))
            evidence += action * scalar_benefit
            mismatch |= bool(np.linalg.norm(residual) > np.linalg.norm(measured) + 1e-12)
        for index, field in enumerate(ACTION_FIELDS):
            if evidence[index] > 1e-12:
                directions[field] = "increase"
            elif evidence[index] < -1e-12:
                directions[field] = "decrease"

        uncertainty = float(state["forward_uncertainty"]["mean"])
        elevated = uncertainty >= self.uncertainty_threshold
        if not valid_history:
            return _base_output(
                decision="run_default_h1",
                objective_profile=objective,
                step_scale="fine" if maximum_error <= 2.0 else "medium",
                risk_mode="conservative",
                confidence="low",
                reason_codes=(reason, "insufficient_history"),
            )
        if elevated or mismatch:
            return _base_output(
                decision="run_default_h1",
                objective_profile=objective,
                directions=directions,
                step_scale="fine",
                risk_mode="conservative",
                confidence="medium",
                reason_codes=(
                    reason,
                    "elevated_uncertainty" if elevated else "recent_response_mismatch",
                ),
            )
        return _base_output(
            decision="run_guided_h1",
            objective_profile=objective,
            mask_profile="all_actuators",
            directions=directions,
            step_scale="medium" if maximum_error > self.medium_scale_threshold else "fine",
            risk_mode="standard",
            confidence="high" if len(valid_history) >= 2 else "medium",
            reason_codes=(reason, "recent_response_consistent"),
        )


class FrequencyMetaBaseline:
    """Most frequent complete valid configuration; ties are lexical."""

    def __init__(self) -> None:
        self.configuration: dict[str, Any] | None = None

    def fit(
        self, targets: Sequence[Mapping[str, Any] | MetaControllerDecision]
    ) -> "FrequencyMetaBaseline":
        encoded = [canonical_configuration(target) for target in targets]
        if not encoded:
            raise ValueError("frequency baseline requires at least one target")
        counts = Counter(encoded)
        selected = min(counts, key=lambda text: (-counts[text], text))
        self.configuration = parse_meta_output(selected).to_dict()
        return self

    def predict(self, runtime_input: Mapping[str, Any]) -> dict[str, Any]:
        parse_meta_input(runtime_input)
        if self.configuration is None:
            raise RuntimeError("frequency baseline must be fit before prediction")
        return json.loads(canonical_configuration(self.configuration))


class RandomMetaBaseline:
    """Order-independent deterministic sampling from observed train configs."""

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.configurations: list[str] = []

    def fit(
        self, targets: Sequence[Mapping[str, Any] | MetaControllerDecision]
    ) -> "RandomMetaBaseline":
        self.configurations = sorted({canonical_configuration(target) for target in targets})
        if not self.configurations:
            raise ValueError("random baseline requires at least one target")
        return self

    def predict(self, runtime_input: Mapping[str, Any], *, sample_id: str) -> dict[str, Any]:
        parse_meta_input(runtime_input)
        if not self.configurations:
            raise RuntimeError("random baseline must be fit before prediction")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        digest = hashlib.sha256(f"{self.seed}\x1f{sample_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(self.configurations)
        return parse_meta_output(self.configurations[index]).to_dict()


class NumericFeatureEncoder:
    """Fixed-order numeric view of every runtime-visible non-image field."""

    def transform_one(self, runtime_input: Mapping[str, Any]) -> np.ndarray:
        state = parse_meta_input(runtime_input).to_dict()
        values: list[float] = []
        for key in ("current_beam_state", "target_beam_state", "normalized_signed_error"):
            values.extend(float(state[key][field]) for field in STATE_FIELDS)
        values.extend(float(state["actuator_positions_mm"][field]) for field in POSITION_FIELDS)
        semantics = {row["action_id"]: row for row in state["actuator_semantics"]}
        for field in ACTION_FIELDS:
            values.extend(float(value) for value in semantics[field]["legal_per_step_bounds_mm"])
            # The positive/negative semantics strings are model-visible. A
            # fixed small SHA-256 projection lets the numeric baseline consume
            # them without a learned tokenizer or a hidden repository ID.
            for key in ("positive_command_semantics", "negative_command_semantics"):
                digest = hashlib.sha256(str(semantics[field][key]).encode("utf-8")).digest()
                values.extend((byte - 127.5) / 127.5 for byte in digest[:8])
        for row in state["history"]:
            values.append(float(row["valid"]))
            values.extend(float(row["executed_action_mm"][field]) for field in ACTION_FIELDS)
            for key in (
                "measured_beam_delta",
                "predicted_beam_delta",
                "prediction_residual",
                "ensemble_uncertainty",
            ):
                values.extend(float(row[key][field]) for field in STATE_FIELDS)
        forward = state["forward_uncertainty"]
        values.extend(float(forward["per_metric"][field]) for field in STATE_FIELDS)
        values.extend((float(forward["mean"]), float(forward["maximum"])))
        values.extend(
            (
                float(state["remaining_budget"]["measurement_steps"]),
                float(state["remaining_budget"]["control_steps"]),
            )
        )
        validity = state["measurement_validity"]
        values.extend(float(validity["state"] == item) for item in MEASUREMENT_STATES)
        values.extend(
            float(validity["supervisor_diagnosis"] == item)
            for item in SUPERVISOR_DIAGNOSES
        )
        values.extend(
            float(validity["measurement_policy"] == item)
            for item in MEASUREMENT_POLICIES
        )
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("numeric feature encoder produced non-finite values")
        return array

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not rows:
            raise ValueError("numeric feature encoder requires at least one row")
        return np.stack([self.transform_one(row) for row in rows], axis=0)


@dataclass(frozen=True)
class MLPConfig:
    hidden_units: int = 64
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 0.001
    seed: int = 2026080201


class NumericMLPMetaBaseline:
    """Two-hidden-layer NumPy ReLU classifier over complete configurations."""

    def __init__(self, config: MLPConfig = MLPConfig()) -> None:
        if config.hidden_units <= 0 or config.epochs <= 0 or config.batch_size <= 0:
            raise ValueError("MLP dimensions, epochs, and batch size must be positive")
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
            raise ValueError("MLP learning rate must be finite and positive")
        self.config = config
        self.encoder = NumericFeatureEncoder()
        self.classes: list[str] = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.parameters: dict[str, np.ndarray] = {}
        self.best_epoch: int | None = None
        self.best_dev_loss: float | None = None
        self.unseen_dev_configuration_count: int = 0

    @staticmethod
    def _loss_and_gradient(
        features: np.ndarray,
        labels: np.ndarray,
        parameters: Mapping[str, np.ndarray],
    ) -> tuple[float, dict[str, np.ndarray]]:
        z1 = features @ parameters["w1"] + parameters["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ parameters["w2"] + parameters["b2"]
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ parameters["w3"] + parameters["b3"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)).mean())
        dlogits = probabilities
        dlogits[np.arange(len(labels)), labels] -= 1.0
        dlogits /= len(labels)
        gradients: dict[str, np.ndarray] = {}
        gradients["w3"] = h2.T @ dlogits
        gradients["b3"] = dlogits.sum(axis=0)
        dh2 = dlogits @ parameters["w3"].T
        dz2 = dh2 * (z2 > 0.0)
        gradients["w2"] = h1.T @ dz2
        gradients["b2"] = dz2.sum(axis=0)
        dh1 = dz2 @ parameters["w2"].T
        dz1 = dh1 * (z1 > 0.0)
        gradients["w1"] = features.T @ dz1
        gradients["b1"] = dz1.sum(axis=0)
        return loss, gradients

    @staticmethod
    def _loss(
        features: np.ndarray, labels: np.ndarray, parameters: Mapping[str, np.ndarray]
    ) -> float:
        loss, _ = NumericMLPMetaBaseline._loss_and_gradient(features, labels, parameters)
        return loss

    @staticmethod
    def _dev_loss_with_unknown(
        features: np.ndarray,
        encoded_targets: Sequence[str],
        class_index: Mapping[str, int],
        parameters: Mapping[str, np.ndarray],
    ) -> float:
        z1 = features @ parameters["w1"] + parameters["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ parameters["w2"] + parameters["b2"]
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ parameters["w3"] + parameters["b3"]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        losses = []
        for index, target in enumerate(encoded_targets):
            if target in class_index:
                probability = probabilities[index, class_index[target]]
            else:
                # Honest unknown handling: the model cannot assign probability
                # to a complete configuration absent from train. Keep it in the
                # full-dev denominator with a fixed near-zero probability.
                probability = 1e-12
            losses.append(-math.log(max(float(probability), 1e-300)))
        return statistics.fmean(losses)

    def fit(
        self,
        train_inputs: Sequence[Mapping[str, Any]],
        train_targets: Sequence[Mapping[str, Any] | MetaControllerDecision],
        *,
        dev_inputs: Sequence[Mapping[str, Any]],
        dev_targets: Sequence[Mapping[str, Any] | MetaControllerDecision],
    ) -> "NumericMLPMetaBaseline":
        if len(train_inputs) != len(train_targets) or not train_inputs:
            raise ValueError("train inputs/targets must be nonempty and aligned")
        if len(dev_inputs) != len(dev_targets) or not dev_inputs:
            raise ValueError("full development inputs/targets must be nonempty and aligned")
        train_encoded = [canonical_configuration(target) for target in train_targets]
        dev_encoded = [canonical_configuration(target) for target in dev_targets]
        self.classes = sorted(set(train_encoded))
        class_index = {value: index for index, value in enumerate(self.classes)}
        unknown = sorted(set(dev_encoded) - set(class_index))
        self.unseen_dev_configuration_count = sum(
            value not in class_index for value in dev_encoded
        )
        train_labels = np.asarray([class_index[value] for value in train_encoded], dtype=np.int64)
        raw_train = self.encoder.transform(train_inputs)
        raw_dev = self.encoder.transform(dev_inputs)
        self.mean = raw_train.mean(axis=0)
        self.scale = raw_train.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        train = (raw_train - self.mean) / self.scale
        dev = (raw_dev - self.mean) / self.scale

        rng = np.random.default_rng(self.config.seed)
        hidden = self.config.hidden_units
        self.parameters = {
            "w1": rng.normal(0.0, np.sqrt(2.0 / train.shape[1]), (train.shape[1], hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0.0, np.sqrt(2.0 / hidden), (hidden, hidden)),
            "b2": np.zeros(hidden),
            "w3": rng.normal(0.0, np.sqrt(2.0 / hidden), (hidden, len(self.classes))),
            "b3": np.zeros(len(self.classes)),
        }
        best_parameters: dict[str, np.ndarray] | None = None
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(1, self.config.epochs + 1):
            order = rng.permutation(len(train))
            for start in range(0, len(order), self.config.batch_size):
                indices = order[start : start + self.config.batch_size]
                _, gradients = self._loss_and_gradient(
                    train[indices], train_labels[indices], self.parameters
                )
                for name in self.parameters:
                    self.parameters[name] -= self.config.learning_rate * gradients[name]
            dev_loss = self._dev_loss_with_unknown(
                dev, dev_encoded, class_index, self.parameters
            )
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch
                best_parameters = {
                    name: value.copy() for name, value in self.parameters.items()
                }
        if best_parameters is None:
            raise RuntimeError("MLP checkpoint selection did not observe a finite dev loss")
        self.parameters = best_parameters
        self.best_epoch = best_epoch
        self.best_dev_loss = best_loss
        return self

    def _standardized(self, inputs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.mean is None or self.scale is None or not self.parameters:
            raise RuntimeError("numeric MLP must be fit before prediction")
        return (self.encoder.transform(inputs) - self.mean) / self.scale

    def predict_proba(self, inputs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        features = self._standardized(inputs)
        h1 = np.maximum(features @ self.parameters["w1"] + self.parameters["b1"], 0.0)
        h2 = np.maximum(h1 @ self.parameters["w2"] + self.parameters["b2"], 0.0)
        logits = h2 @ self.parameters["w3"] + self.parameters["b3"]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        probabilities = self.predict_proba(inputs)
        return [
            parse_meta_output(self.classes[int(index)]).to_dict()
            for index in probabilities.argmax(axis=1)
        ]

    def state_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None or not self.parameters:
            raise RuntimeError("numeric MLP must be fit before serialization")
        return {
            "version": "qwen_h1_meta_numeric_mlp_v0",
            "config": self.config.__dict__,
            "classes": list(self.classes),
            "feature_mean": self.mean.tolist(),
            "feature_scale": self.scale.tolist(),
            "parameters": {name: value.tolist() for name, value in self.parameters.items()},
            "best_epoch": self.best_epoch,
            "best_dev_loss": self.best_dev_loss,
            "unseen_dev_configuration_count": self.unseen_dev_configuration_count,
            "unseen_dev_configuration_policy": (
                "retain_in_full_dev_loss_with_fixed_probability_1e-12; "
                "never synthesize or leak a dev-only class"
            ),
            "standardization": "train_mean_and_population_standard_deviation_only",
            "actuator_semantics_encoding": (
                "first_8_sha256_bytes_per_visible_positive_and_negative_semantics_string"
            ),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "NumericMLPMetaBaseline":
        if state.get("version") != "qwen_h1_meta_numeric_mlp_v0":
            raise ValueError("unexpected numeric MLP state version")
        raw_config = state.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("numeric MLP state config must be an object")
        model = cls(MLPConfig(**{key: raw_config[key] for key in MLPConfig.__dataclass_fields__}))
        classes = state.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("numeric MLP state must contain nonempty classes")
        model.classes = [canonical_configuration(parse_meta_output(value)) for value in classes]
        model.mean = np.asarray(state["feature_mean"], dtype=np.float64)
        model.scale = np.asarray(state["feature_scale"], dtype=np.float64)
        if (
            model.mean.ndim != 1
            or model.scale.shape != model.mean.shape
            or not np.isfinite(model.mean).all()
            or not np.isfinite(model.scale).all()
            or np.any(model.scale <= 0)
        ):
            raise ValueError("numeric MLP state has invalid standardization arrays")
        raw_parameters = state.get("parameters")
        if not isinstance(raw_parameters, Mapping) or set(raw_parameters) != {
            "w1",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
        }:
            raise ValueError("numeric MLP state has incomplete parameters")
        model.parameters = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in raw_parameters.items()
        }
        if not all(np.isfinite(value).all() for value in model.parameters.values()):
            raise ValueError("numeric MLP state has non-finite parameters")
        hidden = model.config.hidden_units
        expected_shapes = {
            "w1": (len(model.mean), hidden),
            "b1": (hidden,),
            "w2": (hidden, hidden),
            "b2": (hidden,),
            "w3": (hidden, len(model.classes)),
            "b3": (len(model.classes),),
        }
        observed_shapes = {
            name: value.shape for name, value in model.parameters.items()
        }
        if observed_shapes != expected_shapes:
            raise ValueError(
                f"numeric MLP parameter shapes changed: {observed_shapes} != {expected_shapes}"
            )
        model.best_epoch = int(state["best_epoch"])
        model.best_dev_loss = float(state["best_dev_loss"])
        model.unseen_dev_configuration_count = int(
            state.get("unseen_dev_configuration_count", 0)
        )
        return model

    def save(self, path: str | Any) -> None:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.state_dict(), ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Any) -> "NumericMLPMetaBaseline":
        from pathlib import Path

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise ValueError("numeric MLP state file must contain one JSON object")
        return cls.from_state_dict(state)


__all__ = [
    "FrequencyMetaBaseline",
    "MLPConfig",
    "NumericFeatureEncoder",
    "NumericMLPMetaBaseline",
    "RandomMetaBaseline",
    "RuleMetaBaseline",
    "canonical_configuration",
]
