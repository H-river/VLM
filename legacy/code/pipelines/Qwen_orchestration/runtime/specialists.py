"""Read-only adapters for the frozen numerical and visual specialists."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .errors import ContractError, SpecialistError
from .numerics import (
    ACTION_FIELDS,
    CLASSES,
    DIRECTION_FIELDS,
    MATCHING_TOLERANCE,
    SETUP_FIELDS,
    STATE_FIELDS,
    classify_residuals,
    fixed_action_grid,
    inverse_feature_vector,
    legacy_to_sensor,
    model_feature_vector,
    movement_mm,
    sensor_to_legacy_initial,
    state_residual,
    strict_numeric_mapping,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
MANIFEST_PATH = PACKAGE_ROOT / "freeze/baseline_manifest.json"

DIRECTION_BUNDLE = (
    REPO_ROOT
    / "optics_understanding_sft/direction_inverse_v1/results/"
    "direction_small_v1/direction_small_ensemble.pkl"
)
GENERAL_FORWARD_BUNDLE = (
    REPO_ROOT
    / "optics_understanding_sft/direction_inverse_v1/results/"
    "forward_hybrid_v1/forward_hybrid.pkl"
)
GRID_FORWARD_BUNDLE = (
    REPO_ROOT
    / "optics_understanding_sft/direction_inverse_v1/results/"
    "forward_grid_hybrid_v1/forward_hybrid.pkl"
)
DIRECT_INVERSE_BUNDLE = (
    REPO_ROOT
    / "optics_understanding_sft/direction_inverse_v1/results/"
    "inverse_direct_v1/inverse_direct.pkl"
)
INVERSE_SUMMARY = (
    REPO_ROOT
    / "optics_understanding_sft/direction_inverse_v1/results/"
    "inverse_ensemble_v1/summary.json"
)
VISUAL_SUMMARY = (
    REPO_ROOT
    / "optics_understanding_sft/direction_inverse_v1/results/"
    "visual_pipeline_sensor_v1/summary.json"
)


@lru_cache(maxsize=1)
def _manifest_files() -> dict[str, dict[str, Any]]:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    return {entry["path"]: entry for entry in manifest["files"]}


@lru_cache(maxsize=None)
def _verify_frozen_file(path_text: str) -> Path:
    path = Path(path_text).resolve()
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise SpecialistError(f"artifact is outside the repository: {path}") from error
    entry = _manifest_files().get(relative)
    if entry is None:
        raise SpecialistError(f"artifact is not in the frozen manifest: {relative}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise SpecialistError(f"frozen artifact integrity failure: {relative}")
    return path


@lru_cache(maxsize=None)
def _load_pickle(path_text: str) -> Any:
    path = _verify_frozen_file(path_text)
    with path.open("rb") as stream:
        return pickle.load(stream)


@lru_cache(maxsize=None)
def _load_json(path_text: str) -> dict[str, Any]:
    path = _verify_frozen_file(path_text)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=-1, keepdims=True)


def _direction_model(torch: Any, input_dim: int) -> Any:
    class DirectionNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.LayerNorm(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(128, 64),
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(),
            )
            self.heads = torch.nn.ModuleList(
                [torch.nn.Linear(64, len(CLASSES)) for _ in DIRECTION_FIELDS]
            )

        def forward(self, values: Any) -> Any:
            hidden = self.encoder(values)
            return torch.stack([head(hidden) for head in self.heads], dim=1)

    return DirectionNet()


def _direction_member_logits(bundle: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = []
    with torch.inference_mode():
        for member in bundle["members"]:
            model = _direction_model(torch, x.shape[1]).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            scaled = (x - member["mean"]) / member["scale"]
            batches = []
            for start in range(0, len(scaled), 1024):
                values = torch.as_tensor(
                    scaled[start : start + 1024],
                    dtype=torch.float32,
                    device=device,
                )
                batches.append(model(values).float().cpu().numpy())
            outputs.append(np.concatenate(batches))
    return np.mean(np.stack(outputs), axis=0)


def _forward_model(torch: Any, input_dim: int) -> Any:
    class ForwardNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.LayerNorm(128),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.08),
                torch.nn.Linear(128, 64),
                torch.nn.LayerNorm(64),
                torch.nn.SiLU(),
            )
            self.regression_head = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, len(STATE_FIELDS)),
            )
            self.direction_heads = torch.nn.ModuleList(
                [torch.nn.Linear(64, len(CLASSES)) for _ in DIRECTION_FIELDS]
            )

        def forward(self, values: Any) -> tuple[Any, Any]:
            hidden = self.encoder(values)
            return self.regression_head(hidden), torch.stack(
                [head(hidden) for head in self.direction_heads], dim=1
            )

    return ForwardNet()


def _forward_member_prediction(
    bundle: Mapping[str, Any], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    changes, logits = [], []
    with torch.inference_mode():
        for member in bundle["members"]:
            model = _forward_model(torch, x.shape[1]).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            scaled = (x - member["mean"]) / member["scale"]
            member_changes, member_logits = [], []
            for start in range(0, len(scaled), 1024):
                values = torch.as_tensor(
                    scaled[start : start + 1024],
                    dtype=torch.float32,
                    device=device,
                )
                change, direction = model(values)
                member_changes.append(change.float().cpu().numpy())
                member_logits.append(direction.float().cpu().numpy())
            changes.append(np.concatenate(member_changes))
            logits.append(np.concatenate(member_logits))
    return np.mean(np.stack(changes), axis=0), np.mean(np.stack(logits), axis=0)


def _engineered_features(x: np.ndarray) -> np.ndarray:
    fields = (*SETUP_FIELDS, *STATE_FIELDS, *ACTION_FIELDS)
    column = {name: index for index, name in enumerate(fields)}
    lx, ly = x[:, column["lens_x_offset_mm"]], x[:, column["lens_y_offset_mm"]]
    cx, cy = x[:, column["camera_x_offset_mm"]], x[:, column["camera_y_offset_mm"]]
    dlx, dly = x[:, column["lens_x_delta_mm"]], x[:, column["lens_y_delta_mm"]]
    dcx, dcy = x[:, column["camera_x_delta_mm"]], x[:, column["camera_y_delta_mm"]]
    pitch_mm = x[:, column["pixel_size_um"]] / 1000.0
    ratio = (
        x[:, column["lens_to_camera_mm"]]
        / x[:, column["lens_focal_length_mm"]]
    ) / pitch_mm
    derived = np.column_stack(
        [
            lx + dlx,
            ly + dly,
            cx + dcx,
            cy + dcy,
            np.square(lx + dlx) - np.square(lx),
            np.square(ly + dly) - np.square(ly),
            np.square(cx + dcx) - np.square(cx),
            np.square(cy + dcy) - np.square(cy),
            np.hypot(lx + dlx, ly + dly),
            np.hypot(lx, ly),
            np.hypot(cx + dcx, cy + dcy),
            np.hypot(cx, cy),
            dlx * ratio,
            dly * ratio,
            dcx / pitch_mm,
            dcy / pitch_mm,
            ratio,
            dlx * lx,
            dly * ly,
            dcx * cx,
            dcy * cy,
            np.abs(lx + dlx) - np.abs(lx),
            np.abs(ly + dly) - np.abs(ly),
            np.abs(cx + dcx) - np.abs(cx),
            np.abs(cy + dcy) - np.abs(cy),
        ]
    ).astype(np.float32)
    return np.concatenate([x, derived], axis=1)


def _predict_forward_bundle(
    bundle: Mapping[str, Any], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if bundle.get("model_kind") == "neural_hist_blend":
        neural_change, neural_logits = _predict_forward_bundle(
            bundle["neural_bundle"], x
        )
        tree_change = np.asarray(
            bundle["hist_model"].predict(_engineered_features(x)), dtype=np.float32
        )
        alpha = float(bundle["neural_weight"])
        return alpha * neural_change + (1.0 - alpha) * tree_change, neural_logits
    return _forward_member_prediction(bundle, x)


def _direction_from_state(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = _load_pickle(str(DIRECTION_BUNDLE))
    x = model_feature_vector(setup, current, action)[None, :]
    logits = _direction_member_logits(bundle, x)[0]
    probabilities = _softmax(logits)
    directions = {
        field: CLASSES[int(np.argmax(probabilities[index]))]
        for index, field in enumerate(DIRECTION_FIELDS)
    }
    by_class = {
        field: {
            class_name: float(probabilities[field_index, class_index])
            for class_index, class_name in enumerate(CLASSES)
        }
        for field_index, field in enumerate(DIRECTION_FIELDS)
    }
    return {
        "directions": directions,
        "probabilities": by_class,
        "model_version": str(bundle["version"]),
        "simulator_at_inference": False,
    }


def _forward_from_state(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    bundle_path: Path = GENERAL_FORWARD_BUNDLE,
) -> dict[str, Any]:
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    current_out = strict_numeric_mapping(current, STATE_FIELDS, "current_beam_state")
    action_out = strict_numeric_mapping(action, ACTION_FIELDS, "action")
    bundle = _load_pickle(str(bundle_path))
    x = model_feature_vector(setup_out, current_out, action_out)[None, :]
    predicted_scaled, logits = _predict_forward_bundle(bundle, x)
    tolerance = np.asarray(
        [1.0, 1.0, 2.0, 2.0, max(0.05 * abs(current_out["peak_intensity"]), 1e-6)],
        dtype=np.float32,
    )
    change_values = predicted_scaled[0] * tolerance
    change = {
        field: float(change_values[index])
        for index, field in enumerate(STATE_FIELDS)
    }
    predicted_state = {
        field: current_out[field] + change[field] for field in STATE_FIELDS
    }
    probabilities = _softmax(logits[0])
    directions = {
        field: CLASSES[int(np.argmax(probabilities[index]))]
        for index, field in enumerate(DIRECTION_FIELDS)
    }
    return {
        "change": change,
        "predicted_beam_state": predicted_state,
        "directions": directions,
        "model_version": str(bundle["version"]),
        "simulator_at_inference": False,
    }


def measure_beam_image(
    image_path: str | Path, calibration: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.is_file():
        raise ContractError(f"image does not exist: {path}")
    expected = {
        "linear_intensity_low",
        "linear_intensity_high",
        "gamma",
        "source_sensor_resolution_px",
    }
    if not isinstance(calibration, Mapping) or set(calibration) != expected:
        raise ContractError(
            "image_calibration fields differ: "
            f"missing={sorted(expected - set(calibration)) if isinstance(calibration, Mapping) else sorted(expected)}, "
            f"extra={sorted(set(calibration) - expected) if isinstance(calibration, Mapping) else []}"
        )
    scalar = strict_numeric_mapping(
        {key: calibration[key] for key in ("linear_intensity_low", "linear_intensity_high", "gamma")},
        ("linear_intensity_low", "linear_intensity_high", "gamma"),
        "image_calibration",
    )
    resolution = calibration["source_sensor_resolution_px"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in resolution)
    ):
        raise ContractError("source_sensor_resolution_px must contain two positive integers")
    if scalar["linear_intensity_high"] <= scalar["linear_intensity_low"]:
        raise ContractError("linear_intensity_high must exceed linear_intensity_low")
    if scalar["gamma"] <= 0:
        raise ContractError("gamma must be positive")

    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    if scalar["gamma"] != 1.0:
        gray = np.power(gray, 1.0 / scalar["gamma"])
    low, high = scalar["linear_intensity_low"], scalar["linear_intensity_high"]
    intensity = low + gray * (high - low)
    weights = np.maximum(intensity - low, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise SpecialistError("image has no positive intensity above calibration floor")
    height, width = weights.shape
    source_width, source_height = map(float, resolution)
    x = (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5
    y = (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5
    marginal_x, marginal_y = weights.sum(axis=0), weights.sum(axis=1)
    cx = float(np.dot(marginal_x, x) / total)
    cy = float(np.dot(marginal_y, y) / total)
    state = {
        "centroid_x_px": cx,
        "centroid_y_px": cy,
        "sigma_x_px": float(np.sqrt(np.dot(marginal_x, np.square(x - cx)) / total)),
        "sigma_y_px": float(np.sqrt(np.dot(marginal_y, np.square(y - cy)) / total)),
        "peak_intensity": float(intensity.max()),
    }
    if not all(math.isfinite(value) for value in state.values()):
        raise SpecialistError("image measurement produced a non-finite value")
    return {
        "beam_state": state,
        "coordinate_frame": "camera_sensor_array",
        "model_version": "deterministic_calibrated_intensity_moments_v1",
        "simulator_at_inference": False,
    }


def _predicted_grid_states(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    bundle_path: Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    current_out = strict_numeric_mapping(current, STATE_FIELDS, "current_beam_state")
    actions = fixed_action_grid()
    x = np.stack(
        [model_feature_vector(setup_out, current_out, action) for action in actions]
    )
    bundle = _load_pickle(str(bundle_path))
    predicted_scaled, _ = _predict_forward_bundle(bundle, x)
    tolerance = np.asarray(
        [1.0, 1.0, 2.0, 2.0, max(0.05 * abs(current_out["peak_intensity"]), 1e-6)],
        dtype=np.float32,
    )
    changes = predicted_scaled * tolerance
    states = [
        {
            field: current_out[field] + float(changes[index, field_index])
            for field_index, field in enumerate(STATE_FIELDS)
        }
        for index in range(len(actions))
    ]
    return actions, states


def _direct_costs(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    desired: Mapping[str, Any],
    actions: list[dict[str, float]],
) -> np.ndarray:
    bundle = _load_pickle(str(DIRECT_INVERSE_BUNDLE))
    x = inverse_feature_vector(setup, current, desired)[None, :]
    probabilities = [model.predict_proba(x) for model in bundle["action_models"]]
    values = (-0.05, 0.0, 0.05), (-0.05, 0.0, 0.05), (-0.02, 0.0, 0.02), (-0.02, 0.0, 0.02)
    costs = []
    for action in actions:
        labels = [
            min(range(3), key=lambda index: abs(action[field] - values[axis][index]))
            for axis, field in enumerate(ACTION_FIELDS)
        ]
        costs.append(
            -sum(
                math.log(max(float(probabilities[axis][0, label]), 1e-12))
                for axis, label in enumerate(labels)
            )
        )
    return np.asarray(costs, dtype=np.float64)


def _inverse_from_states(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    desired: Mapping[str, Any],
) -> dict[str, Any]:
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    current_out = strict_numeric_mapping(current, STATE_FIELDS, "current_beam_state")
    desired_out = strict_numeric_mapping(desired, STATE_FIELDS, "desired_beam_state")
    summary = _load_json(str(INVERSE_SUMMARY))
    choice = summary["selected_action_routing"]
    forward_path = (
        GENERAL_FORWARD_BUNDLE
        if choice["forward_model"] == "general_forward"
        else GRID_FORWARD_BUNDLE
    )
    actions, states = _predicted_grid_states(setup_out, current_out, forward_path)
    residuals = np.asarray(
        [state_residual(state, desired_out) for state in states], dtype=np.float64
    )
    direct = _direct_costs(setup_out, current_out, desired_out, actions)
    scaled = (residuals - residuals.min()) / max(float(residuals.std()), 1e-9)
    combined = direct + float(choice["residual_weight"]) * scaled
    selected = int(np.argmin(combined))

    status_choice = summary["selected_status_routing"]
    if status_choice["source"] == "direct":
        direct_bundle = _load_pickle(str(DIRECT_INVERSE_BUNDLE))
        status_index = int(
            direct_bundle["status_model"].predict(
                inverse_feature_vector(setup_out, current_out, desired_out)[None, :]
            )[0]
        )
        status = ("unique", "ambiguous", "infeasible_within_limits")[status_index]
    else:
        status_forward = (
            residuals
            if status_choice["source"] == choice["forward_model"]
            else np.asarray(
                [
                    state_residual(state, desired_out)
                    for state in _predicted_grid_states(
                        setup_out,
                        current_out,
                        GENERAL_FORWARD_BUNDLE
                        if status_choice["source"] == "general_forward"
                        else GRID_FORWARD_BUNDLE,
                    )[1]
                ],
                dtype=np.float64,
            )
        )
        calibration = status_choice["calibration"]
        status = classify_residuals(
            status_forward,
            float(calibration["feasibility_cutoff"]),
            float(calibration["ambiguity_margin"]),
        )
    return {
        "predicted_status": status,
        "selected_index": selected,
        "selected_action": actions[selected],
        "predicted_beam_state": states[selected],
        "best_predicted_normalized_residual": float(residuals[selected]),
        "action_grid_size": len(actions),
        "matching_tolerance": dict(MATCHING_TOLERANCE),
        "model_version": str(summary["version"]),
        "simulator_at_inference": False,
    }


def _inverse_from_images(
    setup: Mapping[str, Any],
    current_image: str | Path,
    desired_image: str | Path,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    setup_out = strict_numeric_mapping(setup, SETUP_FIELDS, "setup")
    current_sensor = measure_beam_image(current_image, calibration)["beam_state"]
    desired_sensor = measure_beam_image(desired_image, calibration)["beam_state"]
    current_legacy = sensor_to_legacy_initial(current_sensor, setup_out)
    actions, legacy_states = _predicted_grid_states(
        setup_out, current_legacy, GRID_FORWARD_BUNDLE
    )
    sensor_states = [
        legacy_to_sensor(state, setup_out, action)
        for state, action in zip(legacy_states, actions)
    ]
    residuals = np.asarray(
        [state_residual(state, desired_sensor) for state in sensor_states], dtype=np.float64
    )
    selected = min(
        range(len(residuals)),
        key=lambda index: (float(residuals[index]), movement_mm(actions[index]), index),
    )
    summary = _load_json(str(VISUAL_SUMMARY))
    controller = summary["controller_calibration"]
    status = classify_residuals(
        residuals,
        float(controller["feasibility_cutoff"]),
        float(controller["ambiguity_margin"]),
    )
    return {
        "predicted_status": status,
        "selected_index": int(selected),
        "selected_action": actions[selected],
        "measured_current_beam_state": current_sensor,
        "measured_desired_beam_state": desired_sensor,
        "predicted_beam_state": sensor_states[selected],
        "best_predicted_normalized_residual": float(residuals[selected]),
        "action_grid_size": len(actions),
        "matching_tolerance": dict(MATCHING_TOLERANCE),
        "coordinate_frame": "camera_sensor_array",
        "model_version": str(summary["version"]),
        "simulator_at_inference": False,
    }


def run_specialist(
    route_name: str,
    arguments: Mapping[str, Any],
    image_bindings: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Execute one already-validated registered route."""
    if route_name == "measure_beam_profile_v1":
        return measure_beam_image(
            image_bindings["beam"], arguments["image_calibration"]
        )
    if route_name == "predict_direction_from_state_v1":
        return _direction_from_state(
            arguments["setup"], arguments["current_beam_state"], arguments["action"]
        )
    if route_name == "predict_direction_from_image_v1":
        measured = measure_beam_image(
            image_bindings["current_beam"], arguments["image_calibration"]
        )
        legacy = sensor_to_legacy_initial(measured["beam_state"], arguments["setup"])
        result = _direction_from_state(arguments["setup"], legacy, arguments["action"])
        return {**result, "measured_current_beam_state": measured["beam_state"]}
    if route_name == "predict_forward_from_state_v1":
        return _forward_from_state(
            arguments["setup"], arguments["current_beam_state"], arguments["action"]
        )
    if route_name == "predict_forward_from_image_v1":
        measured = measure_beam_image(
            image_bindings["current_beam"], arguments["image_calibration"]
        )
        legacy = sensor_to_legacy_initial(measured["beam_state"], arguments["setup"])
        result = _forward_from_state(arguments["setup"], legacy, arguments["action"])
        return {**result, "measured_current_beam_state": measured["beam_state"]}
    if route_name == "select_inverse_action_from_states_v1":
        return _inverse_from_states(
            arguments["setup"],
            arguments["current_beam_state"],
            arguments["desired_beam_state"],
        )
    if route_name == "select_inverse_action_from_images_v1":
        return _inverse_from_images(
            arguments["setup"],
            image_bindings["current_beam"],
            image_bindings["desired_beam"],
            arguments["image_calibration"],
        )
    raise ContractError(f"unregistered specialist route: {route_name}")
