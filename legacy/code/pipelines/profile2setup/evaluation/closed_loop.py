"""Closed-loop simulator evaluation for trained profile2setup v2 checkpoints."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from profile2setup.evaluation.profile_metrics import compute_profile_features, load_intensity
from profile2setup.inference.controller import (
    Profile2SetupController,
    assert_no_forbidden_v2_fields,
)
from profile2setup.schema import VARIABLE_ORDER
from profile2setup.training.dataset import filter_records, load_jsonl


TASK_TYPES = ["absolute", "edit", "paired_no_setup"]
SIMULATION_POLICIES = {"target_base", "current_base", "auto"}
CONTROLLED_PATHS = [
    ("geometry", "laser_to_lens"),
    ("geometry", "lens_to_camera"),
    ("lens", "focal_length"),
    ("lens", "x_offset"),
    ("lens", "y_offset"),
    ("camera", "x_offset"),
    ("camera", "y_offset"),
]
DERIVED_CONTEXT_PATHS = [
    ("geometry", "effective_camera_distance"),
    ("sensor", "sensor_size"),
]
_EPS = 1.0e-12


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if np.isfinite(obj):
            return float(obj)
        return None
    return obj


def _as_path(path_value: Any) -> Path | None:
    if path_value is None:
        return None
    if not isinstance(path_value, str) or not path_value:
        return None
    return Path(path_value)


def _first_present(record: dict, top_key: str, ref_key: str) -> str | None:
    value = record.get(top_key)
    if value:
        return str(value)
    ref = record.get("profile_loss_reference") or {}
    if isinstance(ref, dict) and ref.get(ref_key):
        return str(ref[ref_key])
    return None


def resolve_metadata_paths(record) -> dict:
    """Resolve target/current metadata and intensity paths from record fields."""
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    return {
        "target_metadata_path": _first_present(record, "target_metadata_path", "target_metadata_path"),
        "current_metadata_path": _first_present(record, "current_metadata_path", "current_metadata_path"),
        "target_profile_path": _first_present(record, "target_profile_path", "target_profile_path"),
        "current_profile_path": _first_present(record, "current_profile_path", "current_profile_path"),
    }


def _require_canonical_setup(setup_physical: dict) -> dict:
    if not isinstance(setup_physical, dict):
        raise ValueError("setup_physical must be a dict")
    assert_no_forbidden_v2_fields(setup_physical)
    keys = set(setup_physical.keys())
    expected = set(VARIABLE_ORDER)
    if keys != expected:
        missing = [name for name in VARIABLE_ORDER if name not in setup_physical]
        extra = sorted(keys - expected)
        raise ValueError(f"setup_physical must use canonical v2 variables; missing={missing}, extra={extra}")
    return {name: float(setup_physical[name]) for name in VARIABLE_ORDER}


def setup_dict_to_sim_update(setup_physical: dict) -> dict:
    """Map profile2setup canonical physical variables to simulator config paths."""
    setup = _require_canonical_setup(setup_physical)
    update = {
        "geometry": {
            "laser_to_lens": setup["source_to_lens"],
            "lens_to_camera": setup["lens_to_camera"],
        },
        "lens": {
            "focal_length": setup["focal_length"],
            "x_offset": setup["lens_x"],
            "y_offset": setup["lens_y"],
        },
        "camera": {
            "x_offset": setup["camera_x"],
            "y_offset": setup["camera_y"],
        },
    }
    assert_no_forbidden_v2_fields(update)
    return update


def _load_json(path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    with open(json_path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Metadata root must be a dict: {json_path}")
    return obj


def load_base_simulator_config_from_metadata(metadata_path) -> dict:
    """Load a full simulator setup config from a metadata.json file."""
    metadata = _load_json(metadata_path)
    setup = metadata.get("setup")
    if not isinstance(setup, dict):
        raise ValueError(f"Metadata missing dict key 'setup': {metadata_path}")
    return copy.deepcopy(setup)


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict):
            child = base.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot update non-dict simulator config key: {key}")
            _deep_update(child, value)
        else:
            base[key] = value
    return base


def apply_predicted_setup_to_sim_config(base_config, predicted_setup_physical) -> dict:
    """Return simulator config with only the seven controlled variables replaced."""
    if not isinstance(base_config, dict):
        raise ValueError("base_config must be a dict")
    cfg = copy.deepcopy(base_config)
    _deep_update(cfg, setup_dict_to_sim_update(predicted_setup_physical))
    return cfg


def simulate_intensity_from_config(config) -> np.ndarray:
    """Run optical_sim from an in-memory config and return the simulated intensity."""
    from optical_sim.src.optical_elements import setup_from_dict
    from optical_sim.src.simulator import run_simulation

    setup = setup_from_dict(config)
    result = run_simulation(setup)
    intensity = np.asarray(result["intensity"], dtype=np.float64)
    if intensity.ndim != 2:
        raise ValueError(f"Simulator returned non-2D intensity with shape {intensity.shape}")
    return np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_for_similarity(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    peak = float(np.max(arr)) if arr.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / (peak + _EPS)


def compute_closed_loop_profile_metrics(predicted_intensity, target_intensity) -> dict:
    """Compute required profile metrics between simulated and target intensity arrays."""
    pred = np.asarray(predicted_intensity, dtype=np.float64)
    target = np.asarray(target_intensity, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Intensity arrays must be 2D; got pred={pred.shape}, target={target.shape}")
    if pred.shape != target.shape:
        raise ValueError(f"Intensity shape mismatch: pred={pred.shape}, target={target.shape}")
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, 0.0, None)
    target = np.clip(target, 0.0, None)

    diff = pred - target
    mse = float(np.mean(np.square(diff)))
    target_power = float(np.mean(np.square(target)))
    pred_features = compute_profile_features(pred)
    target_features = compute_profile_features(target)

    metrics = {
        "mse": mse,
        "normalized_mse": float(mse / (target_power + _EPS)),
        "peak_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "centroid_x_error_px": float(abs(pred_features["centroid_x_px"] - target_features["centroid_x_px"])),
        "centroid_y_error_px": float(abs(pred_features["centroid_y_px"] - target_features["centroid_y_px"])),
        "sigma_x_error_px": float(abs(pred_features["sigma_x_px"] - target_features["sigma_x_px"])),
        "sigma_y_error_px": float(abs(pred_features["sigma_y_px"] - target_features["sigma_y_px"])),
        "total_intensity_error": float(abs(pred_features["total_intensity"] - target_features["total_intensity"])),
        "peak_intensity_error": float(abs(pred_features["peak_intensity"] - target_features["peak_intensity"])),
    }

    try:
        from skimage.metrics import structural_similarity

        pred_norm = _normalize_for_similarity(pred)
        target_norm = _normalize_for_similarity(target)
        metrics["ssim"] = float(structural_similarity(pred_norm, target_norm, data_range=1.0))
    except Exception:
        pass

    return metrics


def _remove_path(obj: dict, path: tuple[str, ...]) -> None:
    cur = obj
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)


def _non_controlled_context(config: dict) -> dict:
    stripped = copy.deepcopy(config)
    for path in CONTROLLED_PATHS + DERIVED_CONTEXT_PATHS:
        _remove_path(stripped, path)
    return stripped


def non_controlled_context_matches(current_config: dict, target_config: dict) -> bool:
    """Compare simulator contexts after removing controlled variable paths."""
    return _jsonable(_non_controlled_context(current_config)) == _jsonable(
        _non_controlled_context(target_config)
    )


def _safe_filename(record_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record_id)).strip("._")
    return safe or "record"


def _parameter_abs_error(predicted: dict, target: dict) -> dict:
    pred = _require_canonical_setup(predicted)
    tgt = _require_canonical_setup(target)
    return {name: float(abs(pred[name] - tgt[name])) for name in VARIABLE_ORDER}


def _mean_metric(rows: list[dict], metric_names: list[str] | None = None) -> dict | None:
    if not rows:
        return None
    keys = list(metric_names or sorted({key for row in rows for key in row.keys()}))
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def _median_metric(rows: list[dict], metric_names: list[str] | None = None) -> dict | None:
    if not rows:
        return None
    keys = list(metric_names or sorted({key for row in rows for key in row.keys()}))
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = float(np.median(values)) if values else None
    return out


def _aggregate_examples(examples: list[dict]) -> dict:
    profile_rows = [example["profile_metrics"] for example in examples]
    parameter_rows = [example["parameter_abs_error"] for example in examples]
    return {
        "profile_metrics_mean": _mean_metric(profile_rows),
        "profile_metrics_median": _median_metric(profile_rows),
        "parameter_error_mean": _mean_metric(parameter_rows, list(VARIABLE_ORDER)),
        "parameter_error_median": _median_metric(parameter_rows, list(VARIABLE_ORDER)),
    }


def _per_task_type_aggregate(examples: list[dict]) -> dict:
    out = {}
    for task_type in TASK_TYPES:
        rows = [example for example in examples if example.get("task_type") == task_type]
        if rows:
            out[task_type] = {
                "num_profile_evaluated": len(rows),
                **_aggregate_examples(rows),
            }
        else:
            out[task_type] = {
                "num_profile_evaluated": 0,
                "profile_metrics_mean": None,
                "profile_metrics_median": None,
                "parameter_error_mean": None,
                "parameter_error_median": None,
            }
    return out


def _skip(skipped: list[dict], record_id: str, reason: str) -> None:
    skipped.append({"record_id": str(record_id), "reason": str(reason)})


def _eligibility_skip_reason(record: dict, paths: dict) -> str | None:
    if not paths.get("target_profile_path"):
        return "missing target_profile_path"
    if not paths.get("target_metadata_path"):
        return "missing target_metadata_path"
    if record.get("target_setup") is None:
        return "missing target_setup"
    target_profile_path = _as_path(paths["target_profile_path"])
    if target_profile_path is None or not target_profile_path.exists():
        return "target_profile_path does not exist"
    target_metadata_path = _as_path(paths["target_metadata_path"])
    if target_metadata_path is None or not target_metadata_path.exists():
        return "target_metadata_path does not exist"
    return None


def _choose_policy_config(paths: dict, target_config: dict, simulation_policy: str) -> tuple[str, dict, list[str]]:
    warnings = []
    if simulation_policy == "target_base":
        return "target_base", target_config, warnings

    current_path = paths.get("current_metadata_path")
    if simulation_policy == "current_base":
        if not current_path:
            raise ValueError("missing current_metadata_path for current_base simulation")
        current_config = load_base_simulator_config_from_metadata(current_path)
        if not non_controlled_context_matches(current_config, target_config):
            warnings.append("current_base uses current non-controlled context; target context differs")
        return "current_base", current_config, warnings

    if simulation_policy == "auto":
        if current_path:
            current_config = load_base_simulator_config_from_metadata(current_path)
            if non_controlled_context_matches(current_config, target_config):
                return "current_base", current_config, warnings
        return "target_base", target_config, warnings

    raise ValueError(f"Unsupported simulation_policy: {simulation_policy}")


def run_closed_loop_evaluation(
    checkpoint_path,
    data_path,
    out_path=None,
    variables_config_path="profile2setup/configs/variables.yaml",
    config_path=None,
    device="auto",
    max_examples=None,
    task_filter=None,
    strict=True,
    simulation_policy="target_base",
    save_predicted_profiles_dir=None,
) -> dict:
    """Evaluate profile success by simulating routed model predictions."""
    if simulation_policy not in SIMULATION_POLICIES:
        raise ValueError(f"simulation_policy must be one of {sorted(SIMULATION_POLICIES)}")

    controller = Profile2SetupController(
        checkpoint_path=checkpoint_path,
        device=device,
        variables_config_path=variables_config_path,
        config_path=config_path,
    )

    records = filter_records(load_jsonl(data_path), task_filter=task_filter)
    if max_examples is not None:
        records = records[: int(max_examples)]

    save_profiles_dir = Path(save_predicted_profiles_dir) if save_predicted_profiles_dir else None
    if save_profiles_dir is not None:
        save_profiles_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    skipped = []

    for idx, record in enumerate(records):
        record_id = str(record.get("id") or f"row_{idx}")
        try:
            assert_no_forbidden_v2_fields(record)
            paths = resolve_metadata_paths(record)
            reason = _eligibility_skip_reason(record, paths)
            if reason is not None:
                _skip(skipped, record_id, reason)
                continue

            target_config = load_base_simulator_config_from_metadata(paths["target_metadata_path"])
            policy_used, base_config, warnings = _choose_policy_config(paths, target_config, simulation_policy)

            prediction = controller.predict_record(record)
            predicted_setup = prediction["predicted_routed_setup_physical"]
            target_setup = prediction["target_setup_physical"] or record.get("target_setup")
            if target_setup is None:
                _skip(skipped, record_id, "missing target_setup")
                continue

            sim_config = apply_predicted_setup_to_sim_config(base_config, predicted_setup)
            predicted_intensity = simulate_intensity_from_config(sim_config)
            target_intensity = load_intensity(paths["target_profile_path"])
            profile_metrics = compute_closed_loop_profile_metrics(predicted_intensity, target_intensity)
            parameter_abs_error = _parameter_abs_error(predicted_setup, target_setup)

            predicted_profile_path = None
            if save_profiles_dir is not None:
                predicted_profile_path = save_profiles_dir / f"{_safe_filename(record_id)}.npy"
                np.save(predicted_profile_path, predicted_intensity)

            example = {
                "record_id": record_id,
                "task_type": prediction["task_type"],
                "prompt": prediction["prompt"],
                "simulation_policy_used": policy_used,
                "target_profile_path": paths["target_profile_path"],
                "target_metadata_path": paths["target_metadata_path"],
                "current_metadata_path": paths.get("current_metadata_path"),
                "predicted_routed_setup_physical": predicted_setup,
                "target_setup_physical": target_setup,
                "parameter_abs_error": parameter_abs_error,
                "profile_metrics": profile_metrics,
            }
            if predicted_profile_path is not None:
                example["predicted_profile_path"] = str(predicted_profile_path)
            if warnings:
                example["warnings"] = warnings
            examples.append(example)
        except Exception as exc:
            if strict:
                raise
            _skip(skipped, record_id, str(exc))

    result = {
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
        "simulation_policy": simulation_policy,
        "num_records_seen": len(records),
        "num_profile_evaluated": len(examples),
        "num_skipped": len(skipped),
        "aggregate": _aggregate_examples(examples),
        "per_task_type": _per_task_type_aggregate(examples),
        "examples": examples,
        "skipped": skipped,
    }
    result = _jsonable(result)
    assert_no_forbidden_v2_fields(result)

    if out_path is not None:
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    return result
