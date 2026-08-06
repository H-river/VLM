"""Shared helpers for the optics-understanding pilot dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from PIL import Image

from optical_sim.src.optical_elements import OpticalSetup, setup_from_dict
from optics_sft.physics.rendering import intensity_to_uint8_image, random_render_params
from optics_sft.physics.sim_adapter import Action, apply_action_to_setup, setup_to_safe_metadata


DATASET_VERSION = "pilot_v1"
TASK_TYPES = (
    "setup_interpretation",
    "information_sufficiency",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "constrained_intervention",
    "counterfactual_reasoning",
)
VISUAL_TASK_TYPES = frozenset(
    {"causal_effects", "forward_prediction", "diagnosis", "counterfactual_reasoning"}
)
ACTION_KEYS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "centroid_error_px",
        "error_vector",
        "after_state",
        "true_control",
        "true_control_plan",
        "ground_truth",
        "private_eval",
        "answer",
        "label",
        "simulator_config",
    }
)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def assert_finite_tree(value: Any, prefix: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_tree(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{prefix}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number at {prefix}: {value}")


def action_dict(**overrides: float) -> dict[str, float]:
    values = {key: 0.0 for key in ACTION_KEYS}
    values.update({key: float(value) for key, value in overrides.items()})
    return values


def action_from_dict(values: Mapping[str, Any]) -> Action:
    return Action(*(float(values.get(key, 0.0)) for key in ACTION_KEYS))


def rounded_state(state: Mapping[str, Any], digits: int = 4) -> dict[str, float]:
    keys = ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity")
    return {key: round(float(state[key]), digits) for key in keys}


def state_change(before: Mapping[str, Any], after: Mapping[str, Any], digits: int = 4) -> dict[str, float]:
    return {
        "centroid_x_px": round(float(after["centroid_x_px"]) - float(before["centroid_x_px"]), digits),
        "centroid_y_px": round(float(after["centroid_y_px"]) - float(before["centroid_y_px"]), digits),
        "sigma_x_px": round(float(after["sigma_x_px"]) - float(before["sigma_x_px"]), digits),
        "sigma_y_px": round(float(after["sigma_y_px"]) - float(before["sigma_y_px"]), digits),
        "peak_intensity": round(float(after["peak_intensity"]) - float(before["peak_intensity"]), digits),
    }


def centroid_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return math.hypot(
        float(a["centroid_x_px"]) - float(b["centroid_x_px"]),
        float(a["centroid_y_px"]) - float(b["centroid_y_px"]),
    )


def safe_metadata(setup: OpticalSetup) -> dict[str, Any]:
    metadata = setup_to_safe_metadata(setup)
    metadata["coordinate_convention"] = "sensor x increases right; sensor y increases down"
    return round_tree(metadata, 6)


def round_tree(value: Any, digits: int = 6) -> Any:
    if isinstance(value, Mapping):
        return {str(key): round_tree(child, digits) for key, child in value.items()}
    if isinstance(value, list):
        return [round_tree(child, digits) for child in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def setup_snapshot(setup: OpticalSetup) -> dict[str, Any]:
    return {
        "source": {
            "type": setup.source.source_type,
            "wavelength": float(setup.source.wavelength),
            "beam_waist": float(setup.source.beam_waist),
            "power": float(setup.source.power),
        },
        "lens": {
            "focal_length": float(setup.lens.focal_length),
            "clear_aperture": float(setup.lens.clear_aperture),
            "diameter": float(setup.lens.diameter),
            "x_offset": float(setup.lens.x_offset),
            "y_offset": float(setup.lens.y_offset),
        },
        "sensor": {
            "resolution": [int(v) for v in setup.sensor.resolution],
            "pixel_pitch": float(setup.sensor.pixel_pitch),
        },
        "geometry": {
            "laser_to_lens": float(setup.laser_to_lens),
            "lens_to_camera": float(setup.lens_to_camera),
        },
        "camera": {
            "x_offset": float(setup.camera.x_offset),
            "y_offset": float(setup.camera.y_offset),
        },
        "alignment": {
            "x_offset": float(setup.alignment.x_offset),
            "y_offset": float(setup.alignment.y_offset),
            "tilt_x": float(setup.alignment.tilt_x),
            "tilt_y": float(setup.alignment.tilt_y),
            "defocus": float(setup.alignment.defocus),
        },
        "simulation": {
            "grid_size": int(setup.grid_size),
            "grid_extent": float(setup.grid_extent),
            "propagation_backend": setup.propagation_backend,
        },
    }


def _nominal_values(base_cfg: Mapping[str, Any]) -> dict[str, float]:
    return {
        "wavelength_nm": float(base_cfg["source"]["wavelength"]) * 1e9,
        "beam_waist_mm": float(base_cfg["source"]["beam_waist"]) * 1e3,
        "lens_focal_length_mm": float(base_cfg["lens"]["focal_length"]) * 1e3,
        "lens_aperture_mm": float(base_cfg["lens"]["clear_aperture"]) * 1e3,
        "source_to_lens_mm": float(base_cfg["geometry"]["laser_to_lens"]) * 1e3,
        "lens_to_camera_mm": float(base_cfg["geometry"]["lens_to_camera"]) * 1e3,
    }


def _set_sampled_value(cfg: dict[str, Any], key: str, value: float) -> None:
    if key == "wavelength_nm":
        cfg["source"]["wavelength"] = value * 1e-9
    elif key == "beam_waist_mm":
        cfg["source"]["beam_waist"] = value * 1e-3
    elif key == "lens_focal_length_mm":
        cfg["lens"]["focal_length"] = value * 1e-3
    elif key == "lens_aperture_mm":
        cfg["lens"]["clear_aperture"] = value * 1e-3
    elif key == "source_to_lens_mm":
        cfg["geometry"]["laser_to_lens"] = value * 1e-3
    elif key == "lens_to_camera_mm":
        cfg["geometry"]["lens_to_camera"] = value * 1e-3
    else:
        raise KeyError(key)


def sample_setup_config(
    base_cfg: Mapping[str, Any],
    simulation_cfg: Mapping[str, Any],
    rng: random.Random,
    *,
    ood_parameter: str | None = None,
    ood_band: int = 0,
) -> tuple[dict[str, Any], dict[str, float]]:
    cfg = copy.deepcopy(dict(base_cfg))
    nominal = _nominal_values(base_cfg)
    sampled: dict[str, float] = {}
    for key, factor_range in simulation_cfg["iid_factors"].items():
        chosen_range = factor_range
        if key == ood_parameter:
            chosen_range = simulation_cfg["ood_factors"][key][ood_band]
        factor = rng.uniform(float(chosen_range[0]), float(chosen_range[1]))
        sampled[key] = nominal[key] * factor
        _set_sampled_value(cfg, key, sampled[key])

    offsets = simulation_cfg["offsets_mm"]
    sampled.update(
        {
            "lens_x_offset_mm": rng.uniform(*map(float, offsets["lens_x"])),
            "lens_y_offset_mm": rng.uniform(*map(float, offsets["lens_y"])),
            "camera_x_offset_mm": rng.uniform(*map(float, offsets["camera_x"])),
            "camera_y_offset_mm": rng.uniform(*map(float, offsets["camera_y"])),
        }
    )
    cfg["lens"]["x_offset"] = sampled["lens_x_offset_mm"] * 1e-3
    cfg["lens"]["y_offset"] = sampled["lens_y_offset_mm"] * 1e-3
    cfg.setdefault("camera", {})["x_offset"] = sampled["camera_x_offset_mm"] * 1e-3
    cfg["camera"]["y_offset"] = sampled["camera_y_offset_mm"] * 1e-3
    return cfg, sampled


def modified_setup_config(config: Mapping[str, Any], parameter: str, factor: float) -> dict[str, Any]:
    copied = copy.deepcopy(dict(config))
    setup = setup_from_dict(copied)
    current = safe_metadata(setup)[parameter]
    _set_sampled_value(copied, parameter, float(current) * factor)
    return copied


def assign_tasks_to_groups(
    group_ids: list[str], counts: Mapping[str, Any], rng: random.Random, questions_per_group: int = 4
) -> dict[str, list[str]]:
    remaining = {task: int(counts[task]) for task in TASK_TYPES}
    if sum(remaining.values()) != len(group_ids) * questions_per_group:
        raise ValueError("Task counts do not match groups times questions_per_group")
    assignments: dict[str, list[str]] = {}
    for group_index, group_id in enumerate(group_ids):
        groups_left = len(group_ids) - group_index
        ranked = sorted(
            TASK_TYPES,
            key=lambda task: (remaining[task] / groups_left, remaining[task], rng.random()),
            reverse=True,
        )
        selected = [task for task in ranked if remaining[task] > 0][:questions_per_group]
        if len(selected) != questions_per_group:
            raise RuntimeError("Could not assign four unique tasks to a scenario")
        assignments[group_id] = selected
        for task in selected:
            remaining[task] -= 1
    if any(remaining.values()):
        raise RuntimeError(f"Task assignment left non-zero counts: {remaining}")
    return assignments


def select_visual_examples(
    assignments: Mapping[str, list[str]], target_count: int, rng: random.Random
) -> set[tuple[str, str]]:
    eligible = [
        (group_id, task)
        for group_id, tasks in assignments.items()
        for task in tasks
        if task in VISUAL_TASK_TYPES
    ]
    if len(eligible) < target_count:
        raise ValueError("Not enough visual-eligible examples")
    rng.shuffle(eligible)
    return set(eligible[:target_count])


def effect_label(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "increase"
    if delta < -threshold:
        return "decrease"
    return "no_change"


def classify_effects(
    before: Mapping[str, Any], after: Mapping[str, Any], label_cfg: Mapping[str, Any]
) -> dict[str, str]:
    peak = max(abs(float(before["peak_intensity"])), 1e-12)
    return {
        "centroid_x": effect_label(
            float(after["centroid_x_px"]) - float(before["centroid_x_px"]),
            float(label_cfg["centroid_effect_threshold_px"]),
        ),
        "centroid_y": effect_label(
            float(after["centroid_y_px"]) - float(before["centroid_y_px"]),
            float(label_cfg["centroid_effect_threshold_px"]),
        ),
        "sigma_x": effect_label(
            float(after["sigma_x_px"]) - float(before["sigma_x_px"]),
            float(label_cfg["sigma_effect_threshold_px"]),
        ),
        "sigma_y": effect_label(
            float(after["sigma_y_px"]) - float(before["sigma_y_px"]),
            float(label_cfg["sigma_effect_threshold_px"]),
        ),
        "peak_intensity": effect_label(
            (float(after["peak_intensity"]) - float(before["peak_intensity"])) / peak,
            float(label_cfg["intensity_relative_threshold"]),
        ),
    }


def prompt_key_hits(value: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_PROMPT_KEYS:
                hits.append(path)
            hits.extend(prompt_key_hits(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(prompt_key_hits(child, f"{prefix}[{index}]"))
    return hits


def render_image(
    intensity: Any,
    output_path: Path,
    rng: random.Random,
    *,
    size_px: int,
    difficulty: str,
) -> dict[str, Any]:
    options = random_render_params(rng, difficulty=difficulty)
    image = intensity_to_uint8_image(intensity, options)
    image = image.resize((size_px, size_px), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return round_tree(options, 8)


def qwen_prompt(text: str, image_count: int) -> list[dict[str, Any]]:
    content = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def qwen_completion(target: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            # Preserve the deliberately constructed target order.  Most tasks
            # still place status first, while action-first control supervision
            # places executable action evidence before the derived status.
            "content": [{"type": "text", "text": json.dumps(target, sort_keys=False)}],
        }
    ]


def make_messages(record: Mapping[str, Any], include_target: bool) -> dict[str, Any]:
    images = list(record["prompt_inputs"].get("images", []))
    messages = qwen_prompt(str(record["prompt"]), len(images))
    if include_target:
        messages += qwen_completion(record["target"])
    exported = {
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "task_type": record["task_type"],
        "images": images,
        "messages": messages,
    }
    match_group_id = record.get("provenance", {}).get("match_group_id")
    if match_group_id is not None:
        exported["match_group_id"] = match_group_id
    return exported


def make_qwen_record(record: Mapping[str, Any], include_target: bool) -> dict[str, Any]:
    images = list(record["prompt_inputs"].get("images", []))
    exported = {
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "task_type": record["task_type"],
        "images": images,
        "prompt": qwen_prompt(str(record["prompt"]), len(images)),
    }
    match_group_id = record.get("provenance", {}).get("match_group_id")
    if match_group_id is not None:
        exported["match_group_id"] = match_group_id
    if include_target:
        exported["completion"] = qwen_completion(record["target"])
    return exported


def apply_action_dict(setup: OpticalSetup, action: Mapping[str, Any]) -> OpticalSetup:
    return apply_action_to_setup(setup, action_from_dict(action))


def axis_action(axis_key: str, value_mm: float) -> dict[str, float]:
    if axis_key not in ACTION_KEYS:
        raise KeyError(axis_key)
    return action_dict(**{axis_key: value_mm})


def discrete_grid(bound_mm: float, count: int) -> list[float]:
    if count < 2:
        return [0.0]
    return [round(float(value), 6) for value in np.linspace(-bound_mm, bound_mm, count)]
