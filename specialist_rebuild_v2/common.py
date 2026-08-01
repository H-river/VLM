"""Shared numerical definitions and compact dataset helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SETUP_FIELDS = (
    "wavelength_nm",
    "beam_waist_mm",
    "power_w",
    "lens_focal_length_mm",
    "lens_aperture_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
    "lens_x_offset_mm",
    "lens_y_offset_mm",
    "camera_x_offset_mm",
    "camera_y_offset_mm",
    "pixel_size_um",
)
STATE_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
DIRECTION_FIELDS = (
    "centroid_x",
    "centroid_y",
    "width_x",
    "width_y",
    "peak_intensity",
)
CLASSES = ("decrease", "no_change", "increase")
STATUSES = ("unique", "ambiguous", "infeasible_within_limits")
MATCHING_TOLERANCE = {
    "centroid_vector_px": 0.5,
    "width_each_px": 1.0,
    "peak_relative": 0.02,
}
CHANGE_TOLERANCE = np.asarray([1.0, 1.0, 2.0, 2.0, 1.0], dtype=np.float32)


def stable_token(*parts: Any) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(int(stable_token(*parts)[:16], 16))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def fixed_action_grid() -> list[dict[str, float]]:
    import itertools

    lens = (-0.05, 0.0, 0.05)
    camera = (-0.02, 0.0, 0.02)
    return [
        dict(zip(ACTION_FIELDS, map(float, values), strict=True))
        for values in itertools.product(lens, lens, camera, camera)
    ]


def setup_array(setup: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(setup[key]) for key in SETUP_FIELDS], dtype=np.float32)


def state_array(state: Mapping[str, Any]) -> np.ndarray:
    values = [float(state[key]) for key in STATE_FIELDS]
    values[-1] = math.log1p(max(values[-1], 0.0))
    return np.asarray(values, dtype=np.float32)


def raw_state_array(state: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(state[key]) for key in STATE_FIELDS], dtype=np.float32)


def action_array(action: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(action[key]) for key in ACTION_FIELDS], dtype=np.float32)


def forward_feature(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    action: Mapping[str, Any],
) -> np.ndarray:
    base = np.concatenate(
        [setup_array(setup), state_array(current), action_array(action)]
    )
    fields = (*SETUP_FIELDS, *STATE_FIELDS, *ACTION_FIELDS)
    col = {name: index for index, name in enumerate(fields)}
    lx, ly = base[col["lens_x_offset_mm"]], base[col["lens_y_offset_mm"]]
    cx, cy = base[col["camera_x_offset_mm"]], base[col["camera_y_offset_mm"]]
    dlx, dly = base[col["lens_x_delta_mm"]], base[col["lens_y_delta_mm"]]
    dcx, dcy = base[col["camera_x_delta_mm"]], base[col["camera_y_delta_mm"]]
    pitch = base[col["pixel_size_um"]] / 1000.0
    ratio = (
        base[col["lens_to_camera_mm"]] / base[col["lens_focal_length_mm"]]
    ) / pitch
    derived = np.asarray(
        [
            lx + dlx,
            ly + dly,
            cx + dcx,
            cy + dcy,
            (lx + dlx) ** 2 - lx**2,
            (ly + dly) ** 2 - ly**2,
            (cx + dcx) ** 2 - cx**2,
            (cy + dcy) ** 2 - cy**2,
            math.hypot(lx + dlx, ly + dly),
            math.hypot(lx, ly),
            math.hypot(cx + dcx, cy + dcy),
            math.hypot(cx, cy),
            dlx * ratio,
            dly * ratio,
            dcx / pitch,
            dcy / pitch,
            ratio,
            dlx * lx,
            dly * ly,
            dcx * cx,
            dcy * cy,
            abs(lx + dlx) - abs(lx),
            abs(ly + dly) - abs(ly),
            abs(cx + dcx) - abs(cx),
            abs(cy + dcy) - abs(cy),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, derived])


def direction_feature(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    action: Mapping[str, Any],
) -> np.ndarray:
    return np.concatenate(
        [setup_array(setup), state_array(current), action_array(action)]
    )


def inverse_context(
    setup: Mapping[str, Any],
    current: Mapping[str, Any],
    desired: Mapping[str, Any],
) -> np.ndarray:
    setup_values = setup_array(setup)
    current_raw = raw_state_array(current)
    desired_raw = raw_state_array(desired)
    delta = desired_raw - current_raw
    peak = max(abs(float(current_raw[-1])), 1e-9)
    derived = np.asarray(
        [
            delta[-1] / peak,
            math.hypot(float(delta[0]), float(delta[1])),
            math.hypot(
                float(setup["lens_x_offset_mm"]),
                float(setup["lens_y_offset_mm"]),
            ),
            math.hypot(
                float(setup["camera_x_offset_mm"]),
                float(setup["camera_y_offset_mm"]),
            ),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            setup_values,
            state_array(current),
            state_array(desired),
            delta,
            derived,
        ]
    )


def change_and_directions(
    current: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, str]]:
    change = {
        key: float(after[key]) - float(current[key]) for key in STATE_FIELDS
    }
    tolerance = {
        "centroid_x_px": 1.0,
        "centroid_y_px": 1.0,
        "sigma_x_px": 2.0,
        "sigma_y_px": 2.0,
        "peak_intensity": max(
            0.05 * abs(float(current["peak_intensity"])), 1e-9
        ),
    }
    source = {
        "centroid_x": "centroid_x_px",
        "centroid_y": "centroid_y_px",
        "width_x": "sigma_x_px",
        "width_y": "sigma_y_px",
        "peak_intensity": "peak_intensity",
    }
    directions = {}
    for output, key in source.items():
        value = change[key]
        directions[output] = (
            "increase"
            if value > tolerance[key]
            else "decrease"
            if value < -tolerance[key]
            else "no_change"
        )
    return change, directions


def matching_mask(
    candidate_states: np.ndarray, desired: Sequence[float]
) -> np.ndarray:
    desired_values = np.asarray(desired, dtype=np.float64)
    centroid = np.hypot(
        candidate_states[:, 0] - desired_values[0],
        candidate_states[:, 1] - desired_values[1],
    )
    width_x = np.abs(candidate_states[:, 2] - desired_values[2])
    width_y = np.abs(candidate_states[:, 3] - desired_values[3])
    peak = np.abs(candidate_states[:, 4] - desired_values[4]) / max(
        abs(float(desired_values[4])), 1e-12
    )
    return (
        (centroid <= MATCHING_TOLERANCE["centroid_vector_px"])
        & (width_x <= MATCHING_TOLERANCE["width_each_px"])
        & (width_y <= MATCHING_TOLERANCE["width_each_px"])
        & (peak <= MATCHING_TOLERANCE["peak_relative"])
    )


def minimum_motion_index(indices: Sequence[int]) -> int:
    actions = fixed_action_grid()
    return min(
        indices,
        key=lambda index: (
            sum(abs(float(actions[index][key])) for key in ACTION_FIELDS),
            index,
        ),
    )

