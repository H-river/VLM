#!/usr/bin/env python3
"""Fast smoke checks for v2 lens/camera offsets and metadata."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optical_sim.src.io_utils import save_run
from optical_sim.src.metrics import compute_metrics
from optical_sim.src.optical_elements import setup_from_dict
from optical_sim.src.simulator import run_simulation


def _base_cfg() -> dict:
    return {
        "source": {
            "type": "gaussian",
            "wavelength": 632.8e-9,
            "beam_waist": 1.0e-3,
            "power": 1.0,
        },
        "lens": {
            "focal_length": 0.1,
            "clear_aperture": 0.025,
            "diameter": 0.0254,
            "x_offset": 0.0,
            "y_offset": 0.0,
        },
        "sensor": {
            "resolution": [64, 64],
            "pixel_pitch": 5.5e-6,
        },
        "geometry": {
            "laser_to_lens": 0.2,
            "lens_to_camera": 0.15,
        },
        "camera": {
            "x_offset": 0.0,
            "y_offset": 0.0,
        },
        "simulation": {
            "grid_size": 128,
            "grid_extent": 0.01,
            "propagation_backend": "fresnel_numpy",
        },
    }


def _simulate(cfg: dict):
    setup = setup_from_dict(cfg)
    result = run_simulation(setup)
    return setup, result


def test_lens_offset_changes_intensity() -> None:
    cfg_ref = _base_cfg()
    cfg_shift = copy.deepcopy(cfg_ref)
    cfg_shift["lens"]["x_offset"] = 1.5e-3

    _, ref = _simulate(cfg_ref)
    _, shifted = _simulate(cfg_shift)
    i_ref = ref["intensity"]
    i_shift = shifted["intensity"]

    assert i_ref.shape == i_shift.shape
    assert not np.allclose(i_ref, i_shift)


def test_camera_offset_changes_intensity() -> None:
    cfg_ref = _base_cfg()
    cfg_shift = copy.deepcopy(cfg_ref)
    cfg_shift["camera"]["x_offset"] = 1.5e-3

    _, ref = _simulate(cfg_ref)
    _, shifted = _simulate(cfg_shift)
    i_ref = ref["intensity"]
    i_shift = shifted["intensity"]

    assert i_ref.shape == i_shift.shape
    assert not np.allclose(i_ref, i_shift)


def test_v2_config_parsing() -> None:
    cfg = _base_cfg()
    cfg["camera"]["x_offset"] = 1.2e-3
    cfg["camera"]["y_offset"] = -9.0e-4
    setup = setup_from_dict(cfg)

    assert np.isclose(setup.camera.x_offset, cfg["camera"]["x_offset"])
    assert np.isclose(setup.camera.y_offset, cfg["camera"]["y_offset"])


def test_v2_metadata_layout() -> None:
    cfg = _base_cfg()
    cfg["lens"]["x_offset"] = 7.5e-4
    cfg["lens"]["y_offset"] = -4.0e-4
    cfg["camera"]["x_offset"] = 1.0e-3
    cfg["camera"]["y_offset"] = -6.0e-4

    setup, result = _simulate(cfg)
    metrics = compute_metrics(result["intensity"], result["sensor_X"], result["sensor_Y"])

    with tempfile.TemporaryDirectory() as tmp:
        meta = save_run(
            tmp,
            "smoke_v2",
            setup,
            result["intensity"],
            metrics,
            metadata_format="v2",
        )
        setup_meta = meta["setup"]
        assert "camera" in setup_meta
        assert "x_offset" in setup_meta["camera"]
        assert "y_offset" in setup_meta["camera"]
        assert "x_offset" in setup_meta["lens"]
        assert "y_offset" in setup_meta["lens"]
        assert "alignment" not in setup_meta

        metadata_path = Path(tmp) / "smoke_v2" / "metadata.json"
        with open(metadata_path, "r") as f:
            written = json.load(f)
        assert "alignment" not in written["setup"]
        assert "camera" in written["setup"]


def main() -> None:
    test_lens_offset_changes_intensity()
    test_camera_offset_changes_intensity()
    test_v2_config_parsing()
    test_v2_metadata_layout()
    print("smoke_v2_offsets: OK")


if __name__ == "__main__":
    main()
