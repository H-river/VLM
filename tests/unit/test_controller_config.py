from __future__ import annotations

from pathlib import Path

from optics_vla.common.config import load_controller_config


ROOT = Path(__file__).resolve().parents[2]


def test_canonical_paths_are_repository_relative() -> None:
    config = load_controller_config(ROOT / "configs/controller/branch_a.json")
    assert config.maximum_horizon == 8
    assert config.initial_control_steps == 4
    for key in ("simulator.config", "simulator.base_config", "dynamics.checkpoint"):
        raw = config.raw
        for part in key.split("."):
            raw = raw[part]
        assert not Path(str(raw)).is_absolute()

