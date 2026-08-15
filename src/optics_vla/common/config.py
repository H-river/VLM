"""Typed access to the canonical, repository-relative controller config."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def repository_root() -> Path:
    """Return the checkout root for an editable/source installation."""

    return Path(__file__).resolve().parents[3]


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class ControllerConfig:
    source: Path
    raw: Mapping[str, Any]

    @property
    def initial_control_steps(self) -> int:
        return int(self.raw["continuation"]["initial_control_steps"])

    @property
    def maximum_horizon(self) -> int:
        return int(self.raw["continuation"]["maximum_horizon"])

    @property
    def minimum_last_step_improvement(self) -> float:
        return float(self.raw["continuation"]["minimum_last_step_improvement"])

    @property
    def maximum_final_distance(self) -> float:
        value = self.raw["continuation"]["maximum_final_distance"]
        return float("inf") if value == "infinity" else float(value)

    def resolve(self, key_path: str) -> Path:
        value: Any = self.raw
        for key in key_path.split("."):
            value = value[key]
        path = Path(str(value))
        if path.is_absolute():
            raise ValueError(f"canonical path must be repository-relative: {key_path}")
        return repository_root() / path

    def checkpoint_available(self) -> bool:
        return self.resolve("dynamics.checkpoint").is_file()

    def verify_checkpoint(self) -> bool:
        path = self.resolve("dynamics.checkpoint")
        if not path.is_file():
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest == str(self.raw["dynamics"]["checkpoint_sha256"])


def load_controller_config(path: Path | None = None) -> ControllerConfig:
    source = path or repository_root() / "configs/controller/branch_a.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "canonical":
        raise ValueError("controller config must be marked canonical")
    continuation = payload["continuation"]
    initial = _positive_int(continuation["initial_control_steps"], "initial_control_steps")
    maximum = _positive_int(continuation["maximum_horizon"], "maximum_horizon")
    if maximum < initial:
        raise ValueError("maximum_horizon must not be below initial_control_steps")
    if float(continuation["minimum_last_step_improvement"]) < 0.0:
        raise ValueError("minimum_last_step_improvement must be nonnegative")
    if int(payload["dynamics"]["model_horizon"]) != 1:
        raise ValueError("the canonical Learned-H1 model must remain one-step")
    if payload["branch_a_preamble"]["probe_design"] != "symmetric_pair":
        raise ValueError("the frozen Branch-A preamble must remain symmetric_pair")
    config = ControllerConfig(source=source.resolve(), raw=payload)
    for key in (
        "simulator.config",
        "simulator.base_config",
        "dynamics.checkpoint",
    ):
        config.resolve(key)
    return config

