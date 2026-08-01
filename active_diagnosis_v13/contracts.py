"""Visibility and leakage contracts for v13 decision features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FORBIDDEN_POLICY_KEYS = {
    "true_gain",
    "evaluator_only_true_gain",
    "true_position",
    "evaluator_only_true_position",
    "q_goal",
    "q_goal_mm",
    "future_simulator_outcome",
    "oracle_result",
    "protected_result",
}


def assert_policy_visible(value: Any, *, path: str = "root") -> None:
    """Reject hidden state and evaluator-only outcomes in deployed features."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_POLICY_KEYS or normalized.startswith(
                "evaluator_only_"
            ):
                raise ValueError(f"forbidden policy feature at {path}.{key}")
            assert_policy_visible(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            assert_policy_visible(nested, path=f"{path}[{index}]")


def require_frozen_branch_a_refinement(gate_dir: Path, model_path: Path) -> dict[str, Any]:
    """Require a formal Branch-A resolution matching the reduced-feature model."""

    resolution_path = gate_dir.resolve() / "branch_a_resolution.json"
    if not resolution_path.exists():
        raise ValueError(
            "reduced Branch-A artifacts are forbidden before branch_a_resolution.json"
        )
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    if not resolution.get("frozen") or resolution.get("selected_primary_branch") != "A":
        raise ValueError("reduced artifacts require a frozen Branch-A resolution")
    if resolution.get("protected_set_used_for_selection", True):
        raise ValueError("Branch-A resolution must be development-only")
    selected = resolution.get("selected_probe")
    if not isinstance(selected, Mapping):
        raise ValueError("Branch-A resolution has no selected probe")
    requested = model_path.resolve()
    frozen = Path(str(selected.get("classifier_bundle", ""))).resolve()
    if requested != frozen:
        raise ValueError("retained-feature model differs from the frozen Branch-A model")
    digest = hashlib.sha256(requested.read_bytes()).hexdigest()
    if digest != str(selected.get("classifier_bundle_sha256", "")):
        raise ValueError("retained-feature model hash differs from the frozen Branch-A model")
    return resolution
