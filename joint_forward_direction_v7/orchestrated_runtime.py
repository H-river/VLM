"""Shared-v7 overlay on the frozen Qwen and numerical-v5 system."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from control_rebuild_v5.forward_runtime import load_forward_runtime_v5
from control_rebuild_v5.inverse_runtime import load_inverse_runtime_v5
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)


class OrchestratedSharedRuntimeV7:
    """Replace forward and direction routes while preserving both inverse routes."""

    FORWARD_OR_DIRECTION = {
        "predict_forward_from_state_v1",
        "predict_forward_from_image_v1",
        "predict_direction_from_state_v1",
        "predict_direction_from_image_v1",
    }

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        self.backend = backend
        self.shared, _ = load_forward_direction_runtime_v7(
            shared_artifact,
            torch,
            device,
        )
        self.v5_forward, _ = load_forward_runtime_v5(v5_forward_artifact)
        self.v5_inverse, _ = load_inverse_runtime_v5(v5_inverse_artifact)

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        route = str(decision.get("route_name"))
        if route in self.FORWARD_OR_DIRECTION:
            self.backend.forward = self.shared
            self.backend.direction = self.shared
        elif route == "select_inverse_action_from_states_v1":
            self.backend.forward = self.v5_forward
            self.backend.inverse = self.v5_inverse
        return self.backend.dispatch(decision, available_images)
