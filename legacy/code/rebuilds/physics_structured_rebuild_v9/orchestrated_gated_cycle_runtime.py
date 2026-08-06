"""Final-cycle orchestration overlay for independently protected candidates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    BoundaryDirectionCorrectionRuntimeV9,
)
from physics_structured_rebuild_v9.inverse_success_runtime import (
    load_inverse_success_ranker_runtime_v9,
)
from physics_structured_rebuild_v9.runtime import (
    OrchestratedForwardSelectorEnsembleRuntimeV9,
)

RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
BOUNDARY_STATE_DIRECTION = (
    RUN_ROOT / "boundary_direction_protected_calibrated_state_v9.pkl"
)
INVERSE_SUCCESS_RANKER = RUN_ROOT / "inverse_success_ranker_v9.pkl"


class OrchestratedGatedCycleRuntimeV9(
    OrchestratedForwardSelectorEnsembleRuntimeV9
):
    """Apply only protected state-direction and numerical-inverse overlays."""

    def __init__(
        self,
        torch: Any,
        backend: OrchestratedSpecialistRuntimeV4,
        shared_artifact: Path,
        v5_forward_artifact: Path,
        v5_inverse_artifact: Path,
        device: Any,
    ) -> None:
        super().__init__(
            torch,
            backend,
            shared_artifact,
            v5_forward_artifact,
            v5_inverse_artifact,
            device,
        )
        self.boundary_state_direction = (
            BoundaryDirectionCorrectionRuntimeV9(
                torch,
                BOUNDARY_STATE_DIRECTION,
                device,
            )
        )
        self.v5_inverse, _ = load_inverse_success_ranker_runtime_v9(
            INVERSE_SUCCESS_RANKER,
            torch,
            device,
        )

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        if (
            str(decision.get("route_name"))
            == "predict_direction_from_state_v1"
        ):
            self.backend.direction = self.boundary_state_direction
            return self.backend.dispatch(decision, available_images)
        return super().dispatch(decision, available_images)
