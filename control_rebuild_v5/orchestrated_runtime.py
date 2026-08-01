"""Numerical-v5 overlay on the frozen Qwen and v4 visual system."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from control_rebuild_v5.forward_runtime import load_forward_runtime_v5
from control_rebuild_v5.inverse_runtime import load_inverse_runtime_v5


class OrchestratedNumericalRuntimeV5:
    """Replace only numerical forward and numerical inverse execution.

    Measurement, direction, and the complete visual-inverse pipeline remain
    exactly as pinned by the supplied v4 direction-overlay manifest.
    """

    def __init__(
        self,
        backend: OrchestratedSpecialistRuntimeV4,
        forward_artifact: Path,
        inverse_artifact: Path,
    ) -> None:
        self.backend = backend
        self.forward, _ = load_forward_runtime_v5(forward_artifact)
        self.inverse, _ = load_inverse_runtime_v5(inverse_artifact)
        # v4 forward routes and the numerical state-inverse route call these
        # public attributes. The visual pipeline has its own frozen instances.
        self.backend.forward = self.forward
        self.backend.inverse = self.inverse

    def dispatch(
        self,
        decision: Mapping[str, Any],
        available_images: Mapping[str, str | Path] | None = None,
    ) -> dict[str, Any]:
        return self.backend.dispatch(decision, available_images)

