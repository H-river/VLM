"""Pure helpers for validation-only, pre-registered checkpoint selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def gate_margin_summary(margins: Mapping[str, float]) -> dict[str, Any]:
    if not margins:
        raise ValueError("checkpoint selection requires at least one gate margin")
    values = {name: float(value) for name, value in margins.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("checkpoint selection gate margins must be finite")
    passed = {name: value >= 0.0 for name, value in values.items()}
    failed = [name for name, value in passed.items() if not value]
    return {
        "margins": values,
        "passed": passed,
        "gate_count": len(values),
        "passed_gate_count": len(values) - len(failed),
        "failed_gates": failed,
        "minimum_margin": min(values.values()),
        "mean_margin": sum(values.values()) / len(values),
    }


def gate_aware_selection_key(
    gate_summary: Mapping[str, Any],
    composite_score: float,
    *tie_breakers: float,
) -> tuple[float, ...]:
    score = float(composite_score)
    if not math.isfinite(score):
        raise ValueError("checkpoint composite score must be finite")
    values = (
        float(gate_summary["passed_gate_count"]),
        float(gate_summary["minimum_margin"]),
        float(gate_summary["mean_margin"]),
        score,
        *(float(value) for value in tie_breakers),
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("checkpoint selection key must be finite")
    return values
