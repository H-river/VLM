"""Candidate-only, fixed-gain reasoning-plan selector experiment.

The optical execution stack is imported lazily so the independent Qwen
training environment does not need simulator-only dependencies.
"""

from .protocol import PLAN_NAMES

__all__ = [
    "FIXED_CONTROLLER_CONFIG",
    "PLAN_NAMES",
    "FixedGainPlanController",
    "PlanCEM",
    "PlanSpec",
    "build_plan_bank",
]


def __getattr__(name: str):
    if name in set(__all__) - {"PLAN_NAMES"}:
        from . import core
        return getattr(core, name)
    raise AttributeError(name)
