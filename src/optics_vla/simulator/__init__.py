"""Canonical simulator identity.

The implementation remains in :mod:`continuous_control_v12.simulator` because
historical manifests and hashes refer to that module.
"""

from continuous_control_v12.simulator import (
    CORRECTED_SEMANTICS_VERSION,
    default_simulator_fixed,
    sample_group_setup,
    simulate_state,
)

__all__ = [
    "CORRECTED_SEMANTICS_VERSION",
    "default_simulator_fixed",
    "sample_group_setup",
    "simulate_state",
]

