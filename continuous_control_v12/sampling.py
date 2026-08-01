"""Deterministic mixed continuous-action sampling within one optical group."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import qmc

from continuous_control_v12.contracts import Bounds, legacy_action_array


def _key(action: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(action, dtype=np.float64), 12).tolist())


def sample_continuous_actions(
    budget: int,
    seed: int,
    bounds: Bounds,
    fractions: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return exactly ``budget`` tagged actions, including paired probes."""

    if budget < 8:
        raise ValueError("continuous probe budget must be at least 8")
    bounds.validate()
    configured = {
        "paired_fraction": 0.1875,
        "axis_fraction": 0.125,
        "fine_fraction": 0.125,
        "legacy_fraction": 0.125,
        "multi_axis_fraction": 0.0625,
        "sobol_fraction": 0.375,
    }
    if fractions is not None:
        configured.update(
            {
                key: float(fractions[key])
                for key in configured
                if key in fractions
            }
        )
    if any(value < 0.0 for value in configured.values()):
        raise ValueError("sampling fractions must be non-negative")
    if sum(configured.values()) > 1.0000001:
        raise ValueError("sampling fractions must sum to at most one")
    rng = np.random.default_rng(seed)
    scale = bounds.action_high
    output: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()

    def add(action: np.ndarray, kind: str, **metadata: Any) -> bool:
        values = np.clip(
            np.asarray(action, dtype=np.float64),
            bounds.action_low,
            bounds.action_high,
        )
        key = _key(values)
        if key in seen or len(output) >= budget:
            return False
        seen.add(key)
        output.append(
            {
                "action": values,
                "sampling": {"kind": kind, **metadata},
            }
        )
        return True

    add(np.zeros(4), "no_op")

    pair_count = max(
        1, int(round(budget * configured["paired_fraction"] / 2.0))
    )
    for pair_index in range(pair_count):
        direction = rng.normal(size=4)
        direction /= max(float(np.max(np.abs(direction))), 1e-12)
        magnitude = rng.uniform(0.25, 0.9)
        base = direction * scale * magnitude
        add(base, "paired", pair_id=f"pair_{pair_index:03d}", sign=1)
        add(-base, "paired", pair_id=f"pair_{pair_index:03d}", sign=-1)

    axis_count = max(2, int(round(budget * configured["axis_fraction"])))
    for index in range(axis_count):
        axis = index % 4
        action = np.zeros(4)
        action[axis] = (
            (-1.0 if (index // 4) % 2 else 1.0)
            * scale[axis]
            * rng.uniform(0.35, 1.0)
        )
        add(action, "axis_aligned", axis=axis)

    fine_count = max(1, int(round(budget * configured["fine_fraction"])))
    for _ in range(fine_count):
        add(rng.uniform(-0.12, 0.12, size=4) * scale, "near_zero_fine")

    legacy = legacy_action_array()
    legacy_order = rng.permutation(len(legacy))
    legacy_count = max(1, int(round(budget * configured["legacy_fraction"])))
    for index in legacy_order[:legacy_count]:
        add(legacy[index], "legacy_grid", legacy_action_index=int(index))

    multi_count = max(
        1, int(round(budget * configured["multi_axis_fraction"]))
    )
    for _ in range(multi_count):
        values = rng.uniform(-1.0, 1.0, size=4) * scale
        small = np.argsort(np.abs(values))[:1]
        values[small] = 0.0
        add(values, "multi_axis")

    remaining = budget - len(output)
    if remaining > 0:
        exponent = max(1, int(math.ceil(math.log2(remaining + 8))))
        sobol = qmc.Sobol(d=4, scramble=True, seed=int(seed % (2**32)))
        values = qmc.scale(
            sobol.random_base2(exponent),
            bounds.action_low,
            bounds.action_high,
        )
        for index, action in enumerate(values):
            add(action, "sobol", sobol_index=index)
            if len(output) == budget:
                break
    while len(output) < budget:
        add(rng.uniform(bounds.action_low, bounds.action_high), "uniform_fill")
    return output
