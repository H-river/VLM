from __future__ import annotations

import pytest

from physics_structured_rebuild_v9.models import (
    build_structured_forward_model,
)
from physics_structured_rebuild_v9.strict_forward_runtime import (
    build_strict_forward_model,
)


@pytest.fixture
def config() -> dict[str, float | int]:
    return {
        "dimension": 32,
        "heads": 4,
        "feedforward": 64,
        "layers": 2,
        "head_hidden": 16,
        "dropout": 0.0,
    }


def test_structured_forward_shapes(
    config: dict[str, float | int],
) -> None:
    import torch

    model = build_structured_forward_model(torch, config)
    values = torch.randn(7, 46)
    change, next_state, direction = model.forward_with_aux(values)
    assert change.shape == (7, 5)
    assert next_state.shape == (7, 5)
    assert direction.shape == (7, 5, 3)


def test_zero_action_has_exact_zero_change(
    config: dict[str, float | int],
) -> None:
    import torch

    model = build_structured_forward_model(torch, config)
    values = torch.randn(3, 46)
    values[:, 17:21] = 0.0
    change, _ = model(values)
    assert torch.equal(change, torch.zeros_like(change))


def test_strict_forward_correction_starts_as_identity() -> None:
    import torch

    model = build_strict_forward_model(
        torch,
        {
            "input_dim": 56,
            "width": 32,
            "depth": 2,
            "dropout": 0.0,
        },
    )
    correction = model(torch.randn(7, 56))
    assert correction.shape == (7, 5)
    assert torch.equal(correction, torch.zeros_like(correction))
