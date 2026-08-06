from __future__ import annotations

import pytest

from control_rebuild_v3.common import inverse_candidate_features
from tabm_transformer_rebuild_v8.inverse_features import (
    inverse_features_torch,
)
from tabm_transformer_rebuild_v8.models import (
    build_forward_direction_model,
    build_inverse_model,
)
from tabm_transformer_rebuild_v8.train_inverse import hard_candidate_indices


@pytest.mark.parametrize("architecture", ["tabm", "transformer"])
def test_forward_direction_shapes(architecture: str) -> None:
    import torch

    config = (
        {"k": 4, "n_blocks": 2, "d_block": 32, "dropout": 0.0}
        if architecture == "tabm"
        else {
            "dimension": 32,
            "heads": 4,
            "feedforward": 64,
            "layers": 2,
            "dropout": 0.0,
        }
    )
    model = build_forward_direction_model(torch, architecture, 46, config)
    changes, directions = model(torch.randn(3, 46))
    if architecture == "tabm":
        assert changes.shape == (3, 4, 5)
        assert directions.shape == (3, 4, 5, 3)
    else:
        assert changes.shape == (3, 5)
        assert directions.shape == (3, 5, 3)


@pytest.mark.parametrize("architecture", ["tabm", "transformer"])
def test_inverse_shapes(architecture: str) -> None:
    import torch

    config = (
        {"k": 4, "n_blocks": 2, "d_block": 32, "dropout": 0.0}
        if architecture == "tabm"
        else {
            "dimension": 32,
            "heads": 4,
            "feedforward": 64,
            "layers": 2,
            "dropout": 0.0,
        }
    )
    model = build_inverse_model(
        torch,
        architecture,
        context_dim=31,
        candidate_dim=23,
        status_dim=12,
        candidate_count=81,
        config=config,
    )
    correction, status = model(
        torch.randn(3, 31),
        torch.randn(3, 81, 23),
        torch.randn(3, 12),
    )
    if architecture == "tabm":
        assert correction.shape == (3, 81, 4)
        assert status.shape == (3, 4, 3)
    else:
        assert correction.shape == (3, 81)
        assert status.shape == (3, 3)


def test_torch_inverse_features_match_registered_numpy_features() -> None:
    import numpy as np
    import torch

    rng = np.random.default_rng(19)
    states = rng.normal(size=(4, 81, 5)).astype(np.float32)
    states[..., 4] += 10.0
    desired = rng.normal(size=(4, 5)).astype(np.float32)
    desired[..., 4] += 10.0
    expected_candidate, expected_cost, expected_status = (
        inverse_candidate_features(states, desired)
    )
    candidate, cost, status = inverse_features_torch(
        torch,
        torch.from_numpy(states),
        torch.from_numpy(desired),
    )
    np.testing.assert_allclose(
        candidate.numpy(),
        expected_candidate,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        cost.numpy(),
        expected_cost,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        status.numpy(),
        expected_status,
        rtol=2e-5,
        atol=2e-5,
    )


def test_hard_candidate_limit_when_many_actions_are_positive() -> None:
    import numpy as np

    states = np.zeros((1, 81, 5), dtype=np.float32)
    desired = np.zeros((1, 5), dtype=np.float32)
    positives = np.ones((1, 81), dtype=np.bool_)
    selected = hard_candidate_indices(
        states,
        desired,
        positives,
        count=32,
    )
    assert selected.shape == (1, 32)
    np.testing.assert_array_equal(selected[0], np.arange(32))
