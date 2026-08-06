from __future__ import annotations

import numpy as np

from joint_forward_direction_v6.models import shared_forward_direction_model
from specialist_rebuild_v2.common import CLASSES, DIRECTION_FIELDS, STATE_FIELDS


def test_shared_model_output_shapes() -> None:
    import torch

    model = shared_forward_direction_model(
        torch,
        51,
        width=32,
        hidden_dim=24,
        residual_blocks=1,
        dropout=0.0,
    )
    values = torch.zeros(7, 51)
    prior = torch.zeros(7, 5)
    changes, logits = model(values, prior)
    assert changes.shape == (7, len(STATE_FIELDS))
    assert logits.shape == (
        7,
        len(DIRECTION_FIELDS),
        len(CLASSES),
    )
    assert torch.isfinite(changes).all()
    assert torch.isfinite(logits).all()


def test_regression_heads_start_as_exact_prior() -> None:
    import torch

    model = shared_forward_direction_model(
        torch,
        51,
        width=32,
        hidden_dim=24,
        residual_blocks=1,
        dropout=0.0,
    )
    prior = torch.randn(11, 5)
    changes, _ = model(torch.randn(11, 51), prior)
    np.testing.assert_allclose(
        changes.detach().numpy(),
        prior.numpy(),
        atol=0.0,
        rtol=0.0,
    )


def test_direction_heads_start_as_exact_threshold_rule() -> None:
    import torch

    model = shared_forward_direction_model(
        torch,
        51,
        width=32,
        hidden_dim=24,
        residual_blocks=1,
        dropout=0.0,
    )
    prior = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [2.0, 1.1, -1.1, 0.5, -0.5],
        ]
    )
    _, logits = model(torch.randn(2, 51), prior)
    expected = torch.tensor(
        [
            [0, 1, 1, 1, 2],
            [2, 2, 0, 1, 1],
        ]
    )
    assert torch.equal(logits.argmax(dim=-1), expected)
