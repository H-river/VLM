from __future__ import annotations

import pytest

from specialist_rebuild_v2.models import (
    direction_model,
    forward_model,
    inverse_ranker,
    measurement_model,
    require_torch,
    visual_inverse_ranker,
)


@pytest.fixture(scope="module")
def torch():
    return require_torch()


def test_direction_shape(torch) -> None:
    assert direction_model(torch)(torch.randn(2, 21)).shape == (2, 5, 3)


def test_forward_shapes(torch) -> None:
    output = forward_model(torch)(torch.randn(2, 46))
    assert [tuple(values.shape) for values in output] == [
        (2, 5),
        (2, 5),
        (2, 5, 3),
    ]


def test_inverse_shapes(torch) -> None:
    output = inverse_ranker(torch)(torch.randn(2, 31), torch.randn(2, 81, 5))
    assert [tuple(values.shape) for values in output] == [(2, 81), (2, 3)]


def test_measurement_shapes(torch) -> None:
    output = measurement_model(torch)(
        torch.randn(2, 1, 192, 192), torch.randn(2, 4)
    )
    assert [tuple(values.shape) for values in output] == [(2, 5), (2, 5)]


def test_visual_inverse_shapes(torch) -> None:
    output = visual_inverse_ranker(torch)(
        torch.randn(2, 1, 192, 192),
        torch.randn(2, 1, 192, 192),
        torch.randn(2, 16),
        torch.randn(2, 81, 4),
    )
    assert [tuple(values.shape) for values in output] == [(2, 81), (2, 3)]
