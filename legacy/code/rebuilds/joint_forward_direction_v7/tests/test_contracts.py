from __future__ import annotations

import numpy as np

from direction_rebuild_v4.data import labels_from_normalized_change
from joint_forward_direction_v7.build_targeted_dataset import (
    category_schedule,
    peak_interaction_setup,
)
from joint_forward_direction_v7.models import (
    shared_forward_direction_model_v7,
)
from joint_forward_direction_v7.runtime import gated_direction_indices
from specialist_rebuild_v2.common import CLASSES, DIRECTION_FIELDS, STATE_FIELDS


def test_shared_model_output_shapes() -> None:
    import torch

    model = shared_forward_direction_model_v7(
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


def test_heads_start_as_exact_prior_and_neutral_correction() -> None:
    import torch

    model = shared_forward_direction_model_v7(
        torch,
        51,
        width=32,
        hidden_dim=24,
        residual_blocks=1,
        dropout=0.0,
    )
    prior = torch.randn(11, 5)
    changes, logits = model(torch.randn(11, 51), prior)
    np.testing.assert_allclose(
        changes.detach().numpy(),
        prior.numpy(),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        logits.detach().numpy(),
        0.0,
        atol=0.0,
        rtol=0.0,
    )


def test_neutral_correction_retains_thresholded_forward_direction() -> None:
    changes = np.asarray(
        [[-2.0, -1.0, 0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    logits = np.zeros((1, 5, 3), dtype=np.float32)
    predicted = gated_direction_indices(
        changes,
        logits,
        minimum_probability=0.34,
        minimum_margin=0.0,
    )
    np.testing.assert_array_equal(
        predicted,
        labels_from_normalized_change(changes),
    )


def test_confident_correction_is_applied() -> None:
    changes = np.zeros((1, 5), dtype=np.float32)
    logits = np.zeros((1, 5, 3), dtype=np.float32)
    logits[0, 0] = np.asarray([8.0, 0.0, -1.0], dtype=np.float32)
    predicted = gated_direction_indices(
        changes,
        logits,
        minimum_probability=0.90,
        minimum_margin=0.50,
    )
    assert predicted[0, 0] == 0
    np.testing.assert_array_equal(predicted[0, 1:], 1)


def test_ambiguous_correction_is_rejected_by_margin() -> None:
    changes = np.zeros((1, 5), dtype=np.float32)
    logits = np.zeros((1, 5, 3), dtype=np.float32)
    logits[0, 0] = np.asarray([2.0, 1.9, -4.0], dtype=np.float32)
    predicted = gated_direction_indices(
        changes,
        logits,
        minimum_probability=0.45,
        minimum_margin=0.10,
    )
    assert predicted[0, 0] == 1


def test_peak_interaction_setup_is_deterministic_and_bounded() -> None:
    first = peak_interaction_setup(17, "train", "group-a")
    second = peak_interaction_setup(17, "train", "group-a")
    assert first == second
    assert 598.0 <= first["wavelength_nm"] <= 672.0
    assert 0.58 <= first["power_w"] <= 1.42
    assert not 0.72 < first["power_w"] < 1.28
    assert 16.0 <= first["lens_aperture_mm"] <= 20.0
    assert 0.22 <= abs(first["lens_x_offset_mm"]) <= 0.32
    assert 0.16 <= abs(first["camera_y_offset_mm"]) <= 0.24


def test_targeted_category_schedule_has_exact_requested_counts() -> None:
    requested = {
        "peak_interaction": 12,
        "hard_interaction": 9,
        "ood_boundary": 6,
        "iid_expanded": 3,
    }
    schedule = category_schedule(19, 30, requested)
    assert len(schedule) == 30
    assert {
        category: schedule.count(category) for category in requested
    } == requested
