"""Torch implementation of the registered numerical-inverse features."""

from __future__ import annotations

from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_NORMALIZED, MOVEMENT


def inverse_features_torch(
    torch: Any,
    states: Any,
    desired: Any,
) -> tuple[Any, Any, Any]:
    """Return 23 candidate features, base cost and 12 request features."""

    peak_scale = torch.clamp(desired[:, 4:5].abs() * 0.02, min=1e-6)
    signed = torch.cat(
        [
            (states[..., 0:2] - desired[:, None, 0:2]) / 0.5,
            states[..., 2:4] - desired[:, None, 2:4],
            (
                states[..., 4:5] - desired[:, None, 4:5]
            )
            / peak_scale[:, None, :],
        ],
        dim=-1,
    )
    absolute = signed.abs()
    squared = signed.square()
    cost = squared.mean(dim=-1).sqrt()
    centroid = (
        signed[..., 0].square() + signed[..., 1].square()
    ).sqrt()
    action = torch.as_tensor(
        ACTION_NORMALIZED,
        dtype=states.dtype,
        device=states.device,
    )[None, :, :].expand(len(states), -1, -1)
    movement = torch.as_tensor(
        MOVEMENT,
        dtype=states.dtype,
        device=states.device,
    )[None, :, None].expand(len(states), -1, -1)
    candidate = torch.cat(
        [
            action,
            signed,
            absolute,
            squared,
            movement,
            cost[..., None],
            absolute.max(dim=-1, keepdim=True).values,
            centroid[..., None],
        ],
        dim=-1,
    )
    ordered = cost.sort(dim=1).values
    predicted_match = (
        (centroid <= 1.0)
        & (absolute[..., 2] <= 1.0)
        & (absolute[..., 3] <= 1.0)
        & (absolute[..., 4] <= 1.0)
    )
    status = torch.stack(
        [
            ordered[:, 0],
            ordered[:, 1],
            ordered[:, 2],
            ordered[:, 1] - ordered[:, 0],
            ordered[:, 2] - ordered[:, 0],
            predicted_match.float().mean(dim=1),
            (cost <= 1.0).float().mean(dim=1),
            centroid.min(dim=1).values,
            absolute[..., 2].min(dim=1).values,
            absolute[..., 3].min(dim=1).values,
            absolute[..., 4].min(dim=1).values,
            cost.mean(dim=1),
        ],
        dim=1,
    )
    return candidate, cost, status


def feature_statistics(
    states: np.ndarray,
    desired: np.ndarray,
    *,
    maximum_requests: int = 20000,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate candidate and status normalization from deterministic requests."""

    from control_rebuild_v3.common import inverse_candidate_features

    if len(states) > maximum_requests:
        selected = np.linspace(
            0,
            len(states) - 1,
            maximum_requests,
            dtype=np.int64,
        )
    else:
        selected = np.arange(len(states), dtype=np.int64)
    candidate_sum = None
    candidate_square_sum = None
    status_sum = None
    status_square_sum = None
    candidate_count = 0
    status_count = 0
    for start in range(0, len(selected), chunk_size):
        index = selected[start : start + chunk_size]
        candidate, _, status = inverse_candidate_features(
            states[index],
            desired[index],
        )
        flat = candidate.reshape(-1, candidate.shape[-1]).astype(np.float64)
        status64 = status.astype(np.float64)
        if candidate_sum is None:
            candidate_sum = flat.sum(axis=0)
            candidate_square_sum = np.square(flat).sum(axis=0)
            status_sum = status64.sum(axis=0)
            status_square_sum = np.square(status64).sum(axis=0)
        else:
            candidate_sum += flat.sum(axis=0)
            candidate_square_sum += np.square(flat).sum(axis=0)
            status_sum += status64.sum(axis=0)
            status_square_sum += np.square(status64).sum(axis=0)
        candidate_count += len(flat)
        status_count += len(status64)
    assert candidate_sum is not None
    assert candidate_square_sum is not None
    assert status_sum is not None
    assert status_square_sum is not None
    candidate_mean = candidate_sum / candidate_count
    candidate_scale = np.sqrt(
        np.maximum(
            candidate_square_sum / candidate_count - candidate_mean**2,
            1e-12,
        )
    )
    status_mean = status_sum / status_count
    status_scale = np.sqrt(
        np.maximum(
            status_square_sum / status_count - status_mean**2,
            1e-12,
        )
    )
    return (
        candidate_mean.astype(np.float32),
        np.maximum(candidate_scale, 1e-6).astype(np.float32),
        status_mean.astype(np.float32),
        np.maximum(status_scale, 1e-6).astype(np.float32),
    )
