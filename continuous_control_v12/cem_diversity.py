"""Deterministic elite-diversity helpers for controlled CEM ablations."""

from __future__ import annotations

import numpy as np


def select_diverse_elite_indices(
    scores: np.ndarray,
    effective_sequences: np.ndarray,
    *,
    elites: int,
    action_scale: np.ndarray,
    minimum_normalized_distance: float,
) -> np.ndarray:
    """Keep the best candidate, then impose a minimum RMS action distance."""
    scores = np.asarray(scores, dtype=np.float64)
    sequences = np.asarray(effective_sequences, dtype=np.float64)
    scale = np.asarray(action_scale, dtype=np.float64)
    if scores.ndim != 1 or sequences.shape[0] != scores.shape[0]:
        raise ValueError("scores and candidate sequences do not align")
    if not 1 <= elites < len(scores):
        raise ValueError("elites must be in [1, population)")
    if scale.shape != (sequences.shape[-1],) or np.any(scale <= 0):
        raise ValueError("action scale must be positive and match the action dimension")
    if minimum_normalized_distance < 0:
        raise ValueError("minimum normalized distance must be nonnegative")
    ordered = np.argsort(scores, kind="stable")
    if minimum_normalized_distance == 0:
        return ordered[:elites]
    normalized = sequences / scale.reshape((1,) * (sequences.ndim - 1) + (-1,))
    flattened = normalized.reshape(len(scores), -1)
    selected: list[int] = [int(ordered[0])]
    for index_value in ordered[1:]:
        index = int(index_value)
        distances = [
            float(np.sqrt(np.mean(np.square(flattened[index] - flattened[other]))))
            for other in selected
        ]
        if min(distances) + 1e-15 >= minimum_normalized_distance:
            selected.append(index)
            if len(selected) == elites:
                break
    if len(selected) < elites:
        selected_set = set(selected)
        selected.extend(
            int(index) for index in ordered if int(index) not in selected_set
        )
    return np.asarray(selected[:elites], dtype=np.int64)


def _pairwise_distances(
    effective_sequences: np.ndarray,
    indices: np.ndarray,
    action_scale: np.ndarray,
) -> list[float]:
    sequences = np.asarray(effective_sequences, dtype=np.float64)[indices]
    scale = np.asarray(action_scale, dtype=np.float64)
    normalized = sequences / scale.reshape((1,) * (sequences.ndim - 1) + (-1,))
    flattened = normalized.reshape(len(sequences), -1)
    return [
        float(np.sqrt(np.mean(np.square(flattened[left] - flattened[right]))))
        for left in range(len(flattened))
        for right in range(left + 1, len(flattened))
    ]


def mean_pairwise_normalized_distance(
    effective_sequences: np.ndarray,
    indices: np.ndarray,
    action_scale: np.ndarray,
) -> float:
    """Return RMS normalized action distance among selected candidates."""
    values = _pairwise_distances(effective_sequences, indices, action_scale)
    return float(np.mean(values)) if values else 0.0


def minimum_pairwise_normalized_distance(
    effective_sequences: np.ndarray,
    indices: np.ndarray,
    action_scale: np.ndarray,
) -> float:
    """Return the closest RMS normalized action distance among elites."""
    values = _pairwise_distances(effective_sequences, indices, action_scale)
    return min(values) if values else 0.0
