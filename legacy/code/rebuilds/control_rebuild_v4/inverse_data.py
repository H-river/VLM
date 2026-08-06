"""Deterministic inverse requests and paired measurement-error augmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from control_rebuild_v3.common import ACTION_GRID, STATUS_INDEX
from measurement_rebuild_v3.common import measurement_tolerance
from specialist_rebuild_v2.common import (
    STATE_FIELDS,
    inverse_context,
    matching_mask,
    minimum_motion_index,
    raw_state_array,
    stable_token,
)


def state_mapping(values: np.ndarray) -> dict[str, float]:
    return {field: float(values[index]) for index, field in enumerate(STATE_FIELDS)}


def derived_inverse_pairs(
    rows: Sequence[Mapping[str, Any]],
    reachable_per_group: int = 4,
    infeasible_per_group: int = 2,
) -> list[dict[str, Any]]:
    """Derive reachable and guaranteed-infeasible requests from true grids."""

    pairs: list[dict[str, Any]] = []
    for row in rows:
        group_id = str(row["group_id"])
        states = np.asarray(
            [
                raw_state_array(candidate["next_state"])
                for candidate in row["candidates"]
            ],
            dtype=np.float32,
        )
        offset = int(stable_token(group_id, "inverse_v4_schedule")[:8], 16)
        used: set[int] = set()
        for request_index in range(reachable_per_group):
            selected_source = (offset + request_index * 23) % len(ACTION_GRID)
            while selected_source in used:
                selected_source = (selected_source + 1) % len(ACTION_GRID)
            used.add(selected_source)
            desired = states[selected_source].copy()
            positive = matching_mask(states, desired)
            matches = np.flatnonzero(positive).tolist()
            status = "unique" if len(matches) == 1 else "ambiguous"
            pairs.append(
                {
                    "request_id": (f"{group_id}:reachable:{request_index:02d}"),
                    "group_id": group_id,
                    "desired_beam_state": state_mapping(desired),
                    "matching_indices": matches,
                    "selected_index": minimum_motion_index(matches),
                    "status": status,
                    "source": "v4_derived_reachable",
                }
            )
        for request_index in range(infeasible_per_group):
            desired = raw_state_array(row["current_beam_state"]).copy()
            axis = (offset + request_index) % 2
            direction = -1 if ((offset // 2 + request_index) % 2) else 1
            if direction > 0:
                desired[axis] = float(states[:, axis].max()) + 1.25
            else:
                desired[axis] = float(states[:, axis].min()) - 1.25
            matches = np.flatnonzero(matching_mask(states, desired)).tolist()
            if matches:
                raise AssertionError(
                    f"{group_id}: constructed infeasible request has matches"
                )
            pairs.append(
                {
                    "request_id": (f"{group_id}:infeasible:{request_index:02d}"),
                    "group_id": group_id,
                    "desired_beam_state": state_mapping(desired),
                    "matching_indices": [],
                    "selected_index": None,
                    "status": "infeasible_within_limits",
                    "source": "v4_derived_infeasible",
                }
            )
    return pairs


def sample_absolute_errors(
    normalized_error_bank: np.ndarray,
    references: np.ndarray,
    rng: np.random.Generator,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_indices = (
        rng.integers(0, len(normalized_error_bank), size=len(references))
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    if len(sampled_indices) != len(references):
        raise ValueError("one measurement-error index is required per state")
    tolerance = np.asarray(
        [measurement_tolerance(value) for value in references],
        dtype=np.float32,
    )
    return (normalized_error_bank[sampled_indices] * tolerance).astype(
        np.float32
    ), sampled_indices


def paired_error_indices(
    error_condition_bank: np.ndarray,
    desired_conditions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty(len(desired_conditions), dtype=np.int64)
    for condition in np.unique(desired_conditions):
        request_mask = desired_conditions == condition
        pool = np.flatnonzero(error_condition_bank == condition)
        if len(pool) == 0:
            raise ValueError(f"measurement-error bank has no condition {condition}")
        output[request_mask] = rng.choice(
            pool, size=int(request_mask.sum()), replace=True
        )
    return output


def physically_valid_states(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    output[..., 2:4] = np.maximum(output[..., 2:4], 0.25)
    output[..., 4] = np.maximum(output[..., 4], 1e-8)
    return output


def request_arrays(
    pairs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    predicted_clean: np.ndarray,
    predicted_noisy: np.ndarray,
    normalized_error_bank: np.ndarray,
    error_condition_bank: np.ndarray,
    seed: int,
    noisy_fraction: float,
) -> dict[str, Any]:
    """Create model inputs while retaining true physical labels.

    For noisy requests, one empirical error perturbs the measured current state
    and a separately sampled error perturbs the desired state.  Candidate
    states come from running the forward model on the perturbed current state.
    Labels remain defined by the unperturbed simulator grid.
    """

    if not 0.0 <= noisy_fraction <= 1.0:
        raise ValueError("noisy_fraction must be between zero and one")
    row_position = {str(row["group_id"]): index for index, row in enumerate(rows)}
    true_current_by_group = np.asarray(
        [raw_state_array(row["current_beam_state"]) for row in rows],
        dtype=np.float32,
    )
    rng = np.random.default_rng(seed)
    group_error, group_error_indices = sample_absolute_errors(
        normalized_error_bank,
        true_current_by_group,
        rng,
    )
    group_error_conditions = error_condition_bank[group_error_indices]
    measured_current_by_group = physically_valid_states(
        true_current_by_group + group_error
    )

    group_indices = np.asarray(
        [row_position[str(pair["group_id"])] for pair in pairs],
        dtype=np.int64,
    )
    desired_true = np.asarray(
        [raw_state_array(pair["desired_beam_state"]) for pair in pairs],
        dtype=np.float32,
    )
    desired_conditions = group_error_conditions[group_indices]
    desired_error_indices = paired_error_indices(
        error_condition_bank, desired_conditions, rng
    )
    desired_error, _ = sample_absolute_errors(
        normalized_error_bank,
        desired_true,
        rng,
        indices=desired_error_indices,
    )
    desired_noisy = physically_valid_states(desired_true + desired_error)
    noisy = rng.random(len(pairs)) < float(noisy_fraction)
    observed_current = true_current_by_group[group_indices].copy()
    observed_current[noisy] = measured_current_by_group[group_indices[noisy]]
    observed_desired = desired_true.copy()
    observed_desired[noisy] = desired_noisy[noisy]

    contexts = np.asarray(
        [
            inverse_context(
                rows[group_indices[index]]["setup"],
                state_mapping(observed_current[index]),
                state_mapping(observed_desired[index]),
            )
            for index in range(len(pairs))
        ],
        dtype=np.float32,
    )
    positives = np.zeros((len(pairs), len(ACTION_GRID)), dtype=np.bool_)
    statuses = np.empty(len(pairs), dtype=np.int64)
    selected = np.full(len(pairs), -1, dtype=np.int64)
    for index, pair in enumerate(pairs):
        matches = np.asarray(pair["matching_indices"], dtype=np.int64)
        positives[index, matches] = True
        statuses[index] = STATUS_INDEX[str(pair["status"])]
        if pair["selected_index"] is not None:
            selected[index] = int(pair["selected_index"])

    candidate_bank = np.concatenate([predicted_clean, predicted_noisy], axis=0).astype(
        np.float32
    )
    candidate_indices = group_indices.copy()
    candidate_indices[noisy] += len(rows)
    return {
        "contexts": contexts,
        "desired_observed": observed_desired,
        "desired_true": desired_true,
        "observed_current": observed_current,
        "candidate_bank": candidate_bank,
        "candidate_indices": candidate_indices,
        "positives": positives,
        "statuses": statuses,
        "selected": selected,
        "noisy": noisy,
        "group_indices": group_indices,
        "measured_current_by_group": measured_current_by_group,
        "measurement_condition_indices": desired_conditions,
    }
