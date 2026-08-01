"""Runtime selector over two forward enumerators and one frozen inverse ranker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from control_rebuild_v4.inverse_data import state_mapping


def selected_values(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return matrix[np.arange(len(indices)), indices]


def margins(scores: np.ndarray) -> np.ndarray:
    ordered = np.partition(scores, kth=-2, axis=1)
    return ordered[:, -1] - ordered[:, -2]


class InverseEnumeratorSelectorRuntimeV9:
    """Select primary or secondary ranker outputs using fixed diagnostics."""

    def __init__(
        self,
        primary_inverse: Any,
        secondary_forward: Any,
        threshold: float | None = None,
        classifier: Any | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        if (threshold is None) == (classifier is None):
            raise ValueError("provide exactly one selector rule")
        self.primary_inverse = primary_inverse
        self.secondary_forward = secondary_forward
        self.threshold = None if threshold is None else float(threshold)
        self.classifier = classifier
        self.feature_names = tuple(
            feature_names
            or (
                "selected_score_advantage",
                "selected_base_cost_advantage",
                "score_margin_advantage",
            )
        )
        supported = {
            "selected_score_advantage",
            "selected_base_cost_advantage",
            "score_margin_advantage",
        }
        if not self.feature_names or not set(self.feature_names) <= supported:
            raise ValueError("inverse selector feature names differ")

    def score_requests(
        self,
        setups: Sequence[Mapping[str, Any]],
        context_current: np.ndarray,
        context_desired: np.ndarray,
        candidate_states: np.ndarray,
        feature_desired: np.ndarray | None = None,
        batch_size: int = 128,
    ) -> dict[str, Any]:
        primary = self.primary_inverse.score_requests(
            setups,
            context_current,
            context_desired,
            candidate_states,
            feature_desired=feature_desired,
            batch_size=batch_size,
        )
        current = np.asarray(context_current, dtype=np.float32)
        rows = [
            {
                "group_id": f"inverse_selector_v9_{index}",
                "setup": dict(setups[index]),
                "current_beam_state": state_mapping(current[index]),
            }
            for index in range(len(setups))
        ]
        secondary_states = self.secondary_forward.predict_states(rows)
        secondary = self.primary_inverse.score_requests(
            setups,
            context_current,
            context_desired,
            secondary_states,
            feature_desired=feature_desired,
            batch_size=batch_size,
        )
        primary_index = np.asarray(
            primary["selected_indices"],
            dtype=np.int64,
        )
        secondary_index = np.asarray(
            secondary["selected_indices"],
            dtype=np.int64,
        )
        primary_scores = np.asarray(primary["scores"], dtype=np.float32)
        secondary_scores = np.asarray(
            secondary["scores"],
            dtype=np.float32,
        )
        primary_costs = np.asarray(primary["base_costs"], dtype=np.float32)
        secondary_costs = np.asarray(
            secondary["base_costs"],
            dtype=np.float32,
        )
        diagnostics = {
            "selected_score_advantage": (
                selected_values(secondary_scores, secondary_index)
                - selected_values(primary_scores, primary_index)
            ),
            "selected_base_cost_advantage": (
                selected_values(primary_costs, primary_index)
                - selected_values(secondary_costs, secondary_index)
            ),
            "score_margin_advantage": (
                margins(secondary_scores) - margins(primary_scores)
            ),
        }
        selector_features = np.column_stack(
            [diagnostics[name] for name in self.feature_names]
        ).astype(np.float32)
        if self.classifier is None:
            assert self.threshold is not None
            choose_secondary = (
                diagnostics["selected_score_advantage"] < self.threshold
            )
        else:
            choose_secondary = (
                np.asarray(self.classifier.predict(selector_features))
                .astype(np.int64)
                == 1
            )
        selected_index = np.where(
            choose_secondary,
            secondary_index,
            primary_index,
        )
        candidate_array = np.asarray(candidate_states).copy()
        candidate_array[choose_secondary] = secondary_states[choose_secondary]

        output = dict(primary)
        output["selected_indices"] = selected_index
        output["selected_actions"] = [
            (
                secondary["selected_actions"][index]
                if choose_secondary[index]
                else primary["selected_actions"][index]
            )
            for index in range(len(choose_secondary))
        ]
        output["predicted_status_indices"] = np.where(
            choose_secondary,
            np.asarray(secondary["predicted_status_indices"]),
            np.asarray(primary["predicted_status_indices"]),
        )
        output["predicted_statuses"] = [
            (
                secondary["predicted_statuses"][index]
                if choose_secondary[index]
                else primary["predicted_statuses"][index]
            )
            for index in range(len(choose_secondary))
        ]
        for key in (
            "scores",
            "base_costs",
            "status_probabilities",
            "status_logits",
        ):
            primary_value = np.asarray(primary[key]).copy()
            secondary_value = np.asarray(secondary[key])
            primary_value[choose_secondary] = secondary_value[
                choose_secondary
            ]
            output[key] = primary_value
        output["enumerator_selected"] = np.where(
            choose_secondary,
            "secondary",
            "primary",
        )
        output["enumerator_score_advantage"] = diagnostics[
            "selected_score_advantage"
        ]
        output["enumerator_selector_features"] = selector_features
        output["enumerator_selector_feature_names"] = self.feature_names
        output["enumerator_threshold"] = self.threshold
        return output
