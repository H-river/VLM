from __future__ import annotations

import pytest

from active_diagnosis_v13.contracts import assert_policy_visible


def test_visible_policy_record_is_accepted() -> None:
    assert_policy_visible(
        {
            "observed_metrics": [1.0, 2.0],
            "gain_belief": 0.75,
            "history": [{"command": [0.01, -0.01]}],
        }
    )


@pytest.mark.parametrize(
    "key",
    ["true_gain", "q_goal_mm", "oracle_result", "evaluator_only_true_position"],
)
def test_hidden_or_oracle_features_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="forbidden policy feature"):
        assert_policy_visible({"nested": [{key: 1.0}]})
