from __future__ import annotations

import numpy as np

from physics_structured_rebuild_v10.contracts import (
    ZERO_ACTION_INDEX,
    action_cardinalities,
    explicit_action_features,
)


def test_canonical_action_cardinality_counts() -> None:
    values, counts = np.unique(action_cardinalities(), return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist())) == {
        0: 1,
        1: 8,
        2: 24,
        3: 32,
        4: 16,
    }


def test_explicit_action_features_are_finite_and_distinct() -> None:
    features = explicit_action_features()
    assert features.shape[0] == 81
    assert np.isfinite(features).all()
    assert len(np.unique(features, axis=0)) == 81
    assert np.all(features[ZERO_ACTION_INDEX, :12] == 0.0)

