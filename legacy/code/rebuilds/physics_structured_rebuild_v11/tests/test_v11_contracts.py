from __future__ import annotations

import json
from pathlib import Path

from physics_structured_rebuild_v11.contracts import (
    LEARNING_CURVE_SIZES,
    OVERFIT_GROUP_SIZES,
)
from physics_structured_rebuild_v11.train import parity_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (REPO_ROOT / "physics_structured_rebuild_v11/config_v11.json").read_text()
)


def test_learning_curve_and_overfit_sizes_are_preregistered() -> None:
    assert LEARNING_CURVE_SIZES == (326, 678, 1200, 2400, 5000, 10500)
    assert OVERFIT_GROUP_SIZES == (32, 64)
    assert tuple(CONFIG["learning_curve_group_sizes"]) == LEARNING_CURVE_SIZES
    assert tuple(CONFIG["overfit_group_sizes"]) == OVERFIT_GROUP_SIZES


def test_weak_baseline_marks_ablations_not_interpretable() -> None:
    weak = {
        "normalized_mae": 10.0,
        "strict_all_five_accuracy": 0.0,
    }
    result = parity_gate(weak, weak, CONFIG)
    assert not result["passed"]
    assert result["ablation_interpretation"] == "not_interpretable"


def test_ablation_matrix_is_matched_around_system_aligned_baseline() -> None:
    ablations = CONFIG["ablations"]
    assert ablations["baseline"] == {
        "loss": "tolerance_normalized",
        "action_representation": "structured",
        "distribution": "system_aligned",
    }
    assert ablations["ordinary_loss"]["action_representation"] == "structured"
    assert ablations["opaque_action"]["loss"] == "tolerance_normalized"
    assert ablations["legacy_distribution"]["distribution"] == "legacy_mixed"

