from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    LEGACY_ACTIONS,
    Bounds,
    action_vector,
    apply_action,
    assert_no_q_star,
    legacy_action_array,
    position_dict,
    project_action,
)
from continuous_control_v12.goal import StructuredGoalAdapter
from continuous_control_v12.reachability import (
    classify_estimated_reachability,
)
from continuous_control_v12.sampling import sample_continuous_actions
from specialist_rebuild_v2.common import fixed_action_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (REPO_ROOT / "continuous_control_v12/config_v12.json").read_text()
)


def test_legacy_grid_has_81_unique_actions_in_historical_order() -> None:
    expected = [
        tuple(map(float, values))
        for values in itertools.product(
            (-0.05, 0.0, 0.05),
            (-0.05, 0.0, 0.05),
            (-0.02, 0.0, 0.02),
            (-0.02, 0.0, 0.02),
        )
    ]
    actual = [
        tuple(float(action[field]) for field in ACTION_FIELDS)
        for action in LEGACY_ACTIONS
    ]
    existing = [
        tuple(float(action[field]) for field in ACTION_FIELDS)
        for action in fixed_action_grid()
    ]
    assert len(actual) == 81
    assert len(set(actual)) == 81
    assert actual == expected == existing
    assert actual[40] == (0.0, 0.0, 0.0, 0.0)


def test_actions_and_updated_positions_respect_both_bounds() -> None:
    bounds = Bounds.from_config(CONFIG)
    position = np.asarray([2.99, -2.99, 2.995, -2.995])
    requested = np.asarray([0.05, -0.05, 0.02, -0.02])
    projected = project_action(position, requested, bounds)
    assert np.all(projected <= bounds.action_high)
    assert np.all(projected >= bounds.action_low)
    updated = apply_action(position, projected, bounds)
    assert np.all(updated <= bounds.position_high + 1e-12)
    assert np.all(updated >= bounds.position_low - 1e-12)
    with pytest.raises(ValueError):
        apply_action(position, requested, bounds)


def test_paired_sampling_and_seed_reproducibility() -> None:
    bounds = Bounds.from_config(CONFIG)
    left = sample_continuous_actions(64, 12345, bounds, CONFIG["sampling"])
    right = sample_continuous_actions(64, 12345, bounds, CONFIG["sampling"])
    assert [item["sampling"] for item in left] == [
        item["sampling"] for item in right
    ]
    assert np.allclose(
        np.stack([item["action"] for item in left]),
        np.stack([item["action"] for item in right]),
    )
    paired = {}
    for item in left:
        if item["sampling"]["kind"] == "paired":
            paired.setdefault(item["sampling"]["pair_id"], []).append(item)
    assert paired
    for pair in paired.values():
        assert len(pair) == 2
        assert {item["sampling"]["sign"] for item in pair} == {-1, 1}
        assert np.allclose(pair[0]["action"], -pair[1]["action"])


def test_q_star_is_excluded_from_deployed_goal_and_inputs() -> None:
    current = {
        "centroid_x_px": 10.0,
        "centroid_y_px": 11.0,
        "sigma_x_px": 2.0,
        "sigma_y_px": 2.5,
        "peak_intensity": 0.8,
    }
    with pytest.raises(ValueError):
        assert_no_q_star({"target_positions_mm": position_dict(np.zeros(4))})
    result = StructuredGoalAdapter().parse(
        {
            "target_metrics": current,
            "q_star": position_dict(np.zeros(4)),
        },
        current_metrics=current,
    )
    assert result.status == "contradictory"
    assert result.goal is None


def test_ambiguous_oracle_runs_are_not_called_infeasible() -> None:
    label, agreement = classify_estimated_reachability([1.3, 2.0])
    assert label == "ambiguous_boundary"
    assert not agreement
    label, agreement = classify_estimated_reachability([1.19, 1.21, 1.20])
    assert label == "ambiguous_boundary"
    assert not agreement
    label, agreement = classify_estimated_reachability([1.31, 1.32, 1.30])
    assert label == "candidate_infeasible"
    assert agreement


def test_group_ids_do_not_overlap_between_splits() -> None:
    splits = {
        "train": {"g0", "g1"},
        "development": {"g2"},
        "test": {"g3"},
    }
    for index, left in enumerate(splits):
        for right in list(splits)[index + 1 :]:
            assert not (splits[left] & splits[right])


def test_default_action_units_are_millimetres() -> None:
    values = legacy_action_array()
    assert np.isclose(values[:, 0].max(), 0.05)
    assert np.isclose(values[:, 2].max(), 0.02)
    assert action_vector(
        {
            "lens_x_delta_mm": 0.05,
            "lens_y_delta_mm": 0.0,
            "camera_x_delta_mm": 0.0,
            "camera_y_delta_mm": 0.0,
        }
    )[0] == 0.05


def test_new_packages_do_not_reference_v10_locked_data() -> None:
    for package in (
        REPO_ROOT / "continuous_control_v12",
        REPO_ROOT / "physics_structured_rebuild_v11",
    ):
        for path in package.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            assert "locked_test.jsonl" not in text
            assert "locked-test.jsonl" not in text
            assert "physics_structured_rebuild_v10.generate_dataset" not in text
