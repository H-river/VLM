from __future__ import annotations

import json

import pytest

from active_diagnosis_v13.run_gate_a import (
    _assert_frozen_full_cases_outside_oof_map,
    _assert_frozen_probe_identity,
    _load_frozen_decision,
    _select_cases,
    _validate_frozen_full_probe_override,
    matched_planner_seed,
    select_gate_branch,
)


def test_frozen_full_probe_override_is_only_for_unseen_development_cases(
    tmp_path,
) -> None:
    model = tmp_path / "probe.joblib"
    model.write_bytes(b"model identity")
    _validate_frozen_full_probe_override(
        enabled=True,
        mode="probe_replan",
        split="development",
        probe_model=model,
        gain_predictions=None,
    )
    with pytest.raises(ValueError, match="requires development probe_replan"):
        _validate_frozen_full_probe_override(
            enabled=True,
            mode="direct",
            split="development",
            probe_model=model,
            gain_predictions=None,
        )
    _assert_frozen_full_cases_outside_oof_map(
        [{"case_id": "fresh_case"}], {"case_to_fold": {"selection_case": 0}}
    )
    with pytest.raises(ValueError, match="outside the OOF selection map"):
        _assert_frozen_full_cases_outside_oof_map(
            [{"case_id": "selection_case"}],
            {"case_to_fold": {"selection_case": 0}},
        )


def test_matched_planner_seed_depends_only_on_case_and_root_seed() -> None:
    config = {"root_seed": 123}
    first = matched_planner_seed(config, "case_a")
    assert first == matched_planner_seed(config, "case_a")
    assert first != matched_planner_seed(config, "case_b")
    assert first != matched_planner_seed({"root_seed": 124}, "case_a")


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ((True, True, True, True), "P"),
        ((True, True, False, True), "A"),
        ((True, True, True, False), "B"),
        ((True, False, False, False), "B"),
        ((False, True, True, True), "C"),
    ],
)
def test_gate_branch_decision_tree(checks, expected) -> None:
    branch, reason = select_gate_branch(*checks)
    assert branch == expected
    assert reason


def test_protected_freeze_must_be_development_only(tmp_path) -> None:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "selected_primary_branch": "A",
                "protected_set_used_for_selection": True,
            }
        )
    )
    with pytest.raises(ValueError, match="development-only"):
        _load_frozen_decision(freeze_path)


def test_protected_probe_must_match_frozen_identity_and_hash(tmp_path) -> None:
    model = tmp_path / "probe.joblib"
    model.write_bytes(b"frozen classifier")
    import hashlib

    freeze = {
        "selected_probe": {
            "classifier_bundle": str(model),
            "classifier_bundle_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "selected_design": "symmetric_pair",
            "selected_fraction": 0.1,
        }
    }
    _assert_frozen_probe_identity(freeze, model, "symmetric_pair", 0.1)
    with pytest.raises(ValueError, match="design"):
        _assert_frozen_probe_identity(freeze, model, "single_positive", 0.1)
    model.write_bytes(b"mutated classifier")
    with pytest.raises(ValueError, match="hash"):
        _assert_frozen_probe_identity(freeze, model, "symmetric_pair", 0.1)


def test_protected_case_selection_enforces_registered_suffix_range(tmp_path) -> None:
    strata = ("interior", "boundary", "multi_step")
    cases = [
        {
            "case_id": f"{stratum}_{suffix:04d}",
            "group_id": f"{stratum}_{suffix:04d}",
            "stratum": stratum,
        }
        for stratum in strata
        for suffix in (10, 11, 12, 13, 14, 16)
    ]
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"cases": cases}))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "frozen": True,
                "selected_primary_branch": "A",
                "protected_set_used_for_selection": False,
            }
        )
    )
    config = {"baseline": {"evaluation_suite": str(suite)}}
    with pytest.raises(ValueError, match="unexpected protected case count: 15"):
        _select_cases(config, "protected", None, freeze)

    for case in cases:
        if case["case_id"].endswith("0016"):
            case["case_id"] = case["case_id"][:-4] + "0015"
            case["group_id"] = case["group_id"][:-4] + "0015"
    suite.write_text(json.dumps({"cases": cases}))
    selected = _select_cases(config, "protected", None, freeze)
    assert len(selected) == 18
    assert {int(case["case_id"].rsplit("_", 1)[1]) for case in selected} == set(
        range(10, 16)
    )
