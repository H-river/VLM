from __future__ import annotations

from active_diagnosis_v13.build_decision_dataset import prompt_for_record, qwen_row


def _row() -> dict:
    return {
        "record_id": "case_a__g0.75__symmetric_pair__f0.1",
        "case_id": "case_a",
        "group_id": "case_a",
        "design": "symmetric_pair",
        "fraction": 0.1,
        "evaluator_only_true_gain": 0.75,
        "policy_record": {
            "feature_names": ["observed_delta.x", "residual.x"],
            "feature_vector": [0.12, -0.03],
        },
    }


def test_prompt_excludes_true_gain_and_randomizes_candidates() -> None:
    prompt = prompt_for_record(_row(), 7)
    assert "evaluator_only_true_gain" not in prompt
    assert "candidate_gains_randomized" in prompt
    assert "visible_probe_features" in prompt


def test_candidate_order_is_independent_of_evaluator_gain_suffix() -> None:
    first = _row()
    second = _row()
    second["record_id"] = "case_a__g1.5__symmetric_pair__f0.1"
    second["evaluator_only_true_gain"] = 1.5
    assert prompt_for_record(first, 7) == prompt_for_record(second, 7)


def test_prompt_can_apply_frozen_feature_filter() -> None:
    prompt = prompt_for_record(_row(), 7, [0])
    assert "observed_delta.x" in prompt
    assert "residual.x" not in prompt


def test_gain_label_appears_only_in_completion() -> None:
    row = qwen_row(_row(), 7)
    assert "0.75" not in row["prompt"][0]["content"][0]["text"].split(
        '"candidate_gains_randomized"'
    )[0]
    assert "0.75" in row["completion"][0]["content"][0]["text"]
