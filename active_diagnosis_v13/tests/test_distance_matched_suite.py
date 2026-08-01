from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from active_diagnosis_v13.select_distance_matched_fresh_suite import STRATA, main


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_distance_matched_suite_uses_covariates_and_preserves_suite_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_rows = []
    candidate_cases = []
    gains = (0.5, 0.75, 1.0, 1.25, 1.5)
    for stratum_index, stratum in enumerate(STRATA):
        for group_index in range(10):
            distance = float(2 + stratum_index * 20 + group_index)
            group = f"reference_{stratum_index:02d}_{group_index:04d}"
            for gain in gains:
                reference_rows.append(
                    {
                        "group_id": group,
                        "stratum": stratum,
                        "initial_normalized_distance": distance,
                        "evaluator_only_true_gain": gain,
                        "strict_success": gain == 1.0,
                    }
                )
        for pool_index in range(20):
            source_distance = float(
                2 + stratum_index * 20 + pool_index
                if pool_index < 10
                else 102 + stratum_index * 20 + pool_index
            )
            source_id = f"candidate_{stratum_index:02d}_{pool_index:04d}"
            candidate_cases.append(
                {
                    "case_id": source_id,
                    "group_id": source_id,
                    "setup_hash": f"hash_{stratum_index}_{pool_index}",
                    "stratum": stratum,
                    "initial_normalized_distance": source_distance,
                    "initial_distance_band": "low" if pool_index < 10 else "high",
                    "exact_q_goal_replay_max_abs_metric_error": 0.0,
                }
            )
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "matched.json"
    _write_jsonl(reference, reference_rows)
    candidate.write_text(
        json.dumps(
            {
                "version": "v12_mpc_h1_h3_evaluation_suite_v1",
                "suite_label": "candidate_pool",
                "cases": candidate_cases,
                "validation": {
                    "groups": 60,
                    "prior_group_id_overlap": 0,
                    "prior_setup_hash_overlap": 0,
                    "q_goal_replay_max_abs_metric_error": 0.0,
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_distance_matched_fresh_suite",
            "--candidate-suite",
            str(candidate),
            "--reference-development-control",
            str(reference),
            "--suite-label",
            "matched_test",
            "--output",
            str(output),
        ],
    )
    main()
    artifact = json.loads(output.read_text())
    assert artifact["version"] == "v12_mpc_h1_h3_evaluation_suite_v1"
    assert artifact["selection_version"].endswith("_v1")
    assert artifact["validation"]["groups"] == 30
    assert artifact["validation"]["stratum_counts"] == {
        stratum: 10 for stratum in STRATA
    }
    matching = artifact["outcome_free_difficulty_matching"]
    assert matching["candidate_pool_groups"] == 60
    assert matching["mean_absolute_log1p_distance_mismatch"] == 0.0
    assert matching["maximum_absolute_log1p_distance_mismatch"] == 0.0
    assert matching["protected_set_used"] is False
    assert matching["selection_or_retuning_on_candidate_outcomes"] is False
    assert {
        int(row["case_id"].rsplit("_", 1)[1]) for row in artifact["cases"]
    } == set(range(10))
