from __future__ import annotations

import json
import sys

import pytest

from active_diagnosis_v13.analyze_boundary_candidate_seed_robustness import (
    _counts_by_gain,
    main as analyze_seed_robustness,
)
from active_diagnosis_v13.analyze_candidate_coverage import summarize_decisions
from active_diagnosis_v13.audit_boundary_candidate_coverage import (
    main as audit_candidate_coverage,
)


def _candidate(rank: int, actual: float) -> dict:
    return {
        "selection_rank": rank,
        "actual_terminal_cost": actual,
        "effective_sequence": [
            {
                "lens_x_delta_mm": 0.01 * rank,
                "lens_y_delta_mm": 0.0,
                "camera_x_delta_mm": 0.0,
                "camera_y_delta_mm": 0.0,
            }
        ],
        "counterfactual_depths": [
            {
                "tolerance_normalized_prediction_residual": {
                    "centroid_x_px": 0.1,
                    "centroid_y_px": 0.2,
                    "sigma_x_px": 0.3,
                    "sigma_y_px": 0.4,
                    "peak_intensity": 0.5,
                }
            }
        ],
    }


def test_best_of_k_recovers_hard_negative() -> None:
    result = summarize_decisions(
        [
            {
                "candidates": [
                    _candidate(1, 1.4),
                    _candidate(2, 0.8),
                    _candidate(3, 1.1),
                ]
            }
        ]
    )
    assert result["h1_cem_top1_strict_success"] == 0.0
    assert result["simulator_oracle_best_of_k_strict_success"] == 1.0
    assert result["matched_hard_negative_recoveries"] == 1
    assert result["top_k_coverage_curve"][1]["strict_success_coverage"] == 1.0


def test_candidate_seed_slice_counts_gain_suffixes_and_rejects_unknowns() -> None:
    assert _counts_by_gain(
        {
            "v12_mpcdiag_primary_02_0001__g0.5",
            "v12_mpcdiag_primary_02_0002__g0.5",
            "v12_mpcdiag_primary_02_0003__g1.25",
        }
    ) == {"0.5": 2, "0.75": 0, "1.25": 1, "1.5": 0}
    with pytest.raises(ValueError, match="unexpected candidate success gain"):
        _counts_by_gain({"v12_mpcdiag_primary_02_0001__g1"})


def test_three_seed_candidate_aggregation_tracks_stable_and_any_unions(
    tmp_path, monkeypatch, capsys
) -> None:
    ids = {
        "a": "v12_mpcdiag_primary_02_0001__g0.5",
        "b": "v12_mpcdiag_primary_02_0002__g0.75",
        "c": "v12_mpcdiag_primary_02_0003__g1.25",
        "d": "v12_mpcdiag_primary_02_0004__g1.5",
        "e": "v12_mpcdiag_primary_02_0005__g0.5",
    }
    arms = {
        "seed1": ([ids["a"], ids["b"]], [ids["c"]]),
        "seed2": ([ids["b"], ids["c"]], [ids["d"]]),
        "seed3": ([ids["b"], ids["e"]], [ids["c"]]),
    }
    arguments = ["analyze", "--output", str(tmp_path / "summary.json")]
    for index, (label, (uniform, conditioned)) in enumerate(arms.items(), 1):
        root_seed_serialized = label != "seed1"
        audit = {
            "protected_set_used": False,
            "episodes": 20,
            "budgets": [96],
            "record_schema_versions": ["test_v1"],
            "planner_root_seed_field_complete": root_seed_serialized,
            "planner_root_seed_values": [index] if root_seed_serialized else [],
            "nested_curve": [
                {
                    "candidate_budget": 96,
                    "proposals": {
                        "uniform_feasible": {
                            "strict_successes": len(uniform),
                            "strict_success_ids": uniform,
                        },
                        "boundary_conditioned": {
                            "strict_successes": len(conditioned),
                            "strict_success_ids": conditioned,
                        },
                    },
                }
            ],
        }
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(audit))
        arguments.extend(["--audit", f"{label}={path}"])

    monkeypatch.setattr(sys, "argv", arguments)
    analyze_seed_robustness()
    capsys.readouterr()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["proposal_union_successes_by_seed"] == {
        "seed1": 3,
        "seed2": 3,
        "seed3": 3,
    }
    assert summary["audit_provenance"]["seed1"] == {
        "planner_root_seed_field_complete": False,
        "planner_root_seed_values": [],
        "record_schema_versions": ["test_v1"],
    }
    assert summary["audit_provenance"]["seed2"][
        "planner_root_seed_field_complete"
    ] is True
    assert summary["stable_union_success_ids_all_seeds"] == [ids["b"], ids["c"]]
    assert summary["union_success_ids_any_seed"] == sorted(ids.values())
    union = summary["seed_robustness_curve"][0]["proposal_union"]
    assert union["stable_to_any_success_ratio"] == pytest.approx(0.4)
    assert union["pairwise_success_jaccard"] == {
        "seed1__seed2": pytest.approx(0.5),
        "seed1__seed3": pytest.approx(0.5),
        "seed2__seed3": pytest.approx(0.5),
    }
    assert summary["maximum_budget_seed_recovery_frequency"] == {
        "recovery_seed_count_by_episode": {
            ids["a"]: 1,
            ids["b"]: 3,
            ids["c"]: 3,
            ids["d"]: 1,
            ids["e"]: 1,
        },
        "episode_count_by_recovery_seed_count": {"1": 3, "2": 0, "3": 2},
        "stable_recovery_ids_all_seeds": [ids["b"], ids["c"]],
        "seed_variable_recovery_ids": [ids["a"], ids["d"], ids["e"]],
    }
    assert summary["maximum_budget_recovery_slices"] == {
        "successes_by_gain_per_seed": {
            "seed1": {"0.5": 1, "0.75": 1, "1.25": 1, "1.5": 0},
            "seed2": {"0.5": 0, "0.75": 1, "1.25": 1, "1.5": 1},
            "seed3": {"0.5": 1, "0.75": 1, "1.25": 1, "1.5": 0},
        },
        "stable_successes_by_gain_all_seeds": {
            "0.5": 0,
            "0.75": 1,
            "1.25": 1,
            "1.5": 0,
        },
        "successes_by_gain_any_seed": {
            "0.5": 2,
            "0.75": 1,
            "1.25": 1,
            "1.5": 1,
        },
        "stable_success_group_ids_all_seeds": [
            "v12_mpcdiag_primary_02_0002",
            "v12_mpcdiag_primary_02_0003",
        ],
        "success_group_ids_any_seed": [
            "v12_mpcdiag_primary_02_0001",
            "v12_mpcdiag_primary_02_0002",
            "v12_mpcdiag_primary_02_0003",
            "v12_mpcdiag_primary_02_0004",
            "v12_mpcdiag_primary_02_0005",
        ],
    }


def test_candidate_audit_rejects_success_flag_distance_mismatch(
    tmp_path, monkeypatch
) -> None:
    rows = []
    for index in range(20):
        case_id = f"boundary_case_{index:02d}"
        candidate = {
            "terminal_normalized_distance": 2.0,
            "strict_success": index == 0,
            "desired_physical_sequence": [{}, {}],
            "issued_command_sequence": [{}, {}],
        }
        rows.append(
            {
                "version": "test_candidate_v2",
                "record_id": f"{case_id}__g0.5__boundary_candidate_coverage",
                "case_id": case_id,
                "group_id": case_id,
                "stratum": "reachable_boundary_or_clipping",
                "evaluator_only_true_gain": 0.5,
                "source_budget4_strict_success": False,
                "sequence_horizon": 2,
                "maximum_budget": 1,
                "proposal_raw_uniforms_matched": True,
                "proposals": {
                    proposal: {
                        "candidates": [candidate],
                        "conditioning_active_candidates": 0,
                        "mean_first_action_pairwise_normalized_distance": 0.0,
                    }
                    for proposal in ("uniform_feasible", "boundary_conditioned")
                },
            }
        )
    records = tmp_path / "records.jsonl"
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--records",
            str(records),
            "--budgets",
            "1",
            "--output",
            str(tmp_path / "audit.json"),
        ],
    )
    with pytest.raises(ValueError, match="invalid candidate semantics"):
        audit_candidate_coverage()
