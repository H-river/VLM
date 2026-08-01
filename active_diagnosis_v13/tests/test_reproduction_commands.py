from active_diagnosis_v13.audit_reproduction_commands import (
    extract_commands,
    parse_module_command,
)
from active_diagnosis_v13.audit_development_integrity import _protected_case_tokens
from active_diagnosis_v13.analyze_guarded_margin_seed_frontier import _is_dominated
from active_diagnosis_v13.analyze_cem_risk_sensitivity import _plan_delta
from active_diagnosis_v13.analyze_policy_delta_failures import _category
from active_diagnosis_v13.compare_validation_subset import _complete_case_ids
from continuous_control_v12.cem_diversity import (
    mean_pairwise_normalized_distance,
    minimum_pairwise_normalized_distance,
    select_diverse_elite_indices,
)
from continuous_control_v12.cem_feasible_proposal import (
    sample_truncated_normal_by_resampling,
)
from active_diagnosis_v13.run_cem_seed_robustness import _population_name
from active_diagnosis_v13.summarize_guarded_margin_seeds import _use_continuous
from active_diagnosis_v13.train_gain_baselines import _model
import numpy as np


def test_extract_commands_only_reads_shell_fences() -> None:
    markdown = """Text
```bash
/env/python -m package.module --input data.json
```
```json
{"ignored": true}
```
"""
    assert extract_commands(markdown) == [
        "/env/python -m package.module --input data.json"
    ]


def test_parse_module_command_handles_environment_and_subcommand() -> None:
    parsed = parse_module_command(
        "FLAG=1 /env/python -m package.module control --config config.json --seed=3"
    )
    assert parsed["status"] == "parsed"
    assert parsed["module"] == "package.module"
    assert parsed["subcommand"] == "control"
    assert parsed["options"] == ["--config", "--seed"]


def test_validation_subset_requires_complete_gain_groups() -> None:
    rows = [
        {"case_id": "case_0000", "evaluator_only_true_gain": gain}
        for gain in (0.5, 0.75, 1.0, 1.25, 1.5)
    ]
    assert _complete_case_ids(rows) == {"case_0000"}


def test_policy_delta_category_is_directional() -> None:
    assert _category({"strict_success": False}, {"strict_success": True}) == "alternate_recovery"
    assert _category({"strict_success": True}, {"strict_success": False}) == "alternate_regression"


def test_reduced_baseline_pipeline_accepts_full_feature_vectors() -> None:
    features = np.asarray([[0.0, 10.0, 1.0], [1.0, 20.0, 0.0], [0.1, 30.0, 0.9], [0.9, 40.0, 0.1]])
    labels = np.asarray(["a", "b", "a", "b"])
    model = _model("linear", 7, [0, 2], 3).fit(features, labels)
    assert model.predict(features).shape == (4,)


def test_guarded_margin_switch_uses_visible_predictions_only() -> None:
    discrete = {"gain_belief": 1.25, "evaluator_only_true_gain": 0.5}
    assert _use_continuous(
        discrete,
        {"gain_belief": 1.125, "evaluator_only_true_gain": 1.5},
        discrete_threshold=1.0,
        downward_margin=0.125,
    )
    assert not _use_continuous(
        discrete,
        {"gain_belief": 1.1249, "evaluator_only_true_gain": 1.25},
        discrete_threshold=1.0,
        downward_margin=0.125,
    )
    assert not _use_continuous(
        {"gain_belief": 0.75},
        {"gain_belief": 1.5},
        discrete_threshold=1.0,
        downward_margin=0.125,
    )


def test_cem_seed_policy_names_preserve_the_completed_base_arm() -> None:
    assert (
        _population_name(101, 101, "direct_population_48", "population48_seed")
        == "direct_population_48"
    )
    assert (
        _population_name(102, 101, "direct_population_48", "population48_seed")
        == "population48_seed_102"
    )


def test_cross_seed_frontier_rejects_equal_success_with_more_saturation() -> None:
    efficient = {
        "control_value_over_direct": {"mean": 0.1},
        "fault_saturation_episode_rate": {"mean": 0.2},
    }
    inefficient = {
        "control_value_over_direct": {"mean": 0.1},
        "fault_saturation_episode_rate": {"mean": 0.3},
    }
    assert _is_dominated(inefficient, [efficient, inefficient])
    assert not _is_dominated(efficient, [efficient, inefficient])


def test_cem_risk_plan_delta_detects_trace_length_changes() -> None:
    step = {
        "command_mm": {
            "lens_x_delta_mm": 0.0,
            "lens_y_delta_mm": 0.0,
            "camera_x_delta_mm": 0.0,
            "camera_y_delta_mm": 0.0,
        }
    }
    assert not _plan_delta({"trace": [step]}, {"trace": [step]})["changed"]
    assert _plan_delta({"trace": [step]}, {"trace": []})["changed"]


def test_cem_elite_diversity_keeps_best_then_separates_candidates() -> None:
    scores = np.asarray([0.0, 0.1, 0.2, 0.3])
    sequences = np.asarray([[[0.0]], [[0.01]], [[0.5]], [[-0.5]]])
    plain = select_diverse_elite_indices(
        scores,
        sequences,
        elites=3,
        action_scale=np.asarray([1.0]),
        minimum_normalized_distance=0.0,
    )
    diverse = select_diverse_elite_indices(
        scores,
        sequences,
        elites=3,
        action_scale=np.asarray([1.0]),
        minimum_normalized_distance=0.25,
    )
    assert plain.tolist() == [0, 1, 2]
    assert diverse.tolist() == [0, 2, 3]
    assert mean_pairwise_normalized_distance(
        sequences, diverse, np.asarray([1.0])
    ) > mean_pairwise_normalized_distance(sequences, plain, np.asarray([1.0]))
    assert minimum_pairwise_normalized_distance(
        sequences, diverse, np.asarray([1.0])
    ) >= 0.25


def test_development_integrity_detects_protected_case_tokens_recursively() -> None:
    assert not _protected_case_tokens({"case_id": "v12_mpcdiag_primary_00_0009"})
    assert _protected_case_tokens(
        {"nested": ["v12_mpcdiag_primary_02_0010__g1.5"]}
    ) == {"0010"}


def test_truncated_cem_proposals_respect_feasible_bounds() -> None:
    values, diagnostic = sample_truncated_normal_by_resampling(
        np.random.default_rng(7),
        np.zeros((1, 2)),
        np.ones((1, 2)),
        population=100,
        lower=np.asarray([[-0.1, -0.2]]),
        upper=np.asarray([[0.1, 0.2]]),
        resample_attempts=8,
    )
    assert np.all(values >= np.asarray([[[-0.1, -0.2]]]))
    assert np.all(values <= np.asarray([[[0.1, 0.2]]]))
    assert diagnostic["initially_out_of_feasible_bounds_fraction"] > 0
    assert diagnostic["remaining_out_of_feasible_bounds_before_clip_fraction"] < diagnostic[
        "initially_out_of_feasible_bounds_fraction"
    ]
