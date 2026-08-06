from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_h1_meta_v0_candidate import training
from qwen_h1_meta_v0_candidate.contracts import parse_meta_input
from qwen_h1_meta_v0_candidate.inference import IndependentMetaAdapter
from qwen_h1_meta_v0_candidate.reasoning_ablations import (
    ABLATIONS,
    ABLATION_SEED,
    EXECUTION_SCOPE,
    LiveReasoningAblationPolicy,
    ReasoningAblationError,
    apply_ablation,
    build_not_run_due_to_data_gate_report,
    ensure_blank_image,
    evaluate_reasoning_ablations,
    generate_reasoning_ablation_predictions,
    normalize_candidate_eval_manifest,
    select_closed_loop_trace,
    setup_aware_donor_map,
)


SEEDS = (2026080201, 2026080202, 2026080203)


def _canonical(value: dict) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _manifest(
    root: Path,
    valid_input: dict,
    valid_output: dict,
    *,
    setup_count: int = 2,
) -> list[dict]:
    from PIL import Image

    records = []
    image_root = root / "candidate_eval_images"
    image_root.mkdir(parents=True, exist_ok=True)
    for index in range(setup_count):
        image = image_root / f"setup_{index}.png"
        Image.new("L", (1024, 1024), 30 + index).save(image)
        runtime = copy.deepcopy(valid_input)
        runtime["current_beam_state"]["centroid_x_px"] += float(index * 5)
        runtime["target_beam_state"]["centroid_y_px"] += float(index * 7)
        runtime["history"][0]["measured_beam_delta"]["centroid_x_px"] = float(index)
        runtime["actuator_positions_mm"]["lens_x_mm"] = float(index) * 0.1
        records.append(
            {
                "schema_version": "qwen_h1_meta_v0_candidate_record_v1",
                "record_id": f"qh1meta_eval_{index:04d}__target-00-near_target",
                "split": "candidate_eval",
                "image": {
                    "storage_path": str(image),
                    "sha256": training.sha256_path(image),
                    "width_px": 1024,
                    "height_px": 1024,
                    "mode": "L",
                },
                "model_visible_input": runtime,
                "oracle_output": copy.deepcopy(valid_output),
                "identity": {"setup_hash": format(index + 10, "064x")},
                "oracle_audit": {"selected_configuration_id": "synthetic"},
                "evaluator_only": {
                    "setup_id": f"qh1meta_eval_{index:04d}",
                    "episode_id": f"qh1meta_eval_{index:04d}__target-00-near_target",
                    "target_counterfactual_id": "near_target",
                    "target_index": 0,
                },
            }
        )
    return records


def test_all_ablation_transforms_are_schema_valid_setup_aware_and_nonmutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_input,
    valid_output,
) -> None:
    monkeypatch.setattr(training, "PACKAGE_ROOT", tmp_path)
    records = _manifest(tmp_path, valid_input, valid_output)
    before = copy.deepcopy(records)
    samples = normalize_candidate_eval_manifest(
        records,
        verify_images=True,
        require_preregistered_cardinality=False,
    )
    donors = setup_aware_donor_map(samples)
    blank_path, blank_hash = ensure_blank_image(tmp_path / "assets" / "blank.png")
    assert ABLATION_SEED == 2026080213
    assert all(
        donors[sample.sample_id].setup_hash != sample.setup_hash for sample in samples
    )

    transformed = {}
    for ablation in ABLATIONS:
        donor = donors[samples[0].sample_id] if "shuffle" in ablation and ablation != "actuator_semantics_shuffle" else None
        transformed[ablation] = apply_ablation(
            samples[0],
            ablation=ablation,
            donor=donor,
            blank_image_path=blank_path,
            blank_image_sha256=blank_hash,
        )
        parse_meta_input(transformed[ablation].runtime_input)

    assert records == before
    assert transformed["blank_image"].image_sha256 == blank_hash
    assert transformed["deterministic_image_shuffle"].donor_setup_hash != samples[0].setup_hash
    assert all(
        value == 0.0
        for group in (
            transformed["metrics_blank"].runtime_input["current_beam_state"],
            transformed["metrics_blank"].runtime_input["target_beam_state"],
            transformed["metrics_blank"].runtime_input["normalized_signed_error"],
        )
        for value in group.values()
    )
    target_current = transformed["target_replace_current"].runtime_input
    assert target_current["target_beam_state"] == target_current["current_beam_state"]
    assert set(target_current["normalized_signed_error"].values()) == {0.0}
    assert all(not row["valid"] for row in transformed["history_blank"].runtime_input["history"])
    assert transformed["actuator_semantics_shuffle"].physical_validation[
        "actuator_semantics"
    ] == "unverified"
    assert transformed["reason_codes_disabled"].runtime_input == transformed[
        "full_input"
    ].runtime_input


def test_generation_uses_three_mock_adapters_and_reuses_full_for_reason_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_input,
    valid_output,
) -> None:
    monkeypatch.setattr(training, "PACKAGE_ROOT", tmp_path)
    records = _manifest(tmp_path, valid_input, valid_output)
    calls: dict[int, list[tuple]] = {seed: [] for seed in SEEDS}

    def adapter(seed: int) -> IndependentMetaAdapter:
        def backend(prompt, image_path, generation_seed, max_new_tokens):
            calls[seed].append((prompt, image_path, generation_seed, max_new_tokens))
            return _canonical(valid_output)

        return IndependentMetaAdapter(backend)

    rows = generate_reasoning_ablation_predictions(
        manifest_records=records,
        adapters={seed: adapter(seed) for seed in SEEDS},
        blank_image_path=tmp_path / "assets" / "blank.png",
        verify_images=True,
        require_preregistered_cardinality=False,
    )
    assert len(rows) == len(records) * len(ABLATIONS) * len(SEEDS)
    # reason_codes_disabled is an audit/control-consumption intervention and
    # reuses exact full bytes; it does not trigger a second model generation.
    assert all(len(calls[seed]) == len(records) * (len(ABLATIONS) - 1) for seed in SEEDS)
    assert all(call[2:] == (seed, 256) for seed in SEEDS for call in calls[seed])
    assert all(row["execution_scope"] == EXECUTION_SCOPE for row in rows)
    assert all(row["hardware_dispatch_allowed"] is False for row in rows)
    for seed in SEEDS:
        for record in records:
            sample_id = record["record_id"]
            full = next(
                row
                for row in rows
                if row["seed"] == seed
                and row["sample_id"] == sample_id
                and row["ablation"] == "full_input"
            )
            disabled = next(
                row
                for row in rows
                if row["seed"] == seed
                and row["sample_id"] == sample_id
                and row["ablation"] == "reason_codes_disabled"
            )
            assert disabled["prediction"] == full["prediction"]
            assert disabled["generation_reused"] is True
            assert disabled["control_fields_unchanged_from_full_input"] is True


def test_offline_diagnostics_and_closed_loop_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_input,
    valid_output,
) -> None:
    monkeypatch.setattr(training, "PACKAGE_ROOT", tmp_path)
    records = _manifest(tmp_path, valid_input, valid_output)

    def backend(prompt, image_path, seed, max_new_tokens):
        return _canonical(valid_output)

    rows = generate_reasoning_ablation_predictions(
        manifest_records=records,
        adapters={seed: IndependentMetaAdapter(backend) for seed in SEEDS},
        blank_image_path=tmp_path / "assets" / "blank.png",
        require_preregistered_cardinality=False,
    )
    for row in rows:
        row["configuration_regret"] = (
            0.0 if row["ablation"] == "full_input" else 0.25
        )
    report = evaluate_reasoning_ablations(
        manifest_records=records,
        prediction_records=rows,
        require_preregistered_cardinality=False,
    )
    assert report["candidate_only"] is True
    assert report["full_input_offline_evaluation"]["version"] == (
        "qwen_h1_meta_v0_candidate_eval_diagnostics_v1"
    )
    assert report["per_ablation"]["target_shuffle"]["paired_vs_full"][
        "configuration_regret_status"
    ] == "evaluated_in_supplied_simulator_or_shadow_records"
    assert report["per_ablation"]["actuator_semantics_shuffle"][
        "physical_validation"
    ]["actuator_semantics"] == "unverified"
    trace = select_closed_loop_trace(
        rows, ablation="target_shuffle", seed=2026080201
    )
    assert len(trace) == len(records)
    assert all(row["step"] == 0 for row in trace)
    assert all(row["ablation"] == "target_shuffle" for row in trace)


def test_live_policy_binds_updated_state_and_simulator_image_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_input,
    valid_output,
) -> None:
    monkeypatch.setattr(training, "PACKAGE_ROOT", tmp_path)
    records = _manifest(tmp_path, valid_input, valid_output)
    calls = []

    def backend(prompt, image_path, seed, max_new_tokens):
        calls.append((prompt, image_path, seed, max_new_tokens))
        return _canonical(valid_output)

    traces: list[dict] = []
    policy = LiveReasoningAblationPolicy(
        adapter=IndependentMetaAdapter(backend),
        seed=2026080201,
        ablation="target_replace_current",
        manifest_records=records,
        trace_sink=traces,
        require_preregistered_cardinality=False,
    )
    updated = copy.deepcopy(records[0]["model_visible_input"])
    updated["current_beam_state"]["centroid_x_px"] = 520.0
    context = SimpleNamespace(
        episode_id=records[0]["record_id"],
        step=2,
        runtime_input=updated,
        sensor_image_normalized=np.full((1024, 1024), 0.25, dtype=np.float64),
    )
    result = policy.decide(context)
    assert result.available is True
    assert len(calls) == 1
    shown_input = training.validate_prompt(calls[0][0])
    assert shown_input["current_beam_state"]["centroid_x_px"] == 520.0
    assert shown_input["target_beam_state"] == shown_input["current_beam_state"]
    assert traces[0]["step"] == 2
    assert traces[0]["hardware_dispatch_allowed"] is False
    assert traces[0]["prediction"] == _canonical(valid_output)

    unavailable = LiveReasoningAblationPolicy(
        adapter=IndependentMetaAdapter(backend),
        seed=2026080201,
        ablation="deterministic_image_shuffle",
        manifest_records=records,
        require_preregistered_cardinality=False,
    ).decide(context)
    assert unavailable.available is False
    assert unavailable.unavailable_reason == (
        "live_image_shuffle_requires_donor_image_provider"
    )
    assert len(calls) == 1


def test_not_run_report_is_explicit_and_contains_no_placeholder_results() -> None:
    report = build_not_run_due_to_data_gate_report(
        gate_evidence={
            "near_visible_conflict_rate": 1.0,
            "visible_grouped_classifier_macro_f1": 0.0192,
        }
    )
    assert report["status"] == "unavailable"
    assert report["reason"] == "not_run_due_to_information_sufficiency_gate"
    assert report["reason_alias"] == "not_run_due_to_data_gate"
    assert report["model_loaded"] is False
    assert report["inference_started"] is False
    assert report["predictions_generated"] is False
    assert report["closed_loop_started"] is False
    assert all(
        value["status"] == "unavailable"
        and value["configuration_regret"] is None
        for value in report["per_ablation"].values()
    )
    assert report["per_ablation"]["actuator_semantics_shuffle"][
        "physical_validation"
    ] == "unverified"


def test_manifest_rejects_forbidden_candidate_image_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_input,
    valid_output,
) -> None:
    monkeypatch.setattr(training, "PACKAGE_ROOT", tmp_path)
    records = _manifest(tmp_path, valid_input, valid_output)
    records[0]["image"]["storage_path"] = str(tmp_path / "protected" / "image.png")
    with pytest.raises((ReasoningAblationError, ValueError), match="forbidden"):
        normalize_candidate_eval_manifest(
            records,
            verify_images=False,
            require_preregistered_cardinality=False,
        )
