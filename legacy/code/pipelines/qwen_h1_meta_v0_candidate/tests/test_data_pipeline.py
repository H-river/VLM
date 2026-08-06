from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
from PIL import Image

from continuous_control_v12.contracts import Bounds
from qwen_h1_meta_v0_candidate import data_pipeline as pipeline
from qwen_h1_meta_v0_candidate.training import validate_prebuilt_row


def _metric(value: float = 0.0) -> dict[str, float]:
    return {field: float(value) for field in pipeline.METRIC_FIELDS}


def _valid_state() -> dict:
    history = [
        {
            "valid": True,
            "executed_action_mm": {
                "lens_x_delta_mm": 0.0,
                "lens_y_delta_mm": 0.0,
                "camera_x_delta_mm": 0.0,
                "camera_y_delta_mm": 0.0,
            },
            "measured_beam_delta": _metric(),
            "predicted_beam_delta": _metric(),
            "prediction_residual": _metric(),
            "ensemble_uncertainty": _metric(),
            "padding_reason": "none",
        }
        for _ in range(3)
    ]
    return {
        "schema_version": "qwen_h1_meta_input_v0",
        "current_beam_image": {
            "role": "current_sensor_frame_beam_image",
            "coordinate_frame": "camera_sensor_array",
            "display_normalization": "per_image_peak_normalized_for_qwen_only",
            "width_px": 1024,
            "height_px": 1024,
        },
        "current_beam_state": _metric(),
        "target_beam_state": _metric(1.0),
        "normalized_signed_error": _metric(1.0),
        "actuator_positions_mm": {
            "lens_x_mm": 0.0,
            "lens_y_mm": 0.0,
            "camera_x_mm": 0.0,
            "camera_y_mm": 0.0,
        },
        "actuator_semantics": pipeline._actuator_semantics([0.05, 0.05, 0.02, 0.02]),
        "history": history,
        "forward_uncertainty": {
            "per_metric": _metric(),
            "mean": 0.0,
            "maximum": 0.0,
        },
        "remaining_budget": {"measurement_steps": 2, "control_steps": 4},
        "measurement_validity": {
            "state": "valid",
            "supervisor_diagnosis": "nominal",
            "measurement_policy": "standard",
        },
    }


def _default_output() -> dict:
    return {
        "schema_version": "qwen_h1_meta_v0",
        "decision": "run_default_h1",
        "observation_request": "reuse_current",
        "objective_profile": "balanced",
        "mask_profile": "default",
        "directional_prior": {field: "unknown" for field in pipeline.ACTION_FIELDS},
        "step_scale": "default",
        "risk_mode": "standard",
        "confidence": "medium",
        "reason_codes": ["recent_response_consistent"],
    }


def _setup_context(index: int = 0) -> dict[str, float]:
    return {
        "wavelength_nm": 630.0 + index * 0.01,
        "beam_waist_mm": 1.0,
        "power_w": 1.0,
        "lens_focal_length_mm": 100.0,
        "lens_aperture_mm": 25.0,
        "source_to_lens_mm": 200.0,
        "lens_to_camera_mm": 150.0,
        "pixel_size_um": 5.5,
    }


def _manifest_row(*, image_path: str, image_hash: str) -> dict:
    return {
        "record_id": "synthetic-row",
        "split": "train",
        "image": {
            "storage_path": image_path,
            "sha256": image_hash,
            "audit_features": {"mean": 0.0, "standard_deviation": 0.0},
        },
        "model_visible_input": _valid_state(),
        "oracle_output": _default_output(),
        "identity": {"setup_hash": "b" * 64},
        "oracle_audit": {"selected_configuration_id": "default"},
        "evaluator_only": {
            "setup_id": "synthetic-setup",
            "setup_context": _setup_context(),
            "target_counterfactual_id": "one_step_mixed",
        },
    }


def test_verify_protocol_checks_single_file_addendum() -> None:
    verified = pipeline.verify_frozen_protocol()
    assert "data_generation_preregistration.json" in verified
    assert verified["data_generation_preregistration.json"] == (
        "beb65dd7f2d56e2426e63adf22f23a042d8d5f8e26c95eb16c37445c9c1a3561"
    )


def test_identity_registry_uses_corrected_v12_fields() -> None:
    registry = pipeline.KnownIdentityRegistry.load(
        pipeline.PACKAGE_ROOT / "configs/known_identity_blocklist.json"
    )
    assert len(registry.setup_hashes) == 298
    assert pipeline.setup_common_fingerprint(_setup_context()) == (
        630.0,
        1.0,
        1.0,
        100.0,
        25.0,
        200.0,
        150.0,
        5.5,
    )


def test_alternate_identity_registry_is_rejected_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = False

    def forbidden_load(cls, path):
        nonlocal loaded
        loaded = True
        raise AssertionError(f"alternate registry was opened: {path}")

    monkeypatch.setattr(
        pipeline.KnownIdentityRegistry, "load", classmethod(forbidden_load)
    )
    with pytest.raises(pipeline.DataGenerationError, match="must equal"):
        pipeline.CandidateDataGenerator(
            output_root=pipeline.PACKAGE_ROOT / "artifacts/never-created-registry-check",
            identity_registry_path=pipeline.PACKAGE_ROOT / "artifacts/arbitrary.json",
        )
    assert loaded is False


def test_frozen_configuration_enumeration_is_1_plus_35() -> None:
    priors = {
        "all_actuators": {field: "increase" for field in pipeline.ACTION_FIELDS},
        "lens_only": {
            field: "increase" if index < 2 else "hold"
            for index, field in enumerate(pipeline.ACTION_FIELDS)
        },
        "camera_only": {
            field: "increase" if index >= 2 else "hold"
            for index, field in enumerate(pipeline.ACTION_FIELDS)
        },
    }
    configurations = pipeline.enumerate_oracle_configurations(priors)
    assert len(configurations) == 36
    assert sum(item["template_family"] == "primary_local_gradient" for item in configurations) == 24
    assert sum(item["template_family"] == "lens_local_gradient" for item in configurations) == 4
    assert sum(item["template_family"] == "camera_local_gradient" for item in configurations) == 4
    assert sum(item["template_family"] == "unbiased_unknown" for item in configurations) == 3


def test_score_and_minimum_improvement_default_tie() -> None:
    score, components = pipeline.canonical_one_step_score(
        next_metrics=[1, 0, 0, 0, 0],
        target_metrics=[0, 0, 0, 0, 0],
        tolerance_reference=[1, 1, 1, 1, 1],
        action_mm=[0, 0, 0, 0],
        action_limit_mm=[0.05, 0.05, 0.02, 0.02],
    )
    assert score == 1.10 + 0.15 / 5
    assert components["max_normalized_error"] == 1.0
    default = {
        "configuration_id": "default",
        "configuration": {"objective_profile": "balanced", "mask_profile": "default", "step_scale": "default", "risk_mode": "standard"},
        "actual_score": 1.0,
    }
    guided = {
        "configuration_id": "guided",
        "configuration": {"objective_profile": "balanced", "mask_profile": "all_actuators", "step_scale": "fine", "risk_mode": "conservative"},
        "actual_score": 0.981,
    }
    selected, audit = pipeline.choose_actual_label(default, [guided])
    assert selected["configuration_id"] == "default"
    assert audit["guided_improvement"] < 0.02


def test_prebuilt_export_roundtrips_candidate_validator() -> None:
    artifact_root = pipeline.PACKAGE_ROOT / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="data-pipeline-test-", dir=artifact_root) as directory:
        image_path = Path(directory) / "current.png"
        Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8), mode="L").save(image_path)
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        relative = str(image_path.relative_to(pipeline.REPO_ROOT))
        manifest_row = _manifest_row(image_path=relative, image_hash=image_hash)
        system_prompt = (pipeline.PROTOCOL_ROOT / "system_prompt.txt").read_text(
            encoding="utf-8"
        )
        prompt_contract = json.loads(
            (pipeline.PROTOCOL_ROOT / "prompt_contract.json").read_text(encoding="utf-8")
        )
        row = pipeline.build_prebuilt_chat_row(
            manifest_row=manifest_row,
            system_prompt=system_prompt,
            prompt_contract=prompt_contract,
        )
        validated = validate_prebuilt_row(
            row,
            allowed_splits={"train"},
            image_root=pipeline.REPO_ROOT,
            verify_image=True,
        )
        assert validated == image_path.resolve()
        assert row["prompt"][0]["content"] == [{"type": "text", "text": system_prompt}]
        assert row["metadata"]["image_sha256"] == image_hash


def test_model_visible_id_leakage_is_rejected() -> None:
    state = _valid_state()
    state["record_id"] = "hidden-id"
    try:
        pipeline.validate_model_visible_state(state)
    except pipeline.DataGenerationError as exc:
        assert "record_id" in str(exc)
    else:
        raise AssertionError("record_id leakage was not rejected")


def test_information_gate_red_stops_conflicting_export() -> None:
    rows = []
    for setup_index in range(24):
        split = "train" if setup_index < 16 else "dev"
        for target_index in range(3):
            row = _manifest_row(image_path="unused.png", image_hash="a" * 64)
            row["record_id"] = f"row-{setup_index}-{target_index}"
            row["split"] = split
            row["identity"] = {"setup_hash": f"{setup_index:064x}"}
            row["evaluator_only"] = {
                "setup_id": f"setup-{setup_index}",
                "setup_context": _setup_context(setup_index),
                "target_counterfactual_id": pipeline.TARGET_VARIANTS[target_index][0],
            }
            if setup_index % 2:
                row["oracle_output"] = pipeline.CandidateDataGenerator._defensive_output(
                    "stop"
                )
                row["oracle_audit"] = {"selected_configuration_id": "stop"}
            rows.append(row)
    audit = pipeline.information_sufficiency_audit(
        rows,
        known_identity_count=298,
        identity_overlap_audit={"known_overlap_count": 0, "cross_split_overlap_count": 0},
    )
    assert audit["gate"] == "RED_STOP"
    assert "exact_visible_label_conflict" in audit["red_reasons"]
    try:
        pipeline.export_prebuilt_chat(
            rows=rows,
            output_root=pipeline.PACKAGE_ROOT / "artifacts/never-written",
            data_audit=audit,
        )
    except pipeline.InformationSufficiencyError:
        pass
    else:
        raise AssertionError("RED gate did not stop SFT export")


class _SyntheticPlanner:
    def plan(self, **kwargs):
        configuration = kwargs["configuration"]
        directions = configuration["directional_prior"]
        mapping = {"increase": 1.0, "decrease": -1.0, "hold": 0.0, "unknown": 0.0}
        scale = 0.01 if configuration["decision"] == "run_guided_h1" else 0.0
        return {
            "selected_action": {
                field: scale * mapping[directions[field]] for field in pipeline.ACTION_FIELDS
            }
        }


def test_mock_oracle_is_deterministic_and_uses_physical_rescore() -> None:
    generator = pipeline.CandidateDataGenerator.__new__(pipeline.CandidateDataGenerator)
    generator.bounds = Bounds(
        action_low=np.asarray([-0.05, -0.05, -0.02, -0.02]),
        action_high=np.asarray([0.05, 0.05, 0.02, 0.02]),
        position_low=np.full(4, -3.0),
        position_high=np.full(4, 3.0),
    )
    generator.planner_backend = _SyntheticPlanner()
    generator.planner_calls = 0

    def synthetic_simulate(self, setup_context, positions):
        vector = np.asarray(positions, dtype=np.float64)
        metrics = np.asarray(
            [10 * vector[0], 10 * vector[1], 2 + 5 * vector[2], 2 + 5 * vector[3], 1 + vector.sum()]
        )
        return {
            "metrics": pipeline._metric_dict(metrics),
            "auxiliary": {
                "clipping_fraction": 0.0,
                "camera_boundary_indicator": False,
                "simulator_valid": True,
            },
        }

    def synthetic_predict(self, *, setup_context, positions, metrics, action):
        current = np.asarray(metrics, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        delta = np.asarray([10 * action[0], 10 * action[1], 5 * action[2], 5 * action[3], action.sum()])
        return {
            "predicted_next_metrics": current + delta,
            "uncertainty": np.full(5, 0.1),
            "auxiliary_predictions": {
                "clipping_fraction": 0.0,
                "camera_boundary_probability": 0.0,
            },
        }

    generator._simulate = MethodType(synthetic_simulate, generator)
    generator._predict = MethodType(synthetic_predict, generator)
    kwargs = {
        "setup_id": "synthetic",
        "target_id": "one_step_mixed",
        "setup_context": _setup_context(),
        "positions": np.zeros(4),
        "current_metrics": np.asarray([0, 0, 2, 2, 1], dtype=np.float64),
        "target_metrics": np.asarray([0.5, -0.5, 2.1, 1.9, 1], dtype=np.float64),
        "tolerance_reference": np.asarray([1, 1, 2, 2, 0.05], dtype=np.float64),
        "history": _valid_state()["history"],
        "current_uncertainty": np.full(5, 0.1),
    }
    first_output, first_audit = generator._oracle_for_target(**kwargs)
    second_output, second_audit = generator._oracle_for_target(**kwargs)
    assert first_output == second_output
    assert first_audit["selected_configuration_id"] == second_audit["selected_configuration_id"]
    assert first_audit["configuration_count"] == 36
    assert first_audit["predicted_shortlist_size"] == 6
    assert len(first_audit["actual_shortlist"]) == 6


def test_mock_generate_gate_export_and_resume_are_deterministic() -> None:
    artifact_root = pipeline.PACKAGE_ROOT / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mock-generate-", dir=artifact_root) as directory:
        output_root = Path(directory).resolve()
        image_path = output_root / "images/mock-current.png"
        image_path.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8), mode="L").save(image_path)
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        storage_path = str(image_path.relative_to(pipeline.REPO_ROOT))

        def make_generator(*, resume: bool):
            generator = pipeline.CandidateDataGenerator.__new__(
                pipeline.CandidateDataGenerator
            )
            generator.output_root = output_root
            generator.resume = resume
            generator.protocol_hashes = pipeline.verify_frozen_protocol()
            generator.source_hashes = pipeline.verify_source_freeze()
            generator.identity_registry = pipeline.KnownIdentityRegistry.load(
                pipeline.PACKAGE_ROOT / "configs/known_identity_blocklist.json"
            )
            generator.simulator_fixed = {"synthetic": True}
            generator.simulator_calls = 0
            generator.forward_calls = 0
            generator.planner_calls = 0
            generator.rejected_known_identity_candidates = 0
            generator.rejected_candidate_identity_candidates = 0

            def sample_setup(
                self,
                *,
                split,
                index,
                accepted_hashes,
                accepted_fingerprints,
            ):
                split_offset = {"train": 0, "dev": 100, "candidate_eval": 200}[split]
                identity_index = split_offset + index + 1
                context = _setup_context(identity_index)
                context["wavelength_nm"] = 750.0 + identity_index
                return context, np.zeros(4), {}, 0, f"{identity_index:064x}"

            def setup_rows(
                self,
                *,
                split,
                index,
                setup_context,
                initial_positions,
                initial_capture,
                setup_attempt,
                identity_hash,
            ):
                setup_id = self._setup_id(split, index)
                rows = []
                for target_index, (target_id, _) in enumerate(pipeline.TARGET_VARIANTS):
                    record_id = f"{setup_id}__target-{target_index:02d}-{target_id}"
                    rows.append(
                        {
                            "schema_version": "qwen_h1_meta_v0_candidate_record_v1",
                            "record_id": record_id,
                            "split": split,
                            "image": {
                                "storage_path": storage_path,
                                "sha256": image_hash,
                                "audit_features": {
                                    "mean": 0.0,
                                    "standard_deviation": 0.0,
                                },
                            },
                            "model_visible_input": _valid_state(),
                            "oracle_output": _default_output(),
                            "identity": {
                                "setup_hash": identity_hash,
                                "common_fields_rounded_9": list(
                                    pipeline.setup_common_fingerprint(setup_context)
                                ),
                            },
                            "oracle_audit": {
                                "selected_configuration_id": "default"
                            },
                            "evaluator_only": {
                                "setup_id": setup_id,
                                "episode_id": record_id,
                                "target_counterfactual_id": target_id,
                                "setup_index": index,
                                "target_index": target_index,
                                "setup_context": dict(setup_context),
                            },
                        }
                    )
                return rows

            generator._sample_setup = MethodType(sample_setup, generator)
            generator._generate_setup_rows = MethodType(setup_rows, generator)
            return generator

        first = make_generator(resume=False).generate()
        assert first["gate"] == "PASS"
        assert first["sft_exported"] is True
        train_path = Path(first["exports"]["train"])
        first_train_bytes = train_path.read_bytes()
        first_manifest_bytes = (output_root / "manifest_train.jsonl").read_bytes()

        second = make_generator(resume=True).generate()
        assert second["gate"] == "PASS"
        assert train_path.read_bytes() == first_train_bytes
        assert (output_root / "manifest_train.jsonl").read_bytes() == first_manifest_bytes
        first_row = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
        validate_prebuilt_row(
            first_row,
            allowed_splits={"train"},
            image_root=pipeline.REPO_ROOT,
            verify_image=True,
        )
