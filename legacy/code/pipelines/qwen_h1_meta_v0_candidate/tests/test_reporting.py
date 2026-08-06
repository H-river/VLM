from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_h1_meta_v0_candidate import reporting


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _metric(value: float) -> dict[str, object]:
    return {"value": value, "reason": None}


def _sources(root: Path, *, data_gate: str = "PASS") -> reporting.ReportSources:
    red = data_gate != "PASS"
    data = _write_json(
        root / "evidence/data_audit.json",
        {
            "version": "qwen_h1_meta_v0_data_audit_v1",
            "candidate_only": True,
            "audited_splits": ["train", "dev"],
            "record_count": 72,
            "identity_overlap": {
                "known_overlap_count": 0,
                "cross_split_overlap_count": 0,
            },
            "exact_visible_collision": {"conflicting_collision_group_count": 0},
            "rounded_4dp_visible_collision": {
                "conflicting_collision_group_count": 0
            },
            "hidden_setup_macro_f1_gain": 0.0,
            "gate": data_gate,
            "red_reasons": ["visible_classifier_macro_f1_below_0.50"] if red else [],
            "sft_export_permitted": not red,
        },
    )
    regression = _write_json(
        root / "evidence/regression.json",
        {
            "version": "qwen_h1_meta_v0_regression_v1",
            "candidate_only": True,
            "formal_frozen_evaluation_enabled": False,
            "status": "PASS",
            "existing_structure_preserved": True,
            "safety": {
                "feature_flag_off_equivalent": True,
                "compiler_bounds_preserved": True,
                "continuous_action_injection_rejected": True,
                "dispatch_gate_enforced": True,
                "h3_disabled": True,
                "existing_state_machine_tests_passed": True,
                "canonical_objective_and_strict_success_preserved": True,
                "frozen_or_protected_content_untouched": True,
            },
            "checks": [{"name": "candidate synthetic", "status": "PASS", "passed": 1, "failed": 0}],
        },
    )
    commands = _write_json(
        root / "evidence/command_log.json",
        {
            "commands": [
                {
                    "command": "python -m qwen_h1_meta_v0_candidate.reporting validate",
                    "purpose": "Validate candidate reports",
                    "status": "passed",
                }
            ]
        },
    )
    if red:
        return reporting.ReportSources(
            data_audit=data, regression=regression, command_log=commands
        )

    training_runs: list[Path] = []
    per_seed: list[dict[str, object]] = []
    for index, seed in enumerate(reporting.TRAINING_SEEDS):
        training_runs.append(
            _write_json(
                root / f"training/seed_{seed}/run_manifest.latest.json",
                {
                    "manifest_version": "qwen_h1_meta_training_run_v0",
                    "independent_adapter_name": "qwen_h1_meta_v0",
                    "status": "completed",
                    "data": {"frozen_or_protected_predictions_opened": False},
                    "training": {"seed": seed},
                    "training_result": {
                        "global_step": 200,
                        "best_dev_checkpoint": f"training/seed_{seed}/checkpoint-100",
                        "runtime_wall_seconds": 10.0 + index,
                        "metrics": {"train_loss": 0.3 - index * 0.01},
                    },
                    "gpu": {
                        "peak_allocated_bytes": 1000 + index,
                        "peak_reserved_bytes": 1200 + index,
                    },
                },
            )
        )
        per_seed.append(
            {
                "seed": seed,
                "valid_json_rate": 1.0,
                "decision_macro_f1": 0.8,
                "full_configuration_exact_match": 0.7,
                "direction_accuracy": 0.9,
                "direction_macro_f1": 0.85,
                "compiled_guidance_validity_rate": 1.0,
                "training_run": {
                    "best_dev_loss": 0.2,
                    "final_dev_loss": 0.21,
                    "wall_time_seconds": 10.0 + index,
                    "peak_memory_bytes": 1000 + index,
                },
            }
        )
    offline = _write_json(
        root / "evidence/offline_reasoning_results.json",
        {
            "version": "qwen_h1_meta_v0_offline_evaluation_v1",
            "scope": "candidate-only synthetic",
            "formal_frozen_evaluation_enabled": False,
            "expected_seeds": list(reporting.TRAINING_SEEDS),
            "per_seed": per_seed,
            "aggregate": {
                "valid_json_rate": {"mean": 1.0, "std_population": 0.0},
                "configuration_regret_mean": {"mean": 0.1, "std_population": 0.01},
            },
        },
    )
    methods: dict[str, object] = {}
    success = {
        "default_h1": 0.5,
        "dual_budget_default_h1": 0.55,
        "rule_guided_h1": 0.58,
        "metrics_mlp_meta_h1": 0.57,
        "oracle_guided_h1": 0.9,
        "random_or_frequency_meta_h1": 0.2,
        **{f"qwen_guided_h1_seed_{seed}": 0.7 for seed in reporting.TRAINING_SEEDS},
    }
    for name, value in success.items():
        methods[name] = {
            "method_input_complete": True,
            "performance": {
                "strict_all_five_success": _metric(value),
                "final_normalized_error": _metric(1.0 - value),
            },
            "audit": {
                "configuration_regret": _metric(0.1),
                "wall_clock_compute_seconds": 1.0,
            },
        }
    closed_loop = _write_json(
        root / "evidence/closed_loop_results.json",
        {
            "version": "qwen_h1_meta_v0_candidate_closed_loop_report_v1",
            "candidate_only": True,
            "formal_frozen_evaluation_enabled": False,
            "scientific_conclusion": False,
            "manifest_record_count": 36,
            "setup_count": 12,
            "methods": methods,
            "wall_clock_seconds": 20.0,
        },
    )
    ablation = _write_json(
        root / "evidence/ablation_results.json",
        {
            "version": "qwen_h1_meta_reasoning_ablation_offline_v1",
            "candidate_only": True,
            "formal_frozen_evaluation_enabled": False,
            "per_seed": [{"seed": seed} for seed in reporting.TRAINING_SEEDS],
            "candidate_level_reasoning_supported": True,
        },
    )
    return reporting.ReportSources(
        data_audit=data,
        training_runs=tuple(training_runs),
        offline=offline,
        closed_loop=closed_loop,
        ablation=ablation,
        regression=regression,
        command_log=commands,
    )


def _fixed_machine(*, provenance: object, verdict: object) -> dict[str, object]:
    return {
        "schema_version": "qwen_h1_meta_v0_machine_summary_v1",
        "status": reporting.STATUS_LINE,
        "candidate_only": True,
        "formal_frozen_evaluation_enabled": False,
        "collection_policy": "synthetic read-only fixture",
        "python": {"version": "fixture"},
        "platform": "fixture",
        "packages": {},
        "gpu": {"available": False, "devices": []},
        "repository": {"commit": "fixture", "branch": "fixture"},
        "evidence_inputs": provenance,
        "report_verdict": verdict,
    }


def test_build_is_deterministic_and_self_hash_is_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporting, "collect_machine_summary", _fixed_machine)
    with tempfile.TemporaryDirectory(
        prefix=".reporting_synth_", dir=reporting.PACKAGE_ROOT
    ) as directory:
        root = Path(directory)
        sources = _sources(root)
        first = reporting.build_reports(sources, output_root=root)
        first_bytes = {
            name: (root / name).read_bytes() for name in reporting.REPORT_FILENAMES
        }
        second = reporting.build_reports(sources, output_root=root)
        second_bytes = {
            name: (root / name).read_bytes() for name in reporting.REPORT_FILENAMES
        }
        assert first["status"] == second["status"] == "YELLOW"
        assert first_bytes == second_bytes
        verdict = json.loads((root / "machine_summary.json").read_text())[
            "report_verdict"
        ]
        # Evidence-declared booleans cannot override missing paired CIs/raw
        # ablation diagnostics.
        assert verdict["qwen_closed_loop_gain_supported"] is False
        assert verdict["candidate_reasoning_supported"] is False
        hashes = json.loads((root / "artifact_hashes.json").read_text())
        assert "artifact_hashes.json" not in hashes["files"]
        assert "tests" not in hashes["inventory_policy"]["exclude_directory_names"]
        assert reporting.validate_generated_artifacts(root)["status"] == "PASS"
        assert (root / "final_report.md").read_text().rstrip().endswith(
            reporting.TERMINAL_LINE
        )


def test_red_information_gate_marks_downstream_as_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporting, "collect_machine_summary", _fixed_machine)
    with tempfile.TemporaryDirectory(
        prefix=".reporting_synth_", dir=reporting.PACKAGE_ROOT
    ) as directory:
        root = Path(directory)
        execution = _write_json(
            root / "evidence/execution_evidence.json",
            {
                "version": "qwen_h1_meta_v0_execution_evidence_v1",
                "candidate_only": True,
                "formal_frozen_evaluation_enabled": False,
                "data_generation": {
                    "command": "python -m qwen_h1_meta_v0_candidate.data_pipeline generate",
                    "exit_status": 3,
                    "exit_status_meaning": "preregistered information-sufficiency RED_STOP",
                    "pipeline_elapsed_seconds": 2.5,
                    "time_wall_clock": "0:02.50",
                    "cpu_percent": 100,
                    "maximum_resident_set_kib": 42,
                    "sft_exported": False,
                },
                "stopped_operations": {
                    key: "not_run_due_to_information_sufficiency_gate"
                    for key in (
                        "qwen_training",
                        "dev_generation",
                        "offline_performance_evaluation",
                        "candidate_closed_loop_evaluation",
                        "reasoning_ablations",
                    )
                },
            },
        )
        sources = replace(
            _sources(root, data_gate="RED_STOP"), execution_evidence=execution
        )
        result = reporting.build_reports(
            sources, output_root=root
        )
        machine = json.loads((root / "machine_summary.json").read_text())
        verdict = machine["report_verdict"]
        assert result["status"] == "RED"
        assert verdict["downstream_status"] == {
            "training": "not_run_due_to_information_sufficiency_gate",
            "offline": "not_run_due_to_information_sufficiency_gate",
            "closed_loop": "not_run_due_to_information_sufficiency_gate",
            "ablation": "not_run_due_to_information_sufficiency_gate",
        }
        report = (root / "final_report.md").read_text()
        assert "not_run_due_to_information_sufficiency_gate" in report
        assert "stable_qwen_closed_loop_gain_not_confirmed" not in verdict["yellow_reasons"]
        assert "data_pipeline generate" in (root / "commands.md").read_text()


def test_evidence_outside_candidate_namespace_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        outside = _write_json(Path(directory) / "data_audit.json", {})
        with pytest.raises(reporting.ReportingError, match="candidate"):
            reporting.build_reports(
                reporting.ReportSources(data_audit=outside),
                output_root=reporting.PACKAGE_ROOT,
            )


def test_hash_validation_detects_report_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporting, "collect_machine_summary", _fixed_machine)
    with tempfile.TemporaryDirectory(
        prefix=".reporting_synth_", dir=reporting.PACKAGE_ROOT
    ) as directory:
        root = Path(directory)
        reporting.build_reports(_sources(root), output_root=root)
        (root / "commands.md").write_text("tampered\n", encoding="utf-8")
        with pytest.raises(reporting.ArtifactValidationError, match="mismatch"):
            reporting.validate_generated_artifacts(root)


def test_formal_frozen_evaluation_flag_forces_red(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporting, "collect_machine_summary", _fixed_machine)
    with tempfile.TemporaryDirectory(
        prefix=".reporting_synth_", dir=reporting.PACKAGE_ROOT
    ) as directory:
        root = Path(directory)
        sources = _sources(root)
        closed = json.loads(Path(sources.closed_loop).read_text())
        closed["formal_frozen_evaluation_enabled"] = True
        _write_json(Path(sources.closed_loop), closed)
        result = reporting.build_reports(sources, output_root=root)
        assert result["status"] == "RED"
        verdict = json.loads((root / "machine_summary.json").read_text())[
            "report_verdict"
        ]
        assert "closed_loop_formal_frozen_evaluation_enabled" in verdict["red_reasons"]


def test_ablation_version_is_allowlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reporting, "collect_machine_summary", _fixed_machine)
    with tempfile.TemporaryDirectory(
        prefix=".reporting_synth_", dir=reporting.PACKAGE_ROOT
    ) as directory:
        root = Path(directory)
        sources = _sources(root)
        ablation = json.loads(Path(sources.ablation).read_text())
        ablation["version"] = "untrusted_ablation_v99"
        _write_json(Path(sources.ablation), ablation)
        with pytest.raises(reporting.ReportingError, match="ablation.version"):
            reporting.build_reports(sources, output_root=root)
