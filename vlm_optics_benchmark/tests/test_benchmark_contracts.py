from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from continuous_control_v12.contracts import Bounds
from continuous_control_v12.world_model import load_forward_ensemble
from vlm_optics_benchmark.visual_anomalies import (
    FAMILIES,
    MATCH_TOLERANCE,
    moment_metrics,
    read_jsonl,
)
from vlm_optics_benchmark.visual_control import execute_episode


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/vlm_optics_benchmark_20260801_154812"
DATA = RUN / "visual_data"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_setup_splits_are_disjoint_and_complete() -> None:
    setups = {}
    expected = {"train": 24, "iid_heldout": 6, "severity_ood": 30}
    for split, count in expected.items():
        rows = read_jsonl(DATA / f"dataset_{split}.jsonl")
        setups[split] = {row["setup_id"] for row in rows}
        assert len(setups[split]) == count
    assert not (setups["train"] & setups["iid_heldout"])
    assert not (setups["train"] & setups["severity_ood"])
    assert not (setups["iid_heldout"] & setups["severity_ood"])


def test_metric_image_consistency_and_pair_matching() -> None:
    for split in ("train", "iid_heldout", "severity_ood"):
        rows = {row["sample_id"]: row for row in read_jsonl(DATA / f"dataset_{split}.jsonl")}
        for row in rows.values():
            image = np.asarray(Image.open(DATA / row["model_input"]["image_ref"]).convert("L"), dtype=float) / 255.0
            delta = np.abs(moment_metrics(image) - np.asarray(row["model_input"]["five_metrics"])) / MATCH_TOLERANCE
            assert float(delta.max()) <= 0.40
        for pair in read_jsonl(DATA / f"pairs_{split}.jsonl"):
            clean = np.asarray(Image.open(DATA / pair["clean_image_ref"]).convert("L"), dtype=float) / 255.0
            anomalous = np.asarray(Image.open(DATA / pair["anomalous_image_ref"]).convert("L"), dtype=float) / 255.0
            delta = np.abs(moment_metrics(clean) - moment_metrics(anomalous)) / MATCH_TOLERANCE
            assert float(delta.max()) <= 0.25
            assert float(np.linalg.norm(delta)) <= 0.40


def test_no_hidden_label_leakage_and_structured_serialization() -> None:
    forbidden = ("fault", "severity", "oracle", "hidden", "outcome", "setup_hash")
    expected_options = {"standard_metrics", "reduce_exposure_reacquire", "primary_spot_specialist", "stop"}
    observed_orders = set()
    for split in ("train", "iid_heldout", "severity_ood"):
        for row in read_jsonl(DATA / f"dataset_{split}.jsonl"):
            model_input = row["model_input"]
            visible_without_path = dict(model_input)
            image_ref = Path(visible_without_path.pop("image_ref"))
            serialized = json.dumps(visible_without_path, sort_keys=True).lower()
            assert not any(token in serialized for token in forbidden)
            assert not any(token in image_ref.name.lower() for token in forbidden)
            assert len(model_input["five_metrics"]) == 5
            assert len(model_input["five_metric_uncertainties"]) == 5
            assert set(model_input["candidate_options"]) == expected_options
            observed_orders.add(tuple(model_input["candidate_options"]))
            assert set(row["supervision"]) == {
                "fault_type",
                "binary_fault_present",
                "oracle_recovery_decision",
            }
    assert len(observed_orders) > 4
    schema = json.loads((RUN / "vlm_dataset_schema.json").read_text())
    assert schema["continuous_control_owner"] == "frozen_h1_cem_sequential_max8"
    assert "free_form_continuous_actuator_values" in schema["forbidden_model_output"]


def test_controller_sources_and_external_configuration_are_frozen() -> None:
    prereg = json.loads((RUN / "external_validation_preregistration.json").read_text())
    identities = prereg["frozen_source_identities"]
    assert _sha(REPO / "active_diagnosis_v13/config_v13.json") == identities["v13_config_sha256"]
    assert _sha(REPO / "runs/active_diagnosis_v13_20260801_010051/development/ablations/models/no_residual_history.joblib") == identities["probe_model_sha256"]
    assert _sha(REPO / "runs/active_diagnosis_v13_20260801_010051/development/sequential_horizon_rule_probe_seed1.json") == identities["sequential_rule_report_sha256"]
    assert _sha(REPO / "vlm_optics_benchmark/external_validation.py") == identities["external_runner_sha256"]
    results = json.loads((RUN / "external_validation_results.json").read_text())
    assert results["backbone_freeze_gate"]["passed"] is True
    assert results["episodes"] == 60


@lru_cache(maxsize=1)
def _control_runtime():
    config = json.loads((REPO / "active_diagnosis_v13/config_v13.json").read_text())
    config = {**config, "root_seed": 2026081301, "baseline": {**config["baseline"], "max_control_steps": 8}}
    v12 = json.loads(Path(config["baseline"]["v12_config"]).read_text())
    return config, Bounds.from_config(v12), load_forward_ensemble(Path(config["baseline"]["checkpoint"]), device_name="cpu")


def test_deterministic_episode_replay_per_stratum_and_anomaly() -> None:
    suite = json.loads((RUN / "external_suite_manifest.json").read_text())
    stored = {
        (row["case_id"], row["family"]): row
        for row in read_jsonl(RUN / "control_value_episodes.jsonl")
        if row["arm"] == "oracle_diagnosis"
    }
    config, bounds, model = _control_runtime()
    chosen = {}
    for case in suite["cases"]:
        chosen.setdefault(case["stratum"], case)
    assert len(chosen) == 3
    for family in FAMILIES:
        for case in chosen.values():
            replay = execute_episode(
                case=case,
                family=family,
                arm="oracle_diagnosis",
                config=config,
                bounds=bounds,
                model=model,
                base_config=REPO / "optical_sim/configs/base_config.yaml",
                diagnostic=None,
            )
            expected = stored[(case["case_id"], family)]
            assert replay["strict_success"] == expected["strict_success"]
            assert replay["control_steps"] == expected["control_steps"]
            assert replay["saturation_count"] == expected["saturation_count"]
            assert replay["final_normalized_target_distance"] == expected["final_normalized_target_distance"]
            assert replay["trace"] == expected["trace"]
