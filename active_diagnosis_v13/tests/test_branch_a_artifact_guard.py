from __future__ import annotations

import hashlib
import json

import pytest

from active_diagnosis_v13.build_decision_dataset import main as build_dataset_main
from active_diagnosis_v13.contracts import require_frozen_branch_a_refinement


def test_reduced_artifacts_require_matching_branch_a_resolution(tmp_path) -> None:
    model = tmp_path / "no_residual.joblib"
    model.write_bytes(b"frozen reduced model")
    with pytest.raises(ValueError, match="forbidden before"):
        require_frozen_branch_a_refinement(tmp_path, model)

    resolution = {
        "frozen": True,
        "selected_primary_branch": "A",
        "protected_set_used_for_selection": False,
        "selected_probe": {
            "classifier_bundle": str(model),
            "classifier_bundle_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "branch_a_resolution.json").write_text(json.dumps(resolution))
    loaded = require_frozen_branch_a_refinement(tmp_path, model)
    assert loaded["selected_primary_branch"] == "A"

    model.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash differs"):
        require_frozen_branch_a_refinement(tmp_path, model)


def test_reduced_artifacts_reject_non_a_or_protected_selection(tmp_path) -> None:
    model = tmp_path / "model.joblib"
    model.write_bytes(b"model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    base = {
        "frozen": True,
        "selected_primary_branch": "P",
        "protected_set_used_for_selection": False,
        "selected_probe": {
            "classifier_bundle": str(model),
            "classifier_bundle_sha256": digest,
        },
    }
    path = tmp_path / "branch_a_resolution.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="frozen Branch-A"):
        require_frozen_branch_a_refinement(tmp_path, model)
    path.write_text(
        json.dumps(
            {
                **base,
                "selected_primary_branch": "A",
                "protected_set_used_for_selection": True,
            }
        )
    )
    with pytest.raises(ValueError, match="development-only"):
        require_frozen_branch_a_refinement(tmp_path, model)


def test_reduced_dataset_rejection_leaves_no_output_directory(
    tmp_path, monkeypatch
) -> None:
    gate = tmp_path / "development"
    probes = gate / "probes"
    probes.mkdir(parents=True)
    (probes / "selected_probe.json").write_text(
        json.dumps({"selected_design": "symmetric_pair", "selected_fraction": 0.1})
    )
    row = {
        "record_id": "case_0000__g1",
        "case_id": "case_0000",
        "group_id": "case_0000",
        "design": "symmetric_pair",
        "fraction": 0.1,
    }
    (probes / "records.jsonl").write_text(json.dumps(row) + "\n")
    model = tmp_path / "model.joblib"
    model.write_bytes(b"not loaded before freeze guard")
    output = tmp_path / "branch_a" / "decision_dataset"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_decision_dataset",
            "--gate-dir",
            str(gate),
            "--output-dir",
            str(output),
            "--retained-feature-model",
            str(model),
        ],
    )
    with pytest.raises(ValueError, match="forbidden before"):
        build_dataset_main()
    assert not output.exists()
