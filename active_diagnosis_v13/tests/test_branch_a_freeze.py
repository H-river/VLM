from __future__ import annotations

import hashlib
import json

import pytest

from active_diagnosis_v13.finalize_branch_a import _validate_formal_gate


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_branch_a_resolution_requires_matching_formal_chain(tmp_path) -> None:
    diagnosis = {
        "selected_primary_branch": "A",
        "split": "development_only",
        "protected_set_used": False,
    }
    diagnosis_path = tmp_path / "gate_a_diagnosis.json"
    diagnosis_path.write_text(json.dumps(diagnosis))
    formal = {
        "frozen": True,
        "selected_primary_branch": "A",
        "protected_set_used_for_selection": False,
        "gate_a_diagnosis_sha256": _sha(diagnosis_path),
        "config_sha256": "locked-config",
    }
    (tmp_path / "frozen_decision.json").write_text(json.dumps(formal))
    _, loaded = _validate_formal_gate(tmp_path, diagnosis_path, diagnosis)
    assert loaded["config_sha256"] == "locked-config"

    diagnosis_path.write_text(json.dumps({**diagnosis, "tampered": True}))
    with pytest.raises(ValueError, match="does not match"):
        _validate_formal_gate(tmp_path, diagnosis_path, {**diagnosis, "tampered": True})


def test_branch_a_resolution_rejects_protected_selection(tmp_path) -> None:
    diagnosis = {
        "selected_primary_branch": "A",
        "split": "development_only",
        "protected_set_used": True,
    }
    diagnosis_path = tmp_path / "gate_a_diagnosis.json"
    diagnosis_path.write_text(json.dumps(diagnosis))
    with pytest.raises(ValueError, match="development-only"):
        _validate_formal_gate(tmp_path, diagnosis_path, diagnosis)
