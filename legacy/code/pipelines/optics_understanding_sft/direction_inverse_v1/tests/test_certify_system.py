from optics_understanding_sft.direction_inverse_v1.certify_system import gate


def test_gate_preserves_rule_and_boolean() -> None:
    result = gate(True, {"score": 0.5}, "score >= 0.5")
    assert result == {"passed": True, "rule": "score >= 0.5", "evidence": {"score": 0.5}}
