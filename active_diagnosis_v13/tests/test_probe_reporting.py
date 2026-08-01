from active_diagnosis_v13.summarize_probe_controls import (
    _group_bootstrap_difference,
    fraction_label,
    output_name,
)


def test_probe_control_output_names_are_stable() -> None:
    assert fraction_label(0.01) == "0p01"
    assert fraction_label(0.1) == "0p1"
    assert output_name("symmetric_pair", 0.05) == "probe_symmetric_pair_f0p05"


def test_probe_control_bootstrap_uses_matched_groups() -> None:
    reference = [
        {"group_id": "a", "strict_success": False},
        {"group_id": "b", "strict_success": True},
    ]
    policy = [
        {"group_id": "a", "strict_success": True},
        {"group_id": "b", "strict_success": True},
    ]
    interval = _group_bootstrap_difference(reference, policy, 7)
    assert interval["estimate"] == 0.5
    assert interval["low"] <= interval["estimate"] <= interval["high"]
