from active_diagnosis_v13.analyze_protected_confirmation import _category


def _row(success: bool):
    return {"strict_success": success}


def test_protected_failure_categories_cover_recovery_and_regression() -> None:
    assert _category(_row(False), _row(True), _row(True)) == (
        "probe_recovered_oracle_recoverable_failure"
    )
    assert _category(_row(False), _row(False), _row(True)) == (
        "probe_recovered_beyond_oracle_policy"
    )
    assert _category(_row(True), _row(True), _row(False)) == (
        "probe_induced_regression"
    )
    assert _category(_row(False), _row(False), _row(False)) == (
        "unrecovered_even_with_oracle_gain"
    )
