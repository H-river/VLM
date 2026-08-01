import pytest

from active_diagnosis_v13.freeze_gate_once import _parse_timestamp


def test_formal_gate_timestamp_requires_timezone() -> None:
    parsed = _parse_timestamp("2026-08-01T05:00:51+08:00")
    assert parsed.utcoffset().total_seconds() == 8 * 3600
    with pytest.raises(ValueError, match="UTC offset"):
        _parse_timestamp("2026-08-01T05:00:51")
