from __future__ import annotations

from active_diagnosis_v13.run_protected_once import validate_protected_rows


def _rows() -> list[dict[str, object]]:
    return [
        {
            "record_id": f"{stratum}_{suffix:04d}__g{gain:g}",
            "case_id": f"{stratum}_{suffix:04d}",
            "stratum": stratum,
            "evaluator_only_true_gain": gain,
        }
        for stratum in ("interior", "boundary", "multi_step")
        for suffix in range(10, 16)
        for gain in (0.5, 0.75, 1.0, 1.25, 1.5)
    ]


def test_exact_protected_support_passes() -> None:
    assert validate_protected_rows(_rows(), "probe_replan") == []


def test_protected_support_rejects_out_of_range_suffix_at_constant_size() -> None:
    rows = _rows()
    for row in rows:
        if row["stratum"] == "boundary" and str(row["case_id"]).endswith("0015"):
            gain = float(row["evaluator_only_true_gain"])
            row["case_id"] = "boundary_0016"
            row["record_id"] = f"boundary_0016__g{gain:g}"
    errors = validate_protected_rows(rows, "probe_replan")
    assert any("expected [10, 11, 12, 13, 14, 15]" in error for error in errors)


def test_protected_support_rejects_duplicate_case_gain_episode() -> None:
    rows = _rows()
    rows[-1]["case_id"] = rows[-2]["case_id"]
    rows[-1]["evaluator_only_true_gain"] = rows[-2]["evaluator_only_true_gain"]
    errors = validate_protected_rows(rows, "direct")
    assert "direct does not contain 90 unique case-gain episodes" in errors
