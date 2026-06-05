"""Pass/fail gates for base vs SFT benchmark comparisons."""

from __future__ import annotations

from typing import Any, Mapping

DEFAULT_SFT_GATES = {
    "json_valid_rate_min": 0.95,
    "overall_lens_sign_accuracy_min": 0.55,
    "overall_lens_sign_accuracy_min_delta_over_base": 0.05,
    "mean_lens_mae_max_ratio_vs_base": 0.95,
}


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def evaluate_sft_gates(
    base: Mapping[str, Any],
    sft: Mapping[str, Any],
    gates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = dict(DEFAULT_SFT_GATES)
    if gates:
        thresholds.update(gates)

    base_json = _num(base.get("json_valid_rate"))
    sft_json = _num(sft.get("json_valid_rate"))
    base_sign = _num(base.get("overall_lens_sign_accuracy"))
    sft_sign = _num(sft.get("overall_lens_sign_accuracy"))
    base_mae = _num(base.get("mean_lens_action_mae"))
    sft_mae = _num(sft.get("mean_lens_action_mae"))

    sign_delta = None if base_sign is None or sft_sign is None else sft_sign - base_sign
    mae_ratio = None if base_mae is None or sft_mae is None or base_mae == 0 else sft_mae / base_mae

    checks = {
        "json_valid_rate_ok": sft_json is not None and sft_json >= float(thresholds["json_valid_rate_min"]),
        "sign_accuracy_absolute_ok": sft_sign is not None and sft_sign >= float(
            thresholds["overall_lens_sign_accuracy_min"]
        ),
        "sign_accuracy_improved_over_base": sign_delta is not None
        and sign_delta >= float(thresholds["overall_lens_sign_accuracy_min_delta_over_base"]),
        "mean_lens_mae_not_regressed": mae_ratio is not None
        and mae_ratio <= float(thresholds["mean_lens_mae_max_ratio_vs_base"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "observed": {
            "base_overall_lens_sign_accuracy": base_sign,
            "sft_overall_lens_sign_accuracy": sft_sign,
            "sign_accuracy_delta": sign_delta,
            "base_mean_lens_action_mae": base_mae,
            "sft_mean_lens_action_mae": sft_mae,
            "mean_lens_mae_ratio_sft_over_base": mae_ratio,
            "sft_json_valid_rate": sft_json,
        },
    }
