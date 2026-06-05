#!/usr/bin/env python3
"""Compare base and SFT benchmark reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.eval.sft_gates import evaluate_sft_gates


METRIC_KEYS = (
    "json_valid_rate",
    "control_plan_present_rate",
    "mean_lens_action_mae",
    "overall_lens_sign_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs SFT text benchmark reports.")
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--sft-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with code 1 when SFT success gates fail.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def delta(base_value: Any, sft_value: Any) -> float | None:
    if not isinstance(base_value, (int, float)) or not isinstance(sft_value, (int, float)):
        return None
    return float(sft_value) - float(base_value)


def relative_improvement(base_value: Any, sft_value: Any, lower_is_better: bool) -> float | None:
    if not isinstance(base_value, (int, float)) or not isinstance(sft_value, (int, float)):
        return None
    base = float(base_value)
    sft = float(sft_value)
    if lower_is_better:
        if base == 0:
            return None
        return (base - sft) / base
    if base == 0:
        return None
    return (sft - base) / base


def main() -> None:
    args = parse_args()
    base = load_report(args.base_report)
    sft = load_report(args.sft_report)

    comparison: dict[str, Any] = {}
    for key in METRIC_KEYS:
        base_value = base.get(key)
        sft_value = sft.get(key)
        comparison[key] = {
            "base": base_value,
            "sft": sft_value,
            "delta": delta(base_value, sft_value),
            "relative_improvement": relative_improvement(
                base_value,
                sft_value,
                lower_is_better=key.endswith("_mae"),
            ),
        }

    sft_gates = evaluate_sft_gates(base, sft)
    report = {
        "base_report": str(args.base_report),
        "sft_report": str(args.sft_report),
        "num_examples": {
            "base": base.get("num_examples"),
            "sft": sft.get("num_examples"),
        },
        "comparison": comparison,
        "action_mae": {
            key: {
                "base": (base.get("action_mae") or {}).get(key),
                "sft": (sft.get("action_mae") or {}).get(key),
                "delta": delta((base.get("action_mae") or {}).get(key), (sft.get("action_mae") or {}).get(key)),
            }
            for key in ("lens_x_delta_mm", "lens_y_delta_mm", "camera_x_delta_mm", "camera_y_delta_mm")
        },
        "sft_success_gates": sft_gates,
        "interpretation": {
            "json_valid_rate": "Higher is better.",
            "mean_lens_action_mae": "Lower is better.",
            "overall_lens_sign_accuracy": "Higher is better.",
            "sft_success_gates": (
                "SFT passes when JSON stays valid, sign accuracy reaches 55% and improves "
                "base by at least 5 points, and lens MAE does not regress beyond 95% of base."
            ),
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_gate and not sft_gates["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
