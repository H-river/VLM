#!/usr/bin/env python3
"""Gate quality-routed visual tools across clean and perturbed conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = ("clean", "noise", "blur", "dimnoise")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_interval(summary: dict, metric: str) -> dict:
    bootstrap = summary.get("group_bootstrap") or {}
    interval = bootstrap.get(f"{metric}_95ci") or {}
    return {
        "low": float(interval["low"]),
        "high": float(interval["high"]),
        "samples": int(bootstrap["samples"]),
        "seed": int(bootstrap["seed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition}-state", type=Path, required=True)
        parser.add_argument(f"--{condition}-pair", type=Path, required=True)
    parser.add_argument("--saturation-state", type=Path)
    parser.add_argument("--saturation-pair", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.saturation_state is None) != (args.saturation_pair is None):
        parser.error("provide both --saturation-state and --saturation-pair")
    conditions = list(CONDITIONS)
    if args.saturation_state is not None:
        conditions.append("saturation")
    metrics = {}
    for condition in conditions:
        state = load(getattr(args, f"{condition}_state"))
        pair = load(getattr(args, f"{condition}_pair"))
        state_macro = float(state["equal_field_macro_f1"])
        pair_macro = float(pair["equal_field_macro_f1"])
        state_macro_ci = bootstrap_interval(state, "equal_field_macro_f1")
        pair_macro_ci = bootstrap_interval(pair, "equal_field_macro_f1")
        state_joint_ci = bootstrap_interval(state, "joint_exact_match")
        pair_joint_ci = bootstrap_interval(pair, "joint_exact_match")
        metrics[condition] = {
            "state_macro_f1": state_macro,
            "state_macro_f1_95ci": state_macro_ci,
            "state_joint_exact": float(state["joint_exact_match"]),
            "state_joint_exact_95ci": state_joint_ci,
            "pair_macro_f1": pair_macro,
            "pair_macro_f1_95ci": pair_macro_ci,
            "pair_joint_exact": float(pair["joint_exact_match"]),
            "pair_joint_exact_95ci": pair_joint_ci,
            "production_visual_macro_f1": (4.0 * state_macro + 5.0 * pair_macro) / 9.0,
            # This weighted bound is intentionally conservative. It combines the
            # two component interval endpoints instead of claiming a joint
            # bootstrap interval across differently sized state and pair rows.
            "production_visual_macro_f1_conservative_95ci": {
                "low": (4.0 * state_macro_ci["low"] + 5.0 * pair_macro_ci["low"])
                / 9.0,
                "high": (4.0 * state_macro_ci["high"] + 5.0 * pair_macro_ci["high"])
                / 9.0,
            },
        }
    checks = {
        "clean_production_macro_at_least_0_90": metrics["clean"][
            "production_visual_macro_f1"
        ]
        >= 0.90,
        "clean_state_joint_at_least_0_90": metrics["clean"]["state_joint_exact"] >= 0.90,
        "clean_pair_joint_at_least_0_90": metrics["clean"]["pair_joint_exact"] >= 0.90,
        "noise_production_macro_at_least_0_80": metrics["noise"][
            "production_visual_macro_f1"
        ]
        >= 0.80,
        "noise_state_joint_at_least_0_65": metrics["noise"]["state_joint_exact"] >= 0.65,
        "noise_pair_joint_at_least_0_80": metrics["noise"]["pair_joint_exact"] >= 0.80,
        "blur_production_macro_at_least_0_85": metrics["blur"][
            "production_visual_macro_f1"
        ]
        >= 0.85,
        "blur_state_joint_at_least_0_85": metrics["blur"]["state_joint_exact"] >= 0.85,
        "blur_pair_joint_at_least_0_80": metrics["blur"]["pair_joint_exact"] >= 0.80,
        "dimnoise_production_macro_at_least_0_80": metrics["dimnoise"][
            "production_visual_macro_f1"
        ]
        >= 0.80,
        "dimnoise_state_joint_at_least_0_60": metrics["dimnoise"]["state_joint_exact"]
        >= 0.60,
        "dimnoise_pair_joint_at_least_0_75": metrics["dimnoise"]["pair_joint_exact"]
        >= 0.75,
        "all_component_bootstraps_use_at_least_1000_samples": all(
            metrics[condition][f"{mode}_{metric}_95ci"]["samples"] >= 1000
            for condition in conditions
            for mode in ("state", "pair")
            for metric in ("macro_f1", "joint_exact")
        ),
    }
    if "saturation" in metrics:
        checks.update(
            {
                "saturation_production_macro_at_least_0_85": metrics["saturation"][
                    "production_visual_macro_f1"
                ]
                >= 0.85,
                "saturation_state_joint_at_least_0_85": metrics["saturation"][
                    "state_joint_exact"
                ]
                >= 0.85,
                "saturation_pair_joint_at_least_0_80": metrics["saturation"][
                    "pair_joint_exact"
                ]
                >= 0.80,
            }
        )
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "average_production_visual_macro_f1": sum(
            value["production_visual_macro_f1"] for value in metrics.values()
        )
        / len(metrics),
        "worst_condition_production_visual_macro_f1": min(
            value["production_visual_macro_f1"] for value in metrics.values()
        ),
        "worst_condition_conservative_production_macro_f1_95ci_low": min(
            value["production_visual_macro_f1_conservative_95ci"]["low"]
            for value in metrics.values()
        ),
        "claim_boundary": (
            "Perturbations are deterministic synthetic image corruptions, not measurements from real "
            "hardware. The quality router preserves the clean interface and selects calibrated fields."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
