#!/usr/bin/env python3
"""Build exact four-through-eight control-horizon curves from matched runs."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.analyze_control_step_budget import (
    _assert_replay,
    _key,
    _read_jsonl,
)

VERSION = "active_diagnosis_v13_control_horizon_curve_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cap(row: dict[str, Any], horizon: int) -> dict[str, Any]:
    trace = row["trace"][:horizon]
    final_distance = (
        float(row["initial_normalized_distance"])
        if not trace
        else float(trace[-1]["actual_target_cost"])
    )
    probe_saturation = (
        0 if row.get("probe_record") is None else int(row["probe_record"]["saturation_count"])
    )
    control_saturation = sum(
        bool(step["actual_step_audit"]["step_saturated"])
        or bool(step["actual_step_audit"]["absolute_position_saturated"])
        for step in trace
    )
    return {
        "case_id": row["case_id"],
        "group_id": row["group_id"],
        "stratum": row["stratum"],
        "evaluator_only_true_gain": float(row["evaluator_only_true_gain"]),
        "strict_success": bool(final_distance <= 1.0),
        "final_normalized_distance": final_distance,
        "control_steps": len(trace),
        "saturation_count": int(probe_saturation + control_saturation),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fault = [row for row in rows if row["evaluator_only_true_gain"] != 1.0]
    boundary = [
        row for row in fault if row["stratum"] == "reachable_boundary_or_clipping"
    ]

    def block(values: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "episodes": len(values),
            "strict_success": float(np.mean([row["strict_success"] for row in values])),
            "mean_final_distance": float(
                np.mean([row["final_normalized_distance"] for row in values])
            ),
            "mean_control_steps": float(np.mean([row["control_steps"] for row in values])),
            "saturation_episode_rate": float(
                np.mean([row["saturation_count"] > 0 for row in values])
            ),
        }

    return {"overall": block(rows), "fault": block(fault), "boundary_fault": block(boundary)}


def _bootstrap(
    baseline: dict[tuple[str, float], dict[str, Any]],
    treatment: dict[tuple[str, float], dict[str, Any]],
    seed: int,
) -> dict[str, float]:
    keys = [key for key in baseline if key[1] != 1.0]
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for key in keys:
        grouped[str(baseline[key]["group_id"])].append(
            float(treatment[key]["strict_success"])
            - float(baseline[key]["strict_success"])
        )
    values = np.asarray(
        [float(np.mean(grouped[group])) for group in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(values[rng.integers(0, len(values), len(values))]))
        for _ in range(4000)
    ]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(np.mean(values)), "low": float(low), "high": float(high)}


def _arm(
    name: str,
    budget4_rows: list[dict[str, Any]],
    budget6_rows: list[dict[str, Any]],
    budget8_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    four = {_key(row): row for row in budget4_rows}
    six = {_key(row): row for row in budget6_rows}
    eight = {_key(row): row for row in budget8_rows}
    if len(four) != 150 or len(six) != 150 or len(eight) != 150:
        raise ValueError(f"{name} horizon arms must each contain 150 rows")
    if set(four) != set(six) or set(four) != set(eight):
        raise ValueError(f"{name} horizon arms are not matched")
    mismatch_four = _assert_replay(four, eight)
    mismatch_six = _assert_replay(six, eight)
    if mismatch_four or mismatch_six:
        raise ValueError(
            f"{name} prefix mismatch: four={mismatch_four[:2]} six={mismatch_six[:2]}"
        )
    capped = {
        horizon: {key: _cap(eight[key], horizon) for key in sorted(eight)}
        for horizon in range(4, 9)
    }
    for key in four:
        for expected, actual in ((four[key], capped[4][key]), (six[key], capped[6][key])):
            if bool(expected["strict_success"]) != bool(actual["strict_success"]):
                raise ValueError(f"{name} capped outcome mismatch at {key}")
            if not np.isclose(
                float(expected["final_normalized_distance"]),
                float(actual["final_normalized_distance"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"{name} capped final distance mismatch at {key}")
            if int(expected["saturation_count"]) != int(actual["saturation_count"]):
                raise ValueError(f"{name} capped saturation mismatch at {key}")
    curve = []
    for horizon in range(4, 9):
        curve.append(
            {
                "horizon": horizon,
                "summary": _summary(list(capped[horizon].values())),
                "fault_difference_vs_four": _bootstrap(
                    capped[4], capped[horizon], seed + horizon
                ),
            }
        )
    marginal = []
    for horizon in range(5, 9):
        previous, current = capped[horizon - 1], capped[horizon]
        recovered = [
            f"{key[0]}__g{key[1]:g}"
            for key in sorted(current)
            if key[1] != 1.0
            and not bool(previous[key]["strict_success"])
            and bool(current[key]["strict_success"])
        ]
        regressed = [
            f"{key[0]}__g{key[1]:g}"
            for key in sorted(current)
            if key[1] != 1.0
            and bool(previous[key]["strict_success"])
            and not bool(current[key]["strict_success"])
        ]
        marginal.append(
            {
                "from_horizon": horizon - 1,
                "to_horizon": horizon,
                "fault_recoveries": len(recovered),
                "fault_regressions": len(regressed),
                "recovered_episode_ids": recovered,
                "regressed_episode_ids": regressed,
            }
        )
    return {
        "arm": name,
        "prefix_replay": {"budget4_exact": True, "budget6_exact": True},
        "curve": curve,
        "marginal_transitions": marginal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("direct", "probe"):
        for horizon in (4, 6, 8):
            parser.add_argument(f"--{arm}-budget{horizon}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "arms": {
            arm: _arm(
                arm,
                _read_jsonl(getattr(args, f"{arm}_budget4")),
                _read_jsonl(getattr(args, f"{arm}_budget6")),
                _read_jsonl(getattr(args, f"{arm}_budget8")),
                2026080180 + offset * 10,
            )
            for offset, arm in enumerate(("direct", "probe"))
        },
        "interpretation_guard": (
            "Budget-five and budget-seven points are exact prefixes of the matched "
            "budget-eight executions. This remains post-freeze development-only evidence."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
