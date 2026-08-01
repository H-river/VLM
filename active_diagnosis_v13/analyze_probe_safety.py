#!/usr/bin/env python3
"""Report transient, cumulative, and final beam disturbance for every probe."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import normalized_distance


VERSION = "active_diagnosis_v13_probe_transient_safety_v1"


def _rows(root: Path) -> list[dict[str, Any]]:
    by_id = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                by_id[row["record_id"]] = row
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    grouped: defaultdict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(root):
        initial = row["step_records"][0]["visible_input"]["observed_metrics"]
        disturbances = [
            normalized_distance(step["observed_metrics"], initial, initial)
            for step in row["step_records"]
        ]
        grouped[(str(row["design"]), float(row["fraction"]))].append(
            {
                "record_id": row["record_id"],
                "peak": max(disturbances),
                "cumulative": sum(disturbances),
                "final": disturbances[-1],
                "saturation_count": int(row["saturation_count"]),
                "constraint_violation_count": int(row["constraint_violation_count"]),
                "all_simulations_valid": bool(row["all_simulations_valid"]),
            }
        )
    cells = []
    for (design, fraction), rows in sorted(grouped.items()):
        peak = np.asarray([row["peak"] for row in rows], dtype=np.float64)
        cumulative = np.asarray([row["cumulative"] for row in rows], dtype=np.float64)
        final = np.asarray([row["final"] for row in rows], dtype=np.float64)
        cells.append(
            {
                "design": design,
                "fraction": fraction,
                "records": len(rows),
                "mean_peak_beam_state_disturbance": float(peak.mean()),
                "p95_peak_beam_state_disturbance": float(np.quantile(peak, 0.95)),
                "maximum_peak_beam_state_disturbance": float(peak.max()),
                "mean_cumulative_beam_state_disturbance": float(cumulative.mean()),
                "mean_final_beam_state_disturbance": float(final.mean()),
                "saturation_episode_rate": float(
                    np.mean([row["saturation_count"] > 0 for row in rows])
                ),
                "constraint_violations": int(
                    sum(row["constraint_violation_count"] for row in rows)
                ),
                "invalid_simulations": int(
                    sum(not row["all_simulations_valid"] for row in rows)
                ),
            }
        )
    report: dict[str, Any] = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "disturbance_units": "per-episode tolerance-normalized L-infinity distance from the pre-probe beam state",
        "cells": cells,
    }
    output = root / "probes" / "probe_safety_summary.json"
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
