#!/usr/bin/env python3
"""Export deduplicated inverse-grid replays as group-split forward transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from optics_understanding_sft.core import read_jsonl, write_jsonl

STATE_KEYS = ("centroid_x_px", "centroid_y_px", "sigma_x_px", "sigma_y_px", "peak_intensity")
DIRECTION_KEYS = ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def direction(value: float, tolerance: float) -> str:
    return "increase" if value > tolerance else "decrease" if value < -tolerance else "no_change"


def records_for_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    public = [row for row in read_jsonl(data_dir / "inverse/canonical" / f"{split}.jsonl")
              if row["task_type"] == "inverse_action_numeric"]
    public_by_pair = {row["match_group_id"]: row for row in public}
    replay = read_jsonl(data_dir / "inverse/private" / f"{split}_replay.jsonl")
    representative: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for private in replay:
        representative.setdefault(private["group_id"], (private, public_by_pair[private["pair_id"]]))
    rows = []
    for group_id, (private, visible) in sorted(representative.items()):
        inputs = visible["prompt_inputs"]
        current = inputs["current_beam_state_A"]
        for index, (action, state) in enumerate(zip(inputs["action_grid"], private["candidate_states"], strict=True)):
            change = {key: float(state[key]) - float(current[key]) for key in STATE_KEYS}
            peak_tolerance = 0.05 * max(abs(float(current["peak_intensity"])), 1e-12)
            tolerances = (1.0, 1.0, 2.0, 2.0, peak_tolerance)
            directions = {name: direction(change[key], tolerance)
                          for name, key, tolerance in zip(DIRECTION_KEYS, STATE_KEYS, tolerances, strict=True)}
            rows.append({
                "distribution": visible.get("distribution", "iid"),
                "example_id": f"gridv1_{group_id}_{index:02d}", "group_id": group_id,
                "inputs": {"setup": inputs["setup"], "current_beam_state": current, "action": action},
                "ood_parameter": visible.get("ood_parameter"),
                "target": {"after_state": state, "change": change, "directions": directions},
            })
    return rows


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {split: records_for_split(args.data_dir, split)
              for split in ("train", "val", "eval_iid", "eval_ood")}
    groups = {split: {row["group_id"] for row in rows} for split, rows in splits.items()}
    overlap = {f"{a}:{b}": len(groups[a] & groups[b]) for i, a in enumerate(groups)
               for b in list(groups)[i + 1:]}
    if any(overlap.values()):
        raise RuntimeError(f"group leakage: {overlap}")
    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    audit = {"version": "direction_inverse_v1_grid_transitions", "passed": True,
             "simulator_at_inference": False, "record_counts": {s: len(r) for s, r in splits.items()},
             "group_counts": {s: len(g) for s, g in groups.items()}, "group_overlap": overlap,
             "actions_per_group": 81}
    (args.output_dir / "audit_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
