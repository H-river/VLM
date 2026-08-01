#!/usr/bin/env python3
"""Repair visual inverse labels into the camera-sensor frame and rerender losslessly."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from optics_understanding_sft.core import read_jsonl, stable_json_hash, write_jsonl
from optics_understanding_sft.direction_inverse_v1.build_inverse import (
    inverse_prompt, matching_indices, public_target, qwen_export, render_pair,
    select_minimum_motion,
)
from optics_understanding_sft.build_dataset import simulator_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-rerender", action="store_true")
    return parser.parse_args()


def sensor_state(state: Mapping[str, Any], setup: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, float]:
    pitch_mm = float(setup["pixel_size_um"]) / 1000.0
    result = {key: float(value) for key, value in state.items()}
    result["centroid_x_px"] -= (
        float(setup["camera_x_offset_mm"]) + float(action["camera_x_delta_mm"])
    ) / pitch_mm
    result["centroid_y_px"] -= (
        float(setup["camera_y_offset_mm"]) + float(action["camera_y_delta_mm"])
    ) / pitch_mm
    return {key: round(value, 6) for key, value in result.items()}


def zero_action() -> dict[str, float]:
    return {"lens_x_delta_mm": 0.0, "lens_y_delta_mm": 0.0,
            "camera_x_delta_mm": 0.0, "camera_y_delta_mm": 0.0}


def _rerender(job: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    before = simulator_result(job["setup_config"])
    target = simulator_result(job["setup_config"], job["source_target_action"])
    calibration = render_pair(before, target, Path(job["before_path"]), Path(job["target_path"]),
                              int(job["size_px"]))
    return str(job["pair_id"]), calibration


def repair_split(data_dir: Path, split: str, workers: int, rerender: bool = True) -> dict[str, Any]:
    canonical_path = data_dir / "inverse/canonical" / f"{split}.jsonl"
    rows = read_jsonl(canonical_path)
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["match_group_id"], {})[row["modality"]] = row
    replay_rows = read_jsonl(data_dir / "inverse/private" / f"{split}_replay.jsonl")
    replay = {row["pair_id"]: row for row in replay_rows}
    jobs = []
    for pair_id, private in replay.items():
        visual = by_pair[pair_id]["visual"]
        images = visual["prompt_inputs"]["images"]
        jobs.append({"pair_id": pair_id, "setup_config": private["setup_config"],
                     "source_target_action": private["source_target_action"],
                     "before_path": str(data_dir / images[0]), "target_path": str(data_dir / images[1]),
                     "size_px": visual["prompt_inputs"]["image_calibration"]["rendered_size_px"][0]})
    if not rerender:
        calibration = {
            pair_id: {
                key: by_pair[pair_id]["visual"]["prompt_inputs"]["image_calibration"][key]
                for key in ("linear_intensity_low", "linear_intensity_high", "gamma")
            }
            for pair_id in replay
        }
    elif workers <= 1:
        calibration = dict(_rerender(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            calibration = dict(pool.map(_rerender, jobs, chunksize=1))

    visual_private = []
    transitions = Counter()
    for pair_id, private in replay.items():
        numeric = by_pair[pair_id]["text"]
        visual = by_pair[pair_id]["visual"]
        inputs = numeric["prompt_inputs"]
        setup, actions, tolerance = inputs["setup"], inputs["action_grid"], inputs["matching_tolerance"]
        candidates = [sensor_state(state, setup, action)
                      for state, action in zip(private["candidate_states"], actions, strict=True)]
        before = sensor_state(inputs["current_beam_state_A"], setup, zero_action())
        target = sensor_state(private["target_state"], setup, private["source_target_action"])
        matches = matching_indices(candidates, target, tolerance)
        selected = select_minimum_motion(actions, candidates, target, matches, tolerance)
        status = "unique" if len(matches) == 1 else "ambiguous" if matches else "infeasible_within_limits"
        transitions[(visual["target"]["status"], status)] += 1
        visual["target"] = public_target(status, actions, candidates, target, matches, selected, tolerance)
        visual_inputs = visual["prompt_inputs"]
        visual_inputs["coordinate_frame"] = "camera_sensor_array"
        visual_inputs["image_calibration"].update(calibration[pair_id])
        visual_inputs["image_calibration"]["coordinate_frame"] = "camera_sensor_array"
        visual["prompt"] = inverse_prompt(visual_inputs, True)
        visual_private.append({"pair_id": pair_id, "group_id": private["group_id"], "split": split,
                               "setup_config": private["setup_config"],
                               "source_target_action": private["source_target_action"],
                               "before_state": before, "candidate_states": candidates,
                               "target_state": target, "matching_indices": matches,
                               "selected_index": selected})
    write_jsonl(canonical_path, rows)
    write_jsonl(data_dir / "exports/qwen" / f"inverse_{split}.jsonl", map(qwen_export, rows))
    write_jsonl(data_dir / "inverse/private" / f"{split}_visual_replay.jsonl", visual_private)
    visual_counts = Counter(by_pair[pair]["visual"]["target"]["status"] for pair in by_pair)
    numeric_counts = Counter(by_pair[pair]["text"]["target"]["status"] for pair in by_pair)
    return {"records": len(rows), "pairs": len(by_pair), "numeric_status_counts": dict(numeric_counts),
            "visual_sensor_status_counts": dict(visual_counts),
            "status_transitions": {f"{a}->{b}": count for (a, b), count in transitions.items()},
            "visual_private_records": len(visual_private)}


def main() -> None:
    args = parse_args()
    summary = {split: repair_split(args.data_dir, split, args.workers, not args.skip_rerender)
               for split in ("train", "val", "eval_iid", "eval_ood")}
    report = {"version": "direction_inverse_v1_visual_sensor_frame_repair",
              "passed": True, "coordinate_frame": "camera_sensor_array",
              "rendering": "shared absolute maximum, invertible gamma 0.5, RGB8 PNG",
              "simulator_role": "offline rerender only", "splits": summary}
    path = args.data_dir / "inverse/visual_sensor_frame_audit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    main_audit_path = args.data_dir / "inverse/audit_report.json"
    main_audit = json.loads(main_audit_path.read_text(encoding="utf-8"))
    main_audit["record_hashes"] = {
        split: stable_json_hash(read_jsonl(args.data_dir / "inverse/canonical" / f"{split}.jsonl"))
        for split in ("train", "val", "eval_iid", "eval_ood")
    }
    main_audit["visual_sensor_frame_audit"] = {
        "passed": True, "path": "inverse/visual_sensor_frame_audit.json",
        "coordinate_frame": "camera_sensor_array",
    }
    main_audit_path.write_text(json.dumps(main_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
