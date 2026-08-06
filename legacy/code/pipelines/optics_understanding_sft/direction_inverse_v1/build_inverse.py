#!/usr/bin/env python3
"""Build ambiguity-aware numeric and visual A-to-B inverse-action records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from optics_sft.physics.rendering import intensity_to_uint8_image
from optics_understanding_sft.build_dataset import simulator_result
from optics_understanding_sft.core import load_yaml, read_jsonl, stable_json_hash, write_jsonl
from optics_understanding_sft.hybrid_direction_magnitude_v1 import full_setup
from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml


ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
STATE_FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
STATUSES = ("unique", "ambiguous", "infeasible_within_limits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-groups", type=int, help="Per-split smoke/debug cap")
    return parser.parse_args()


def rounded_state(state: Mapping[str, Any]) -> dict[str, float]:
    return {field: round(float(state[field]), 6) for field in STATE_FIELDS}


def action_grid(lens_values: Sequence[Any], camera_values: Sequence[Any]) -> list[dict[str, float]]:
    return [
        dict(zip(ACTION_FIELDS, map(float, values)))
        for values in itertools.product(lens_values, lens_values, camera_values, camera_values)
    ]


def action_direction(action: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for field in ACTION_FIELDS:
        value = float(action[field])
        result[field.removesuffix("_delta_mm")] = (
            "increase" if value > 0 else "decrease" if value < 0 else "no_change"
        )
    return result


def state_errors(
    candidate: Mapping[str, Any], target: Mapping[str, Any], tolerance: Mapping[str, Any]
) -> dict[str, float]:
    peak_scale = max(abs(float(target["peak_intensity"])), 1e-12)
    return {
        "centroid_vector_px": math.hypot(
            float(candidate["centroid_x_px"]) - float(target["centroid_x_px"]),
            float(candidate["centroid_y_px"]) - float(target["centroid_y_px"]),
        ),
        "width_x_px": abs(float(candidate["sigma_x_px"]) - float(target["sigma_x_px"])),
        "width_y_px": abs(float(candidate["sigma_y_px"]) - float(target["sigma_y_px"])),
        "peak_relative": abs(
            float(candidate["peak_intensity"]) - float(target["peak_intensity"])
        )
        / peak_scale,
    }


def state_matches(
    candidate: Mapping[str, Any], target: Mapping[str, Any], tolerance: Mapping[str, Any]
) -> bool:
    errors = state_errors(candidate, target, tolerance)
    return (
        errors["centroid_vector_px"] <= float(tolerance["centroid_vector_px"])
        and errors["width_x_px"] <= float(tolerance["width_each_px"])
        and errors["width_y_px"] <= float(tolerance["width_each_px"])
        and errors["peak_relative"] <= float(tolerance["peak_relative"])
    )


def normalized_residual(
    candidate: Mapping[str, Any], target: Mapping[str, Any], tolerance: Mapping[str, Any]
) -> float:
    error = state_errors(candidate, target, tolerance)
    values = (
        error["centroid_vector_px"] / float(tolerance["centroid_vector_px"]),
        error["width_x_px"] / float(tolerance["width_each_px"]),
        error["width_y_px"] / float(tolerance["width_each_px"]),
        error["peak_relative"] / float(tolerance["peak_relative"]),
    )
    return math.sqrt(sum(value * value for value in values) / len(values))


def movement_mm(action: Mapping[str, Any]) -> float:
    return sum(abs(float(action[field])) for field in ACTION_FIELDS)


def matching_indices(
    states: Sequence[Mapping[str, Any]], target: Mapping[str, Any], tolerance: Mapping[str, Any]
) -> list[int]:
    return [index for index, state in enumerate(states) if state_matches(state, target, tolerance)]


def select_minimum_motion(
    actions: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    matches: Sequence[int],
    tolerance: Mapping[str, Any],
) -> int | None:
    if not matches:
        return None
    return min(
        matches,
        key=lambda index: (
            movement_mm(actions[index]),
            normalized_residual(states[index], target, tolerance),
            index,
        ),
    )


def config_from_visible(setup: Mapping[str, Any], base_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg["source"]["wavelength"] = float(setup["wavelength_nm"]) * 1e-9
    cfg["source"]["beam_waist"] = float(setup["beam_waist_mm"]) * 1e-3
    cfg["source"]["power"] = float(setup["power_w"])
    cfg["lens"]["focal_length"] = float(setup["lens_focal_length_mm"]) * 1e-3
    cfg["lens"]["clear_aperture"] = float(setup["lens_aperture_mm"]) * 1e-3
    cfg["geometry"]["laser_to_lens"] = float(setup["source_to_lens_mm"]) * 1e-3
    cfg["geometry"]["lens_to_camera"] = float(setup["lens_to_camera_mm"]) * 1e-3
    cfg["lens"]["x_offset"] = float(setup["lens_x_offset_mm"]) * 1e-3
    cfg["lens"]["y_offset"] = float(setup["lens_y_offset_mm"]) * 1e-3
    cfg["camera"]["x_offset"] = float(setup["camera_x_offset_mm"]) * 1e-3
    cfg["camera"]["y_offset"] = float(setup["camera_y_offset_mm"]) * 1e-3
    cfg["sensor"]["pixel_pitch"] = float(setup["pixel_size_um"]) * 1e-6
    cfg["sensor"]["resolution"] = [int(value) for value in setup["sensor_resolution_px"]]
    return cfg


def load_scenarios(cfg: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sources = cfg["sources"]
    pilot = read_jsonl(Path(sources["pilot_root"]) / "master/cases.jsonl")
    dev = read_jsonl(Path(sources["iid_eval_root"]) / "master/cases.jsonl")
    eval_iid_ids = {
        row["group_id"]
        for row in read_jsonl(Path(sources["transition_root"]) / "eval_iid.jsonl")
    }
    base_cfg = load_sim_yaml(Path("optical_sim/configs/base_config.yaml"))
    ood_rows = read_jsonl(Path(sources["ood_eval_jsonl"]))
    result = {
        "train": [
            {"group_id": row["group_id"], "config": row["setup_config"], "distribution": "iid"}
            for row in pilot
            if row["split"] == "train"
        ],
        "val": [
            {"group_id": row["group_id"], "config": row["setup_config"], "distribution": "iid"}
            for row in pilot
            if row["split"] == "val"
        ],
        "eval_iid": [
            {"group_id": row["group_id"], "config": row["setup_config"], "distribution": "iid"}
            for row in dev
            if row["group_id"] in eval_iid_ids
        ],
        "eval_ood": [
            {
                "group_id": row["group_id"],
                "config": config_from_visible(row["inputs"]["setup"], base_cfg),
                "distribution": "ood",
                "ood_parameter": row.get("ood_parameter"),
            }
            for row in ood_rows
        ],
    }
    return result


def nontrivial(before: Mapping[str, Any], target: Mapping[str, Any], tolerance: Mapping[str, Any]) -> bool:
    return not state_matches(before, target, tolerance)


def stable_pick(values: Sequence[int], token: str) -> int | None:
    if not values:
        return None
    ordered = sorted(
        values,
        key=lambda value: hashlib.sha256(f"{token}:{value}".encode()).hexdigest(),
    )
    return ordered[0]


def off_grid_actions(group_id: str, seed: int, count: int = 24) -> list[dict[str, float]]:
    token = hashlib.sha256(f"{seed}:{group_id}:infeasible".encode()).digest()
    rng = random.Random(int.from_bytes(token[:8], "big"))
    actions = []
    for index in range(count):
        scale = 1.15 + 0.35 * (index % 3)
        actions.append(
            {
                "lens_x_delta_mm": round(rng.uniform(-0.05 * scale, 0.05 * scale), 6),
                "lens_y_delta_mm": round(rng.uniform(-0.05 * scale, 0.05 * scale), 6),
                "camera_x_delta_mm": round(rng.uniform(-0.02 * scale, 0.02 * scale), 6),
                "camera_y_delta_mm": round(rng.uniform(-0.02 * scale, 0.02 * scale), 6),
            }
        )
    return actions


def render_pair(
    before_result: Mapping[str, Any],
    target_result: Mapping[str, Any],
    before_path: Path,
    target_path: Path,
    size_px: int,
) -> dict[str, Any]:
    before_array = np.asarray(before_result["intensity"], dtype=np.float64)
    target_array = np.asarray(target_result["intensity"], dtype=np.float64)
    # Preserve absolute peak and tail information needed by quantitative image
    # metrology. A square-root display gamma retains faint Gaussian tails in an
    # 8-bit PNG while remaining exactly invertible from the declared calibration.
    high = max(float(before_array.max()), float(target_array.max()), 1e-12)
    options = {
        "normalize": False,
        "normalization_bounds": [0.0, high],
        "gamma": 0.5,
        "background": 0.0,
        "read_noise_std": 0.0,
        "blur_sigma_px": 0.0,
        "saturation": 0.0,
    }
    for result, path in ((before_result, before_path), (target_result, target_path)):
        image = intensity_to_uint8_image(result["intensity"], options)
        image = image.resize((size_px, size_px), Image.Resampling.LANCZOS)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG", optimize=True)
    return {"linear_intensity_low": 0.0, "linear_intensity_high": high, "gamma": 0.5}


def public_target(
    status: str,
    actions: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    target_state: Mapping[str, Any],
    matches: Sequence[int],
    selected: int | None,
    tolerance: Mapping[str, Any],
) -> dict[str, Any]:
    if selected is None:
        best = min(
            range(len(states)),
            key=lambda index: (normalized_residual(states[index], target_state, tolerance), movement_mm(actions[index]), index),
        )
        return {
            "status": status,
            "answer": {
                "matching_actions": [],
                "selected_minimum_motion_action": None,
                "action_directions": None,
                "best_grid_action": dict(actions[best]),
                "best_normalized_residual": round(normalized_residual(states[best], target_state, tolerance), 6),
            },
        }
    return {
        "status": status,
        "answer": {
            "matching_actions": [dict(actions[index]) for index in matches],
            "selected_minimum_motion_action": dict(actions[selected]),
            "action_directions": action_direction(actions[selected]),
            "selected_normalized_residual": round(
                normalized_residual(states[selected], target_state, tolerance), 6
            ),
        },
    }


def inverse_prompt(inputs: Mapping[str, Any], visual: bool) -> str:
    visible = {key: value for key, value in inputs.items() if key != "images"}
    observation = (
        "The first image is current beam A and the second image is desired beam B."
        if visual
        else "The current and desired calibrated beam states are provided numerically."
    )
    contract = {
        "status": "unique | ambiguous | infeasible_within_limits",
        "answer": {
            "matching_actions": ["zero or more complete four-actuator action objects"],
            "selected_minimum_motion_action": "action object | null",
            "action_directions": "four direction fields | null",
            "selected_normalized_residual": "number | null",
            "best_grid_action": "action object | null",
            "best_normalized_residual": "number | null",
        },
    }
    return (
        "Determine how to change the optical setup so that current beam A becomes desired beam B. "
        "Evaluate only the declared discrete action grid. Return every matching action and select the matching "
        "action with minimum total absolute motion. If none matches, report infeasible_within_limits and the "
        f"best grid action. {observation}\n\nInput data:\n{json.dumps(visible, indent=2, sort_keys=True)}\n\n"
        "Return only strict JSON matching this contract:\n"
        f"{json.dumps(contract, indent=2, sort_keys=True)}"
    )


def _process_scenario(job: Mapping[str, Any]) -> dict[str, Any]:
    group_id = str(job["group_id"])
    split = str(job["split"])
    config = job["config"]
    actions = job["actions"]
    tolerance = job["tolerance"]
    seed = int(job["seed"])
    output_dir = Path(job["output_dir"])
    size_px = int(job["image_size_px"])

    before_result = simulator_result(config)
    before_state = rounded_state(before_result["state"])
    grid_states = [rounded_state(simulator_result(config, action)["state"]) for action in actions]
    zero_index = next(
        index for index, action in enumerate(actions) if all(float(action[field]) == 0.0 for field in ACTION_FIELDS)
    )

    by_status: dict[str, dict[str, Any]] = {}
    match_sets = [matching_indices(grid_states, target, tolerance) for target in grid_states]
    unique_candidates = [
        index
        for index, matches in enumerate(match_sets)
        if len(matches) == 1 and index != zero_index and nontrivial(before_state, grid_states[index], tolerance)
    ]
    ambiguous_candidates = []
    for index, matches in enumerate(match_sets):
        if len(matches) <= 1 or not nontrivial(before_state, grid_states[index], tolerance):
            continue
        selected = select_minimum_motion(actions, grid_states, grid_states[index], matches, tolerance)
        if selected is not None and selected != zero_index:
            ambiguous_candidates.append(index)

    for status, candidates in (("unique", unique_candidates), ("ambiguous", ambiguous_candidates)):
        chosen = stable_pick(candidates, f"{seed}:{group_id}:{status}")
        if chosen is None:
            continue
        target_state = grid_states[chosen]
        matches = match_sets[chosen]
        selected = select_minimum_motion(actions, grid_states, target_state, matches, tolerance)
        by_status[status] = {
            "source_action": actions[chosen],
            "target_state": target_state,
            "matching_indices": matches,
            "selected_index": selected,
        }

    for candidate_action in off_grid_actions(group_id, seed):
        result = simulator_result(config, candidate_action)
        target_state = rounded_state(result["state"])
        matches = matching_indices(grid_states, target_state, tolerance)
        if not matches and nontrivial(before_state, target_state, tolerance):
            by_status["infeasible_within_limits"] = {
                "source_action": candidate_action,
                "target_state": target_state,
                "matching_indices": [],
                "selected_index": None,
            }
            break

    pairs = []
    for status, item in by_status.items():
        target_result = simulator_result(config, item["source_action"])
        stem = f"{group_id}_{status}"
        before_rel = Path("images") / split / f"{stem}_A.png"
        target_rel = Path("images") / split / f"{stem}_B.png"
        calibration = render_pair(
            before_result,
            target_result,
            output_dir / before_rel,
            output_dir / target_rel,
            size_px,
        )
        target = public_target(
            status,
            actions,
            grid_states,
            item["target_state"],
            item["matching_indices"],
            item["selected_index"],
            tolerance,
        )
        pairs.append(
            {
                "pair_id": f"invv1_{stem}",
                "group_id": group_id,
                "split": split,
                "distribution": job["distribution"],
                "ood_parameter": job.get("ood_parameter"),
                "setup": full_setup(config),
                "before_state": before_state,
                "target_state": item["target_state"],
                "images": [before_rel.as_posix(), target_rel.as_posix()],
                "image_calibration": {
                    "rendered_size_px": [size_px, size_px],
                    "source_sensor_resolution_px": list(config["sensor"]["resolution"]),
                    **calibration,
                },
                "action_grid": [dict(action) for action in actions],
                "matching_tolerance": dict(tolerance),
                "target": target,
                "private": {
                    "setup_config": config,
                    "source_target_action": item["source_action"],
                    "candidate_states": grid_states,
                    "target_state": item["target_state"],
                    "matching_indices": item["matching_indices"],
                    "selected_index": item["selected_index"],
                },
            }
        )
    return {"group_id": group_id, "split": split, "pairs": pairs}


def make_record(pair: Mapping[str, Any], visual: bool) -> dict[str, Any]:
    task_type = "inverse_action_visual" if visual else "inverse_action_numeric"
    inputs: dict[str, Any] = {
        "setup": pair["setup"],
        "action_grid": pair["action_grid"],
        "matching_tolerance": pair["matching_tolerance"],
        "images": pair["images"] if visual else [],
    }
    if visual:
        inputs["observation_format"] = "image A current beam, followed by image B desired beam"
        inputs["image_calibration"] = pair["image_calibration"]
    else:
        inputs["current_beam_state_A"] = pair["before_state"]
        inputs["desired_beam_state_B"] = pair["target_state"]
    return {
        "example_id": f"{pair['pair_id']}_{'visual' if visual else 'numeric'}",
        "match_group_id": pair["pair_id"],
        "group_id": pair["group_id"],
        "split": pair["split"],
        "distribution": pair["distribution"],
        "ood_parameter": pair.get("ood_parameter"),
        "task_type": task_type,
        "modality": "visual" if visual else "text",
        "prompt_inputs": inputs,
        "prompt": inverse_prompt(inputs, visual),
        "target": pair["target"],
    }


def balanced_pairs(results: list[dict[str, Any]], split: str, seed: int) -> list[dict[str, Any]]:
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for pair in result["pairs"]:
            by_status[pair["target"]["status"]].append(pair)
    quota = min(len(by_status[status]) for status in STATUSES)
    selected = []
    for status in STATUSES:
        ordered = sorted(
            by_status[status],
            key=lambda pair: hashlib.sha256(
                f"{seed}:{split}:{status}:{pair['pair_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[:quota])
    random.Random(seed + len(split)).shuffle(selected)
    return selected


def qwen_export(record: Mapping[str, Any]) -> dict[str, Any]:
    images = list(record["prompt_inputs"]["images"])
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": record["prompt"]})
    return {
        "example_id": record["example_id"],
        "group_id": record["group_id"],
        "task_type": record["task_type"],
        "images": images,
        "prompt": [{"role": "user", "content": content}],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(record["target"], sort_keys=False)}],
            }
        ],
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    inv_cfg = cfg["inverse"]
    actions = action_grid(inv_cfg["grid"]["lens_mm"], inv_cfg["grid"]["camera_mm"])
    if len(actions) != 81:
        raise ValueError(f"expected a 3^4=81 action grid, got {len(actions)}")
    scenarios = load_scenarios(cfg)
    if args.max_groups is not None:
        scenarios = {split: rows[: args.max_groups] for split, rows in scenarios.items()}
    jobs = [
        {
            **scenario,
            "split": split,
            "actions": actions,
            "tolerance": inv_cfg["matching_tolerance"],
            "seed": seed,
            "output_dir": str(args.output_dir),
            "image_size_px": int(inv_cfg["image_size_px"]),
        }
        for split, rows in scenarios.items()
        for scenario in rows
    ]
    if args.workers <= 1:
        processed = [_process_scenario(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            processed = list(pool.map(_process_scenario, jobs, chunksize=1))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in processed:
        by_split[result["split"]].append(result)

    selected_pairs = {
        split: balanced_pairs(results, split, seed) for split, results in by_split.items()
    }
    public_records = {
        split: [make_record(pair, visual) for pair in pairs for visual in (False, True)]
        for split, pairs in selected_pairs.items()
    }
    for split, records in public_records.items():
        write_jsonl(args.output_dir / "inverse/canonical" / f"{split}.jsonl", records)
        write_jsonl(args.output_dir / "exports/qwen" / f"inverse_{split}.jsonl", map(qwen_export, records))
    for split, pairs in selected_pairs.items():
        write_jsonl(
            args.output_dir / "inverse/private" / f"{split}_replay.jsonl",
            (
                {
                    "pair_id": pair["pair_id"],
                    "group_id": pair["group_id"],
                    "split": pair["split"],
                    **pair["private"],
                }
                for pair in pairs
            ),
        )

    group_sets = {split: {row["group_id"] for row in rows} for split, rows in scenarios.items()}
    overlap = {
        f"{a}:{b}": len(group_sets[a] & group_sets[b])
        for i, a in enumerate(group_sets)
        for b in list(group_sets)[i + 1 :]
    }
    image_paths = [
        args.output_dir / image
        for records in public_records.values()
        for record in records
        if record["modality"] == "visual"
        for image in record["prompt_inputs"]["images"]
    ]
    image_errors = []
    for path in image_paths:
        if not path.exists():
            image_errors.append(f"missing:{path}")
            continue
        with Image.open(path) as image:
            if image.size != (int(inv_cfg["image_size_px"]), int(inv_cfg["image_size_px"])):
                image_errors.append(f"size:{path}:{image.size}")
    status_counts = {
        split: dict(Counter(pair["target"]["status"] for pair in pairs))
        for split, pairs in selected_pairs.items()
    }
    source_availability = {
        split: dict(
            Counter(pair["target"]["status"] for result in results for pair in result["pairs"])
        )
        for split, results in by_split.items()
    }
    forbidden = ("setup_state_handle", "candidate_states", "source_target_action", "selected_index")
    prompt_hits = {
        token: sum(token in record["prompt"] for records in public_records.values() for record in records)
        for token in forbidden
    }
    minimums = inv_cfg.get("minimum_selected_pairs_per_status", {})
    minimum_failures = {}
    if args.max_groups is None:
        for split, minimum in minimums.items():
            for status in STATUSES:
                actual = status_counts.get(split, {}).get(status, 0)
                if actual < int(minimum):
                    minimum_failures[f"{split}:{status}"] = {"actual": actual, "required": int(minimum)}
    passed = (
        not any(overlap.values())
        and not image_errors
        and not any(prompt_hits.values())
        and not minimum_failures
    )
    audit = {
        "version": cfg["version"],
        "seed": seed,
        "passed": passed,
        "simulator_role": "offline label generation and replay audit only",
        "action_grid_size": len(actions),
        "scenario_counts": {split: len(rows) for split, rows in scenarios.items()},
        "source_status_availability": source_availability,
        "selected_pair_counts": {split: len(rows) for split, rows in selected_pairs.items()},
        "selected_status_counts": status_counts,
        "public_record_counts": {split: len(rows) for split, rows in public_records.items()},
        "group_overlap": overlap,
        "image_reference_count": len(image_paths),
        "image_errors": image_errors,
        "prompt_forbidden_token_hits": prompt_hits,
        "minimum_status_count_failures": minimum_failures,
        "record_hashes": {split: stable_json_hash(rows) for split, rows in public_records.items()},
    }
    (args.output_dir / "inverse/audit_report.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inverse/audit_report.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("inverse dataset audit failed")


if __name__ == "__main__":
    main()
