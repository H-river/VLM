#!/usr/bin/env python3
"""Audit structural integrity, leakage resistance, and simulator replay."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import simulate_and_measure

from .core import (
    ACTION_KEYS,
    apply_action_dict,
    assert_finite_tree,
    centroid_distance,
    prompt_key_hits,
    read_jsonl,
)


MULTI_STATUS_TASKS = frozenset(
    {"information_sufficiency", "diagnosis", "constrained_intervention"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def _load_canonical(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": read_jsonl(dataset_dir / "canonical" / "train.jsonl"),
        "val": read_jsonl(dataset_dir / "canonical" / "val.jsonl"),
        "test": read_jsonl(dataset_dir / "canonical" / "test_prompts.jsonl"),
    }


def _test_targets(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(dataset_dir / "private" / "test_labels.jsonl")}


def _record_target(record: Mapping[str, Any], test_targets: Mapping[str, Any]) -> Mapping[str, Any]:
    target = record.get("target")
    if isinstance(target, Mapping):
        return target
    return test_targets[record["example_id"]]["target"]


def _visual_checks(dataset_dir: Path, rows: Mapping[str, list[dict[str, Any]]]) -> tuple[list[str], int]:
    failures: list[str] = []
    seen: set[str] = set()
    count = 0
    for split_rows in rows.values():
        for row in split_rows:
            images = row["prompt_inputs"].get("images", [])
            if row["modality"] == "visual":
                count += 1
                if not images:
                    failures.append(f"visual record has no images: {row['example_id']}")
            elif images:
                failures.append(f"text record unexpectedly has images: {row['example_id']}")
            for relative in images:
                if relative in seen:
                    failures.append(f"image reused across records: {relative}")
                seen.add(relative)
                path = dataset_dir / relative
                if not path.exists():
                    failures.append(f"missing image: {relative}")
                    continue
                with Image.open(path) as image:
                    if image.size != (384, 384):
                        failures.append(f"wrong image size {image.size}: {relative}")
    return failures, count


def _split_overlap(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, int]:
    groups = {split: {row["group_id"] for row in values} for split, values in rows.items()}
    ids = {split: {row["example_id"] for row in values} for split, values in rows.items()}
    return {
        "train_val_groups": len(groups["train"] & groups["val"]),
        "train_test_groups": len(groups["train"] & groups["test"]),
        "val_test_groups": len(groups["val"] & groups["test"]),
        "train_val_ids": len(ids["train"] & ids["val"]),
        "train_test_ids": len(ids["train"] & ids["test"]),
        "val_test_ids": len(ids["val"] & ids["test"]),
    }


def _match_group_checks(rows: Mapping[str, list[dict[str, Any]]]) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    match_splits: dict[str, set[str]] = defaultdict(set)
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, split_rows in rows.items():
        for row in split_rows:
            match_id = row.get("provenance", {}).get("match_group_id")
            if match_id is None:
                continue
            match_id = str(match_id)
            match_splits[match_id].add(split)
            members[match_id].append(row)
    for match_id, splits in match_splits.items():
        if len(splits) != 1:
            failures.append(f"match group crosses splits: {match_id} -> {sorted(splits)}")
    expected_size = {
        "information_sufficiency": 2,
        "constrained_intervention": 2,
        "diagnosis": 3,
    }
    for match_id, group_rows in members.items():
        tasks = {row["task_type"] for row in group_rows}
        if len(tasks) != 1:
            failures.append(f"match group mixes tasks: {match_id} -> {sorted(tasks)}")
            continue
        task = next(iter(tasks))
        kinds = {row.get("provenance", {}).get("match_group_kind") for row in group_rows}
        if len(kinds) > 1:
            failures.append(f"match group mixes kinds: {match_id} -> {sorted(map(str, kinds))}")
            continue
        kind = next(iter(kinds), None)
        if kind in {"action_diversity_pair", "action_diversity_triplet"}:
            expected = 2 if kind == "action_diversity_pair" else 3
            if task != "constrained_intervention" or len(group_rows) != expected:
                failures.append(f"invalid {kind}: {match_id} has task={task}, size={len(group_rows)}")
                continue
            feasible = [row for row in group_rows if row["target"]["status"] == "feasible"]
            infeasible = [
                row for row in group_rows if row["target"]["status"] == "infeasible_within_limits"
            ]
            if len(feasible) != 2 or len(infeasible) != expected - 2:
                failures.append(f"invalid status composition for {match_id}")
            modes = {row["provenance"].get("control_action_mode") for row in feasible}
            signs = {row["provenance"].get("control_action_sign") for row in feasible}
            actuators = {
                row["prompt_inputs"]["actuator_constraints"]["active_actuator"] for row in group_rows
            }
            if modes != {"minimum_motion", "near_boundary"}:
                failures.append(f"action pair lacks both movement modes: {match_id} -> {sorted(map(str, modes))}")
            if signs != {"positive", "negative"}:
                failures.append(f"action pair lacks opposite signs: {match_id} -> {sorted(map(str, signs))}")
            if len(actuators) != 1:
                failures.append(f"action pair mixes actuators: {match_id} -> {sorted(actuators)}")
            continue
        if task in expected_size and len(group_rows) != expected_size[task]:
            failures.append(f"wrong match group size: {match_id} has {len(group_rows)}")
        if (
            task in MULTI_STATUS_TASKS
            and all("target" in row for row in group_rows)
            and len({row["target"]["status"] for row in group_rows}) != len(group_rows)
        ):
            failures.append(f"match group statuses are not distinct: {match_id}")
    return failures, {
        "group_count": len(members),
        "record_count": sum(len(group_rows) for group_rows in members.values()),
        "cross_split_count": sum(len(splits) > 1 for splits in match_splits.values()),
    }


def _balance_report(
    rows: Mapping[str, list[dict[str, Any]]], test_targets: Mapping[str, Any]
) -> dict[str, Any]:
    statuses: dict[str, Counter[str]] = defaultdict(Counter)
    causal_labels: Counter[str] = Counter()
    for split_rows in rows.values():
        for row in split_rows:
            target = _record_target(row, test_targets)
            statuses[row["task_type"]][str(target["status"])] += 1
            if row["task_type"] == "causal_effects":
                causal_labels.update(target["answer"]["effects"].values())
    return {
        "statuses": {key: dict(value) for key, value in statuses.items()},
        "causal_direction_labels": dict(causal_labels),
    }


def _counterbalanced_pair_count(
    rows: Mapping[str, list[dict[str, Any]]], test_targets: Mapping[str, Any]
) -> int:
    controls: list[tuple[float, float, str, float]] = []
    for split_rows in rows.values():
        for row in split_rows:
            if row["task_type"] != "constrained_intervention":
                continue
            inputs = row["prompt_inputs"]
            current = inputs["current_observation"]
            target_obs = inputs["target_observation"]
            error = math.hypot(
                float(current["centroid_x_px"]) - float(target_obs["centroid_x_px"]),
                float(current["centroid_y_px"]) - float(target_obs["centroid_y_px"]),
            )
            target = _record_target(row, test_targets)
            plan = target["answer"].get("control_plan")
            if not isinstance(plan, Mapping):
                continue
            actuator = inputs["actuator_constraints"]["active_actuator"]
            value = float(plan[actuator])
            focal = float(inputs["setup"]["lens_focal_length_mm"])
            controls.append((error, value, actuator, focal))
    pairs = 0
    for index, left in enumerate(controls):
        for right in controls[index + 1 :]:
            if left[2] != right[2]:
                continue
            if abs(left[0] - right[0]) <= 1.5 and abs(left[1] - right[1]) >= 0.01 and abs(left[3] - right[3]) >= 5.0:
                pairs += 1
    return pairs


def _fixed_gain_success(dataset_dir: Path, rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    labels = _test_targets(dataset_dir)
    successes = 0
    total = 0
    for row in rows["test"]:
        if row["task_type"] != "constrained_intervention":
            continue
        label = labels[row["example_id"]]
        private = label["private_eval"]
        baseline_spec = private["replay_specs"][0]
        setup = setup_from_dict(copy.deepcopy(baseline_spec["setup_config"]))
        inputs = row["prompt_inputs"]
        actuator = inputs["actuator_constraints"]["active_actuator"]
        allowed = [float(v) for v in inputs["actuator_constraints"]["allowed_values_mm"]]
        current = inputs["current_observation"]
        target = inputs["target_observation"]
        state_axis = "centroid_x_px" if "_x_" in actuator else "centroid_y_px"
        error = float(current[state_axis]) - float(target[state_axis])
        proposed = -0.002 * error
        quantized = min(allowed, key=lambda value: abs(value - proposed))
        action = {key: 0.0 for key in ACTION_KEYS}
        action[actuator] = quantized
        result = simulate_and_measure(apply_action_dict(setup, action))
        if centroid_distance(result["state"], target) <= float(inputs["actuator_constraints"]["success_tolerance_px"]):
            successes += 1
        total += 1
    return {
        "successes": successes,
        "count": total,
        "success_rate": successes / total if total else None,
    }


def _replay_one(job: tuple[str, Mapping[str, Any]]) -> tuple[str, str, float]:
    example_id, spec = job
    setup = setup_from_dict(copy.deepcopy(spec["setup_config"]))
    if "action" in spec:
        setup = apply_action_dict(setup, spec["action"])
    result = simulate_and_measure(setup)["state"]
    expected = spec["expected_state"]
    error = max(abs(float(result[key]) - float(expected[key])) for key in expected)
    return example_id, str(spec["name"]), error


def _replay_master(dataset_dir: Path, workers: int) -> dict[str, Any]:
    masters = read_jsonl(dataset_dir / "master" / "cases.jsonl")
    jobs = [
        (item["record"]["example_id"], spec)
        for master in masters
        for item in master["records"]
        for spec in item["private_eval"].get("replay_specs", [])
    ]
    failures: list[dict[str, Any]] = []
    replay_count = 0
    max_abs_error = 0.0
    worker_count = max(1, int(workers))
    if worker_count == 1:
        results = map(_replay_one, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        results = executor.map(_replay_one, jobs, chunksize=2)
    try:
        for example_id, spec_name, error in results:
            max_abs_error = max(max_abs_error, error)
            replay_count += 1
            if error > 0.001:
                failures.append(
                    {"example_id": example_id, "spec": spec_name, "max_abs_error": error}
                )
    finally:
        if executor is not None:
            executor.shutdown()
    return {
        "count": replay_count,
        "workers": worker_count,
        "max_abs_error": max_abs_error,
        "failures": failures[:20],
        "failure_count": len(failures),
    }


def audit(dataset_dir: Path, *, replay: bool, strict: bool, replay_workers: int = 1) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = _load_canonical(dataset_dir)
    test_targets = _test_targets(dataset_dir)
    failures: list[str] = []

    record_counts = {split: len(values) for split, values in rows.items()}
    visual_counts = {split: sum(row["modality"] == "visual" for row in values) for split, values in rows.items()}
    expected_records = {split: int(manifest["record_counts"].get(split, 0)) for split in rows}
    expected_visual = {split: int(manifest["visual_counts"].get(split, 0)) for split in rows}
    if strict and record_counts != expected_records:
        failures.append(f"record counts differ: {record_counts}")
    if strict and visual_counts != expected_visual:
        failures.append(f"visual counts differ: {visual_counts}")
    if len(test_targets) != record_counts["test"]:
        failures.append("test prompt/label count mismatch")
    if any("target" in row for row in rows["test"]):
        failures.append("public test prompts contain targets")

    overlap = _split_overlap(rows)
    if any(overlap.values()):
        failures.append(f"split overlap detected: {overlap}")

    match_failures, match_report = _match_group_checks(rows)
    failures.extend(match_failures[:20])

    visual_failures, _ = _visual_checks(dataset_dir, rows)
    failures.extend(visual_failures[:20])
    prompt_leaks: list[str] = []
    for split_rows in rows.values():
        for row in split_rows:
            try:
                assert_finite_tree(row)
            except ValueError as exc:
                failures.append(str(exc))
            hits = prompt_key_hits(row["prompt_inputs"])
            if hits:
                prompt_leaks.append(f"{row['example_id']}: {hits}")
            prompt = str(row.get("prompt", ""))
            if "Task output contract (all fields shown; use null when not applicable):" not in prompt:
                prompt_leaks.append(f"{row['example_id']}: missing label-independent output contract")
            if row.get("provenance", {}).get("prompt_contract_version") != 2:
                prompt_leaks.append(f"{row['example_id']}: unexpected prompt contract version")
            if row["task_type"] in MULTI_STATUS_TASKS:
                status = str(_record_target(row, test_targets)["status"])
                if f'"status": "{status}"' in prompt:
                    prompt_leaks.append(f"{row['example_id']}: exact target status leaked into prompt")
    if prompt_leaks:
        failures.extend(prompt_leaks[:20])

    balance = _balance_report(rows, test_targets)
    if strict:
        action_first_focused = manifest.get("version") == "action_first_v3"
        configured_status_counts = manifest.get("configured_status_counts", {})
        for split, task_expectations in configured_status_counts.items():
            split_rows = rows.get(split, [])
            for task, expected in task_expectations.items():
                observed = Counter(
                    _record_target(row, test_targets)["status"]
                    for row in split_rows
                    if row["task_type"] == task
                )
                normalized_expected = Counter({str(key): int(value) for key, value in expected.items()})
                if observed != normalized_expected:
                    failures.append(
                        f"configured status counts differ for {split}/{task}: {dict(observed)} vs {dict(normalized_expected)}"
                    )
        suff = balance["statuses"].get("information_sufficiency", {})
        if not {"answerable", "insufficient_information"}.issubset(suff):
            failures.append(f"sufficiency labels not balanced: {suff}")
        control = balance["statuses"].get("constrained_intervention", {})
        if not {"feasible", "infeasible_within_limits"}.issubset(control):
            failures.append(f"control feasibility labels not balanced: {control}")
        else:
            minimum_control_fraction = 0.25 if action_first_focused else 0.35
            if min(control.values()) / sum(control.values()) < minimum_control_fraction:
                failures.append(
                    f"control feasibility minority class below {minimum_control_fraction:.0%}: {control}"
                )
        diagnosis = balance["statuses"].get("diagnosis", {})
        diagnosis_present = any(
            row["task_type"] == "diagnosis" for split_rows in rows.values() for row in split_rows
        )
        if diagnosis_present and len(diagnosis) < 2:
            failures.append(f"diagnosis status diversity too low: {diagnosis}")
        causal_present = any(
            row["task_type"] == "causal_effects" for split_rows in rows.values() for row in split_rows
        )
        if causal_present and not {"increase", "decrease", "no_change"}.issubset(
            balance["causal_direction_labels"]
        ):
            failures.append(f"causal direction diversity too low: {balance['causal_direction_labels']}")

    counterbalanced_pairs = _counterbalanced_pair_count(rows, test_targets)
    audit_requirements = manifest.get("audit_requirements", {})
    default_pair_minimum = 100 if expected_records["test"] else 0
    required_pairs = int(audit_requirements.get("counterbalanced_pair_minimum", default_pair_minimum))
    if strict and counterbalanced_pairs < required_pairs:
        failures.append(f"only {counterbalanced_pairs} counterbalanced pairs; expected at least {required_pairs}")

    fixed_gain = _fixed_gain_success(dataset_dir, rows) if strict else {"skipped": True}
    if strict and fixed_gain.get("success_rate") is not None and fixed_gain["success_rate"] >= 0.60:
        failures.append(f"fixed-gain success too high: {fixed_gain['success_rate']:.3f}")

    replay_report = _replay_master(dataset_dir, replay_workers) if replay else {"skipped": True}
    if replay and replay_report["failure_count"]:
        failures.append(f"simulator replay failures: {replay_report['failure_count']}")

    return {
        "passed": not failures,
        "failures": failures,
        "dataset": manifest["dataset"],
        "expected_record_counts": expected_records,
        "expected_visual_counts": expected_visual,
        "record_counts": record_counts,
        "visual_counts": visual_counts,
        "split_overlap": overlap,
        "match_groups": match_report,
        "task_counts": {split: dict(Counter(row["task_type"] for row in values)) for split, values in rows.items()},
        "balance": balance,
        "counterbalanced_pair_count": counterbalanced_pairs,
        "counterbalanced_pair_minimum": required_pairs,
        "fixed_gain_baseline": fixed_gain,
        "replay": replay_report,
    }


def main() -> None:
    args = parse_args()
    report = audit(
        args.dataset_dir,
        replay=args.replay,
        strict=not args.no_strict,
        replay_workers=args.replay_workers,
    )
    (args.dataset_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
