#!/usr/bin/env python3
"""Audit balance, minimal-pair integrity, leakage, and simulator replay for v4."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .audit_dataset import _replay_master
from .core import assert_finite_tree, centroid_distance, prompt_key_hits, read_jsonl
from .evaluate import schema_valid


EXPECTED_STATUSES = {
    "constrained_intervention": {"feasible", "infeasible_within_limits"},
    "information_sufficiency": {"answerable", "insufficient_information"},
}
EXPECTED_PROMPT_DELTAS = {
    "constrained_intervention": {"target_observation"},
    "information_sufficiency": {"compatible_hidden_values_mm"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--replay-workers", type=int, default=4)
    parser.add_argument("--skip-replay", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def best_threshold_accuracy(values: list[tuple[float, bool]]) -> float | None:
    if not values:
        return None
    unique = sorted({value for value, _ in values})
    cuts = [unique[0] - 1.0]
    cuts.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    cuts.append(unique[-1] + 1.0)
    return max(
        sum(((value <= cut) if low_is_positive else (value > cut)) == label for value, label in values)
        / len(values)
        for cut in cuts
        for low_is_positive in (True, False)
    )


def private_index(masters: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item["record"]["example_id"]: item["private_eval"]
        for master in masters
        for item in master["records"]
    }


def replay_directions(row: Mapping[str, Any], private: Mapping[str, Any]) -> set[str]:
    current_x = float(row["prompt_inputs"]["current_observation"]["centroid_x_px"])
    result: set[str] = set()
    for spec in private["replay_specs"]:
        delta = float(spec["expected_state"]["centroid_x_px"]) - current_x
        result.add("increase" if delta > 1.0 else "decrease" if delta < -1.0 else "no_change")
    return result


def reference_overlap(
    rows: list[dict[str, Any]], masters: list[dict[str, Any]], reference_dirs: list[Path]
) -> dict[str, Any]:
    example_ids = {row["example_id"] for row in rows}
    group_ids = {row["group_id"] for row in rows}
    scenario_seeds = {int(master["scenario_seed"]) for master in masters}
    result: dict[str, Any] = {}
    for directory in reference_dirs:
        reference_rows = read_jsonl(directory / "canonical" / "train.jsonl")
        reference_masters = read_jsonl(directory / "master" / "cases.jsonl")
        result[directory.name] = {
            "example_id_overlap": len(example_ids & {row["example_id"] for row in reference_rows}),
            "group_id_overlap": len(group_ids & {row["group_id"] for row in reference_rows}),
            "scenario_seed_overlap": len(
                scenario_seeds & {int(master["scenario_seed"]) for master in reference_masters}
            ),
        }
    return result


def audit(
    dataset_dir: Path,
    reference_dirs: list[Path],
    replay_workers: int,
    *,
    replay_enabled: bool = True,
) -> dict[str, Any]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(dataset_dir / "canonical" / "train.jsonl")
    qwen = read_jsonl(dataset_dir / "exports" / "qwen" / "train.jsonl")
    masters = read_jsonl(dataset_dir / "master" / "cases.jsonl")
    private = private_index(masters)
    failures: list[str] = []
    expected_scenarios = int(manifest["scenario_count"])
    expected_per_task = expected_scenarios * 2

    if len(rows) != expected_scenarios * 4 or len(masters) != expected_scenarios:
        failures.append(f"unexpected dataset size: {len(rows)} rows, {len(masters)} scenarios")
    if len(qwen) != len(rows) or len({row["example_id"] for row in rows}) != len(rows):
        failures.append("Qwen/canonical counts differ or example IDs are not unique")
    if any(row.get("images") for row in qwen):
        failures.append("hard-pair dataset unexpectedly contains images")

    task_counts = Counter(row["task_type"] for row in rows)
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    schema_failures = 0
    prompt_leaks = 0
    nonfinite = 0
    compact_target_failures = 0
    for row in rows:
        task = str(row["task_type"])
        status_counts[task][str(row["target"]["status"])] += 1
        groups[str(row["provenance"].get("match_group_id"))].append(row)
        valid, _ = schema_valid(task, row["target"])
        schema_failures += not valid
        compact_target_failures += set(row["target"]) != {"status", "answer"}
        prompt_leaks += bool(prompt_key_hits(row["prompt_inputs"]))
        try:
            assert_finite_tree(row)
        except ValueError:
            nonfinite += 1

    if task_counts != Counter(
        {
            "constrained_intervention": expected_per_task,
            "information_sufficiency": expected_per_task,
        }
    ):
        failures.append(f"unexpected task counts: {dict(task_counts)}")
    for task, statuses in EXPECTED_STATUSES.items():
        if status_counts[task] != Counter({status: expected_scenarios for status in statuses}):
            failures.append(f"unbalanced {task}: {dict(status_counts[task])}")
    if schema_failures or compact_target_failures:
        failures.append(
            f"target contract failures: schema={schema_failures}, compact={compact_target_failures}"
        )
    if prompt_leaks or nonfinite:
        failures.append(f"prompt/nonfinite failures: prompt={prompt_leaks}, nonfinite={nonfinite}")

    pair_prompt_delta_counts: Counter[str] = Counter()
    target_error_differences: list[float] = []
    target_error_labels: list[tuple[float, bool]] = []
    sufficiency_order_buckets: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
    sufficiency_sign_buckets: dict[tuple[int, int, int], Counter[str]] = defaultdict(Counter)
    sufficiency_hidden_fields: Counter[str] = Counter()
    evidence_failures = 0
    control_grid_failures = 0
    member_order_counts: Counter[str] = Counter()
    for match_id, members in groups.items():
        if len(members) != 2:
            failures.append(f"pair {match_id} has {len(members)} members")
            continue
        task = str(members[0]["task_type"])
        if {row["target"]["status"] for row in members} != EXPECTED_STATUSES[task]:
            failures.append(f"pair {match_id} does not contain both statuses")
        if len({row["group_id"] for row in members}) != 1:
            failures.append(f"pair {match_id} crosses physical scenarios")
        deltas = {
            key
            for key in set(members[0]["prompt_inputs"]) | set(members[1]["prompt_inputs"])
            if members[0]["prompt_inputs"].get(key) != members[1]["prompt_inputs"].get(key)
        }
        pair_prompt_delta_counts[",".join(sorted(deltas))] += 1
        if deltas != EXPECTED_PROMPT_DELTAS[task]:
            failures.append(f"pair {match_id} changes prompt fields {sorted(deltas)}")

        if task == "constrained_intervention":
            constraints = [row["prompt_inputs"]["actuator_constraints"] for row in members]
            control_grid_failures += constraints[0] != constraints[1]
            errors = [
                centroid_distance(
                    row["prompt_inputs"]["current_observation"],
                    row["prompt_inputs"]["target_observation"],
                )
                for row in members
            ]
            target_error_differences.append(abs(errors[0] - errors[1]))
            target_error_labels.extend(
                (error, row["target"]["status"] == "feasible")
                for error, row in zip(errors, members)
            )
            for row in members:
                scores = [float(value) for value in private[row["example_id"]]["grid_scores"]]
                feasible = min(scores) <= float(
                    row["prompt_inputs"]["actuator_constraints"]["success_tolerance_px"]
                )
                evidence_failures += feasible != (row["target"]["status"] == "feasible")
        else:
            for row in members:
                sufficiency_hidden_fields[
                    str(row["prompt_inputs"]["hidden_action_field"])
                ] += 1
                values = [float(value) for value in row["prompt_inputs"]["compatible_hidden_values_mm"]]
                ranks = {value: index for index, value in enumerate(sorted(values))}
                permutation = tuple(ranks[value] for value in values)
                sufficiency_order_buckets[permutation][row["target"]["status"]] += 1
                signature = (
                    sum(value < 0 for value in values),
                    sum(value == 0 for value in values),
                    sum(value > 0 for value in values),
                )
                sufficiency_sign_buckets[signature][row["target"]["status"]] += 1
                directions = replay_directions(row, private[row["example_id"]])
                invariant = len(directions) == 1
                evidence_failures += invariant != (row["target"]["status"] == "answerable")

    for master in masters:
        statuses = [item["record"]["target"]["status"] for item in master["records"]]
        member_order_counts["|".join(statuses)] += 1

    if len(groups) != expected_scenarios * 2:
        failures.append(f"expected {expected_scenarios * 2} minimal pairs, got {len(groups)}")
    if evidence_failures or control_grid_failures:
        failures.append(
            f"grounding/pair-grid failures: evidence={evidence_failures}, grid={control_grid_failures}"
        )
    error_mean = statistics.fmean(target_error_differences) if target_error_differences else None
    error_p95 = percentile(target_error_differences, 0.95)
    target_error_threshold_accuracy = best_threshold_accuracy(target_error_labels)
    order_shortcut_accuracy = (
        sum(max(counts.values()) for counts in sufficiency_order_buckets.values())
        / sum(sum(counts.values()) for counts in sufficiency_order_buckets.values())
        if sufficiency_order_buckets
        else None
    )
    sign_shortcut_accuracy = (
        sum(max(counts.values()) for counts in sufficiency_sign_buckets.values())
        / sum(sum(counts.values()) for counts in sufficiency_sign_buckets.values())
        if sufficiency_sign_buckets
        else None
    )
    if error_mean is None or error_mean > 1.0 or error_p95 is None or error_p95 > 2.0:
        failures.append(f"control pair target-error mismatch: mean={error_mean}, p95={error_p95}")
    if len(masters) >= 50 and (
        target_error_threshold_accuracy is None
        or target_error_threshold_accuracy > 0.65
        or order_shortcut_accuracy != 0.5
        or sign_shortcut_accuracy != 0.5
    ):
        failures.append(
            "simple shortcut accuracy too high: "
            f"target_error={target_error_threshold_accuracy}, order={order_shortcut_accuracy}, "
            f"sign={sign_shortcut_accuracy}"
        )
    if sufficiency_hidden_fields != Counter({"lens_x_delta_mm": len(masters) * 2}):
        failures.append(f"unexpected sufficiency hidden fields: {dict(sufficiency_hidden_fields)}")

    overlap = reference_overlap(rows, masters, reference_dirs)
    if any(any(value for value in counts.values()) for counts in overlap.values()):
        failures.append(f"reference overlap detected: {overlap}")
    replay = (
        _replay_master(dataset_dir, replay_workers)
        if replay_enabled
        else {"skipped": True}
    )
    if replay_enabled and replay["failure_count"]:
        failures.append(f"simulator replay failures: {replay['failure_count']}")

    return {
        "passed": not failures,
        "failures": failures[:50],
        "dataset_dir": str(dataset_dir.resolve()),
        "manifest_record_count": manifest["record_count"],
        "record_count": len(rows),
        "scenario_count": len(masters),
        "minimal_pair_count": len(groups),
        "task_counts": dict(task_counts),
        "status_counts": {task: dict(values) for task, values in status_counts.items()},
        "schema_failure_count": schema_failures,
        "prompt_leak_count": prompt_leaks,
        "nonfinite_count": nonfinite,
        "pair_prompt_delta_counts": dict(pair_prompt_delta_counts),
        "member_order_counts": dict(member_order_counts),
        "control_target_error_difference_px_mean": error_mean,
        "control_target_error_difference_px_p95": error_p95,
        "control_target_error_threshold_accuracy": target_error_threshold_accuracy,
        "sufficiency_order_shortcut_accuracy": order_shortcut_accuracy,
        "sufficiency_sign_signature_shortcut_accuracy": sign_shortcut_accuracy,
        "sufficiency_hidden_field_counts": dict(sufficiency_hidden_fields),
        "grounding_failure_count": evidence_failures,
        "control_grid_failure_count": control_grid_failures,
        "reference_overlap": overlap,
        "replay": replay,
    }


def main() -> None:
    args = parse_args()
    result = audit(
        args.dataset_dir,
        args.reference_dir,
        args.replay_workers,
        replay_enabled=not args.skip_replay,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
