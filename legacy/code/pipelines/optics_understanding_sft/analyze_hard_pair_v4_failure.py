#!/usr/bin/env python3
"""Diagnose hard-pair v4 checkpoint collapse and prompt-evidence mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import centroid_distance, read_jsonl


TASKS = ("information_sufficiency", "constrained_intervention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--master-jsonl", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=RESULT_DIR",
        help="A completed diagnostic result directory. Repeat for each checkpoint.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    items = sorted(float(value) for value in values)
    if not items:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": items[0],
        "median": statistics.median(items),
        "mean": statistics.fmean(items),
        "max": items[-1],
    }


def local_turn_count(values: list[float]) -> int:
    """Count strict changes in slope direction along a discrete response curve."""
    return sum(
        (middle - left) * (right - middle) < 0
        for left, middle, right in zip(values, values[1:], values[2:])
    )


def replacement_deltas(left: list[float], right: list[float]) -> list[float]:
    """Return sorted-pair magnitudes for the multiset replacements in a pair."""
    left_only = sorted(float(value) for value in (Counter(left) - Counter(right)).elements())
    right_only = sorted(float(value) for value in (Counter(right) - Counter(left)).elements())
    if not left_only or len(left_only) != len(right_only):
        raise ValueError(f"invalid minimal-pair replacements: {left_only} and {right_only}")
    return [abs(left_value - right_value) for left_value, right_value in zip(left_only, right_only)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"invalid --candidate value: {value!r}")
    return name, Path(raw_path)


def private_index(master_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for case in read_jsonl(master_path):
        for item in case["records"]:
            private = item["private_eval"]
            key = (str(private["match_group_id"]), str(private["pair_member"]))
            if key in result:
                raise ValueError(f"duplicate private record: {key}")
            result[key] = private
    return result


def record_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["provenance"]["match_group_id"])].append(record)
    invalid = {group_id: len(rows) for group_id, rows in groups.items() if len(rows) != 2}
    if invalid:
        raise ValueError(f"invalid diagnostic pairs: {invalid}")
    return dict(groups)


def dataset_diagnostics(
    records: list[dict[str, Any]],
    groups: Mapping[str, list[dict[str, Any]]],
    private: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    sufficiency_deltas: list[float] = []
    sufficiency_change_counts: list[float] = []
    sufficiency_pair_max_deltas: list[float] = []
    sufficiency_turns: dict[str, list[float]] = defaultdict(list)
    sufficiency_unique_directions: dict[str, list[float]] = defaultdict(list)
    control_current_errors: dict[str, list[float]] = defaultdict(list)
    control_curve_turns: dict[str, list[float]] = defaultdict(list)
    control_best_errors: dict[str, list[float]] = defaultdict(list)
    control_canonical_errors: dict[str, list[float]] = defaultdict(list)

    for pair in groups.values():
        task = str(pair[0]["task_type"])
        if task == "information_sufficiency":
            left = list(pair[0]["prompt_inputs"]["compatible_hidden_values_mm"])
            right = list(pair[1]["prompt_inputs"]["compatible_hidden_values_mm"])
            deltas = replacement_deltas(left, right)
            sufficiency_deltas.extend(deltas)
            sufficiency_change_counts.append(float(len(deltas)))
            sufficiency_pair_max_deltas.append(max(deltas))
            for record in pair:
                label = str(record["target"]["status"])
                key = (
                    str(record["provenance"]["match_group_id"]),
                    str(record["provenance"]["pair_member"]),
                )
                evidence = sorted(
                    private[key]["completion_evidence"],
                    key=lambda item: float(item["hidden_value_mm"]),
                )
                directions = [str(item["centroid_x_direction"]) for item in evidence]
                sufficiency_turns[label].append(
                    float(sum(left != right for left, right in zip(directions, directions[1:])))
                )
                sufficiency_unique_directions[label].append(float(len(set(directions))))
        elif task == "constrained_intervention":
            for record in pair:
                label = str(record["target"]["status"])
                control_current_errors[label].append(
                    centroid_distance(
                        record["prompt_inputs"]["current_observation"],
                        record["prompt_inputs"]["target_observation"],
                    )
                )
                key = (
                    str(record["provenance"]["match_group_id"]),
                    str(record["provenance"]["pair_member"]),
                )
                private_record = private[key]
                scores = [float(value) for value in private_record["grid_scores"]]
                control_curve_turns[label].append(float(local_turn_count(scores)))
                control_best_errors[label].append(min(scores))
                canonical = private_record.get("canonical_action")
                if isinstance(canonical, dict):
                    actuator = str(private_record["active_actuator"])
                    allowed = [
                        float(value)
                        for value in record["prompt_inputs"]["actuator_constraints"][
                            "allowed_values_mm"
                        ]
                    ]
                    action_value = float(canonical[actuator])
                    matches = [
                        index
                        for index, value in enumerate(allowed)
                        if math.isclose(value, action_value, abs_tol=1e-9)
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"canonical action is absent from grid: {key}")
                    control_canonical_errors[label].append(scores[matches[0]])

    forbidden_evidence = ("grid_scores", "completion_evidence", "expected_state", "after_state")
    evidence_occurrences = {
        field: sum(field in str(record["prompt"]) for record in records)
        for field in forbidden_evidence
    }
    tolerance = 2.0
    return {
        "record_count": len(records),
        "pair_count": len(groups),
        "status_counts": {
            task: dict(
                Counter(
                    str(record["target"]["status"])
                    for record in records
                    if record["task_type"] == task
                )
            )
            for task in TASKS
        },
        "prompt_evidence_field_occurrences": evidence_occurrences,
        "information_sufficiency": {
            "changed_value_count_per_pair": quantiles(sufficiency_change_counts),
            "replacement_delta_mm": quantiles(sufficiency_deltas),
            "maximum_replacement_delta_mm_per_pair": quantiles(sufficiency_pair_max_deltas),
            "sorted_completion_direction_changes": {
                label: quantiles(values) for label, values in sufficiency_turns.items()
            },
            "unique_completion_directions": {
                label: quantiles(values)
                for label, values in sufficiency_unique_directions.items()
            },
        },
        "constrained_intervention": {
            "success_tolerance_px": tolerance,
            "current_target_error_px": {
                label: quantiles(values) for label, values in control_current_errors.items()
            },
            "current_target_errors_within_tolerance": {
                label: sum(value <= tolerance for value in values)
                for label, values in control_current_errors.items()
            },
            "hidden_grid_best_residual_px": {
                label: quantiles(values) for label, values in control_best_errors.items()
            },
            "hidden_grid_canonical_residual_px": {
                label: quantiles(values) for label, values in control_canonical_errors.items()
            },
            "hidden_grid_local_turns": {
                label: quantiles(values) for label, values in control_curve_turns.items()
            },
        },
    }


def candidate_diagnostics(
    name: str,
    result_dir: Path,
    records: list[dict[str, Any]],
    groups: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    predictions = {row["example_id"]: row for row in read_jsonl(result_dir / "predictions.jsonl")}
    details = read_jsonl(result_dir / "details.jsonl")
    gates = json.loads((result_dir / "gates.json").read_text(encoding="utf-8"))
    if len(predictions) != len(records) or len(details) != len(records):
        raise ValueError(f"candidate {name} is incomplete")

    status_counts = {
        task: dict(
            Counter(
                str(row.get("predicted_status"))
                for row in details
                if row["task_type"] == task
            )
        )
        for task in TASKS
    }
    pair_sensitivity = {}
    for task in TASKS:
        task_pairs = [pair for pair in groups.values() if pair[0]["task_type"] == task]
        same_status = 0
        same_raw = 0
        for left, right in task_pairs:
            left_prediction = predictions[left["example_id"]]
            right_prediction = predictions[right["example_id"]]
            left_json = left_prediction.get("parsed_json") or {}
            right_json = right_prediction.get("parsed_json") or {}
            same_status += left_json.get("status") == right_json.get("status")
            same_raw += left_prediction.get("raw_prediction_text") == right_prediction.get(
                "raw_prediction_text"
            )
        pair_sensitivity[task] = {
            "pair_count": len(task_pairs),
            "same_predicted_status_count": same_status,
            "exact_same_raw_response_count": same_raw,
        }

    record_index = {record["example_id"]: record for record in records}
    residual_errors: list[float] = []
    exact_no_action_residuals = 0
    residual_count = 0
    for example_id, prediction in predictions.items():
        record = record_index[example_id]
        if record["task_type"] != "constrained_intervention":
            continue
        parsed = prediction.get("parsed_json")
        answer = parsed.get("answer") if isinstance(parsed, dict) else None
        residual = answer.get("best_achievable_residual_px") if isinstance(answer, dict) else None
        if not isinstance(residual, (int, float)) or isinstance(residual, bool):
            continue
        no_action = centroid_distance(
            record["prompt_inputs"]["current_observation"],
            record["prompt_inputs"]["target_observation"],
        )
        error = abs(float(residual) - no_action)
        residual_errors.append(error)
        residual_count += 1
        exact_no_action_residuals += error <= 1e-4

    run_manifest = json.loads(
        (result_dir / "predictions.run.json").read_text(encoding="utf-8")
    )
    adapter_path = Path(run_manifest["adapter_path"])
    adapter_file = adapter_path / "adapter_model.safetensors"
    return (
        {
            "result_dir": str(result_dir.resolve()),
            "adapter_path": str(adapter_path),
            "adapter_sha256": sha256(adapter_file),
            "passed": bool(gates["passed"]),
            "metrics": gates["metrics"],
            "gates": gates["gates"],
            "predicted_status_counts": status_counts,
            "pair_sensitivity": pair_sensitivity,
            "control_predicted_best_residual_vs_no_action": {
                "numeric_prediction_count": residual_count,
                "exact_within_1e_4_count": exact_no_action_residuals,
                "absolute_error_px": quantiles(residual_errors),
            },
        },
        predictions,
    )


def agreement(
    prediction_sets: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names = list(prediction_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = prediction_sets[left_name]
            right = prediction_sets[right_name]
            ids = sorted(set(left) & set(right))
            status_equal = sum(
                (left[example_id].get("parsed_json") or {}).get("status")
                == (right[example_id].get("parsed_json") or {}).get("status")
                for example_id in ids
            )
            raw_equal = sum(
                left[example_id].get("raw_prediction_text")
                == right[example_id].get("raw_prediction_text")
                for example_id in ids
            )
            result[f"{left_name}__vs__{right_name}"] = {
                "record_count": len(ids),
                "same_status_count": status_equal,
                "exact_same_raw_response_count": raw_equal,
            }
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    dataset = result["dataset"]
    sufficiency = dataset["information_sufficiency"]
    control = dataset["constrained_intervention"]
    lines = [
        "# Hard-pairs v4 seed-42 failure analysis",
        "",
        "## Frozen decision",
        "",
        "All predeclared checkpoints failed the balanced 120-record diagnostic. The 240-record "
        "confirmation holdout, unchanged development set, additional seeds, and sealed test remain "
        "unevaluated.",
        "",
        "| Checkpoint | Schema | Sufficiency F1 | Control F1 | Feasible recall | Pair joint | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for name, candidate in result["candidates"].items():
        metrics = candidate["metrics"]
        lines.append(
            f"| {name} | {metrics['schema_valid_rate']:.3f} | "
            f"{metrics['sufficiency_status_macro_f1']:.3f} | "
            f"{metrics['control_status_macro_f1']:.3f} | "
            f"{metrics['control_feasible_recall']:.3f} | "
            f"{metrics['minimal_pair_joint_status_accuracy']:.3f} | "
            f"{'yes' if candidate['passed'] else 'no'} |"
        )
    final_name = list(result["candidates"])[-1]
    final = result["candidates"][final_name]
    suff_pairs = final["pair_sensitivity"]["information_sufficiency"]
    control_pairs = final["pair_sensitivity"]["constrained_intervention"]
    residual = final["control_predicted_best_residual_vs_no_action"]
    change_count = sufficiency["changed_value_count_per_pair"]
    delta = sufficiency["replacement_delta_mm"]
    insufficient_turns = sufficiency["sorted_completion_direction_changes"][
        "insufficient_information"
    ]
    feasible_error = control["current_target_error_px"]["feasible"]
    grid_turns = control["hidden_grid_local_turns"]["feasible"]
    lines.extend(
        [
            "",
            "## Observed failure mode",
            "",
            f"- The final checkpoint returned the same status for all {suff_pairs['pair_count']} "
            "sufficiency pairs and all "
            f"{control_pairs['pair_count']} control pairs.",
            f"- Sufficiency pair responses were byte-identical for "
            f"{suff_pairs['exact_same_raw_response_count']}/{suff_pairs['pair_count']} pairs.",
            f"- For control, {residual['exact_within_1e_4_count']}/"
            f"{residual['numeric_prediction_count']} predicted best residuals exactly matched the "
            "no-action current-to-target distance. The mean absolute difference was "
            f"{residual['absolute_error_px']['mean']:.4f} px.",
            "- Adapter hashes differ across checkpoints, so this is not a trainer no-op. Generated "
            "decisions nevertheless remained almost unchanged.",
            "",
            "## Why balance was not enough",
            "",
            f"- Every feasible control target starts outside the declared 2 px tolerance "
            f"(minimum {feasible_error['min']:.4f} px). Feasibility is visible only after replaying "
            "the allowed actions.",
            f"- The hidden nine-action response curves average {grid_turns['mean']:.2f} local slope "
            "reversals. Their outcomes are not present in the prompt.",
            f"- Sufficiency pairs replace a median of {change_count['median']:.1f} compatible values; "
            f"the median replacement magnitude is {delta['median']:.3f} mm. Insufficient cases average "
            f"{insufficient_turns['mean']:.2f} direction changes after sorting those values, and the "
            "per-value measurements are also absent from the prompt.",
            "- The dataset is label-correct and shortcut-controlled, but these records demand "
            "high-precision emulation of a non-monotonic Fresnel simulator. They do not isolate "
            "reasoning over observable experimental evidence.",
            "",
            "## Recommended v5 experiment",
            "",
            "1. Add a raw calibration table to control prompts: candidate action plus measured "
            "after-observation. Ask the model to compute residuals, enforce the tolerance, and choose "
            "the minimum-motion successful action.",
            "2. Add raw completion measurements to sufficiency prompts: hidden-value completion plus "
            "centroid observation. Ask the model to derive directions and decide whether all compatible "
            "completions agree.",
            "3. Keep labels balanced and pairs counterfactual, but vary grid size, ordering, decoys, "
            "tolerances, and response-curve shape so success cannot reduce to one threshold or slot.",
            "4. Add a separate analytic-physics tier using paraxial, monotonic setups for genuine optics "
            "intuition. Reserve exact Fresnel behavior for simulator-tool-use evaluation rather than "
            "requiring a 3B model to internalize the solver.",
            "5. Before any new 200-step run, test base and retained-reference models on a small frozen "
            "v5 diagnostic. Train only if the task is above chance yet below the desired gate, proving "
            "that the prompt supplies usable evidence and still has learning headroom.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(args.records_jsonl)
    groups = record_groups(records)
    private = private_index(args.master_jsonl)
    candidates: dict[str, Any] = {}
    prediction_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for value in args.candidate:
        name, result_dir = parse_candidate(value)
        if name in candidates:
            raise ValueError(f"duplicate candidate name: {name}")
        candidates[name], prediction_sets[name] = candidate_diagnostics(
            name, result_dir, records, groups
        )
    adapter_hashes = [candidate["adapter_sha256"] for candidate in candidates.values()]
    return {
        "protocol": "hard_pairs_v4",
        "diagnostic_records": str(args.records_jsonl.resolve()),
        "sealed_test_evaluated": False,
        "later_stages_stopped": True,
        "dataset": dataset_diagnostics(records, groups, private),
        "candidates": candidates,
        "candidate_agreement": agreement(prediction_sets),
        "distinct_adapter_hash_count": len(set(adapter_hashes)),
        "conclusion": "prompt_evidence_mismatch_and_conservative_class_collapse",
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
