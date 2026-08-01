#!/usr/bin/env python3
"""Build fresh, same-setup hard pairs for the two failed status decisions.

Every independently sampled physical setup produces two minimal pairs:

* control: identical setup/current state/action grid, with only the target
  observation changing between feasible and infeasible;
* sufficiency: identical setup/current state/masked field, with only the set of
  compatible hidden values changing between invariant and answer-changing.

Ground truth and replay provenance come from the existing simulator builders.
Targets are compacted to the unchanged dev-v2 response schema so decision
tokens are not diluted by training-only evidence arrays or envelope fields.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml
from optical_sim.src.optical_elements import setup_from_dict
from optics_sft.physics.sim_adapter import simulate_and_measure

from .build_dataset import (
    TEMPLATE_PATH,
    canonical_public,
    constrained_intervention,
    information_sufficiency,
    prepare_output_dir,
    simulator_result,
)
from .core import (
    file_sha256,
    action_dict,
    load_yaml,
    make_messages,
    make_qwen_record,
    round_tree,
    rounded_state,
    sample_setup_config,
    setup_snapshot,
    stable_json_hash,
    write_jsonl,
)


CONTROL_CONTRACT = {
    "status": "feasible | infeasible_within_limits",
    "answer": {
        "control_plan": "object with numeric lens_x_delta_mm, lens_y_delta_mm, camera_x_delta_mm, camera_y_delta_mm | null",
        "expected_residual_px": "number | null",
        "best_achievable_residual_px": "number | null",
    },
}
SUFFICIENCY_CONTRACT = {
    "status": "answerable | insufficient_information",
    "answer": {
        "centroid_x_direction": "increase | decrease | no_change | null",
        "missing_fields": ["field_name"],
        "nonidentifiable_output": "field_name | null",
    },
}
CONTRACT_MARKER = "Task output contract (all fields shown; use null when not applicable):"
INPUT_MARKER = "Input data:\n"
RETURN_MARKER = "\n\nReturn only strict JSON"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def replace_contract(prompt: str, contract: Mapping[str, Any]) -> str:
    if CONTRACT_MARKER not in prompt:
        raise ValueError("generated prompt lacks output-contract marker")
    return (
        prompt.split(CONTRACT_MARKER, 1)[0]
        + CONTRACT_MARKER
        + "\n"
        + json.dumps(contract, indent=2, sort_keys=True)
    )


def refresh_prompt_inputs(record: dict[str, Any]) -> None:
    prompt = str(record["prompt"])
    if INPUT_MARKER not in prompt or RETURN_MARKER not in prompt:
        raise ValueError("generated prompt lacks input-data markers")
    prefix, remainder = prompt.split(INPUT_MARKER, 1)
    _, suffix = remainder.split(RETURN_MARKER, 1)
    visible = {key: value for key, value in record["prompt_inputs"].items() if key != "images"}
    record["prompt"] = (
        prefix
        + INPUT_MARKER
        + json.dumps(visible, indent=2, sort_keys=True)
        + RETURN_MARKER
        + suffix
    )


def counterbalance_hidden_value_order(
    record: dict[str, Any], private: dict[str, Any], *, permutation_seed: int
) -> None:
    """Remove sorted-vs-unsorted status leakage while preserving replay alignment."""
    values = [float(value) for value in record["prompt_inputs"]["compatible_hidden_values_mm"]]
    specs = list(private["replay_specs"])
    if len(values) != 5 or len(specs) != 5:
        raise ValueError("sufficiency hard pairs require five values and replay specs")
    ordered = sorted(zip(values, specs), key=lambda item: item[0])
    ranks = list(range(5))
    random.Random(permutation_seed).shuffle(ranks)
    reordered = [ordered[index] for index in ranks]
    record["prompt_inputs"]["compatible_hidden_values_mm"] = [value for value, _ in reordered]
    private["compatible_values_mm"] = [value for value, _ in reordered]
    private["replay_specs"] = [
        {**spec, "name": f"completion_{index}"}
        for index, (_, spec) in enumerate(reordered)
    ]
    refresh_prompt_inputs(record)


def sign_signature(values: list[float]) -> tuple[int, int, int]:
    return (
        sum(value < 0 for value in values),
        sum(value == 0 for value in values),
        sum(value > 0 for value in values),
    )


def choose_counterbalanced_sufficiency_indices(
    values: list[float], directions: list[str], *, seed: int
) -> dict[str, list[int]]:
    """Choose opposite labels with an identical visible sign signature."""
    invariant: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
    changing: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
    for indices in combinations(range(len(values)), 5):
        selected_values = [values[index] for index in indices]
        signature = sign_signature(selected_values)
        selected_directions = {directions[index] for index in indices}
        destination = invariant if len(selected_directions) == 1 else changing
        destination.setdefault(signature, []).append(indices)
    common = sorted(set(invariant) & set(changing))
    if not common:
        raise RuntimeError("could not construct sign-counterbalanced sufficiency pair")
    rng = random.Random(seed)
    signature = common[rng.randrange(len(common))]
    answerable_candidates = invariant[signature]
    insufficient_candidates = changing[signature]
    answerable = answerable_candidates[rng.randrange(len(answerable_candidates))]

    def features(indices: tuple[int, ...]) -> tuple[float, float]:
        selected = [values[index] for index in indices]
        return max(selected) - min(selected), sum(abs(value) for value in selected) / len(selected)

    answerable_features = features(answerable)
    insufficient = min(
        insufficient_candidates,
        key=lambda indices: (
            abs(features(indices)[0] - answerable_features[0])
            + abs(features(indices)[1] - answerable_features[1]),
            indices,
        ),
    )
    return {
        "answerable": list(answerable),
        "insufficient_information": list(insufficient),
    }


def compact_target(record: dict[str, Any]) -> None:
    status = str(record["target"]["status"])
    if record["task_type"] == "constrained_intervention":
        source = record["target"]["answer"]
        answer = {
            "control_plan": source.get("control_plan"),
            "expected_residual_px": source.get("expected_residual_px"),
            "best_achievable_residual_px": source.get("best_achievable_residual_px"),
        }
        contract = CONTROL_CONTRACT
    elif record["task_type"] == "information_sufficiency":
        source = record["target"]["answer"]
        answer = {
            "centroid_x_direction": source.get("centroid_x_direction"),
            "missing_fields": source.get("missing_fields", []),
            "nonidentifiable_output": source.get("nonidentifiable_output"),
        }
        contract = SUFFICIENCY_CONTRACT
    else:
        raise ValueError(f"unexpected hard-pair task: {record['task_type']}")
    record["target"] = {"status": status, "answer": answer}
    record["prompt"] = replace_contract(str(record["prompt"]), contract)


def finalize_record(
    record: dict[str, Any],
    *,
    dataset_version: str,
    scenario_attempt: int,
    pair_id: str,
    pair_member: str,
) -> None:
    record["example_id"] = f"{record['example_id']}__{pair_member}"
    record["provenance"].update(
        {
            "dataset_version": dataset_version,
            "scenario_sampling_attempt": scenario_attempt,
            "match_group_id": pair_id,
            "match_group_kind": "same_setup_minimal_pair",
            "pair_member": pair_member,
            "compact_dev_v2_target": True,
        }
    )
    compact_target(record)


def paired_records(
    *,
    group_id: str,
    scenario_seed: int,
    scenario_attempt: int,
    task_index: int,
    sampled_cfg: Mapping[str, Any],
    base: Mapping[str, Any],
    templates: Mapping[str, Any],
    label_cfg: Mapping[str, Any],
    dataset_version: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    specifications = (
        (
            "constrained_intervention",
            constrained_intervention,
            ("feasible", "infeasible_within_limits"),
            101,
        ),
        (
            "information_sufficiency",
            information_sufficiency,
            ("answerable", "insufficient_information"),
            202,
        ),
    )
    for task_name, builder, statuses, rng_offset in specifications:
        pair_id = f"{dataset_version}:train:{task_name}:{task_index:04d}"
        members: list[tuple[dict[str, Any], dict[str, Any]]] = []
        shared_control_grid: list[float] | None = None
        sufficiency_cache: dict[str, Any] | None = None
        sufficiency_indices: dict[str, list[int]] | None = None
        if task_name == "information_sufficiency":
            hidden_field = "lens_x_delta_mm"
            bound = 0.06
            values = [round(-bound + (2.0 * bound / 12.0) * index, 6) for index in range(13)]
            actions = [action_dict(**{hidden_field: value}) for value in values]
            results = [simulator_result(sampled_cfg, action) for action in actions]
            directions = []
            for result_state in results:
                delta = float(result_state["state"]["centroid_x_px"]) - float(
                    base["state"]["centroid_x_px"]
                )
                directions.append(
                    "increase" if delta > 1.0 else "decrease" if delta < -1.0 else "no_change"
                )
            sufficiency_cache = {
                "hidden_field": hidden_field,
                "grid_values": values,
                "grid_actions": actions,
                "grid_results": results,
                "grid_directions": directions,
            }
            sufficiency_indices = choose_counterbalanced_sufficiency_indices(
                values, directions, seed=scenario_seed + rng_offset + 29
            )
        for status in statuses:
            # Cloned RNGs make the nuisance variables and prompt paraphrase
            # identical across the pair while the requested label changes.
            member_rng = random.Random(scenario_seed + rng_offset)
            builder_kwargs: dict[str, Any] = {}
            if task_name == "constrained_intervention" and shared_control_grid is not None:
                builder_kwargs["forced_control_grid"] = shared_control_grid
            if task_name == "information_sufficiency":
                builder_kwargs["precomputed_sufficiency_grid"] = sufficiency_cache
                if sufficiency_indices is None:
                    raise AssertionError("sufficiency indices were not prepared")
                builder_kwargs["forced_sufficiency_indices"] = sufficiency_indices[status]
            record, private = builder(
                group_id=group_id,
                split="train",
                task_index=task_index,
                scenario_seed=scenario_seed,
                config=sampled_cfg,
                base=base,
                templates=templates,
                rng=member_rng,
                label_cfg=label_cfg,
                desired_status=status,
                **builder_kwargs,
            )
            if task_name == "information_sufficiency":
                counterbalance_hidden_value_order(
                    record,
                    private,
                    permutation_seed=scenario_seed + rng_offset + 17,
                )
            if task_name == "constrained_intervention" and shared_control_grid is None:
                shared_control_grid = list(
                    record["prompt_inputs"]["actuator_constraints"]["allowed_values_mm"]
                )
            finalize_record(
                record,
                dataset_version=dataset_version,
                scenario_attempt=scenario_attempt,
                pair_id=pair_id,
                pair_member=status,
            )
            private["match_group_id"] = pair_id
            private["pair_member"] = status
            members.append((record, private))
        # Counterbalance both task and label order across physical scenarios.
        if (task_index + (task_name == "information_sufficiency")) % 2:
            members.reverse()
        result.extend(members)
    if task_index % 2:
        result = result[2:] + result[:2]
    return result


def build_one_scenario(
    job: tuple[
        int,
        int,
        int,
        str,
        str,
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
    ]
) -> dict[str, Any]:
    (
        local_index,
        seed,
        start,
        prefix,
        version,
        base_cfg,
        simulation_cfg,
        label_cfg,
        templates,
    ) = job
    group_id = f"{prefix}_{start + local_index:06d}"
    last_error: RuntimeError | None = None
    for attempt in range(64):
        scenario_seed = seed * 1_000_000 + start + local_index + attempt * 1_000_000_000_000
        scenario_rng = random.Random(scenario_seed)
        sampled_cfg, sampled_values = sample_setup_config(base_cfg, simulation_cfg, scenario_rng)
        setup = setup_from_dict(copy.deepcopy(sampled_cfg))
        baseline = simulate_and_measure(setup)
        try:
            members = paired_records(
                group_id=group_id,
                scenario_seed=scenario_seed,
                scenario_attempt=attempt,
                task_index=local_index,
                sampled_cfg=sampled_cfg,
                base=baseline,
                templates=templates,
                label_cfg=label_cfg,
                dataset_version=version,
            )
            break
        except RuntimeError as error:
            last_error = error
    else:
        raise RuntimeError(f"failed to build hard pairs for {group_id}") from last_error
    return {
        "group_id": group_id,
        "split": "train",
        "scenario_seed": scenario_seed,
        "scenario_sampling_attempt": attempt,
        "distribution": "iid",
        "sampled_parameters": round_tree(sampled_values, 8),
        "setup_config": setup_snapshot(setup),
        "baseline_state": rounded_state(baseline["state"]),
        "records": [
            {"record": record, "private_eval": private}
            for record, private in members
        ],
    }


def build(
    config: dict[str, Any],
    output_dir: Path,
    force: bool,
    *,
    config_path: Path,
    workers: int = 1,
) -> dict[str, Any]:
    prepare_output_dir(output_dir, force)
    dataset_cfg = config["dataset"]
    simulation_cfg = config["simulation"]
    label_cfg = config["labels"]
    seed = int(dataset_cfg["seed"])
    count = int(dataset_cfg["scenarios"])
    start = int(dataset_cfg["scenario_start_index"])
    prefix = str(dataset_cfg["group_prefix"])
    version = str(dataset_cfg["version"])
    templates = load_yaml(TEMPLATE_PATH)
    base_cfg = load_sim_yaml(str(Path(__file__).resolve().parents[1] / simulation_cfg["base_config"]))

    jobs = [
        (
            local_index,
            seed,
            start,
            prefix,
            version,
            base_cfg,
            simulation_cfg,
            label_cfg,
            templates,
        )
        for local_index in range(count)
    ]
    worker_count = max(1, int(workers))
    if worker_count == 1:
        generated = map(build_one_scenario, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        generated = executor.map(build_one_scenario, jobs, chunksize=1)
    masters = []
    try:
        for completed, master in enumerate(generated, start=1):
            masters.append(master)
            if completed % 10 == 0 or completed == count:
                print(f"[{completed}/{count}] generated {master['group_id']}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    rows = [item["record"] for master in masters for item in master["records"]]

    write_jsonl(output_dir / "master" / "cases.jsonl", masters)
    write_jsonl(output_dir / "canonical" / "train.jsonl", (canonical_public(row, True) for row in rows))
    write_jsonl(output_dir / "canonical" / "val.jsonl", [])
    write_jsonl(output_dir / "canonical" / "test_prompts.jsonl", [])
    write_jsonl(output_dir / "private" / "test_labels.jsonl", [])
    write_jsonl(output_dir / "exports" / "messages" / "train.jsonl", (make_messages(row, True) for row in rows))
    write_jsonl(output_dir / "exports" / "messages" / "val.jsonl", [])
    write_jsonl(output_dir / "exports" / "messages" / "test.jsonl", [])
    write_jsonl(output_dir / "exports" / "qwen" / "train.jsonl", (make_qwen_record(row, True) for row in rows))
    write_jsonl(output_dir / "exports" / "qwen" / "val.jsonl", [])
    write_jsonl(output_dir / "exports" / "qwen" / "test.jsonl", [])

    statuses = {
        task: dict(sorted(Counter(row["target"]["status"] for row in rows if row["task_type"] == task).items()))
        for task in ("information_sufficiency", "constrained_intervention")
    }
    manifest = {
        "dataset": str(dataset_cfg["name"]),
        "version": version,
        "seed": seed,
        "scenario_count": count,
        "record_count": len(rows),
        "records_per_scenario": 4,
        "task_counts": dict(sorted(Counter(row["task_type"] for row in rows).items())),
        "status_counts": statuses,
        "same_setup_pair_count": count * 2,
        "visual_record_count": 0,
        "generation_workers": worker_count,
        "scenario_ids_hash": stable_json_hash([master["group_id"] for master in masters]),
        "example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
        "config_sha256": file_sha256(config_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_targets = [
        output_dir / "manifest.json",
        output_dir / "master" / "cases.jsonl",
        output_dir / "canonical" / "train.jsonl",
        output_dir / "exports" / "qwen" / "train.jsonl",
    ]
    checksum_lines = [f"{file_sha256(path)}  {path.relative_to(output_dir)}" for path in checksum_targets]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build(
        load_yaml(args.config),
        args.output_dir,
        args.force,
        config_path=args.config,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
