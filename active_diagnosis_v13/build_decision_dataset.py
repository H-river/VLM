#!/usr/bin/env python3
"""Serialize the selected visible probe evidence for cheap models and Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from active_diagnosis_v13.contracts import (
    assert_policy_visible,
    require_frozen_branch_a_refinement,
)

VERSION = "active_diagnosis_v13_decision_dataset_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_for_record(
    row: dict[str, Any],
    seed: int,
    retained_feature_indices: Sequence[int] | None = None,
) -> str:
    policy = row["policy_record"]
    indices = (
        list(range(len(policy["feature_names"])))
        if retained_feature_indices is None
        else list(retained_feature_indices)
    )
    visible = {
        policy["feature_names"][index]: round(
            float(policy["feature_vector"][index]), 6
        )
        for index in indices
    }
    assert_policy_visible(visible)
    hypotheses = [0.5, 0.75, 1.0, 1.25, 1.5]
    digest = hashlib.sha256(
        # Case identity is fixed before the fault is assigned. Do not seed from
        # record_id because its ``__g...`` suffix contains evaluator-only gain.
        f"{seed}:{row['case_id']}:candidate_order".encode()
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    rng.shuffle(hypotheses)
    request = {
        "task": "estimate_hidden_global_actuator_gain_from_safe_probe",
        "probe_design": row["design"],
        "probe_fraction": float(row["fraction"]),
        "candidate_gains_randomized": hypotheses,
        "visible_probe_features": visible,
    }
    assert_policy_visible(request)
    return (
        "Use only the visible probe evidence below. Select exactly one candidate "
        "gain. Return strict JSON with keys task, estimated_gain, confidence, "
        "and next_action. next_action must be replan_with_gain_belief.\n\n"
        + json.dumps(request, sort_keys=True, separators=(",", ":"))
    )


def qwen_row(
    row: dict[str, Any],
    seed: int,
    retained_feature_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    prompt = prompt_for_record(row, seed, retained_feature_indices)
    label = {
        "task": "hidden_global_actuator_gain_estimation",
        "estimated_gain": float(row["evaluator_only_true_gain"]),
        "confidence": 1.0,
        "next_action": "replan_with_gain_belief",
    }
    return {
        "example_id": row["record_id"],
        "group_id": row["group_id"],
        "images": [],
        "prompt": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps(label, sort_keys=True)}
                ],
            }
        ],
        "provenance": {
            "dataset_version": VERSION,
            "split_source": "development_only",
            "protected_set_used": False,
            "serialization_seed": seed,
            "retained_feature_indices": (
                None
                if retained_feature_indices is None
                else list(map(int, retained_feature_indices))
            ),
        },
    }


def _load_rows(gate_dir: Path) -> list[dict[str, Any]]:
    result = {}
    for path in sorted((gate_dir / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                result[row["record_id"]] = row
    return list(result.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080104)
    parser.add_argument("--retained-feature-model", type=Path)
    args = parser.parse_args()
    gate_dir = args.gate_dir.resolve()
    selection = json.loads(
        (gate_dir / "probes" / "selected_probe.json").read_text()
    )
    rows = [
        row
        for row in _load_rows(gate_dir)
        if row["design"] == selection["selected_design"]
        and float(row["fraction"]) == float(selection["selected_fraction"])
    ]
    by_group: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    groups = sorted(by_group)
    validation_groups = {
        group for index, group in enumerate(groups) if index % 5 == 0
    }
    output_dir = args.output_dir.resolve()
    retained_indices = None
    feature_filter = None
    if args.retained_feature_model is not None:
        import joblib

        feature_model_path = args.retained_feature_model.resolve()
        require_frozen_branch_a_refinement(gate_dir, feature_model_path)
        bundle = joblib.load(feature_model_path)
        retained_indices = list(map(int, bundle["retained_feature_indices"]))
        feature_filter = {
            "ablation": bundle["ablation"],
            "retained_feature_count": len(retained_indices),
            "source_model": str(feature_model_path),
            "source_model_sha256": _sha256(feature_model_path),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    group_sets = {}
    for split in ("train", "validation"):
        selected_groups = (
            validation_groups if split == "validation" else set(groups) - validation_groups
        )
        examples = [
            qwen_row(row, args.seed, retained_indices)
            for group in sorted(selected_groups)
            for row in sorted(by_group[group], key=lambda item: item["record_id"])
        ]
        temporary = output_dir / f"{split}.jsonl.tmp.{os.getpid()}"
        with temporary.open("w") as stream:
            for example in examples:
                stream.write(json.dumps(example, sort_keys=True) + "\n")
        os.replace(temporary, output_dir / f"{split}.jsonl")
        counts[split] = len(examples)
        group_sets[split] = sorted(selected_groups)
    manifest = {
        "version": VERSION,
        "source_gate_dir": str(gate_dir),
        "selected_probe": selection,
        "counts": counts,
        "groups": {key: len(value) for key, value in group_sets.items()},
        "group_ids": group_sets,
        "group_overlap": sorted(set(group_sets["train"]) & set(group_sets["validation"])),
        "candidate_order_randomized": True,
        "candidate_order_seed_basis": "root_seed_and_case_id_only_hidden_gain_independent",
        "hidden_gain_present_in_prompt": False,
        "hidden_gain_present_only_in_assistant_supervision": True,
        "feature_filter": feature_filter,
        "protected_set_used": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
