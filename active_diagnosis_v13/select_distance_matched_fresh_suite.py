#!/usr/bin/env python3
"""Select an outcome-free fresh suite matched to development setup difficulty."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from active_diagnosis_v13.analyze_control_step_budget import _read_jsonl

VERSION = "active_diagnosis_v13_distance_matched_fresh_suite_v1"
STRATA = (
    "one_step_reachable_interior",
    "multi_step_reachable_interior",
    "reachable_boundary_or_clipping",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reference_setups(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path.resolve())
    if len(rows) != 150:
        raise ValueError("reference control source must contain 150 development episodes")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    if len(groups) != 30:
        raise ValueError("reference control source must contain 30 development groups")
    setups = []
    for group, values in sorted(groups.items()):
        distances = {float(row["initial_normalized_distance"]) for row in values}
        strata = {str(row["stratum"]) for row in values}
        if len(values) != 5 or len(distances) != 1 or len(strata) != 1:
            raise ValueError(f"reference group {group} is inconsistent")
        setups.append(
            {
                "group_id": group,
                "stratum": next(iter(strata)),
                "initial_normalized_distance": next(iter(distances)),
            }
        )
    if Counter(row["stratum"] for row in setups) != Counter({name: 10 for name in STRATA}):
        raise ValueError("reference development setups are not ten-per-stratum")
    return setups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-suite", type=Path, required=True)
    parser.add_argument("--reference-development-control", type=Path, required=True)
    parser.add_argument("--suite-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate_suite.resolve().read_text())
    cases = candidate["cases"]
    if len(cases) < 60 or len(cases) % len(STRATA):
        raise ValueError("candidate suite must provide a large balanced setup pool")
    if int(candidate["validation"]["prior_group_id_overlap"]) != 0 or int(
        candidate["validation"]["prior_setup_hash_overlap"]
    ) != 0:
        raise ValueError("candidate pool overlaps an excluded suite")
    per_stratum = len(cases) // len(STRATA)
    if Counter(str(row["stratum"]) for row in cases) != Counter(
        {name: per_stratum for name in STRATA}
    ):
        raise ValueError("candidate pool is not balanced by stratum")
    references = _reference_setups(args.reference_development_control)
    selected = []
    matching_rows = []
    for stratum_index, stratum in enumerate(STRATA):
        target = sorted(
            (row for row in references if row["stratum"] == stratum),
            key=lambda row: (row["initial_normalized_distance"], row["group_id"]),
        )
        pool = sorted(
            (row for row in cases if str(row["stratum"]) == stratum),
            key=lambda row: (float(row["initial_normalized_distance"]), row["group_id"]),
        )
        target_distance = np.log1p(
            np.asarray([row["initial_normalized_distance"] for row in target])
        )
        pool_distance = np.log1p(
            np.asarray([float(row["initial_normalized_distance"]) for row in pool])
        )
        cost = np.abs(target_distance[:, None] - pool_distance[None, :])
        reference_indices, candidate_indices = linear_sum_assignment(cost)
        pairs = sorted(
            zip(reference_indices, candidate_indices, strict=True),
            key=lambda pair: target[pair[0]]["initial_normalized_distance"],
        )
        for output_index, (reference_index, candidate_index) in enumerate(pairs):
            source = copy.deepcopy(pool[candidate_index])
            source_case_id = str(source["case_id"])
            source_group_id = str(source["group_id"])
            new_id = f"v12_mpcdiag_{args.suite_label}_{stratum_index:02d}_{output_index:04d}"
            source["selection_source_case_id"] = source_case_id
            source["selection_source_group_id"] = source_group_id
            source["case_id"] = new_id
            source["group_id"] = new_id
            selected.append(source)
            matching_rows.append(
                {
                    "stratum": stratum,
                    "reference_group_id": target[reference_index]["group_id"],
                    "reference_initial_normalized_distance": float(
                        target[reference_index]["initial_normalized_distance"]
                    ),
                    "candidate_source_group_id": source_group_id,
                    "selected_group_id": new_id,
                    "selected_setup_hash": source["setup_hash"],
                    "selected_initial_normalized_distance": float(
                        source["initial_normalized_distance"]
                    ),
                    "absolute_log1p_distance_mismatch": float(
                        cost[reference_index, candidate_index]
                    ),
                }
            )
    if len(selected) != 30 or len({row["setup_hash"] for row in selected}) != 30:
        raise ValueError("matching did not select thirty unique setups")
    distances = [float(row["initial_normalized_distance"]) for row in selected]
    validation = {
        **candidate["validation"],
        "groups": 30,
        "unique_group_ids": 30,
        "unique_setup_hashes": 30,
        "initial_strict_successes": 0,
        "minimum_initial_distance": min(distances),
        "maximum_initial_distance": max(distances),
        "distance_band_counts": dict(Counter(row["initial_distance_band"] for row in selected)),
        "stratum_counts": dict(Counter(row["stratum"] for row in selected)),
        "prior_group_id_overlap": 0,
        "prior_setup_hash_overlap": 0,
        "q_goal_deployed_input": False,
        "q_goal_replay_max_abs_metric_error": max(
            float(row["exact_q_goal_replay_max_abs_metric_error"]) for row in selected
        ),
    }
    mean_mismatch = float(
        np.mean([row["absolute_log1p_distance_mismatch"] for row in matching_rows])
    )
    maximum_mismatch = max(
        float(row["absolute_log1p_distance_mismatch"]) for row in matching_rows
    )
    output = {
        **candidate,
        "selection_version": VERSION,
        "suite_label": args.suite_label,
        "cases": selected,
        "validation": validation,
        "outcome_free_difficulty_matching": {
            "protected_set_used": False,
            "selection_or_retuning_on_candidate_outcomes": False,
            "candidate_suite": str(args.candidate_suite.resolve()),
            "candidate_suite_sha256": _sha256(args.candidate_suite),
            "reference_development_control": str(
                args.reference_development_control.resolve()
            ),
            "reference_development_control_sha256": _sha256(
                args.reference_development_control
            ),
            "reference_groups": 30,
            "candidate_pool_groups": len(cases),
            "selected_groups": 30,
            "matching_objective": (
                "minimum total absolute log1p initial-normalized-distance mismatch "
                "within each stratum via deterministic linear assignment"
            ),
            "mean_absolute_log1p_distance_mismatch": mean_mismatch,
            "maximum_absolute_log1p_distance_mismatch": maximum_mismatch,
            "matches": matching_rows,
            "interpretation_guard": (
                "Selection uses only pre-treatment stratum and initial normalized distance "
                "from development setups. No candidate control outcome, protected setup "
                "covariate, protected result, probe label, or sequential-rule result is used; "
                "protected IDs and hashes are exclusion-only in the upstream pool builder."
            ),
        },
    }
    _atomic_json(args.output.resolve(), output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": _sha256(args.output),
                "selected_groups": 30,
                "mean_absolute_log1p_distance_mismatch": mean_mismatch,
                "maximum_absolute_log1p_distance_mismatch": maximum_mismatch,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
