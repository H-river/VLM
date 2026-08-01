#!/usr/bin/env python3
"""Bind a fresh group-disjoint diagnostic suite into a post-freeze v13 config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

VERSION = "active_diagnosis_v13_fresh_holdout_config_v1"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--fresh-suite", type=Path, required=True)
    parser.add_argument("--reference-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.base_config.resolve().read_text())
    fresh = json.loads(args.fresh_suite.resolve().read_text())
    reference = json.loads(args.reference_suite.resolve().read_text())
    if int(fresh["validation"]["groups"]) != 30:
        raise ValueError("fresh holdout suite must contain exactly thirty setups")
    if int(fresh["validation"]["prior_group_id_overlap"]) != 0 or int(
        fresh["validation"]["prior_setup_hash_overlap"]
    ) != 0:
        raise ValueError("fresh suite reports overlap with an excluded source")
    matching = fresh.get("outcome_free_difficulty_matching")
    if matching is not None and (
        matching.get("protected_set_used") is not False
        or matching.get("selection_or_retuning_on_candidate_outcomes") is not False
        or int(matching.get("candidate_pool_groups", 0)) < 60
        or int(matching.get("selected_groups", 0)) != 30
    ):
        raise ValueError("difficulty-matched fresh suite lacks outcome-free provenance")
    fresh_groups = {str(row["group_id"]) for row in fresh["cases"]}
    fresh_hashes = {str(row["setup_hash"]) for row in fresh["cases"]}
    reference_groups = {str(row["group_id"]) for row in reference["cases"]}
    reference_hashes = {str(row["setup_hash"]) for row in reference["cases"]}
    if fresh_groups & reference_groups or fresh_hashes & reference_hashes:
        raise ValueError("fresh and reference suites overlap")
    if {int(row["case_id"].rsplit("_", 1)[1]) for row in fresh["cases"]} != set(
        range(10)
    ):
        raise ValueError("fresh suite must contain suffixes 0000 through 0009")
    baseline = dict(config["baseline"])
    baseline["evaluation_suite"] = str(args.fresh_suite.resolve())
    baseline["evaluation_suite_sha256"] = _sha256(args.fresh_suite)
    output = {
        **config,
        "baseline": baseline,
        "split": {
            **config["split"],
            "rule": (
                "fresh_group_and_setup_hash_disjoint_nonprotected_suite_suffixes_"
                "0000_to_0009_used_once_postfreeze"
            ),
            "development_groups": 30,
            "protected_heldout_groups": 0,
            "protected_results_may_not_be_loaded_or_generated_before_freeze": True,
        },
        "postfreeze_holdout": {
            "version": VERSION,
            "selection_or_retuning_allowed": False,
            "protected_set_used": False,
            "base_config": str(args.base_config.resolve()),
            "base_config_sha256": _sha256(args.base_config),
            "reference_suite": str(args.reference_suite.resolve()),
            "reference_suite_sha256": _sha256(args.reference_suite),
            "fresh_suite": str(args.fresh_suite.resolve()),
            "fresh_suite_sha256": _sha256(args.fresh_suite),
            "group_id_overlap": 0,
            "setup_hash_overlap": 0,
            "outcome_free_difficulty_matching": matching,
        },
    }
    _atomic_json(args.output.resolve(), output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": _sha256(args.output),
                "fresh_groups": len(fresh_groups),
                "group_id_overlap": 0,
                "setup_hash_overlap": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
