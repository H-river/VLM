#!/usr/bin/env python3
"""Audit direct-reasoning orchestration reconstruction, registries, and prompt leakage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .core import read_jsonl
from .direct_reasoning_tools_v11 import materialize_direct_answer, run_mapped_direct_tool
from .evaluate_direct_tool_orchestration_v11 import (
    load_observation_registry,
    load_setup_registry,
)


FORBIDDEN_PROMPT_TERMS = (
    "setup_config",
    "expected_state",
    "private_eval",
    "replay_specs",
    "observation_registry",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    records = read_jsonl(args.dataset_dir / "canonical" / "dev.jsonl")
    setups = load_setup_registry(args.dataset_dir / "private" / "setup_registry.jsonl")
    observations = load_observation_registry(
        args.dataset_dir / "private" / "observation_registry.jsonl"
    )
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_source[str(row["source_example_id"])].append(row)
    failures: list[dict[str, str]] = []

    def fail(source: str, kind: str, detail: str) -> None:
        failures.append({"source_example_id": source, "kind": kind, "detail": detail})

    for source, rows in sorted(by_source.items()):
        stages = {str(row["stage"]): row for row in rows}
        if set(stages) != {
            "tool_choice",
            "tool_source_mapping",
            "compact_tool_result_interpretation",
        }:
            fail(source, "stage_set", repr(sorted(stages)))
            continue
        choice = stages["tool_choice"]
        mapping = stages["tool_source_mapping"]
        interpretation = stages["compact_tool_result_interpretation"]
        tool_name = str(choice["target"].get("tool_name"))
        if mapping["target"].get("tool_name") != tool_name:
            fail(source, "tool_name_mismatch", tool_name)
            continue
        try:
            result = run_mapped_direct_tool(
                tool_name,
                mapping["prompt_inputs"]["visible_evidence"],
                mapping["target"]["source_map"],
                setup_registry=setups,
                observation_registry=observations,
            )
        except Exception as exc:  # audit should record the exact failed source
            fail(source, "tool_execution", repr(exc))
            continue
        if result != interpretation["prompt_inputs"].get("tool_result"):
            fail(source, "cached_tool_result", "execution differs from interpretation input")
        if materialize_direct_answer(tool_name, result) != interpretation["target"]:
            fail(source, "target_reconstruction", "materialized answer differs from target")
        public_text = "\n".join(str(row["prompt"]) for row in rows)
        for term in FORBIDDEN_PROMPT_TERMS:
            if term in public_text:
                fail(source, "prompt_leakage", term)

    expected_stages = 3 * len(by_source)
    report = {
        "passed": not failures and len(records) == expected_stages,
        "record_count": len(records),
        "source_count": len(by_source),
        "expected_record_count": expected_stages,
        "setup_registry_count": len(setups),
        "observation_registry_count": len(observations),
        "failure_count": len(failures),
        "failures": failures[:50],
        "checks": {
            "three_stages_per_source": len(records) == expected_stages,
            "tool_execution_reconstructs_cached_results": not any(
                item["kind"] in {"tool_execution", "cached_tool_result"} for item in failures
            ),
            "materialized_targets_reconstruct_exactly": not any(
                item["kind"] == "target_reconstruction" for item in failures
            ),
            "no_private_prompt_terms": not any(
                item["kind"] == "prompt_leakage" for item in failures
            ),
        },
        "sealed_evaluation_opened": False,
    }
    output = args.output_json or args.dataset_dir / "audit_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
