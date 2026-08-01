#!/usr/bin/env python3
"""Build a decision warm-up and a disjoint mixed-preservation v4 curriculum."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_corrective_curriculum import completion_target
from .core import file_sha256, read_jsonl, stable_json_hash, write_jsonl


ANCHOR_TASKS = (
    "setup_interpretation",
    "causal_effects",
    "forward_prediction",
    "diagnosis",
    "counterfactual_reasoning",
)
VISUAL_QUOTAS = {
    "setup_interpretation": 0,
    "causal_effects": 2,
    "forward_prediction": 3,
    "diagnosis": 2,
    "counterfactual_reasoning": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-jsonl", type=Path, required=True)
    parser.add_argument("--hard-canonical-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--warmup-output-jsonl", type=Path, required=True)
    parser.add_argument("--mixed-output-jsonl", type=Path, required=True)
    parser.add_argument("--diagnostic-output-jsonl", type=Path, required=True)
    parser.add_argument("--holdout-output-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9041)
    return parser.parse_args()


def grouped_hard_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    if any(len(members) != 4 for members in groups.values()):
        raise ValueError("every hard-pair physical scenario must contain four records")
    return groups


def prepare_anchor(row: Mapping[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    original_id = str(item["example_id"])
    item["source_example_id"] = str(item.get("source_example_id", original_id))
    item["example_id"] = f"{original_id}__hard_v4_anchor"
    item["curriculum_source"] = "hard_v4_preservation_anchor"
    item.pop("match_group_id", None)
    for image in item.get("images", []):
        if not Path(image).exists():
            raise FileNotFoundError(image)
    return item


def select_anchors(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for index, task in enumerate(ANCHOR_TASKS):
        candidates = [row for row in rows if row["task_type"] == task]
        visual = sorted((row for row in candidates if row.get("images")), key=lambda row: row["example_id"])
        text = sorted((row for row in candidates if not row.get("images")), key=lambda row: row["example_id"])
        rng = random.Random(seed + index * 1009)
        rng.shuffle(visual)
        rng.shuffle(text)
        visual_quota = VISUAL_QUOTAS[task]
        text_quota = 40 - visual_quota
        if len(visual) < visual_quota or len(text) < text_quota:
            raise ValueError(f"insufficient text/visual anchors for {task}")
        selected = visual[:visual_quota] + text[:text_quota]
        rng.shuffle(selected)
        result[task] = [prepare_anchor(row) for row in selected]
    return result


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        result[f"task:{row['task_type']}"] += 1
        result[f"status:{row['task_type']}:{completion_target(row)['status']}"] += 1
        result[f"modality:{'visual' if row.get('images') else 'text'}"] += 1
        result[f"source:{row.get('curriculum_source', 'hard_pairs_v4')}"] += 1
    return dict(sorted(result.items()))


def write_with_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    name: str,
    seed: int,
    source_hashes: Mapping[str, str],
) -> None:
    write_jsonl(path, rows)
    manifest = {
        "name": name,
        "seed": seed,
        "record_count": len(rows),
        "unique_example_count": len({row["example_id"] for row in rows}),
        "unique_source_count": len(
            {str(row.get("source_example_id", row["example_id"])) for row in rows}
        ),
        "physical_group_count": len({row.get("group_id") for row in rows if row.get("curriculum_source") != "hard_v4_preservation_anchor"}),
        "example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
        "counts": counts(rows),
        **source_hashes,
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(
    hard_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    groups = grouped_hard_rows(hard_rows)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    if len(group_ids) < 150:
        raise ValueError("need at least 150 disjoint hard-pair scenarios")

    warmup = [row for group_id in group_ids[:100] for row in groups[group_id]]
    anchors = select_anchors(anchor_rows, seed + 1)
    anchor_stream = [
        anchors[ANCHOR_TASKS[index % len(ANCHOR_TASKS)]][index // len(ANCHOR_TASKS)]
        for index in range(200)
    ]
    mixed: list[dict[str, Any]] = []
    for index, group_id in enumerate(group_ids[100:150]):
        mixed.extend(groups[group_id])
        mixed.extend(anchor_stream[index * 4 : index * 4 + 4])

    if len(warmup) != 400 or len(mixed) != 400:
        raise AssertionError("v4 curricula must each contain 400 records")
    if {row["group_id"] for row in warmup} & {
        row["group_id"]
        for row in mixed
        if row.get("curriculum_source") != "hard_v4_preservation_anchor"
    }:
        raise AssertionError("warm-up and mixed hard scenarios overlap")
    return warmup, mixed, group_ids[150:180], group_ids[180:]


def main() -> None:
    args = parse_args()
    hard_rows = read_jsonl(args.hard_jsonl)
    hard_canonical = read_jsonl(args.hard_canonical_jsonl)
    anchor_rows = read_jsonl(args.anchor_jsonl)
    warmup, mixed, diagnostic_group_ids, holdout_group_ids = build(
        hard_rows, anchor_rows, args.seed
    )
    canonical_groups = grouped_hard_rows(hard_canonical)
    diagnostic = [
        row for group_id in diagnostic_group_ids for row in canonical_groups[group_id]
    ]
    holdout = [row for group_id in holdout_group_ids for row in canonical_groups[group_id]]
    source_hashes = {
        "hard_pool_sha256": file_sha256(args.hard_jsonl),
        "anchor_pool_sha256": file_sha256(args.anchor_jsonl),
    }
    write_with_manifest(
        args.warmup_output_jsonl,
        warmup,
        name="hard_pairs_v4_decision_warmup",
        seed=args.seed,
        source_hashes=source_hashes,
    )
    write_with_manifest(
        args.mixed_output_jsonl,
        mixed,
        name="hard_pairs_v4_mixed_preservation",
        seed=args.seed,
        source_hashes=source_hashes,
    )
    for path, rows, groups, name in (
        (
            args.diagnostic_output_jsonl,
            diagnostic,
            diagnostic_group_ids,
            "hard_pairs_v4_checkpoint_diagnostic",
        ),
        (
            args.holdout_output_jsonl,
            holdout,
            holdout_group_ids,
            "hard_pairs_v4_scenario_disjoint_holdout",
        ),
    ):
        write_jsonl(path, rows)
        path.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "seed": args.seed,
                    "record_count": len(rows),
                    "physical_group_count": len(groups),
                    "example_ids_hash": stable_json_hash([row["example_id"] for row in rows]),
                    "source_sha256": file_sha256(args.hard_canonical_jsonl),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "warmup": counts(warmup),
                "mixed": counts(mixed),
                "diagnostic_records": len(diagnostic),
                "holdout_records": len(holdout),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
