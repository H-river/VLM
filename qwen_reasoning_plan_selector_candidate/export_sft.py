#!/usr/bin/env python3
"""Export real-rollout-only Qwen-VL plan-ranking supervision after the gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .core import build_plan_bank
from .protocol import PLAN_NAMES, SYSTEM_PROMPT, USER_PREFIX
from .selector_contract import canonical_selector_text, parse_selector_output


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPOSITORY_ROOT / "artifacts/qwen_reasoning_plan_selector"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, path)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _plan_definitions() -> list[dict[str, Any]]:
    return [build_plan_bank()[name].to_dict() for name in PLAN_NAMES]


def _visible_state(
    group: Mapping[str, Any], representative: Mapping[str, Any]
) -> dict[str, Any]:
    if not representative["trace"]:
        raise ValueError(f"training group has no executed trace: {group['group_id']}")
    initial = representative["trace"][0]
    probe = representative["initial_probe_audit"]
    return {
        "current_five_dimensional_measurement": initial["measured_metrics_before"],
        "target_five_dimensional_measurement": group["target_metrics"],
        "actuator_state_mm": group["initial_positions_mm"],
        "legal_bounds": group["bounds"],
        "learned_h1_probes": probe["rows"],
        "probe_safe_subset": probe["safe_subset"],
        "ensemble_uncertainty_summary": {
            row["actuator"]: row["mean_uncertainty"] for row in probe["rows"]
        },
        "executable_plan_definitions": _plan_definitions(),
        "controller_invariants": {
            "gain_and_action_bound_policy": "identical_for_all_plans",
            "cem_population": 24,
            "cem_iterations": 3,
            "horizon": 1,
            "maximum_control_steps": 4,
            "strict_all_five_tolerance": 1.0,
        },
    }


def _validate_row(row: Mapping[str, Any]) -> None:
    if set(row) != {"example_id", "split", "images", "prompt", "completion", "metadata"}:
        raise ValueError("unexpected SFT row fields")
    if row["split"] not in {"train", "dev"}:
        raise ValueError("only candidate train/dev SFT rows are allowed")
    if [message["role"] for message in row["prompt"]] != ["system", "user"]:
        raise ValueError("prompt roles changed")
    if row["prompt"][0]["content"] != [{"type": "text", "text": SYSTEM_PROMPT}]:
        raise ValueError("system prompt changed")
    user = row["prompt"][1]["content"]
    if len(user) != 2 or user[0] != {"type": "image"}:
        raise ValueError("one image must precede structured state")
    text = user[1]["text"]
    if not text.startswith(USER_PREFIX):
        raise ValueError("user prefix changed")
    visible = json.loads(text[len(USER_PREFIX) :])
    forbidden = {"group_id", "base_state_id", "split", "visual_family", "oracle", "outcome", "q_goal"}
    if forbidden & set(visible):
        raise ValueError("hidden identity/outcome leaked into prompt")
    completion = row["completion"]
    if len(completion) != 1 or completion[0]["role"] != "assistant":
        raise ValueError("completion shape changed")
    target_text = completion[0]["content"][0]["text"]
    if canonical_selector_text(parse_selector_output(target_text)) != target_text:
        raise ValueError("completion is not canonical selector JSON")
    image_path = REPOSITORY_ROOT / row["images"][0]
    if not image_path.is_file() or sha256_path(image_path) != row["metadata"]["image_sha256"]:
        raise ValueError("training image identity mismatch")


def export(args: argparse.Namespace) -> None:
    artifact = args.artifact.resolve()
    audit = json.loads((artifact / "anti_collapse_audit.json").read_text())
    if not bool(audit.get("gate_passed", False)):
        raise RuntimeError("anti-collapse gate did not pass; Qwen SFT is forbidden")
    confirmation_path = artifact / "split_manifests/confirmation_frozen.json"
    rollout_config = json.loads((artifact / "rollout_config.json").read_text())
    if sha256_path(confirmation_path) != rollout_config["confirmation_manifest_sha256"]:
        raise RuntimeError("confirmation manifest changed before training")
    groups = read_jsonl(artifact / "split_manifests/groups.jsonl")
    group_by_id = {str(row["group_id"]): row for row in groups}
    rollouts = read_jsonl(artifact / "rollout_results.jsonl")
    representatives = {
        str(row["group_id"]): row
        for row in rollouts
        if row["plan_name"] == "direct_all_five" and int(row["seed_index"]) == 0
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id, ranking in audit["group_rankings"].items():
        group = group_by_id[group_id]
        if group["split"] == "candidate_confirmation" or not bool(ranking["decisive"]):
            continue
        split = "train" if group["split"] == "candidate_train" else "dev"
        ordered_plans = sorted(
            [row for row in read_plan_outcomes(artifact / "plan_outcomes.csv", group_id)],
            key=lambda row: int(row["rank"]),
        )
        plan_ranking = [str(row["plan_name"]) for row in ordered_plans]
        target = {"plan_ranking": plan_ranking, "selected_plan": plan_ranking[0]}
        image_path = artifact / str(group["initial_sensor_image"])
        visible = _visible_state(group, representatives[group_id])
        row = {
            "example_id": f"qrps_{hashlib.sha256(group_id.encode()).hexdigest()[:20]}",
            "split": split,
            "images": [_repo_relative(image_path)],
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": USER_PREFIX + json.dumps(visible, sort_keys=True, separators=(",", ":"), allow_nan=False),
                        },
                    ],
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": canonical_selector_text(target)}],
                }
            ],
            "metadata": {
                "candidate_only": True,
                "source_group_id": group_id,
                "base_state_id": group["base_state_id"],
                "image_sha256": sha256_path(image_path),
                "label_source": "three_seed_real_corrected_simulator_closed_loop_ranking",
                "rollout_sha256": sha256_path(artifact / "rollout_results.jsonl"),
                "confirmation_excluded": True,
            },
        }
        _validate_row(row)
        rows_by_split[split].append(row)
    for split in rows_by_split:
        rows_by_split[split].sort(key=lambda row: row["example_id"])
    if not rows_by_split["train"] or not rows_by_split["dev"]:
        raise RuntimeError("decisive train/dev exports must both be nonempty")
    training_dir = artifact / "training"
    train_path = training_dir / "sft_train.jsonl"
    dev_path = training_dir / "sft_dev.jsonl"
    write_jsonl(train_path, rows_by_split["train"])
    write_jsonl(dev_path, rows_by_split["dev"])
    (training_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    report = {
        "version": "qwen_reasoning_plan_selector_sft_export_v1",
        "candidate_only": True,
        "gate_sha256": sha256_path(artifact / "anti_collapse_audit.json"),
        "confirmation_manifest_sha256": sha256_path(confirmation_path),
        "hard_labels_from_real_rollouts_only": True,
        "ambiguous_groups_excluded": True,
        "records": {
            "train": {"path": _repo_relative(train_path), "count": len(rows_by_split["train"]), "sha256": sha256_path(train_path)},
            "dev": {"path": _repo_relative(dev_path), "count": len(rows_by_split["dev"]), "sha256": sha256_path(dev_path)},
        },
    }
    atomic_json(training_dir / "sft_export_report.json", report)
    print(json.dumps(report, sort_keys=True))


def read_plan_outcomes(path: Path, group_id: str) -> list[dict[str, Any]]:
    import csv

    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["group_id"] == group_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
