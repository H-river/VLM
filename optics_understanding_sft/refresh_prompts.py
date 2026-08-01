#!/usr/bin/env python3
"""Refresh label-independent prompts and exports without rerunning optical physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .build_dataset import PROMPT_CONTRACT_VERSION, canonical_public, prompt_text
from .core import file_sha256, load_yaml, make_messages, make_qwen_record, read_jsonl, write_jsonl


TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "templates.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--add-setup-actuator-interface",
        action="store_true",
        help="Make setup adjustability prompt-visible for revised training-only datasets.",
    )
    return parser.parse_args()


def template_for_record(record: Mapping[str, Any], templates: Mapping[str, Any]) -> str:
    template_id = str(record["provenance"]["template_id"])
    task_type, group, raw_index = template_id.rsplit(":", 2)
    if task_type != record["task_type"]:
        raise ValueError(f"Template task mismatch for {record['example_id']}: {template_id}")
    return str(templates[task_type][group][int(raw_index)])


def refresh(dataset_dir: Path, *, add_setup_actuator_interface: bool = False) -> dict[str, Any]:
    templates = load_yaml(TEMPLATE_PATH)
    masters = read_jsonl(dataset_dir / "master" / "cases.jsonl")
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for master in masters:
        for item in master["records"]:
            record = item["record"]
            if add_setup_actuator_interface and record["task_type"] == "setup_interpretation":
                record["prompt_inputs"]["actuator_interface"] = {
                    "adjustable_parameters": ["lens_x", "lens_y", "camera_x", "camera_y"],
                    "fixed_during_alignment": [
                        "lens_focal_length_mm",
                        "source_to_lens_mm",
                        "lens_to_camera_mm",
                        "beam_waist_mm",
                        "wavelength_nm",
                    ],
                }
            template = template_for_record(record, templates)
            record["prompt"] = prompt_text(template, record["task_type"], record["prompt_inputs"])
            record["provenance"]["prompt_contract_version"] = PROMPT_CONTRACT_VERSION
            rows_by_split[record["split"]].append(record)
    for rows in rows_by_split.values():
        rows.sort(key=lambda row: row["example_id"])

    write_jsonl(dataset_dir / "master" / "cases.jsonl", masters)
    write_jsonl(dataset_dir / "canonical" / "train.jsonl", (canonical_public(row, True) for row in rows_by_split["train"]))
    write_jsonl(dataset_dir / "canonical" / "val.jsonl", (canonical_public(row, True) for row in rows_by_split["val"]))
    write_jsonl(dataset_dir / "canonical" / "test_prompts.jsonl", (canonical_public(row, False) for row in rows_by_split["test"]))
    write_jsonl(
        dataset_dir / "private" / "test_labels.jsonl",
        (
            {
                "example_id": item["record"]["example_id"],
                "group_id": master["group_id"],
                "target": item["record"]["target"],
                "private_eval": item["private_eval"],
            }
            for master in masters
            if master["split"] == "test"
            for item in master["records"]
        ),
    )
    for split, rows in rows_by_split.items():
        include_target = split != "test"
        write_jsonl(dataset_dir / "exports" / "messages" / f"{split}.jsonl", (make_messages(row, include_target) for row in rows))
        write_jsonl(dataset_dir / "exports" / "qwen" / f"{split}.jsonl", (make_qwen_record(row, include_target) for row in rows))

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_contract_version"] = PROMPT_CONTRACT_VERSION
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_paths = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", "audit_report.json"}
    )
    lines = [f"{file_sha256(path)}  {path.relative_to(dataset_dir).as_posix()}" for path in checksum_paths]
    (dataset_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "record_counts": {split: len(rows) for split, rows in rows_by_split.items()},
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            refresh(
                args.dataset_dir,
                add_setup_actuator_interface=args.add_setup_actuator_interface,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
