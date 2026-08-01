"""Deterministically export supervisor manifests to Qwen2.5-VL chat JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .contracts import SYSTEM_PROMPT, TARGET_KEYS, canonical_target_text, model_visible_state, parse_target_strict
from .validate_manifest import load_manifest, validate_manifest_files


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"qwen_vl_supervisor_v1\x1f{seed}\x1f{sample_id}".encode()).hexdigest()


def user_text(record: dict[str, Any]) -> str:
    state = model_visible_state(record)
    return "Current supervisor state (all coordinates and units are explicit):\n" + json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def export_record(record: dict[str, Any], *, manifest_sha256: str) -> dict[str, Any]:
    target = {key: record["target"][key] for key in TARGET_KEYS}
    completion = canonical_target_text(target)
    parse_target_strict(completion)
    visible_user = user_text(record)
    return {
        "example_id": record["sample_id"],
        "split": record["split"],
        "images": [record["assets"]["current_image_path"]],
        "prompt": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": visible_user},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": completion}],
            }
        ],
        "metadata": {
            "manifest_schema_version": record["schema_version"],
            "manifest_sha256": manifest_sha256,
            "image_sha256": record["assets"]["current_image_sha256"],
            "counterfactual_pair_id": record["counterfactual_pair_id"],
            "target_field_mask": record["target"]["field_mask"],
            "anomaly_family": record["provenance"]["anomaly_family"],
            "width_quartile": record["provenance"]["width_quartile"],
            "boundary_status": record["provenance"]["boundary_status"],
            "severity_bucket": record["provenance"]["severity_bucket"],
        },
    }


def select_balanced(records: list[dict[str, Any]], *, per_class: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for diagnosis in ("nominal", "sensor_saturation", "secondary_reflection"):
        candidates = [row for row in records if row["target"]["diagnosis"] == diagnosis]
        candidates.sort(key=lambda row: (stable_rank(seed, row["sample_id"]), row["sample_id"]))
        if len(candidates) < per_class:
            raise ValueError(f"need {per_class} {diagnosis} examples, found {len(candidates)}")
        selected.extend(candidates[:per_class])
    selected.sort(key=lambda row: (stable_rank(seed + 1, row["sample_id"]), row["sample_id"]))
    return selected


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _visible_text(row: dict[str, Any], role: str) -> str:
    return "\n".join(
        item["text"]
        for message in row["prompt"] + row["completion"]
        if message["role"] == role
        for item in message["content"]
        if item["type"] == "text"
    )


def validate_export_row(row: dict[str, Any], *, repository_root: Path) -> None:
    if set(row) != {"example_id", "split", "images", "prompt", "completion", "metadata"}:
        raise ValueError(f"{row.get('example_id')}: unexpected export fields")
    if len(row["images"]) != 1:
        raise ValueError(f"{row['example_id']}: exactly one current image is required")
    image_slots = sum(
        item.get("type") == "image"
        for message in row["prompt"]
        for item in message.get("content", [])
    )
    if image_slots != 1:
        raise ValueError(f"{row['example_id']}: expected one image placeholder, got {image_slots}")
    image_path = (repository_root / row["images"][0]).resolve()
    if not image_path.is_file():
        raise ValueError(f"{row['example_id']}: missing image {image_path}")
    with Image.open(image_path) as image:
        image.load()
    if sha256_path(image_path) != row["metadata"]["image_sha256"]:
        raise ValueError(f"{row['example_id']}: exported image hash mismatch")
    roles = [message.get("role") for message in row["prompt"] + row["completion"]]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{row['example_id']}: roles must be system,user,assistant")
    user = _visible_text(row, "user")
    state_prefix = "Current supervisor state (all coordinates and units are explicit):\n"
    if not user.startswith(state_prefix):
        raise ValueError(f"{row['example_id']}: missing structured-state prefix")
    visible_state = json.loads(user[len(state_prefix):])
    if set(visible_state) != {
        "decision_context", "current_metrics", "goal_metrics", "recent_history",
        "remaining_step_budget", "actuator_constraints",
    }:
        raise ValueError(f"{row['example_id']}: visible state allow-list changed")
    required_current = {"centroid_x", "centroid_y", "width_x", "width_y", "peak_intensity", "coordinate_frame"}
    if set(visible_state["current_metrics"]) != required_current or set(visible_state["goal_metrics"]) != required_current:
        raise ValueError(f"{row['example_id']}: current/goal metrics were lost")
    forbidden_fragments = (
        "source_cohort", "source_manifest", "source_sample", "setup_hash", "setup_id",
        "pair_id", "counterfactual", "generator_version", "severity_bucket", "boundary_status",
        "width_quartile", "oracle", "frozen_iid", "frozen_ood", "current_image_path", ".png",
    )
    lowered_user = user.lower()
    found = [fragment for fragment in forbidden_fragments if fragment in lowered_user]
    if found:
        raise ValueError(f"{row['example_id']}: hidden metadata leaked into user prompt: {found}")
    assistant = _visible_text(row, "assistant")
    parse_target_strict(assistant)


def render_example(row: dict[str, Any]) -> str:
    system = _visible_text(row, "system")
    user = _visible_text(row, "user")
    assistant = _visible_text(row, "assistant")
    return (
        "# Redacted model-visible example\n\n"
        "This rendering contains only the messages visible to the model. The image is represented by a placeholder; paths and hidden manifest metadata are omitted.\n\n"
        "## System\n\n```text\n" + system + "\n```\n\n"
        "## User\n\n`[CURRENT_BEAM_IMAGE]`\n\n```text\n" + user + "\n```\n\n"
        "## Assistant\n\n```json\n" + assistant + "\n```\n"
    )


def export(
    *,
    repository_root: Path,
    train_manifest: Path,
    dev_manifest: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    train_manifest = train_manifest.resolve()
    dev_manifest = dev_manifest.resolve()
    validate_manifest_files([train_manifest, dev_manifest], repository_root=repository_root)
    train_records = load_manifest(train_manifest)
    dev_records = load_manifest(dev_manifest)
    train_hash = sha256_path(train_manifest)
    dev_hash = sha256_path(dev_manifest)
    train_export = [export_record(row, manifest_sha256=train_hash) for row in train_records]
    dev_export = [export_record(row, manifest_sha256=dev_hash) for row in dev_records]
    smoke_train_records = select_balanced(train_records, per_class=12, seed=seed)
    smoke_dev_records = select_balanced(dev_records, per_class=4, seed=seed + 1000)
    smoke_train = [export_record(row, manifest_sha256=train_hash) for row in smoke_train_records]
    smoke_dev = [export_record(row, manifest_sha256=dev_hash) for row in smoke_dev_records]
    for row in train_export + dev_export:
        validate_export_row(row, repository_root=repository_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": (output_dir / "sft_train.jsonl", train_export),
        "dev": (output_dir / "sft_dev.jsonl", dev_export),
        "smoke_train": (output_dir / "sft_smoke_train.jsonl", smoke_train),
        "smoke_dev": (output_dir / "sft_smoke_dev.jsonl", smoke_dev),
        "sample": (output_dir / "sample_qwen_vl.jsonl", [
            next(row for row in train_export if parse_target_strict(_visible_text(row, "assistant"))["diagnosis"] == diagnosis)
            for diagnosis in ("nominal", "sensor_saturation", "secondary_reflection")
        ]),
    }
    for _, (path, rows) in outputs.items():
        write_jsonl(path, rows)

    view_dir = output_dir / "manifest_views"
    smoke_manifest_views = {
        "smoke_train": (view_dir / "manifest_smoke_train_view.jsonl", smoke_train_records),
        "smoke_dev": (view_dir / "manifest_smoke_dev_view.jsonl", smoke_dev_records),
    }
    for _, (path, rows) in smoke_manifest_views.items():
        write_jsonl(path, rows)

    rendered_dir = output_dir / "rendered_examples"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: dict[str, str] = {}
    for diagnosis in ("nominal", "sensor_saturation", "secondary_reflection"):
        example = next(
            row for row in train_export
            if parse_target_strict(_visible_text(row, "assistant"))["diagnosis"] == diagnosis
        )
        path = rendered_dir / f"{diagnosis}.md"
        path.write_text(render_example(example), encoding="utf-8")
        rendered_paths[diagnosis] = path.relative_to(output_dir).as_posix()

    report = {
        "version": "qwen_vl_supervisor_sft_export_v1",
        "seed": seed,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "records": {
            name: {
                "count": len(rows),
                "class_counts": dict(sorted(Counter(
                    parse_target_strict(_visible_text(row, "assistant"))["diagnosis"] for row in rows
                ).items())),
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_path(path),
            }
            for name, (path, rows) in outputs.items()
        },
        "rendered_examples": rendered_paths,
        "manifest_views": {
            name: {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_path(path),
                "records": len(rows),
                "note": "sampling view of an existing validated split; not a new source manifest or split",
            }
            for name, (path, rows) in smoke_manifest_views.items()
        },
        "visible_image_slots_per_record": 1,
        "visible_paths_or_hidden_metadata": 0,
        "target_format": "whole-string compact canonical JSON; exact three keys; no rationale/confidence",
        "truncation_policy": "disabled (training max_length=null); no critical field may be dropped",
        "smoke_subset_note": "balanced sampling views of existing train/dev splits; not new splits; pair split assignment is unchanged",
    }
    report_path = output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    module_root = Path(__file__).resolve().parent
    parser.add_argument("--repository-root", type=Path, default=module_root.parent)
    parser.add_argument("--train-manifest", type=Path, default=module_root / "manifests/manifest_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=module_root / "manifests/manifest_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=module_root / "sft")
    parser.add_argument("--seed", type=int, default=2026080101)
    args = parser.parse_args()
    report = export(
        repository_root=args.repository_root,
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
