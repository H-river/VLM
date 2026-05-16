"""Build a stratified Stage 1 understanding benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from profile2setup.evaluation.param_metrics import load_tolerances
from profile2setup.schema import VARIABLE_ORDER, validate_dataset_record, validate_setup_dict
from profile2setup.training.normalization import load_variables_config


CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
DEFAULT_COMPOSITION = {
    "normal_edit": 20,
    "absolute": 15,
    "paired_no_setup": 15,
    "constraint": 10,
    "invalid": 10,
    "ambiguous_multi_intent": 10,
}

INVALID_PROMPTS = [
    ("change the wavelength", "outside_canonical_setup:wavelength"),
    ("increase laser power", "outside_canonical_setup:laser_power"),
    ("change the beam color", "outside_canonical_setup:beam_color"),
    ("rotate the optical table", "outside_canonical_setup:optical_table_rotation"),
    ("switch to a different laser source", "outside_canonical_setup:laser_source"),
    ("increase aperture size", "outside_canonical_setup:aperture"),
    ("change the sensor exposure time", "outside_canonical_setup:exposure_time"),
    ("add a second lens", "outside_canonical_setup:optical_element_count"),
    ("change the wavelength and laser power", "outside_canonical_setup:wavelength_and_power"),
    ("tilt the camera sensor", "outside_canonical_setup:camera_tilt"),
]

AMBIGUOUS_PROMPTS = [
    "move the beam left and right",
    "make it wider and smaller",
    "keep everything fixed but move the beam",
    "move the beam up and down at the same time",
    "make the beam tighter but also wider",
    "change only focal_length and do not change focal_length",
    "keep camera fixed but move camera_x",
    "increase and decrease lens_to_camera",
    "move the beam left, unless it should go right",
    "match the target but keep all setup variables unchanged",
]

CONSTRAINT_SPECS = [
    {
        "prompt": "increase focal_length only",
        "directions": {"focal_length": "increase"},
        "fixed": ["source_to_lens", "lens_to_camera", "lens_x", "lens_y", "camera_x", "camera_y"],
        "notes": "Synthetic single-variable constraint.",
    },
    {
        "prompt": "decrease focal_length only",
        "directions": {"focal_length": "decrease"},
        "fixed": ["source_to_lens", "lens_to_camera", "lens_x", "lens_y", "camera_x", "camera_y"],
        "notes": "Synthetic single-variable constraint.",
    },
    {
        "prompt": "increase lens_x but keep the camera fixed",
        "directions": {"lens_x": "increase"},
        "fixed": ["camera_x", "camera_y"],
        "notes": "Synthetic constraint with explicit fixed camera variables.",
    },
    {
        "prompt": "decrease lens_y but keep the camera fixed",
        "directions": {"lens_y": "decrease"},
        "fixed": ["camera_x", "camera_y"],
        "notes": "Synthetic constraint with explicit fixed camera variables.",
    },
    {
        "prompt": "increase source_to_lens and keep lens_to_camera fixed",
        "directions": {"source_to_lens": "increase"},
        "fixed": ["lens_to_camera"],
        "notes": "Synthetic constraint with one changed geometry variable and one fixed geometry variable.",
    },
    {
        "prompt": "decrease lens_to_camera and keep source_to_lens fixed",
        "directions": {"lens_to_camera": "decrease"},
        "fixed": ["source_to_lens"],
        "notes": "Synthetic constraint with one changed geometry variable and one fixed geometry variable.",
    },
    {
        "prompt": "increase camera_x only",
        "directions": {"camera_x": "increase"},
        "fixed": ["source_to_lens", "lens_to_camera", "focal_length", "lens_x", "lens_y", "camera_y"],
        "notes": "Synthetic single-variable constraint.",
    },
    {
        "prompt": "decrease camera_y only",
        "directions": {"camera_y": "decrease"},
        "fixed": ["source_to_lens", "lens_to_camera", "focal_length", "lens_x", "lens_y", "camera_x"],
        "notes": "Synthetic single-variable constraint.",
    },
    {
        "prompt": "decrease focal_length without changing lens_x or lens_y",
        "directions": {"focal_length": "decrease"},
        "fixed": ["lens_x", "lens_y"],
        "notes": "Synthetic constraint based on the requested no-lens-offset-change prompt pattern.",
    },
    {
        "prompt": "increase lens_to_camera but keep source_to_lens fixed",
        "directions": {"lens_to_camera": "increase"},
        "fixed": ["source_to_lens"],
        "notes": "Synthetic constraint with one changed geometry variable and one fixed geometry variable.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 1 understanding benchmark JSONL files.")
    parser.add_argument("--source", required=True, help="Source all_modes test JSONL")
    parser.add_argument("--variables-config", required=True, help="Variables YAML config")
    parser.add_argument("--out", required=True, help="Output benchmark JSONL")
    parser.add_argument("--labels-out", required=True, help="Output labels JSONL")
    parser.add_argument("--manifest-out", required=True, help="Output manifest JSON")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for deterministic sampling")
    parser.add_argument("--target-size", type=int, default=80, help="Total benchmark size")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(obj)
    return records


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _composition_for_size(target_size: int) -> dict[str, int]:
    default_total = sum(DEFAULT_COMPOSITION.values())
    if target_size == default_total:
        return dict(DEFAULT_COMPOSITION)
    if target_size <= 0:
        raise ValueError("--target-size must be positive")

    scaled: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for category, count in DEFAULT_COMPOSITION.items():
        exact = count * target_size / default_total
        whole = int(exact)
        scaled[category] = whole
        assigned += whole
        remainders.append((exact - whole, category))
    for _, category in sorted(remainders, reverse=True)[: target_size - assigned]:
        scaled[category] += 1
    return scaled


def _direction(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "increase"
    if value < -tolerance:
        return "decrease"
    return "unchanged"


def _unchanged_direction() -> dict[str, str]:
    return {name: "unchanged" for name in CANONICAL_VARIABLE_ORDER}


def _directions_from_delta(delta: dict[str, Any], tolerances: dict[str, float]) -> dict[str, str]:
    return {
        name: _direction(float(delta[name]), float(tolerances[name]))
        for name in CANONICAL_VARIABLE_ORDER
    }


def _changed_from_directions(directions: dict[str, str]) -> list[str]:
    return [name for name in CANONICAL_VARIABLE_ORDER if directions[name] != "unchanged"]


def _valid_delta(delta: Any) -> dict[str, float] | None:
    if delta is None or not isinstance(delta, dict):
        return None
    if not validate_setup_dict(delta):
        return None
    return {name: float(delta[name]) for name in CANONICAL_VARIABLE_ORDER}


def _label_from_source(
    *,
    benchmark_id: str,
    category: str,
    record: dict,
    tolerances: dict[str, float],
    notes: str,
) -> dict:
    delta = _valid_delta(record.get("target_delta"))
    if delta is None:
        directions = _unchanged_direction()
        changed: list[str] = []
        label_note = (
            notes
            + " No target_delta is available for this task, so setup-change labels are marked unchanged."
        )
    else:
        directions = _directions_from_delta(delta, tolerances)
        changed = _changed_from_directions(directions)
        label_note = notes + " Labels derived from target_delta using per-variable tolerances."

    return {
        "record_id": benchmark_id,
        "source_record_id": record["id"],
        "category": category,
        "task_type": record.get("task_type"),
        "prompt": record.get("prompt"),
        "valid_request": True,
        "expected_changed_variables": changed,
        "expected_change_direction": directions,
        "expected_fixed_variables": [],
        "expected_rejection_reason": "",
        "notes": label_note.strip(),
    }


def _clone_record(record: dict, *, benchmark_id: str, prompt: str | None = None) -> dict:
    cloned = copy.deepcopy(record)
    cloned["id"] = benchmark_id
    if prompt is not None:
        cloned["prompt"] = prompt
    validate_dataset_record(cloned, strict=True)
    return cloned


def _manual_label(
    *,
    benchmark_id: str,
    source_record: dict,
    category: str,
    prompt: str,
    valid_request: bool,
    directions: dict[str, str],
    fixed: list[str] | None = None,
    rejection_reason: str = "",
    notes: str = "",
) -> dict:
    return {
        "record_id": benchmark_id,
        "source_record_id": source_record["id"],
        "category": category,
        "task_type": source_record.get("task_type"),
        "prompt": prompt,
        "valid_request": bool(valid_request),
        "expected_changed_variables": _changed_from_directions(directions),
        "expected_change_direction": directions,
        "expected_fixed_variables": sorted(fixed or []),
        "expected_rejection_reason": rejection_reason,
        "notes": notes,
    }


def _sample_by_task(records: list[dict], task_type: str, count: int, rng: random.Random) -> list[dict]:
    pool = [record for record in records if record.get("task_type") == task_type]
    if len(pool) < count:
        raise ValueError(f"not enough {task_type} records: need {count}, found {len(pool)}")
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:count]


def _make_benchmark_id(category: str, idx: int, source_id: str) -> str:
    safe_source = str(source_id).replace("/", "_")
    return f"stage1_{category}_{idx:03d}__{safe_source}"


def build_stage1_understanding_set(
    *,
    source_path: Path,
    variables_config_path: Path,
    seed: int,
    target_size: int,
) -> tuple[list[dict], list[dict], dict, str]:
    source_records = _load_jsonl(source_path)
    variables_config = load_variables_config(variables_config_path)
    tolerances = load_tolerances(variables_config)
    composition = _composition_for_size(target_size)
    rng = random.Random(seed)

    output_rows: list[dict] = []
    label_rows: list[dict] = []
    source_usage: dict[str, list[str]] = {category: [] for category in composition}
    synthesis_notes: list[str] = []

    edit_needed = (
        composition["normal_edit"]
        + composition["constraint"]
        + composition["invalid"]
        + composition["ambiguous_multi_intent"]
    )
    edit_records = _sample_by_task(source_records, "edit", edit_needed, rng)
    normal_edit_records = edit_records[: composition["normal_edit"]]
    absolute_records = _sample_by_task(source_records, "absolute", composition["absolute"], rng)
    paired_records = _sample_by_task(source_records, "paired_no_setup", composition["paired_no_setup"], rng)
    synthetic_base = edit_records[composition["normal_edit"] :]

    for category, records, notes in [
        ("normal_edit", normal_edit_records, "Ordinary edit source record."),
        ("absolute", absolute_records, "Ordinary absolute source record."),
        ("paired_no_setup", paired_records, "Ordinary paired_no_setup source record."),
    ]:
        for idx, record in enumerate(records, start=1):
            benchmark_id = _make_benchmark_id(category, idx, record["id"])
            output_rows.append(_clone_record(record, benchmark_id=benchmark_id))
            label_rows.append(
                _label_from_source(
                    benchmark_id=benchmark_id,
                    category=category,
                    record=record,
                    tolerances=tolerances,
                    notes=notes,
                )
            )
            source_usage[category].append(record["id"])

    offset = 0
    constraint_count = composition["constraint"]
    if constraint_count > len(CONSTRAINT_SPECS):
        raise ValueError(
            f"constraint target {constraint_count} exceeds built-in constraint specs {len(CONSTRAINT_SPECS)}"
        )
    for idx, spec in enumerate(CONSTRAINT_SPECS[:constraint_count], start=1):
        record = synthetic_base[offset]
        offset += 1
        benchmark_id = _make_benchmark_id("constraint", idx, record["id"])
        directions = _unchanged_direction()
        for name, direction in spec["directions"].items():
            directions[name] = direction
        output_rows.append(_clone_record(record, benchmark_id=benchmark_id, prompt=spec["prompt"]))
        label_rows.append(
            _manual_label(
                benchmark_id=benchmark_id,
                source_record=record,
                category="constraint",
                prompt=spec["prompt"],
                valid_request=True,
                directions=directions,
                fixed=spec["fixed"],
                notes=spec["notes"]
                + " Prompt was synthesized while preserving the source profile paths and setup fields.",
            )
        )
        source_usage["constraint"].append(record["id"])

    invalid_count = composition["invalid"]
    if invalid_count > len(INVALID_PROMPTS):
        raise ValueError(f"invalid target {invalid_count} exceeds built-in prompts {len(INVALID_PROMPTS)}")
    for idx, (prompt, reason) in enumerate(INVALID_PROMPTS[:invalid_count], start=1):
        record = synthetic_base[offset]
        offset += 1
        benchmark_id = _make_benchmark_id("invalid", idx, record["id"])
        output_rows.append(_clone_record(record, benchmark_id=benchmark_id, prompt=prompt))
        label_rows.append(
            _manual_label(
                benchmark_id=benchmark_id,
                source_record=record,
                category="invalid",
                prompt=prompt,
                valid_request=False,
                directions=_unchanged_direction(),
                fixed=[],
                rejection_reason=reason,
                notes=(
                    "Synthetic invalid/out-of-scope prompt. Profile paths and setup fields are preserved, "
                    "but the request refers to controls outside the canonical seven setup variables."
                ),
            )
        )
        source_usage["invalid"].append(record["id"])

    ambiguous_count = composition["ambiguous_multi_intent"]
    if ambiguous_count > len(AMBIGUOUS_PROMPTS):
        raise ValueError(
            f"ambiguous target {ambiguous_count} exceeds built-in prompts {len(AMBIGUOUS_PROMPTS)}"
        )
    for idx, prompt in enumerate(AMBIGUOUS_PROMPTS[:ambiguous_count], start=1):
        record = synthetic_base[offset]
        offset += 1
        benchmark_id = _make_benchmark_id("ambiguous_multi_intent", idx, record["id"])
        output_rows.append(_clone_record(record, benchmark_id=benchmark_id, prompt=prompt))
        label_rows.append(
            _manual_label(
                benchmark_id=benchmark_id,
                source_record=record,
                category="ambiguous_multi_intent",
                prompt=prompt,
                valid_request=False,
                directions=_unchanged_direction(),
                fixed=[],
                rejection_reason="ambiguous_or_conflicting_request",
                notes=(
                    "Synthetic ambiguous or contradictory prompt. Profile paths and setup fields are preserved, "
                    "but expected behavior is rejection/clarification rather than numerical setup regression."
                ),
            )
        )
        source_usage["ambiguous_multi_intent"].append(record["id"])

    if len(output_rows) != target_size or len(label_rows) != target_size:
        raise AssertionError(
            f"internal size mismatch: output={len(output_rows)} labels={len(label_rows)} target={target_size}"
        )

    label_ids = [row["record_id"] for row in label_rows]
    output_ids = [row["id"] for row in output_rows]
    if label_ids != output_ids:
        raise AssertionError("label row order does not match benchmark row order")

    if composition["constraint"] or composition["invalid"] or composition["ambiguous_multi_intent"]:
        synthesis_notes.append(
            "Source test data did not contain explicit constraint, invalid, or ambiguous categories; "
            "these rows were synthesized by replacing prompts on sampled edit records."
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "variables_config_path": str(variables_config_path),
        "seed": seed,
        "target_size": target_size,
        "composition_target": composition,
        "composition_actual": {
            category: sum(1 for label in label_rows if label["category"] == category)
            for category in composition
        },
        "variable_order": CANONICAL_VARIABLE_ORDER,
        "tolerances": {name: float(tolerances[name]) for name in CANONICAL_VARIABLE_ORDER},
        "source_usage": source_usage,
        "synthetic_categories": ["constraint", "invalid", "ambiguous_multi_intent"],
        "notes": synthesis_notes,
    }
    return output_rows, label_rows, manifest, _readme_text(manifest)


def _readme_text(manifest: dict) -> str:
    composition_lines = [
        f"| `{category}` | {count} |"
        for category, count in manifest["composition_actual"].items()
    ]
    tolerance_lines = [
        f"| `{name}` | {manifest['tolerances'][name]} |"
        for name in CANONICAL_VARIABLE_ORDER
    ]
    return "\n".join(
        [
            "# Stage 1 Understanding Benchmark",
            "",
            "This directory contains a stratified benchmark for comparing explicit LLM/API understanding outputs with non-LLM local-model understanding proxies.",
            "",
            "## Files",
            "",
            "- `stage1_understanding_80.jsonl`: benchmark records consumed by inference/evaluation code.",
            "- `stage1_understanding_labels.jsonl`: understanding labels keyed by `record_id`.",
            "- `stage1_understanding_manifest.json`: build metadata, composition, tolerances, and source record usage.",
            "- `stage1_understanding_readme.md`: this build note.",
            "",
            "## Composition",
            "",
            "| Category | Count |",
            "|---|---:|",
            *composition_lines,
            "",
            "## Label Semantics",
            "",
            "- `valid_request` is false for invalid/out-of-scope and intentionally contradictory prompts.",
            "- `expected_changed_variables` and `expected_change_direction` are derived from `target_delta` for normal edit records.",
            "- Absolute and paired-no-setup source records do not contain setup deltas, so their setup-change directions are marked `unchanged` with notes in the label file.",
            "- Constraint, invalid, and ambiguous rows are synthesized prompt variants over existing edit records. Their profile paths and setup fields are preserved, while the labels reflect the synthetic prompt meaning.",
            "",
            "## Tolerances",
            "",
            "| Variable | Tolerance |",
            "|---|---:|",
            *tolerance_lines,
            "",
            "## Limitations",
            "",
            "- Synthetic prompt labels test instruction understanding, not whether the preserved target profile physically matches the synthetic prompt.",
            "- Invalid and ambiguous examples should be evaluated against the label file, not against the preserved target setup.",
            "- Absolute and paired-no-setup rows are useful for format and task understanding, but they do not provide ground-truth setup-change deltas.",
            "",
            "## Rebuild Command",
            "",
            "```bash",
            "/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.build_stage1_understanding_set_cli \\",
            "  --source profile2setup/data/all_modes/test.jsonl \\",
            "  --variables-config profile2setup/configs/variables.yaml \\",
            "  --out profile2setup/data/stage1_understanding/stage1_understanding_80.jsonl \\",
            "  --labels-out profile2setup/data/stage1_understanding/stage1_understanding_labels.jsonl \\",
            "  --manifest-out profile2setup/data/stage1_understanding/stage1_understanding_manifest.json \\",
            f"  --seed {manifest['seed']} \\",
            f"  --target-size {manifest['target_size']}",
            "```",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    labels_path = Path(args.labels_out)
    manifest_path = Path(args.manifest_out)
    readme_path = manifest_path.with_name("stage1_understanding_readme.md")

    rows, labels, manifest, readme = build_stage1_understanding_set(
        source_path=Path(args.source),
        variables_config_path=Path(args.variables_config),
        seed=args.seed,
        target_size=args.target_size,
    )
    manifest["out_path"] = str(out_path)
    manifest["labels_out_path"] = str(labels_path)
    manifest["manifest_out_path"] = str(manifest_path)
    manifest["readme_out_path"] = str(readme_path)

    _write_jsonl(rows, out_path)
    _write_jsonl(labels, labels_path)
    _write_json(manifest, manifest_path)
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(out_path),
                "labels_out": str(labels_path),
                "manifest_out": str(manifest_path),
                "readme_out": str(readme_path),
                "composition_actual": manifest["composition_actual"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
