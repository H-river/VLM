"""Build physics-understanding diagnostic probe JSONL files."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from legacy.physics_understanding.evaluation.physics_understanding_schema import (
    CANONICAL_VARIABLE_ORDER,
    write_probe_jsonl,
)
from profile2setup.schema import compute_delta_setup, validate_setup_dict

FALLBACK_TOLERANCES = {
    "source_to_lens": 0.01,
    "lens_to_camera": 0.01,
    "focal_length": 0.005,
    "lens_x": 0.0005,
    "lens_y": 0.0005,
    "camera_x": 0.0005,
    "camera_y": 0.0005,
}
CAMERA_VARIABLES = ["camera_x", "camera_y"]
LENS_OFFSET_VARIABLES = ["lens_x", "lens_y"]
LENS_VARIABLES = ["source_to_lens", "lens_to_camera", "focal_length", "lens_x", "lens_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline physics-understanding diagnostic probes from profile2setup records."
    )
    parser.add_argument("--data", required=True, help="Input profile2setup JSONL")
    parser.add_argument("--out", required=True, help="Output probe JSONL")
    parser.add_argument("--variables-config", required=True, help="Variables YAML with optional tolerances")
    parser.add_argument("--max-base-records", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output summary JSON. Defaults to <out stem>_summary.json.",
    )
    parser.add_argument(
        "--markdown-out",
        default=None,
        help="Output summary Markdown. Defaults to <out stem>_summary.md.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} record must be a JSON object")
            records.append(record)
    return records


def _load_tolerances(path: Path) -> dict[str, float]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read --variables-config") from exc

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"variables config must be a dict: {path}")
    order = config.get("variable_order")
    if order is not None and list(order) != list(CANONICAL_VARIABLE_ORDER):
        raise ValueError(
            f"variables config order must be canonical {CANONICAL_VARIABLE_ORDER}; got {order}"
        )
    variables = config.get("variables") or {}
    tolerances: dict[str, float] = {}
    for name in CANONICAL_VARIABLE_ORDER:
        spec = variables.get(name) or {}
        tolerances[name] = float(spec.get("tolerance", FALLBACK_TOLERANCES[name]))
    return tolerances


def _profile_path(record: dict, key: str) -> str | None:
    value = record.get(key)
    if isinstance(value, str) and value.endswith(".npy"):
        return value
    return None


def _current_setup(record: dict) -> dict | None:
    value = record.get("current_setup")
    if isinstance(value, dict) and validate_setup_dict(value):
        return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}
    return None


def _setup_delta(record: dict) -> dict | None:
    explicit = record.get("target_delta")
    if isinstance(explicit, dict) and validate_setup_dict(explicit):
        return {name: float(explicit[name]) for name in CANONICAL_VARIABLE_ORDER}
    current = record.get("current_setup")
    target = record.get("target_setup")
    if isinstance(current, dict) and isinstance(target, dict):
        if validate_setup_dict(current) and validate_setup_dict(target):
            return compute_delta_setup(current, target)
    return None


def _direction(value: float, tolerance: float) -> str:
    if value > tolerance:
        return "increase"
    if value < -tolerance:
        return "decrease"
    return "unchanged"


def _expected_direction(delta: dict, tolerances: dict[str, float]) -> dict[str, str]:
    return {
        name: _direction(float(delta[name]), tolerances[name])
        for name in CANONICAL_VARIABLE_ORDER
    }


def _changed_variables(direction: dict[str, str]) -> list[str]:
    return [name for name in CANONICAL_VARIABLE_ORDER if direction[name] != "unchanged"]


def _metric_delta(record: dict, *names: str) -> float | None:
    metrics = record.get("metrics_delta")
    if not isinstance(metrics, dict):
        return None
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _signed_metric_label(delta: float | None, threshold: float, negative: str, positive: str) -> str | None:
    if delta is None:
        return None
    if delta > threshold:
        return positive
    if delta < -threshold:
        return negative
    return None


def _beam_movement(record: dict) -> dict[str, str]:
    x_delta = _metric_delta(record, "centroid_x_px", "centroid_x")
    y_delta = _metric_delta(record, "centroid_y_px", "centroid_y")
    sx_delta = _metric_delta(record, "sigma_x_px", "sigma_x")
    sy_delta = _metric_delta(record, "sigma_y_px", "sigma_y")
    centroid_threshold = 0.5 if x_delta is not None or y_delta is not None else 1e-6
    sigma_threshold = 0.3 if sx_delta is not None or sy_delta is not None else 1e-6
    movement: dict[str, str] = {}
    x_label = _signed_metric_label(x_delta, centroid_threshold, "left", "right")
    y_label = _signed_metric_label(y_delta, centroid_threshold, "up", "down")
    if x_label:
        movement["centroid_x"] = x_label
    if y_label:
        movement["centroid_y"] = y_label
    width_values = [value for value in (sx_delta, sy_delta) if value is not None]
    if width_values:
        avg_width_delta = sum(width_values) / float(len(width_values))
        width_label = _signed_metric_label(avg_width_delta, sigma_threshold, "tighter", "wider")
        if width_label:
            movement["beam_width"] = width_label
    return movement


def _primary_movement(record: dict) -> str | None:
    movement = _beam_movement(record)
    for key in ("centroid_x", "centroid_y", "beam_width"):
        if key in movement:
            return movement[key]
    return None


def _opposite_movement(direction: str) -> str:
    return {
        "right": "left",
        "left": "right",
        "up": "down",
        "down": "up",
        "tighter": "wider",
        "wider": "tighter",
    }[direction]


def _movement_paraphrases(direction: str) -> list[str]:
    phrases = {
        "right": [
            "move the beam right",
            "shift the spot rightward",
            "translate the beam centroid toward positive x",
            "move the profile center to the right",
        ],
        "left": [
            "move the beam left",
            "shift the spot leftward",
            "translate the beam centroid toward negative x",
            "move the profile center to the left",
        ],
        "up": [
            "move the beam up",
            "shift the spot upward",
            "translate the beam centroid toward negative y",
            "move the profile center upward",
        ],
        "down": [
            "move the beam down",
            "shift the spot downward",
            "translate the beam centroid toward positive y",
            "move the profile center downward",
        ],
        "tighter": [
            "make the beam tighter",
            "reduce the beam width",
            "make the spot more compact",
            "narrow the intensity profile",
        ],
        "wider": [
            "make the beam wider",
            "increase the beam width",
            "make the spot more spread out",
            "broaden the intensity profile",
        ],
    }
    return phrases[direction]


def _clean_record_id(record: dict) -> str | None:
    value = record.get("id")
    return value if isinstance(value, str) and value else None


def _base_probe(
    *,
    probe_id: str,
    probe_type: str,
    record: dict | None,
    task_type: str,
    prompt: str,
    input_mode: str,
    expected_valid: bool,
    expected_changed_variables: list[str] | None = None,
    expected_change_direction: dict[str, str] | None = None,
    fixed_variables: list[str] | None = None,
    allowed_variables: list[str] | None = None,
    expected_rejection_keywords: list[str] | None = None,
    notes: str = "",
    current_profile_path: str | None = None,
    target_profile_path: str | None = None,
    current_setup: dict | None = None,
) -> dict:
    if record is not None:
        current_profile_path = current_profile_path if current_profile_path is not None else _profile_path(record, "current_profile_path")
        target_profile_path = target_profile_path if target_profile_path is not None else _profile_path(record, "target_profile_path")
        current_setup = current_setup if current_setup is not None else _current_setup(record)
    if input_mode == "prompt_only":
        current_profile_path = None
        target_profile_path = None
        current_setup = None
    return {
        "probe_id": probe_id,
        "probe_type": probe_type,
        "base_record_id": _clean_record_id(record) if record is not None else None,
        "task_type": task_type,
        "prompt": prompt,
        "input_mode": input_mode,
        "current_profile_path": current_profile_path,
        "target_profile_path": target_profile_path,
        "current_setup": current_setup,
        "expected_valid": expected_valid,
        "expected_changed_variables": expected_changed_variables,
        "expected_change_direction": expected_change_direction,
        "fixed_variables": fixed_variables or [],
        "allowed_variables": allowed_variables,
        "expected_rejection_keywords": expected_rejection_keywords or [],
        "notes": notes,
    }


def _has_profile_pair(record: dict) -> bool:
    return _profile_path(record, "current_profile_path") is not None and _profile_path(record, "target_profile_path") is not None


def _summary_paths(out_path: Path, summary_arg: str | None, markdown_arg: str | None) -> tuple[Path, Path]:
    if summary_arg:
        summary_path = Path(summary_arg)
    else:
        summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    if markdown_arg:
        markdown_path = Path(markdown_arg)
    else:
        markdown_path = out_path.with_name(f"{out_path.stem}_summary.md")
    return summary_path, markdown_path


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_markdown(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Physics Understanding Probe Build Summary",
        "",
        f"- Input data: `{summary['input_data']}`",
        f"- Output probes: `{summary['probe_jsonl']}`",
        f"- Variables config: `{summary['variables_config']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Loaded records: `{summary['loaded_records']}`",
        f"- Selected base records: `{summary['selected_base_records']}`",
        f"- Generated probes: `{summary['generated_probes']}`",
        "",
        "## Counts By Probe Type",
        "",
    ]
    for probe_type, count in summary["counts_by_probe_type"].items():
        lines.append(f"- `{probe_type}`: `{count}`")
    lines.extend(["", "## Counts By Input Mode", ""])
    for input_mode, count in summary["counts_by_input_mode"].items():
        lines.append(f"- `{input_mode}`: `{count}`")
    lines.extend(["", "## Source Records Used By Probe Type", ""])
    for probe_type, count in summary["base_records_used_by_probe_type"].items():
        lines.append(f"- `{probe_type}`: `{count}`")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


class ProbeBuilder:
    def __init__(self, tolerances: dict[str, float]) -> None:
        self.tolerances = tolerances
        self.probes: list[dict] = []
        self._next_id = 1

    def _probe_id(self, probe_type: str) -> str:
        safe_type = re.sub(r"[^A-Za-z0-9_]+", "_", probe_type).strip("_")
        value = f"{safe_type}_{self._next_id:06d}"
        self._next_id += 1
        return value

    def add(self, probe_type: str, **kwargs: Any) -> None:
        self.probes.append(
            _base_probe(
                probe_id=self._probe_id(probe_type),
                probe_type=probe_type,
                **kwargs,
            )
        )

    def add_paraphrase_consistency(self, record: dict) -> None:
        delta = _setup_delta(record)
        movement = _primary_movement(record)
        if delta is None or movement is None:
            return
        direction = _expected_direction(delta, self.tolerances)
        changed = _changed_variables(direction)
        if not changed:
            return
        for prompt in _movement_paraphrases(movement):
            self.add(
                "paraphrase_consistency",
                record=record,
                task_type=str(record.get("task_type")),
                prompt=prompt,
                input_mode="prompt_plus_images",
                expected_valid=True,
                expected_changed_variables=changed,
                expected_change_direction=direction,
                notes=(
                    "Paraphrased beam-motion prompt; expected setup-variable target is inherited "
                    "from the same base record delta."
                ),
            )

    def add_prompt_sensitivity(self, record: dict) -> None:
        if not _has_profile_pair(record):
            return
        prompts = [
            (
                "Match the target using camera movement if possible.",
                [],
                CAMERA_VARIABLES,
                "Preference probe for camera-motion solution routes.",
            ),
            (
                "Match the target using lens movement if possible.",
                [],
                LENS_VARIABLES,
                "Preference probe for lens-motion solution routes.",
            ),
            (
                "Keep camera_x and camera_y fixed.",
                CAMERA_VARIABLES,
                None,
                "Constraint probe requiring camera offsets to remain fixed.",
            ),
        ]
        for prompt, fixed, allowed, notes in prompts:
            self.add(
                "prompt_sensitivity",
                record=record,
                task_type=str(record.get("task_type")),
                prompt=prompt,
                input_mode="prompt_plus_images",
                expected_valid=True,
                expected_changed_variables=None,
                expected_change_direction=None,
                fixed_variables=fixed,
                allowed_variables=allowed,
                notes=notes,
            )

    def add_fixed_variable_constraint(self, record: dict) -> None:
        if not _has_profile_pair(record):
            return
        prompts = [
            ("Match the target but keep camera_x and camera_y fixed.", CAMERA_VARIABLES),
            ("Do not change focal_length.", ["focal_length"]),
            ("Keep all lens offsets fixed.", LENS_OFFSET_VARIABLES),
        ]
        for prompt, fixed in prompts:
            self.add(
                "fixed_variable_constraint",
                record=record,
                task_type=str(record.get("task_type")),
                prompt=prompt,
                input_mode="prompt_plus_images",
                expected_valid=True,
                expected_changed_variables=None,
                expected_change_direction=None,
                fixed_variables=fixed,
                notes="Fixed-variable constraint probe; metric should check that fixed variables stay unchanged.",
            )

    def add_allowed_variable_constraint(self, record: dict) -> None:
        if not _has_profile_pair(record):
            return
        prompts = [
            ("Only adjust camera_x.", ["camera_x"]),
            ("Only adjust lens_x and lens_y.", LENS_OFFSET_VARIABLES),
            ("Only adjust lens_to_camera.", ["lens_to_camera"]),
        ]
        for prompt, allowed in prompts:
            self.add(
                "allowed_variable_constraint",
                record=record,
                task_type=str(record.get("task_type")),
                prompt=prompt,
                input_mode="prompt_plus_images",
                expected_valid=True,
                expected_changed_variables=None,
                expected_change_direction=None,
                allowed_variables=allowed,
                notes="Allowed-variable constraint probe; metric should check changes stay inside allowed_variables.",
            )

    def add_prompt_image_conflict(self, record: dict) -> None:
        if not _has_profile_pair(record):
            return
        movement = _primary_movement(record)
        if movement is None:
            return
        opposite = _opposite_movement(movement)
        prompt = f"Move the beam {opposite}."
        self.add(
            "prompt_image_conflict",
            record=record,
            task_type=str(record.get("task_type")),
            prompt=prompt,
            input_mode="conflict",
            expected_valid=False,
            expected_changed_variables=None,
            expected_change_direction=None,
            expected_rejection_keywords=["contradict", "conflict", "inconsistent"],
            notes=(
                f"Base profile metrics indicate clear {movement} movement, while prompt requests "
                f"the opposite direction: {opposite}."
            ),
        )

    def add_contradiction_detection(self, record: dict) -> None:
        prompts = [
            ("Match the target profile, but do not change any variable.", list(CANONICAL_VARIABLE_ORDER)),
            ("Move the beam right, but keep all variables fixed.", list(CANONICAL_VARIABLE_ORDER)),
        ]
        for prompt, fixed in prompts:
            self.add(
                "contradiction_detection",
                record=record if _has_profile_pair(record) else None,
                task_type=str(record.get("task_type")) if record is not None else "edit",
                prompt=prompt,
                input_mode="prompt_plus_images" if _has_profile_pair(record) else "prompt_only",
                expected_valid=False,
                expected_changed_variables=None,
                expected_change_direction=None,
                fixed_variables=fixed,
                expected_rejection_keywords=["contradict", "conflict", "inconsistent", "fixed"],
                notes="Contradictory instruction probe with movement/matching request and no-variable-change constraint.",
            )

    def add_physics_causal_questions(self) -> None:
        self.add(
            "physics_causal_question",
            record=None,
            task_type="edit",
            prompt=(
                "If camera_x increases while all other variables are fixed, "
                "which centroid direction should change?"
            ),
            input_mode="prompt_only",
            expected_valid=True,
            expected_changed_variables=["camera_x"],
            expected_change_direction={"camera_x": "increase"},
            fixed_variables=[name for name in CANONICAL_VARIABLE_ORDER if name != "camera_x"],
            allowed_variables=["camera_x"],
            notes="Prompt-only qualitative causal probe about camera_x and centroid direction.",
        )
        self.add(
            "physics_causal_question",
            record=None,
            task_type="edit",
            prompt="If lens_to_camera changes, which profile property is most likely affected?",
            input_mode="prompt_only",
            expected_valid=True,
            expected_changed_variables=["lens_to_camera"],
            expected_change_direction=None,
            fixed_variables=[],
            allowed_variables=["lens_to_camera"],
            notes="Prompt-only qualitative causal probe about axial lens-camera distance and profile shape.",
        )


def build_probes(records: list[dict], *, max_base_records: int, seed: int, tolerances: dict[str, float]) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    selected = shuffled[: max(0, max_base_records)]
    builder = ProbeBuilder(tolerances)

    for record in selected:
        builder.add_paraphrase_consistency(record)
        builder.add_prompt_sensitivity(record)
        builder.add_fixed_variable_constraint(record)
        builder.add_allowed_variable_constraint(record)
        builder.add_prompt_image_conflict(record)
        builder.add_contradiction_detection(record)
    builder.add_physics_causal_questions()
    return builder.probes


def _count_distinct_base_records(probes: list[dict], probe_type: str) -> int:
    return len(
        {
            probe["base_record_id"]
            for probe in probes
            if probe["probe_type"] == probe_type and probe["base_record_id"] is not None
        }
    )


def _build_summary(
    *,
    data_path: Path,
    out_path: Path,
    variables_config_path: Path,
    seed: int,
    max_base_records: int,
    loaded_records: int,
    selected_base_records: int,
    probes: list[dict],
) -> dict:
    counts_by_probe_type = Counter(probe["probe_type"] for probe in probes)
    counts_by_input_mode = Counter(probe["input_mode"] for probe in probes)
    return {
        "input_data": str(data_path),
        "probe_jsonl": str(out_path),
        "variables_config": str(variables_config_path),
        "seed": int(seed),
        "max_base_records": int(max_base_records),
        "loaded_records": int(loaded_records),
        "selected_base_records": int(selected_base_records),
        "generated_probes": int(len(probes)),
        "counts_by_probe_type": dict(sorted(counts_by_probe_type.items())),
        "counts_by_input_mode": dict(sorted(counts_by_input_mode.items())),
        "base_records_used_by_probe_type": {
            probe_type: _count_distinct_base_records(probes, probe_type)
            for probe_type in sorted(counts_by_probe_type)
        },
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)
    variables_config_path = Path(args.variables_config)
    summary_path, markdown_path = _summary_paths(out_path, args.summary_out, args.markdown_out)

    records = _load_jsonl(data_path)
    tolerances = _load_tolerances(variables_config_path)
    selected_count = min(max(0, int(args.max_base_records)), len(records))
    probes = build_probes(
        records,
        max_base_records=selected_count,
        seed=int(args.seed),
        tolerances=tolerances,
    )

    written = write_probe_jsonl(probes, out_path)
    if written != len(probes):
        raise RuntimeError(f"expected to write {len(probes)} probes, wrote {written}")

    summary = _build_summary(
        data_path=data_path,
        out_path=out_path,
        variables_config_path=variables_config_path,
        seed=int(args.seed),
        max_base_records=int(args.max_base_records),
        loaded_records=len(records),
        selected_base_records=selected_count,
        probes=probes,
    )
    summary["summary_json"] = str(summary_path)
    summary["summary_markdown"] = str(markdown_path)
    _write_json(summary, summary_path)
    _write_markdown(summary, markdown_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
