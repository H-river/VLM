"""Strict record and cross-split validation for supervisor-v1 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from .contracts import (
    DIAGNOSES,
    MEASUREMENT_POLICIES,
    POLICY_TO_STATIC_ACTION,
    SCHEMA_VERSION,
    SUPERVISOR_ACTIONS,
    TARGET_KEYS,
    TASK_TYPE,
    validate_target,
)

TOP_KEYS = {
    "schema_version", "sample_id", "task_type", "split", "setup_hash",
    "episode_hash", "step_index", "counterfactual_pair_id", "augmented_base_hash",
    "assets", "model_input", "target", "provenance",
}
ASSET_KEYS = {"current_image_path", "current_image_sha256", "format", "mode", "width", "height"}
MODEL_INPUT_KEYS = {"current_metrics", "goal_metrics", "recent_history", "remaining_step_budget", "actuator_constraints"}
METRIC_KEYS = {"coordinate_frame", "centroid_x", "centroid_y", "width_x", "width_y", "peak_intensity"}
CONSTRAINT_KEYS = {"units", "per_step_delta_limits", "absolute_position_limits", "absolute_limit_source", "continuous_actions_selected_by"}
AXIS_KEYS = {"lens_x", "lens_y", "camera_x", "camera_y"}
TARGET_RECORD_KEYS = set(TARGET_KEYS) | {"field_mask"}
PROVENANCE_KEYS = {
    "source_cohort", "source_manifest_path", "source_manifest_sha256", "source_sample_id",
    "source_split", "anomaly_family", "source_fault_type", "target_provenance",
    "generator_version", "generator_sha256", "controller_version", "controller_hashes",
    "width_quartile", "boundary_status", "severity_bucket", "severity_value",
    "counterfactual_metric_distances", "source_same_state_metrics_excluded_from_history",
}
DISTANCE_KEYS = {
    "per_metric_absolute_difference_tolerances", "maximum_absolute_difference_tolerances",
    "total_l2_distance_tolerances", "passes_frozen_match",
}
CONTROLLER_HASH_KEYS = {"forward_ensemble", "v12_config", "v13_config", "visual_controller"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAMPLE_ID = re.compile(r"^qvlsup1_[0-9a-f]{24}$")
PAIR_ID = re.compile(r"^pair_[0-9a-f]{24}$")
SPLITS = {"train", "dev", "frozen_iid", "frozen_ood"}
SOURCE_SPLITS = {"train", "development", "iid_heldout", "severity_ood"}

# Model inputs are allow-listed, and this secondary recursive deny-list catches
# accidental future additions before they can enter exports.
LEAKY_KEY_PARTS = {
    "anomaly", "fault", "label", "generator", "severity", "setup", "pair",
    "episode_id", "oracle", "future", "outcome", "post_decision", "filename",
    "image_path", "split", "target", "provenance", "source_cohort",
}


class ManifestValidationError(ValueError):
    pass


def _fail(context: str, message: str) -> None:
    raise ManifestValidationError(f"{context}: {message}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = child
    return value


def _exact_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(context, "must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(context, f"field mismatch; missing={missing}, extra={extra}")
    return value


def _finite_number(value: Any, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(context, f"must be numeric, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        _fail(context, f"must be finite, got {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metrics(value: Any, frame: str, context: str) -> None:
    metrics = _exact_keys(value, METRIC_KEYS, context)
    if metrics["coordinate_frame"] != frame:
        _fail(context, f"coordinate_frame must be {frame!r}")
    for name in METRIC_KEYS - {"coordinate_frame"}:
        _finite_number(metrics[name], f"{context}.{name}")


def _validate_limits(value: Any, expected: dict[str, list[float]], context: str) -> None:
    limits = _exact_keys(value, AXIS_KEYS, context)
    for axis, expected_range in expected.items():
        actual = limits[axis]
        if not isinstance(actual, list) or len(actual) != 2:
            _fail(f"{context}.{axis}", "must be a two-number array")
        for index, number in enumerate(actual):
            _finite_number(number, f"{context}.{axis}[{index}]")
        if [float(item) for item in actual] != expected_range:
            _fail(f"{context}.{axis}", f"expected frozen limits {expected_range}, got {actual}")


def _validate_no_leaky_keys(value: Any, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in LEAKY_KEY_PARTS):
                _fail(context, f"forbidden model-input key {key!r}")
            _validate_no_leaky_keys(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_leaky_keys(child, f"{context}[{index}]")


def validate_record(record: Any, *, repository_root: Path, context: str) -> None:
    record = _exact_keys(record, TOP_KEYS, context)
    if record["schema_version"] != SCHEMA_VERSION:
        _fail(context, f"schema_version must be {SCHEMA_VERSION!r}")
    if not isinstance(record["sample_id"], str) or not SAMPLE_ID.fullmatch(record["sample_id"]):
        _fail(context, "invalid sample_id")
    if record["task_type"] != TASK_TYPE:
        _fail(context, f"task_type must be {TASK_TYPE!r}")
    if record["split"] not in SPLITS:
        _fail(context, f"invalid split {record['split']!r}")
    if not isinstance(record["setup_hash"], str) or not HEX64.fullmatch(record["setup_hash"]):
        _fail(context, "setup_hash must be irreversible 64-character lowercase hex")
    if record["episode_hash"] is not None and (
        not isinstance(record["episode_hash"], str) or not HEX64.fullmatch(record["episode_hash"])
    ):
        _fail(context, "episode_hash must be null or 64-character lowercase hex")
    if isinstance(record["step_index"], bool) or not isinstance(record["step_index"], int) or record["step_index"] < 0:
        _fail(context, "step_index must be a non-negative integer")
    if not isinstance(record["counterfactual_pair_id"], str) or not PAIR_ID.fullmatch(record["counterfactual_pair_id"]):
        _fail(context, "invalid counterfactual_pair_id")
    if not isinstance(record["augmented_base_hash"], str) or not HEX64.fullmatch(record["augmented_base_hash"]):
        _fail(context, "invalid augmented_base_hash")

    assets = _exact_keys(record["assets"], ASSET_KEYS, f"{context}.assets")
    if not isinstance(assets["current_image_path"], str) or not assets["current_image_path"]:
        _fail(f"{context}.assets.current_image_path", "must be a non-empty path")
    image_path = (repository_root / assets["current_image_path"]).resolve()
    try:
        image_path.relative_to(repository_root.resolve())
    except ValueError:
        _fail(f"{context}.assets.current_image_path", "path escapes repository root")
    if not image_path.is_file():
        _fail(f"{context}.assets.current_image_path", f"missing image {image_path}")
    image_hash = _sha256(image_path)
    if assets["current_image_sha256"] != image_hash:
        _fail(f"{context}.assets.current_image_sha256", f"hash mismatch for {image_path}")
    if record["augmented_base_hash"] != image_hash:
        _fail(context, "static record augmented_base_hash must equal the current image hash")
    with Image.open(image_path) as image:
        observed = (image.format, image.mode, image.width, image.height)
        expected = (assets["format"], assets["mode"], assets["width"], assets["height"])
        if observed != expected:
            _fail(f"{context}.assets", f"image metadata mismatch: manifest={expected}, actual={observed}")
    if (assets["format"], assets["mode"], assets["width"], assets["height"]) != ("PNG", "L", 128, 128):
        _fail(f"{context}.assets", "v1 requires grayscale 128x128 PNG")

    model_input = _exact_keys(record["model_input"], MODEL_INPUT_KEYS, f"{context}.model_input")
    _validate_no_leaky_keys(model_input, f"{context}.model_input")
    _validate_metrics(model_input["current_metrics"], "diagnostic_image_128px", f"{context}.model_input.current_metrics")
    _validate_metrics(model_input["goal_metrics"], "lab_sensor_1024px_and_raw_peak", f"{context}.model_input.goal_metrics")
    history = model_input["recent_history"]
    if not isinstance(history, list) or len(history) > 8:
        _fail(f"{context}.model_input.recent_history", "must be an array with at most eight observations")
    previous_index = -1
    for history_index, item in enumerate(history):
        item_context = f"{context}.model_input.recent_history[{history_index}]"
        item = _exact_keys(item, {"observation_index", "observed_metrics", "previous_high_level_action"}, item_context)
        observation_index = item["observation_index"]
        if isinstance(observation_index, bool) or not isinstance(observation_index, int):
            _fail(item_context, "observation_index must be integer")
        if observation_index <= previous_index or observation_index >= record["step_index"]:
            _fail(item_context, "history extends beyond current decision time or is not strictly ordered")
        previous_index = observation_index
        _validate_metrics(item["observed_metrics"], "diagnostic_image_128px", f"{item_context}.observed_metrics")
        if item["previous_high_level_action"] not in SUPERVISOR_ACTIONS:
            _fail(item_context, "invalid previous_high_level_action")
    if record["task_type"] == TASK_TYPE and record["step_index"] == 0 and history:
        _fail(context, "static step-zero records must not fabricate temporal history")
    budget = model_input["remaining_step_budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 8:
        _fail(f"{context}.model_input.remaining_step_budget", "must be integer in [0,8]")
    if record["step_index"] == 0 and budget != 8:
        _fail(context, "static initial records must expose the frozen eight-step maximum budget")
    constraints = _exact_keys(model_input["actuator_constraints"], CONSTRAINT_KEYS, f"{context}.model_input.actuator_constraints")
    if constraints["units"] != "mm":
        _fail(context, "actuator constraint unit must be mm")
    _validate_limits(
        constraints["per_step_delta_limits"],
        {"lens_x": [-0.05, 0.05], "lens_y": [-0.05, 0.05], "camera_x": [-0.02, 0.02], "camera_y": [-0.02, 0.02]},
        f"{context}.model_input.actuator_constraints.per_step_delta_limits",
    )
    _validate_limits(
        constraints["absolute_position_limits"],
        {axis: [-3.0, 3.0] for axis in AXIS_KEYS},
        f"{context}.model_input.actuator_constraints.absolute_position_limits",
    )
    if constraints["absolute_limit_source"] != "repository_sampling_domain_not_hardware_limit":
        _fail(context, "absolute limit source must not be represented as a hardware limit")
    if constraints["continuous_actions_selected_by"] != "frozen_h1_one_step_cem":
        _fail(context, "continuous actions must remain owned by frozen H1 one-step CEM")

    target_record = _exact_keys(record["target"], TARGET_RECORD_KEYS, f"{context}.target")
    target = {key: target_record[key] for key in TARGET_KEYS}
    try:
        validate_target(target)
    except ValueError as exc:
        _fail(f"{context}.target", str(exc))
    field_mask = _exact_keys(target_record["field_mask"], set(TARGET_KEYS), f"{context}.target.field_mask")
    if any(value is not True for value in field_mask.values()):
        _fail(context, "v1 static records must have all three mapped target fields enabled")
    if target["supervisor_action"] != POLICY_TO_STATIC_ACTION[target["measurement_policy"]]:
        _fail(context, "target violates the frozen static policy/action mapping")
    expected_diagnosis_for_policy = {
        "standard": "nominal",
        "lower_exposure_reacquire": "sensor_saturation",
        "primary_spot": "secondary_reflection",
    }
    if target["diagnosis"] != expected_diagnosis_for_policy[target["measurement_policy"]]:
        _fail(context, "target diagnosis and measurement policy are inconsistent")
    if target["supervisor_action"] in {"continue", "stop"}:
        _fail(context, "static records cannot supervise continue/stop")

    provenance = _exact_keys(record["provenance"], PROVENANCE_KEYS, f"{context}.provenance")
    for name in ("source_manifest_sha256", "generator_sha256"):
        if not isinstance(provenance[name], str) or not HEX64.fullmatch(provenance[name]):
            _fail(f"{context}.provenance.{name}", "must be SHA-256")
    if provenance["source_split"] not in SOURCE_SPLITS:
        _fail(context, f"invalid source split {provenance['source_split']!r}")
    if record["split"] in {"train", "dev"} and provenance["source_split"] in {"iid_heldout", "severity_ood"}:
        _fail(context, "protected/evaluation source entered a train or dev manifest")
    if record["split"] == "frozen_iid" and provenance["source_split"] != "iid_heldout":
        _fail(context, "frozen_iid record does not come from an IID-heldout source")
    if record["split"] == "frozen_ood" and provenance["source_split"] != "severity_ood":
        _fail(context, "frozen_ood record does not come from a severity-OOD source")
    if provenance["anomaly_family"] not in {"sensor_saturation", "secondary_reflection"}:
        _fail(context, "invalid anomaly family")
    if provenance["anomaly_family"] == "secondary_reflection":
        if not provenance["source_cohort"].startswith("width_relative_"):
            _fail(context, "old fixed-pixel reflection is forbidden")
        if provenance["generator_version"] != "secondary_reflection_primary_sigma_direction_v1":
            _fail(context, "reflection generator is not the frozen width-relative implementation")
    expected_source_fault = "clean" if target["diagnosis"] == "nominal" else target["diagnosis"]
    if provenance["source_fault_type"] != expected_source_fault:
        _fail(context, "source fault and target diagnosis disagree")
    target_provenance = _exact_keys(provenance["target_provenance"], set(TARGET_KEYS), f"{context}.provenance.target_provenance")
    if not all(isinstance(value, str) and value for value in target_provenance.values()):
        _fail(context, "target provenance entries must be non-empty strings")
    hashes = _exact_keys(provenance["controller_hashes"], CONTROLLER_HASH_KEYS, f"{context}.provenance.controller_hashes")
    if not all(isinstance(value, str) and HEX64.fullmatch(value) for value in hashes.values()):
        _fail(context, "controller hashes must be SHA-256")
    if provenance["width_quartile"] not in {None, "Q1", "Q2", "Q3", "Q4"}:
        _fail(context, "invalid width quartile")
    if provenance["boundary_status"] not in {"boundary", "non_boundary", "not_applicable"}:
        _fail(context, "invalid boundary status")
    if provenance["anomaly_family"] == "sensor_saturation" and (
        provenance["width_quartile"] is not None or provenance["boundary_status"] != "not_applicable"
    ):
        _fail(context, "saturation cannot carry reflection width/boundary labels")
    if not isinstance(provenance["severity_bucket"], str) or not provenance["severity_bucket"]:
        _fail(context, "severity bucket must be non-empty")
    if provenance["severity_value"] is not None:
        _finite_number(provenance["severity_value"], f"{context}.provenance.severity_value")
    distance = _exact_keys(provenance["counterfactual_metric_distances"], DISTANCE_KEYS, f"{context}.provenance.counterfactual_metric_distances")
    values = distance["per_metric_absolute_difference_tolerances"]
    if not isinstance(values, list) or len(values) != 5:
        _fail(context, "counterfactual per-metric distance must have five values")
    for index, value in enumerate(values):
        _finite_number(value, f"{context}.provenance.counterfactual_metric_distances[{index}]")
    for name in ("maximum_absolute_difference_tolerances", "total_l2_distance_tolerances"):
        _finite_number(distance[name], f"{context}.provenance.counterfactual_metric_distances.{name}")
    if distance["passes_frozen_match"] is not True:
        _fail(context, "counterfactual pair failed the frozen metric-match gate")
    if provenance["source_same_state_metrics_excluded_from_history"] is not True:
        _fail(context, "same-state source metrics must be excluded from temporal history")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite constant {item}")),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                _fail(f"{path}:{line_number}", f"invalid JSON: {exc}")
            records.append(value)
    if not records:
        _fail(str(path), "manifest is empty")
    return records


def _one_split(groups: Mapping[str, set[str]], name: str) -> None:
    bad = {key: sorted(splits) for key, splits in groups.items() if len(splits) > 1}
    if bad:
        preview = dict(list(sorted(bad.items()))[:5])
        raise ManifestValidationError(f"{name} overlap across splits: {preview}")


def validate_manifest_files(paths: Iterable[Path], *, repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    paths = [Path(path).resolve() for path in paths]
    all_records: list[dict[str, Any]] = []
    manifest_counts: dict[str, int] = {}
    for path in paths:
        rows = load_manifest(path)
        manifest_counts[path.name] = len(rows)
        for index, record in enumerate(rows, start=1):
            validate_record(record, repository_root=repository_root, context=f"{path}:{index}")
        all_records.extend(rows)

    ids = [record["sample_id"] for record in all_records]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ManifestValidationError(f"duplicate sample IDs: {duplicate_ids[:10]}")

    setup_splits: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    episode_splits: dict[str, set[str]] = defaultdict(set)
    base_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    pair_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        split = record["split"]
        setup_splits[record["setup_hash"]].add(split)
        pair_splits[record["counterfactual_pair_id"]].add(split)
        if record["episode_hash"] is not None:
            episode_splits[record["episode_hash"]].add(split)
        base_splits[record["augmented_base_hash"]].add(split)
        image_splits[record["assets"]["current_image_sha256"]].add(split)
        pair_records[record["counterfactual_pair_id"]].append(record)
    _one_split(setup_splits, "setup")
    _one_split(pair_splits, "counterfactual pair")
    _one_split(episode_splits, "episode")
    _one_split(base_splits, "augmented/base observation")
    _one_split(image_splits, "exact image hash")

    broken_pairs: dict[str, Any] = {}
    for pair_id, rows in pair_records.items():
        families = {row["provenance"]["anomaly_family"] for row in rows}
        diagnoses = [row["target"]["diagnosis"] for row in rows]
        setups = {row["setup_hash"] for row in rows}
        expected = {"nominal", next(iter(families))} if len(families) == 1 else set()
        if len(rows) != 2 or len(families) != 1 or set(diagnoses) != expected or len(setups) != 1:
            broken_pairs[pair_id] = {
                "record_count": len(rows), "families": sorted(families),
                "diagnoses": sorted(diagnoses), "setup_count": len(setups),
            }
    if broken_pairs:
        raise ManifestValidationError(f"broken pair membership: {dict(list(sorted(broken_pairs.items()))[:5])}")

    class_counts = Counter(record["target"]["diagnosis"] for record in all_records)
    split_counts = Counter(record["split"] for record in all_records)
    return {
        "status": "pass",
        "records": len(all_records),
        "manifest_counts": dict(sorted(manifest_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "unique_sample_ids": len(ids),
        "unique_setups": len(setup_splits),
        "unique_pairs": len(pair_records),
        "unique_image_hashes": len(image_splits),
        "setup_overlap_across_splits": 0,
        "pair_overlap_across_splits": 0,
        "episode_overlap_across_splits": 0,
        "augmented_base_overlap_across_splits": 0,
        "image_hash_overlap_across_splits": 0,
        "protected_rows_in_train_or_dev": 0,
        "fixed_pixel_reflection_rows": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_manifest_files(args.manifests, repository_root=args.repository_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
