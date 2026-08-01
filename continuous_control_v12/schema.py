"""JSON-schema and physical-contract validation for v12 datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    OUTPUT_FIELDS,
    Bounds,
    action_vector,
    apply_action,
    assert_no_q_star,
    context_hash,
    position_vector,
    setup_hash,
    split_hash,
    validate_action,
    validate_positions,
)

SCHEMA_DIR = Path(__file__).with_name("schemas")


def _schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_json_schema(value: Any, schema_name: str) -> None:
    import jsonschema

    schema = _schema(schema_name)
    jsonschema.Draft202012Validator(schema).validate(value)


def deployed_transition_input(row: Mapping[str, Any]) -> dict[str, Any]:
    """The only fields a deployed world model/controller may consume."""

    return {
        "setup_context": row["setup_context"],
        "simulator_fixed": row["simulator_fixed"],
        "positions_mm": row["positions_mm"],
        "image_ref": row["image_ref"],
        "metrics": row["metrics"],
        "action_mm": row["action_mm"],
    }


def validate_transition(row: Mapping[str, Any], bounds: Bounds) -> None:
    validate_json_schema(row, "transition_v12.schema.json")
    validate_action(row["action_mm"], bounds)
    validate_positions(row["positions_mm"], bounds)
    validate_positions(row["next_positions_mm"], bounds)
    expected = apply_action(row["positions_mm"], row["action_mm"], bounds)
    actual = position_vector(row["next_positions_mm"])
    if not np.allclose(expected, actual, atol=1e-10, rtol=0.0):
        raise ValueError(f"{row['transition_id']}: next position mismatch")
    if row["setup_hash"] != setup_hash(
        row["setup_context"], row["simulator_fixed"]
    ):
        raise ValueError(f"{row['transition_id']}: setup hash mismatch")
    if row["context_hash"] != context_hash(
        row["setup_context"],
        row["simulator_fixed"],
        row["positions_mm"],
        row["metrics"],
    ):
        raise ValueError(f"{row['transition_id']}: context hash mismatch")
    if row["schema_version"] == "v12.1.0":
        fixed = row["simulator_fixed"]
        if row["simulator_semantics_version"] != fixed.get(
            "simulator_semantics_version"
        ):
            raise ValueError(
                f"{row['transition_id']}: simulator semantics mismatch"
            )
        if row["sampling_method"] != fixed.get("sensor_sampling_method"):
            raise ValueError(f"{row['transition_id']}: sampling method mismatch")
        if row["power_semantics"] != fixed.get("power_semantics"):
            raise ValueError(f"{row['transition_id']}: power semantics mismatch")
        if row["intensity_normalization"] != fixed.get(
            "intensity_normalization"
        ):
            raise ValueError(
                f"{row['transition_id']}: intensity normalization mismatch"
            )
        requested_next = position_vector(row["positions_mm"]) + action_vector(
            row["requested_action_mm"]
        )
        if not np.allclose(
            requested_next,
            position_vector(row["requested_next_positions_mm"]),
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(
                f"{row['transition_id']}: requested next position mismatch"
            )
        for legacy_name, explicit_name in (
            ("metrics", "metrics_lab_frame"),
            ("next_metrics", "next_metrics_lab_frame"),
        ):
            if any(
                float(row[legacy_name][field])
                != float(row[explicit_name][field])
                for field in OUTPUT_FIELDS
            ):
                raise ValueError(
                    f"{row['transition_id']}: explicit lab metrics mismatch"
                )
        auxiliary = row["auxiliary"]
        if not np.isclose(
            float(auxiliary["captured_power"]),
            float(auxiliary["captured_power_w"]),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{row['transition_id']}: captured power aliases differ"
            )
        if not np.isclose(
            float(auxiliary["source_integrated_power_w"]),
            float(row["setup_context"]["power_w"]),
            rtol=2e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"{row['transition_id']}: source power normalization mismatch"
            )
        if "minimum_initial_captured_power_fraction" in fixed:
            sampling = row["sampling"]
            if not {
                "setup_resample_attempt",
                "initial_captured_power_fraction",
            }.issubset(sampling):
                raise ValueError(
                    f"{row['transition_id']}: setup signal metadata missing"
                )
            attempt = int(sampling["setup_resample_attempt"])
            fraction = float(sampling["initial_captured_power_fraction"])
            maximum_attempts = int(fixed["maximum_setup_resample_attempts"])
            if attempt < 0 or attempt >= maximum_attempts:
                raise ValueError(
                    f"{row['transition_id']}: setup retry index out of range"
                )
            if not np.isfinite(fraction) or fraction < float(
                fixed["minimum_initial_captured_power_fraction"]
            ):
                raise ValueError(
                    f"{row['transition_id']}: initial captured-power "
                    "fraction violates configured floor"
                )
    assert_no_q_star(deployed_transition_input(row))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_dataset(data_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    if "physics_structured_rebuild_v10" in data_dir.parts:
        raise ValueError("v12 validation refuses every v10 data path")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_json_schema(manifest, "manifest_v12.schema.json")
    if manifest["schema_version"] != config["schema_version"]:
        raise ValueError("manifest/config schema version mismatch")
    fixed = manifest["generator"]["simulator_fixed"]
    if manifest["schema_version"] == "v12.1.0":
        semantics = config["simulator"]["semantics"]
        if manifest["simulator_semantics_version"] != semantics[
            "simulator_semantics_version"
        ]:
            raise ValueError("manifest/config simulator semantics mismatch")
        if manifest.get("power_affects_current_simulator") is not True:
            raise ValueError("v12.1 manifest must attest active power semantics")
        if manifest.get("scale") == "smoke":
            expected_grid_size = int(config["smoke"]["simulator_grid_size"])
            expected_grid_extent = float(
                config["smoke"].get(
                    "simulator_grid_extent_mm",
                    config["simulator"]["data_grid_extent_mm"],
                )
            )
            expected_sensor = list(config["smoke"]["sensor_resolution"])
        else:
            expected_grid_size = int(config["simulator"]["data_grid_size"])
            expected_grid_extent = float(
                config["simulator"]["data_grid_extent_mm"]
            )
            expected_sensor = list(
                config["simulator"]["data_sensor_resolution"]
            )
        if int(fixed["grid_size"]) != expected_grid_size:
            raise ValueError("manifest/config simulation grid size mismatch")
        if not np.isclose(
            float(fixed["grid_extent_mm"]),
            expected_grid_extent,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("manifest/config simulation grid extent mismatch")
        if list(fixed["sensor_resolution_px"]) != expected_sensor:
            raise ValueError("manifest/config sensor resolution mismatch")
        for field in (
            "minimum_initial_captured_power_fraction",
            "maximum_setup_resample_attempts",
        ):
            if field in semantics and not np.isclose(
                float(fixed[field]),
                float(semantics[field]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"manifest/config corrected setup policy mismatch: {field}"
                )
    bounds = Bounds.from_config(config)
    bounds.validate()
    group_sets: dict[str, set[str]] = {}
    reports: dict[str, Any] = {}
    setup_signal_by_group: dict[str, tuple[int, float]] = {}
    for split in ("train", "development", "test"):
        path = data_dir / "transitions" / f"{split}.jsonl"
        rows = read_jsonl(path)
        group_ids: set[str] = set()
        transition_ids: set[str] = set()
        for row in rows:
            if row["split"] != split:
                raise ValueError(f"{row['transition_id']}: split mismatch")
            validate_transition(row, bounds)
            if row["simulator_fixed"] != manifest["generator"][
                "simulator_fixed"
            ]:
                raise ValueError(
                    f"{row['transition_id']}: simulator fixed values differ "
                    "from manifest"
                )
            transition_id = str(row["transition_id"])
            if transition_id in transition_ids:
                raise ValueError(f"duplicate transition ID: {transition_id}")
            transition_ids.add(transition_id)
            group_id = str(row["group_id"])
            group_ids.add(group_id)
            if "minimum_initial_captured_power_fraction" in fixed:
                setup_signal = (
                    int(row["sampling"]["setup_resample_attempt"]),
                    float(row["sampling"]["initial_captured_power_fraction"]),
                )
                previous_signal = setup_signal_by_group.setdefault(
                    group_id, setup_signal
                )
                if setup_signal != previous_signal:
                    raise ValueError(
                        f"{group_id}: inconsistent setup signal metadata"
                    )
            for field in ("image_ref", "next_image_ref"):
                reference = row[field]
                if reference is not None and not (data_dir / reference).is_file():
                    raise ValueError(
                        f"{row['transition_id']}: missing image {reference}"
                    )
                if reference is not None and row["schema_version"] == "v12.1.0":
                    with np.load(data_dir / reference, allow_pickle=False) as image:
                        required = {
                            "image_raw",
                            "image_normalized",
                            "intensity",
                            "valid_region_mask",
                        }
                        if not required.issubset(image.files):
                            raise ValueError(
                                f"{row['transition_id']}: incomplete v12.1 image"
                            )
        expected = manifest["split_summary"][split]
        if len(rows) != int(expected["transitions"]):
            raise ValueError(f"{split}: transition count differs")
        if len(group_ids) != int(expected["groups"]):
            raise ValueError(f"{split}: group count differs")
        if split_hash(sorted(group_ids)) != manifest["split_hashes"][split]:
            raise ValueError(f"{split}: group hash differs")
        if sha256_file(path) != expected["jsonl_sha256"]:
            raise ValueError(f"{split}: JSONL hash differs")
        group_sets[split] = group_ids
        reports[split] = {
            "groups": len(group_ids),
            "transitions": len(rows),
            "jsonl_sha256": sha256_file(path),
        }
    for index, left in enumerate(group_sets):
        for right in list(group_sets)[index + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise ValueError(
                    f"group overlap between {left}/{right}: {sorted(overlap)[:3]}"
                )
    return {
        "version": "continuous_control_v12_validation",
        "complete": True,
        "reports": reports,
        "cross_split_group_overlap": 0,
        "q_star_deployed_input": False,
        "setup_signal_quality": {
            "policy_active": bool(
                "minimum_initial_captured_power_fraction" in fixed
            ),
            "minimum_initial_captured_power_fraction": (
                None
                if not setup_signal_by_group
                else min(value[1] for value in setup_signal_by_group.values())
            ),
            "groups_resampled": sum(
                value[0] > 0 for value in setup_signal_by_group.values()
            ),
            "maximum_setup_resample_attempt_used": (
                0
                if not setup_signal_by_group
                else max(value[0] for value in setup_signal_by_group.values())
            ),
        },
    }
