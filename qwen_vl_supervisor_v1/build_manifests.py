"""Build the deterministic, setup-disjoint supervisor-v1 source manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .contracts import (
    POLICY_TO_STATIC_ACTION,
    SCHEMA_VERSION,
    SOURCE_POLICY_TO_CANONICAL,
    STATIC_ACTION_MAPPING_VERSION,
    TASK_TYPE,
)

SOURCE_SPECS = {
    "saturation_train_pool": {
        "path": "runs/vlm_optics_benchmark_20260801_154812/visual_data/dataset_train.jsonl",
        "sha256": "e5a437a0af27339ecf6958e2762a4fc8018f8b7358fb39d4fe1f3e3372958c7a",
        "family_audit": "sensor_saturation",
        "anomaly_family": "sensor_saturation",
        "pair_path": "runs/vlm_optics_benchmark_20260801_154812/paired_counterfactuals.jsonl",
        "pair_sha256": "1bc370d80bff5a1c834d6dc43f82f0731c3d4d368011772076ba6ce8762bd31d",
    },
    "saturation_iid": {
        "path": "runs/vlm_optics_benchmark_20260801_154812/visual_data/dataset_iid_heldout.jsonl",
        "sha256": "4300c5aa770a94ac208049188ab23f8ba6fda043c1ab972cc6ba65464433e225",
        "family_audit": "sensor_saturation",
        "anomaly_family": "sensor_saturation",
        "pair_path": "runs/vlm_optics_benchmark_20260801_154812/paired_counterfactuals.jsonl",
        "pair_sha256": "1bc370d80bff5a1c834d6dc43f82f0731c3d4d368011772076ba6ce8762bd31d",
    },
    "saturation_ood": {
        "path": "runs/vlm_optics_benchmark_20260801_154812/visual_data/dataset_severity_ood.jsonl",
        "sha256": "771a289a3b254c3fb99add1f0088c03aa868b55180fcaabe2d2f43a92c1929a1",
        "family_audit": "sensor_saturation",
        "anomaly_family": "sensor_saturation",
        "pair_path": "runs/vlm_optics_benchmark_20260801_154812/paired_counterfactuals.jsonl",
        "pair_sha256": "1bc370d80bff5a1c834d6dc43f82f0731c3d4d368011772076ba6ce8762bd31d",
    },
    "width_relative_train": {
        "path": "reflection_width_relative/data/dataset_train.jsonl",
        "sha256": "4c2e70767a4c017a2cb42cecc90c91dc7a7efb4e2e1ac50b8fca84335013c2ef",
        "family_audit": "secondary_reflection_width_relative",
        "anomaly_family": "secondary_reflection",
        "pair_path": "reflection_width_relative/data/pairs_train.jsonl",
        "pair_sha256": "d5c9b65fd523b9aabe4e2bcee232bade97df192d3a1d13cb6e2a3a813f87da4f",
    },
    "width_relative_dev": {
        "path": "reflection_width_relative/data/dataset_development.jsonl",
        "sha256": "a4462f184308836425d9f8d98370d88f08dc9e27b7a1b17950e43d5031635de5",
        "family_audit": "secondary_reflection_width_relative",
        "anomaly_family": "secondary_reflection",
        "pair_path": "reflection_width_relative/data/pairs_development.jsonl",
        "pair_sha256": "f10a1776c4598eb5e22f9dce19e9b7790911be348a72617366087644bfb6b75f",
    },
    "width_relative_iid": {
        "path": "reflection_width_relative/data/dataset_iid_heldout.jsonl",
        "sha256": "2a3c273d252234f001f2b46a0feaec2c4758012e99273974677bc0c88e4133e8",
        "family_audit": "secondary_reflection_width_relative",
        "anomaly_family": "secondary_reflection",
        "pair_path": "reflection_width_relative/data/pairs_iid_heldout.jsonl",
        "pair_sha256": "990c10db150c849b6419d1bec8f090a8b7533549e8b42199aa3787448ac62a45",
    },
}

GENERATOR_PROVENANCE = {
    "sensor_saturation": {
        "version": "visual_anomalies_sensor_saturation_behavior_freeze",
        "sha256": "959e6bfee86c4658dfdb0fb0d1ef51dc74074534464294b38d11ee6e7804c281",
    },
    "secondary_reflection": {
        "version": "secondary_reflection_primary_sigma_direction_v1",
        "sha256": "ae3aaa76c93327366b011868f7c3adbe22e7e7252abe86c22a9a5b1b9b57f747",
    },
}

CONTROLLER_HASHES = {
    "forward_ensemble": "d9b30627c80817f6ecade1959d8cc9e91e7a9de9cc51485153fdbfaa173aca2e",
    "v12_config": "77cb8bfc8cc064e32ab3cbf473e10f94df5c4ca53869ae235bb5eee782b81ff8",
    "v13_config": "4f5652b0532c039d22684dc89e3c4c90aa2c34f5226c86a85db6f71ae65e3a18",
    "visual_controller": "c309a29e96913730b6b0bf015c92bd865415f50d6350ef0f0287dde9397f2a50",
}

ACTUATOR_CONSTRAINTS = {
    "units": "mm",
    "per_step_delta_limits": {
        "lens_x": [-0.05, 0.05],
        "lens_y": [-0.05, 0.05],
        "camera_x": [-0.02, 0.02],
        "camera_y": [-0.02, 0.02],
    },
    "absolute_position_limits": {
        "lens_x": [-3.0, 3.0],
        "lens_y": [-3.0, 3.0],
        "camera_x": [-3.0, 3.0],
        "camera_y": [-3.0, 3.0],
    },
    "absolute_limit_source": "repository_sampling_domain_not_hardware_limit",
    "continuous_actions_selected_by": "frozen_h1_one_step_cem",
}

SOURCE_SPLIT_NAME = {
    "saturation_train_pool": "train",
    "saturation_iid": "iid_heldout",
    "saturation_ood": "severity_ood",
    "width_relative_train": "train",
    "width_relative_dev": "development",
    "width_relative_iid": "iid_heldout",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = child
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=unique_object,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite constant {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def load_pair_index(
    root: Path, pair_path: str, pair_sha256: str, family: str
) -> dict[str, dict[str, Any]]:
    source = root / pair_path
    actual_hash = sha256_path(source)
    if actual_hash != pair_sha256:
        raise RuntimeError(
            f"pair metadata hash drift for {source}: expected {pair_sha256}, got {actual_hash}"
        )
    rows = read_jsonl(source)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_family = row.get("family", row.get("anomaly_type"))
        if family == "sensor_saturation" and row_family != "sensor_saturation":
            continue
        if family == "secondary_reflection" and row_family != "secondary_reflection_width_relative":
            continue
        index[row["pair_id"]] = row
    return index


def current_metrics(values: Iterable[Any]) -> dict[str, Any]:
    values = list(values)
    if len(values) != 5:
        raise ValueError(f"expected five current metrics, got {len(values)}")
    return {
        "coordinate_frame": "diagnostic_image_128px",
        "centroid_x": values[0],
        "centroid_y": values[1],
        "width_x": values[2],
        "width_y": values[3],
        "peak_intensity": values[4],
    }


def goal_metrics(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "coordinate_frame": "lab_sensor_1024px_and_raw_peak",
        "centroid_x": target["centroid_x_px"],
        "centroid_y": target["centroid_y_px"],
        "width_x": target["sigma_x_px"],
        "width_y": target["sigma_y_px"],
        "peak_intensity": target["peak_intensity"],
    }


def normalize_pair_distance(pair: dict[str, Any], family: str) -> dict[str, Any]:
    if family == "sensor_saturation":
        value = pair["normalized_metric_distance"]
    else:
        value = pair["post_serialization_metric_distance"]
    return {
        "per_metric_absolute_difference_tolerances": list(value["per_metric_absolute_difference_tolerances"]),
        "maximum_absolute_difference_tolerances": value["maximum_absolute_difference_tolerances"],
        "total_l2_distance_tolerances": value["total_l2_distance_tolerances"],
        "passes_frozen_match": bool(value["passes_frozen_match"]),
    }


def severity(row: dict[str, Any], pair: dict[str, Any], family: str) -> tuple[str, float | None]:
    if family == "sensor_saturation":
        raw = row.get("severity_scalar")
        if raw is None:
            raw = pair.get("severity", {}).get("clip_level_fraction_of_peak")
        if raw is None:
            return "unknown", None
        value = float(raw)
        if value < 0.30:
            return "saturation_severe_clip", value
        if value < 0.45:
            return "saturation_medium_clip", value
        return "saturation_mild_clip", value
    params = row.get("generator_parameters") or pair.get("generator_parameters", {})
    raw = params.get("relative_reflection_amplitude")
    if raw is None:
        return "unknown", None
    value = float(raw)
    if value <= 0.30:
        return "reflection_amplitude_low", value
    if value <= 0.40:
        return "reflection_amplitude_medium", value
    return "reflection_amplitude_high", value


def supervisor_record(
    *,
    root: Path,
    source_name: str,
    row: dict[str, Any],
    pair: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[source_name]
    family = spec["anomaly_family"]
    fault_type = row["supervision"]["fault_type"]
    diagnosis = "nominal" if fault_type == "clean" else family
    source_policy = row["supervision"]["oracle_recovery_decision"]
    try:
        policy = SOURCE_POLICY_TO_CANONICAL[source_policy]
    except KeyError as exc:
        raise ValueError(f"unmapped source policy {source_policy!r}") from exc
    action = POLICY_TO_STATIC_ACTION[policy]

    source_path = root / spec["path"]
    image_path = (source_path.parent / row["model_input"]["image_ref"]).resolve()
    try:
        image_rel = image_path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"image escapes repository: {image_path}") from exc
    image_hash = sha256_path(image_path)
    with Image.open(image_path) as image:
        image_format, image_mode, width, height = image.format, image.mode, image.width, image.height

    pair_id = "pair_" + stable_hash("qwen_vl_supervisor_v1", family, row["pair_id"])[:24]
    sample_id = "qvlsup1_" + stable_hash(spec["sha256"], row["sample_id"], pair_id)[:24]
    severity_bucket, severity_value = severity(row, pair, family)
    width_quartile = row.get("beam_width_quartile") if family == "secondary_reflection" else None
    boundary_status = row.get("boundary_status", "not_applicable") if family == "secondary_reflection" else "not_applicable"

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "task_type": TASK_TYPE,
        "split": split,
        "setup_hash": row["setup_hash"],
        "episode_hash": None,
        "step_index": 0,
        "counterfactual_pair_id": pair_id,
        "augmented_base_hash": image_hash,
        "assets": {
            "current_image_path": image_rel,
            "current_image_sha256": image_hash,
            "format": image_format,
            "mode": image_mode,
            "width": width,
            "height": height,
        },
        "model_input": {
            "current_metrics": current_metrics(row["model_input"]["five_metrics"]),
            "goal_metrics": goal_metrics(row["model_input"]["target"]),
            "recent_history": [],
            "remaining_step_budget": 8,
            "actuator_constraints": ACTUATOR_CONSTRAINTS,
        },
        "target": {
            "diagnosis": diagnosis,
            "measurement_policy": policy,
            "supervisor_action": action,
            "field_mask": {
                "diagnosis": True,
                "measurement_policy": True,
                "supervisor_action": True,
            },
        },
        "provenance": {
            "source_cohort": source_name,
            "source_manifest_path": spec["path"],
            "source_manifest_sha256": spec["sha256"],
            "source_sample_id": row["sample_id"],
            "source_split": SOURCE_SPLIT_NAME[source_name],
            "anomaly_family": family,
            "source_fault_type": fault_type,
            "target_provenance": {
                "diagnosis": "direct:supervision.fault_type+clean_to_nominal_mapping_v1",
                "measurement_policy": "direct:supervision.oracle_recovery_decision+reversible_policy_mapping_v1",
                "supervisor_action": f"derived:{STATIC_ACTION_MAPPING_VERSION}",
            },
            "generator_version": GENERATOR_PROVENANCE[family]["version"],
            "generator_sha256": GENERATOR_PROVENANCE[family]["sha256"],
            "controller_version": "frozen_h1_one_step_cem_gainaware_sequential_v13_horizon8",
            "controller_hashes": CONTROLLER_HASHES,
            "width_quartile": width_quartile,
            "boundary_status": boundary_status,
            "severity_bucket": severity_bucket,
            "severity_value": severity_value,
            "counterfactual_metric_distances": normalize_pair_distance(pair, family),
            "source_same_state_metrics_excluded_from_history": True,
        },
    }


def filtered_rows(root: Path, source_name: str) -> list[dict[str, Any]]:
    spec = SOURCE_SPECS[source_name]
    source_path = root / spec["path"]
    actual_hash = sha256_path(source_path)
    if actual_hash != spec["sha256"]:
        raise RuntimeError(
            f"source hash drift for {source_path}: expected {spec['sha256']}, got {actual_hash}"
        )
    rows = [row for row in read_jsonl(source_path) if row.get("family_audit") == spec["family_audit"]]
    if spec["family_audit"] == "secondary_reflection_width_relative":
        forbidden = [row for row in rows if row["supervision"]["fault_type"] not in {"clean", "secondary_reflection"}]
        if forbidden:
            raise RuntimeError("width-relative cohort contains a non-width-relative target")
    return rows


def saturation_train_dev_pairs(rows: list[dict[str, Any]], dev_pairs: int = 6) -> set[str]:
    pair_setup: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_setup[row["pair_id"]].add(row["setup_hash"])
    bad = {pair: values for pair, values in pair_setup.items() if len(values) != 1}
    if bad:
        raise RuntimeError(f"saturation pair spans setup hashes: {bad}")
    ranked = sorted(
        pair_setup,
        key=lambda pair: (stable_hash("qwen_vl_supervisor_v1_saturation_dev", pair), pair),
    )
    if len(ranked) < dev_pairs:
        raise RuntimeError(f"need {dev_pairs} saturation dev pairs, only {len(ranked)} available")
    return set(ranked[:dev_pairs])


def build(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "dev", "frozen_iid", "frozen_ood")}

    sat_pool = filtered_rows(root, "saturation_train_pool")
    sat_dev_ids = saturation_train_dev_pairs(sat_pool, dev_pairs=6)
    source_routes = [
        ("saturation_iid", "frozen_iid"),
        ("saturation_ood", "frozen_ood"),
        ("width_relative_train", "train"),
        ("width_relative_dev", "dev"),
        ("width_relative_iid", "frozen_iid"),
    ]

    pair_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for source_name in SOURCE_SPECS:
        spec = SOURCE_SPECS[source_name]
        family = spec["anomaly_family"]
        cache_key = f"{spec['pair_path']}::{spec['pair_sha256']}::{family}"
        if cache_key not in pair_cache:
            pair_cache[cache_key] = load_pair_index(
                root, spec["pair_path"], spec["pair_sha256"], family
            )
        pair_cache[source_name] = pair_cache[cache_key]

    for row in sat_pool:
        split = "dev" if row["pair_id"] in sat_dev_ids else "train"
        pair = pair_cache["saturation_train_pool"].get(row["pair_id"])
        if pair is None:
            raise RuntimeError(f"missing saturation pair metadata for {row['pair_id']}")
        records[split].append(
            supervisor_record(root=root, source_name="saturation_train_pool", row=row, pair=pair, split=split)
        )

    for source_name, split in source_routes:
        for row in filtered_rows(root, source_name):
            pair = pair_cache[source_name].get(row["pair_id"])
            if pair is None:
                raise RuntimeError(f"missing pair metadata for {source_name}:{row['pair_id']}")
            records[split].append(
                supervisor_record(root=root, source_name=source_name, row=row, pair=pair, split=split)
            )

    paths: dict[str, Path] = {}
    for split, split_rows in records.items():
        split_rows.sort(key=lambda row: row["sample_id"])
        path = output_dir / f"manifest_{split}.jsonl"
        text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in split_rows)
        path.write_text(text, encoding="utf-8")
        paths[split] = path

    from .validate_manifest import validate_manifest_files

    audit = validate_manifest_files(list(paths.values()), repository_root=root)
    index = {
        "schema_version": SCHEMA_VERSION,
        "builder": "qwen_vl_supervisor_v1.build_manifests",
        "saturation_dev_pair_selection": {
            "algorithm": "six lowest sha256('qwen_vl_supervisor_v1_saturation_dev' + pair_id)",
            "count": len(sat_dev_ids),
            "selected_pair_hashes": sorted(
                stable_hash("selected_saturation_dev_pair", pair_id) for pair_id in sat_dev_ids
            ),
        },
        "manifests": {
            split: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_path(path),
                "records": len(records[split]),
                "class_counts": dict(sorted(Counter(row["target"]["diagnosis"] for row in records[split]).items())),
                "setup_count": len({row["setup_hash"] for row in records[split]}),
                "pair_count": len({row["counterfactual_pair_id"] for row in records[split]}),
            }
            for split, path in paths.items()
        },
        "validation": audit,
        "source_hashes": {name: spec["sha256"] for name, spec in sorted(SOURCE_SPECS.items())},
        "source_pair_hashes": {
            name: {"path": spec["pair_path"], "sha256": spec["pair_sha256"]}
            for name, spec in sorted(SOURCE_SPECS.items())
        },
        "excluded": {
            "old_fixed_pixel_reflection": True,
            "width_relative_severity_ood": "no valid dataset; partial images excluded",
            "controller_episode_histories": "no temporally paired anomaly images",
        },
    }
    index_path = output_dir / "manifest_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "manifests")
    args = parser.parse_args()
    index = build(args.repository_root, args.output_dir)
    print(json.dumps(index["manifests"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
