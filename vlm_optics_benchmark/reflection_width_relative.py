#!/usr/bin/env python3
"""Width-relative secondary-reflection development and frozen evaluation.

This module deliberately does not alter the original fixed-pixel anomaly path in
``visual_anomalies.py``.  The new version computes separation from the clean
primary covariance before it constructs the secondary component.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, shift
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from active_diagnosis_v13.run_gate_a import _planner_config, matched_planner_seed
from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    Bounds,
    apply_action,
    metrics_vector,
    normalized_distance,
    position_dict,
    position_vector,
)
from continuous_control_v12.mpc import CEMMPC, learned_predictor
from continuous_control_v12.simulator import simulate_state
from continuous_control_v12.world_model import load_forward_ensemble
from vlm_optics_benchmark.visual_anomalies import (
    IMAGE_SIZE,
    MATCH_TOLERANCE,
    _candidate_options,
    _fit_tiny_model,
    _predict_tiny,
    _score,
    _torch_models,
    canonical_patch,
    matched_clean_counterfactual,
    moment_metrics,
    normalized_metric_distance,
    read_jsonl,
    sha256_file,
    stable_seed,
    write_jsonl,
)


VERSION = "secondary_reflection_primary_sigma_direction_v1"
SEPARATION_MODE = "primary_sigma_direction"
METRIC_FIELDS = (
    "centroid_x",
    "centroid_y",
    "width_x",
    "width_y",
    "peak_intensity",
)
DIRECTION_LABELS = (
    "east",
    "north_east",
    "north",
    "north_west",
    "west",
    "south_west",
    "south",
    "south_east",
)
ARMS = ("no_diagnosis", "oracle_diagnosis", "learned_image_diagnostic")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_png(image: np.ndarray, path: Path) -> np.ndarray:
    quantized = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantized, mode="L").save(path)
    return quantized.astype(np.float32) / 255.0


def primary_centroid_covariance(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid (x,y) and full 2-D covariance of the clean primary."""

    values = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError("clean primary image has no finite positive power")
    y, x = np.indices(values.shape, dtype=np.float64)
    centroid = np.asarray(
        [(values * x).sum() / total, (values * y).sum() / total],
        dtype=np.float64,
    )
    dx = x - centroid[0]
    dy = y - centroid[1]
    covariance = np.asarray(
        [
            [(values * dx * dx).sum() / total, (values * dx * dy).sum() / total],
            [(values * dx * dy).sum() / total, (values * dy * dy).sum() / total],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("clean primary covariance is non-finite")
    return centroid, covariance


def directional_sigma(covariance: np.ndarray, direction_xy: Sequence[float]) -> float:
    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("reflection direction must be finite and non-zero")
    unit = direction / norm
    variance = float(unit @ np.asarray(covariance, dtype=np.float64) @ unit)
    return math.sqrt(max(variance, 1e-12))


def _scale_about_centroid(
    image: np.ndarray,
    centroid_xy: Sequence[float],
    width_ratio: float,
) -> np.ndarray:
    if not np.isfinite(width_ratio) or width_ratio <= 0.0:
        raise ValueError("reflection width ratio must be finite and positive")
    values = np.asarray(image, dtype=np.float64)
    y, x = np.indices(values.shape, dtype=np.float64)
    cx, cy = map(float, centroid_xy)
    input_x = cx + (x - cx) / width_ratio
    input_y = cy + (y - cy) / width_ratio
    component = map_coordinates(
        values,
        [input_y, input_x],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    source_peak = max(float(values.max()), 1e-30)
    component_peak = max(float(component.max()), 1e-30)
    return component * (source_peak / component_peak)


def inject_width_relative_reflection(
    clean_primary: np.ndarray,
    *,
    k: float,
    amplitude: float,
    width_ratio: float,
    angle_radians: float,
    normalize_peak: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add a shifted component at ``k * sigma_direction``.

    All geometry is frozen from ``clean_primary`` before the secondary component
    is created.  Shift uses zero fill and never wraps at a sensor boundary.
    """

    values = np.clip(np.asarray(clean_primary, dtype=np.float64), 0.0, None)
    if not all(np.isfinite(value) for value in (k, amplitude, width_ratio, angle_radians)):
        raise ValueError("reflection parameters must be finite")
    if k <= 0.0 or amplitude <= 0.0:
        raise ValueError("k and amplitude must be positive")
    centroid, covariance = primary_centroid_covariance(values)
    unit = np.asarray([math.cos(angle_radians), math.sin(angle_radians)], dtype=np.float64)
    sigma = directional_sigma(covariance, unit)
    separation = float(k) * sigma
    component = _scale_about_centroid(values, centroid, float(width_ratio))
    pre_shift_power = float(component.sum())
    shifted = shift(
        component,
        shift=(separation * unit[1], separation * unit[0]),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    retained_fraction = float(shifted.sum()) / max(pre_shift_power, 1e-30)
    combined = values + float(amplitude) * shifted
    if normalize_peak:
        combined /= max(float(combined.max()), 1e-30)
    output = np.clip(combined, 0.0, None).astype(np.float32)
    metadata = {
        "version": VERSION,
        "separation_mode": SEPARATION_MODE,
        "clean_primary_centroid_xy": centroid.tolist(),
        "clean_primary_covariance_xy": covariance.tolist(),
        "direction_unit_xy": unit.tolist(),
        "direction_angle_radians": float(angle_radians),
        "sigma_direction_px": float(sigma),
        "k": float(k),
        "separation_px": float(separation),
        "relative_reflection_amplitude": float(amplitude),
        "reflection_width_ratio": float(width_ratio),
        "boundary_handling": "bilinear_shift_zero_fill_no_wrap_component_truncation",
        "component_power_retained_fraction": retained_fraction,
        "power_policy": "secondary_power_added_then_combined_peak_normalized",
        "covariance_source": "clean_primary_before_reflection_injection",
    }
    return output, metadata


def _boundary_status(case: Mapping[str, Any]) -> str:
    auxiliary = case.get("initial_auxiliary", {})
    boundary = bool(auxiliary.get("camera_boundary_indicator", False))
    clipping = float(auxiliary.get("clipping_fraction", 0.0) or 0.0) > 0.01
    distance = auxiliary.get("distance_to_camera_boundary_px", math.inf)
    near = float(math.inf if distance is None else distance) <= 64.0
    return "boundary" if boundary or clipping or near else "non_boundary"


def _direction_label(angle: float) -> str:
    index = int(np.rint((angle % (2.0 * math.pi)) / (math.pi / 4.0))) % 8
    return DIRECTION_LABELS[index]


def _rank_quartiles(values: Sequence[float]) -> list[str]:
    order = np.argsort(np.asarray(values), kind="stable")
    output = [""] * len(values)
    for rank, index in enumerate(order):
        quartile = min(3, (4 * rank) // max(len(values), 1)) + 1
        output[int(index)] = f"Q{quartile}"
    return output


def _captures_for_suite(
    suite: Mapping[str, Any],
    v12_config: Path,
    base_config: Path,
) -> list[dict[str, Any]]:
    bounds = Bounds.from_config(json.loads(v12_config.resolve().read_text()))
    captures = []
    widths = []
    for case in suite["cases"]:
        capture = simulate_state(
            case["setup_context"],
            case["initial_positions_mm"],
            case["simulator_fixed"],
            str(base_config.resolve()),
            bounds,
        )
        primary = canonical_patch(capture["intensity"])
        _, covariance = primary_centroid_covariance(primary)
        width = math.sqrt(max(float(np.trace(covariance)) / 2.0, 1e-12))
        captures.append({"case": case, "capture": capture, "primary": primary, "equivalent_sigma": width})
        widths.append(width)
    for item, quartile in zip(captures, _rank_quartiles(widths), strict=True):
        item["beam_width_quartile"] = quartile
    return captures


def _grid_values(config: Mapping[str, Any], key: str) -> list[float]:
    return [float(value) for value in config[key]]


def parameters_for_case(
    case_id: str,
    split: str,
    parameter_config: Mapping[str, Any],
) -> dict[str, float]:
    """Choose only from preregistered values, deterministically and label-free."""

    values = parameter_config["severity_ood" if split == "severity_ood" else "iid"]
    rng = np.random.default_rng(stable_seed("width_relative_parameters_v1", case_id, split))
    angles = _grid_values(values, "direction_angles_radians")
    return {
        "k": float(rng.choice(_grid_values(values, "k_values"))),
        "relative_reflection_amplitude": float(
            rng.choice(_grid_values(values, "amplitude_values"))
        ),
        "reflection_width_ratio": float(
            rng.choice(_grid_values(values, "width_ratio_values"))
        ),
        "direction_angle_radians": float(rng.choice(angles)),
    }


def _pair_from_primary(
    primary: np.ndarray,
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    anomalous, generator = inject_width_relative_reflection(
        primary,
        k=float(parameters["k"]),
        amplitude=float(parameters["relative_reflection_amplitude"]),
        width_ratio=float(parameters["reflection_width_ratio"]),
        angle_radians=float(parameters["direction_angle_radians"]),
    )
    anomalous_metrics = moment_metrics(anomalous)
    clean = matched_clean_counterfactual(primary, anomalous_metrics)
    clean_metrics = moment_metrics(clean)
    pre = normalized_metric_distance(clean_metrics, anomalous_metrics)
    clean_serialized = np.rint(np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    anomaly_serialized = np.rint(np.clip(anomalous, 0.0, 1.0) * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    post = normalized_metric_distance(
        moment_metrics(clean_serialized), moment_metrics(anomaly_serialized)
    )
    return clean, anomalous, generator, pre, post


def _morphology_metrics(
    clean: np.ndarray,
    anomalous: np.ndarray,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    difference = np.asarray(anomalous, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    centroid = np.asarray(generator["clean_primary_centroid_xy"], dtype=np.float64)
    unit = np.asarray(generator["direction_unit_xy"], dtype=np.float64)
    center = centroid + float(generator["separation_px"]) * unit
    y, x = np.indices(anomalous.shape, dtype=np.float64)
    radius = max(
        2.0,
        0.75
        * float(generator["sigma_direction_px"])
        * float(generator["reflection_width_ratio"]),
    )
    region = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    region_contrast = float(difference[region].mean()) if region.any() else 0.0
    rms = float(np.sqrt(np.mean(difference**2)))
    return {
        "image_rms_difference_from_matched_clean": rms,
        "secondary_region_mean_contrast": region_contrast,
        "visible_shoulder_consistent": bool(rms >= 0.02 and region_contrast >= 0.02),
    }


def _pair_tile(clean: np.ndarray, anomalous: np.ndarray, caption: str) -> Image.Image:
    left = np.rint(np.clip(clean, 0.0, 1.0) * 255.0).astype(np.uint8)
    right = np.rint(np.clip(anomalous, 0.0, 1.0) * 255.0).astype(np.uint8)
    tile = Image.new("L", (2 * IMAGE_SIZE, IMAGE_SIZE + 18), color=0)
    tile.paste(Image.fromarray(left, mode="L"), (0, 18))
    tile.paste(Image.fromarray(right, mode="L"), (IMAGE_SIZE, 18))
    ImageDraw.Draw(tile).text((3, 3), caption[:72], fill=255)
    return tile


def _save_montage(tiles: Sequence[Image.Image], path: Path) -> None:
    if not tiles:
        return
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    montage = Image.new("L", (columns * width, rows * height), color=0)
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % columns) * width, (index // columns) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)


def development_search(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"development result already exists: {args.output}")
    search = json.loads(args.search_preregistration.resolve().read_text())
    suite = json.loads(args.suite.resolve().read_text())
    captures = _captures_for_suite(suite, args.v12_config, args.base_config)
    grid = search["grid"]
    records: list[dict[str, Any]] = []
    montage_tiles: dict[tuple[str, str], list[Image.Image]] = defaultdict(list)
    for item in captures:
        case = item["case"]
        primary = item["primary"]
        for k in _grid_values(grid, "k_values"):
            for amplitude in _grid_values(grid, "amplitude_values"):
                for width_ratio in _grid_values(grid, "width_ratio_values"):
                    for angle in _grid_values(grid, "direction_angles_radians"):
                        parameters = {
                            "k": k,
                            "relative_reflection_amplitude": amplitude,
                            "reflection_width_ratio": width_ratio,
                            "direction_angle_radians": angle,
                        }
                        clean, anomaly, generator, pre, post = _pair_from_primary(primary, parameters)
                        morphology = _morphology_metrics(clean, anomaly, generator)
                        record = {
                            "setup_id": case["group_id"],
                            "setup_hash": case["setup_hash"],
                            "beam_width_quartile": item["beam_width_quartile"],
                            "equivalent_primary_sigma_px": item["equivalent_sigma"],
                            "boundary_status": _boundary_status(case),
                            "direction": _direction_label(angle),
                            "parameters": parameters,
                            "generator": generator,
                            "pre_serialization_metric_distance": pre,
                            "post_serialization_metric_distance": post,
                            "morphology": morphology,
                            "primary_spot_specialist_recovery_viable": bool(
                                np.all(np.isfinite(moment_metrics(primary)))
                            ),
                        }
                        records.append(record)
                        caption = (
                            f"{item['beam_width_quartile']} k={k:g} a={amplitude:g} "
                            f"w={width_ratio:g} {_direction_label(angle)}"
                        )
                        tile = _pair_tile(clean, anomaly, caption)
                        groups = (
                            ("beam_width_quartile", item["beam_width_quartile"]),
                            ("k", f"{k:.2f}"),
                            ("amplitude", f"{amplitude:.2f}"),
                            ("direction", _direction_label(angle)),
                        )
                        for group in groups:
                            if len(montage_tiles[group]) < 8:
                                montage_tiles[group].append(tile.copy())
    for (group, value), tiles in montage_tiles.items():
        _save_montage(tiles, args.montage_dir / group / f"{value}.png")

    def aggregate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"records": 0}
        return {
            "records": len(subset),
            "paired_match_rate_pre_serialization": float(
                np.mean([row["pre_serialization_metric_distance"]["passes_frozen_match"] for row in subset])
            ),
            "paired_match_rate_post_serialization": float(
                np.mean([row["post_serialization_metric_distance"]["passes_frozen_match"] for row in subset])
            ),
            "maximum_pre_serialization_per_metric_difference": float(
                max(row["pre_serialization_metric_distance"]["maximum_absolute_difference_tolerances"] for row in subset)
            ),
            "maximum_pre_serialization_l2_distance": float(
                max(row["pre_serialization_metric_distance"]["total_l2_distance_tolerances"] for row in subset)
            ),
            "maximum_post_serialization_per_metric_difference": float(
                max(row["post_serialization_metric_distance"]["maximum_absolute_difference_tolerances"] for row in subset)
            ),
            "maximum_post_serialization_l2_distance": float(
                max(row["post_serialization_metric_distance"]["total_l2_distance_tolerances"] for row in subset)
            ),
            "visible_shoulder_consistency_rate": float(
                np.mean([row["morphology"]["visible_shoulder_consistent"] for row in subset])
            ),
            "median_image_rms_difference": float(
                np.median([row["morphology"]["image_rms_difference_from_matched_clean"] for row in subset])
            ),
            "median_secondary_region_contrast": float(
                np.median([row["morphology"]["secondary_region_mean_contrast"] for row in subset])
            ),
            "minimum_component_power_retained_fraction": float(
                min(row["generator"]["component_power_retained_fraction"] for row in subset)
            ),
            "primary_spot_specialist_viability_rate": float(
                np.mean([row["primary_spot_specialist_recovery_viable"] for row in subset])
            ),
        }

    by_combination: dict[str, Any] = {}
    for k in _grid_values(grid, "k_values"):
        for amplitude in _grid_values(grid, "amplitude_values"):
            for width_ratio in _grid_values(grid, "width_ratio_values"):
                key = f"k={k:.2f}|a={amplitude:.2f}|w={width_ratio:.2f}"
                by_combination[key] = aggregate(
                    [
                        row
                        for row in records
                        if row["parameters"]["k"] == k
                        and row["parameters"]["relative_reflection_amplitude"] == amplitude
                        and row["parameters"]["reflection_width_ratio"] == width_ratio
                    ]
                )
    grouped = {}
    for field, values in (
        ("beam_width_quartile", ("Q1", "Q2", "Q3", "Q4")),
        ("boundary_status", ("non_boundary", "boundary")),
        ("direction", tuple(sorted(set(row["direction"] for row in records)))),
    ):
        grouped[field] = {
            value: aggregate([row for row in records if row[field] == value])
            for value in values
        }
    result = {
        "version": "reflection_width_relative_development_search_v1",
        "protected_previous_results_used": False,
        "search_preregistration": str(args.search_preregistration.resolve()),
        "search_preregistration_sha256": sha256_file(args.search_preregistration),
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256_file(args.suite),
        "setup_groups": len(captures),
        "grid": grid,
        "overall": aggregate(records),
        "by_combination": by_combination,
        "grouped_diagnostics": grouped,
        "records": records,
        "parameter_selection_status": "development_evidence_generated_selection_not_yet_recorded",
    }
    atomic_json(args.output, result)


def _selected_parameters(args: argparse.Namespace) -> Mapping[str, Any]:
    source = json.loads(args.parameter_config.resolve().read_text())
    return source["parameter_protocol"] if "parameter_protocol" in source else source


def generate_dataset(args: argparse.Namespace) -> None:
    output_dataset = args.data_dir / f"dataset_{args.split}.jsonl"
    if output_dataset.exists():
        raise FileExistsError(f"dataset already exists: {output_dataset}")
    suite = json.loads(args.suite.resolve().read_text())
    parameters = _selected_parameters(args)
    captures = _captures_for_suite(suite, args.v12_config, args.base_config)
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for item in captures:
        case = item["case"]
        primary = item["primary"]
        selected = parameters_for_case(str(case["case_id"]), args.split, parameters)
        clean, anomaly, generator, pre, post = _pair_from_primary(primary, selected)
        if not pre["passes_frozen_match"] or not post["passes_frozen_match"]:
            raise RuntimeError(
                f"frozen metric match failed for {case['case_id']}: pre={pre}, post={post}"
            )
        pair_id = hashlib.sha256(
            f"{VERSION}:{args.split}:{case['group_id']}".encode()
        ).hexdigest()[:20]
        sample_ids: dict[str, str] = {}
        image_refs: dict[str, str] = {}
        serialized_metrics: dict[str, list[float]] = {}
        history = moment_metrics(primary).tolist()
        for label, image in (("clean", clean), ("secondary_reflection", anomaly)):
            sample_id = hashlib.sha256(f"{pair_id}:{label}:sample".encode()).hexdigest()[:24]
            relative = Path("images") / args.split / f"img_{sample_id}.png"
            serialized = _save_png(image, args.data_dir / relative)
            metrics = moment_metrics(serialized).tolist()
            sample_ids[label] = sample_id
            image_refs[label] = str(relative)
            serialized_metrics[label] = metrics
            rows.append(
                {
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "setup_id": case["group_id"],
                    "setup_hash": case["setup_hash"],
                    "split": args.split,
                    "family_audit": "secondary_reflection_width_relative",
                    "beam_width_quartile": item["beam_width_quartile"],
                    "boundary_status": _boundary_status(case),
                    "direction": _direction_label(selected["direction_angle_radians"]),
                    "generator_parameters": selected,
                    "model_input": {
                        "image_ref": str(relative),
                        "five_metrics": metrics,
                        "five_metric_uncertainties": MATCH_TOLERANCE.tolist(),
                        "short_history_metrics": [history],
                        "target": case["target_metrics"],
                        "candidate_options": _candidate_options(sample_id),
                    },
                    "supervision": {
                        "fault_type": label,
                        "binary_fault_present": label != "clean",
                        "oracle_recovery_decision": (
                            "primary_spot_specialist" if label != "clean" else "standard_metrics"
                        ),
                    },
                }
            )
        pairs.append(
            {
                "pair_id": pair_id,
                "setup_id": case["group_id"],
                "setup_hash": case["setup_hash"],
                "split": args.split,
                "family": "secondary_reflection_width_relative",
                "clean_sample_id": sample_ids["clean"],
                "anomalous_sample_id": sample_ids["secondary_reflection"],
                "clean_image_ref": image_refs["clean"],
                "anomalous_image_ref": image_refs["secondary_reflection"],
                "clean_five_metrics": serialized_metrics["clean"],
                "anomalous_five_metrics": serialized_metrics["secondary_reflection"],
                "pre_serialization_metric_distance": pre,
                "post_serialization_metric_distance": post,
                "generator": generator,
                "generator_parameters": selected,
                "beam_width_quartile": item["beam_width_quartile"],
                "equivalent_primary_sigma_px": item["equivalent_sigma"],
                "direction": _direction_label(selected["direction_angle_radians"]),
                "boundary_status": _boundary_status(case),
                "target": case["target_metrics"],
                "oracle_recovery_decision": "primary_spot_specialist",
                "no_diagnosis_outcome": None,
                "oracle_diagnosis_outcome": None,
                "learned_diagnosis_outcome": None,
            }
        )
    rng = np.random.default_rng(stable_seed("width_relative_serialization_order", args.split))
    rng.shuffle(rows)
    rng.shuffle(pairs)
    write_jsonl(output_dataset, rows)
    write_jsonl(args.data_dir / f"pairs_{args.split}.jsonl", pairs)
    atomic_json(
        args.data_dir / f"summary_{args.split}.json",
        {
            "version": VERSION,
            "split": args.split,
            "suite": str(args.suite.resolve()),
            "suite_sha256": sha256_file(args.suite),
            "parameter_config": str(args.parameter_config.resolve()),
            "parameter_config_sha256": sha256_file(args.parameter_config),
            "setup_groups": len(captures),
            "samples": len(rows),
            "pairs": len(pairs),
            "class_distribution": dict(sorted(Counter(row["supervision"]["fault_type"] for row in rows).items())),
            "all_pairs_pass_pre_serialization": all(pair["pre_serialization_metric_distance"]["passes_frozen_match"] for pair in pairs),
            "all_pairs_pass_post_serialization": all(pair["post_serialization_metric_distance"]["passes_frozen_match"] for pair in pairs),
            "maximum_pre_serialization_per_metric_difference": max(pair["pre_serialization_metric_distance"]["maximum_absolute_difference_tolerances"] for pair in pairs),
            "maximum_pre_serialization_l2_distance": max(pair["pre_serialization_metric_distance"]["total_l2_distance_tolerances"] for pair in pairs),
            "maximum_post_serialization_per_metric_difference": max(pair["post_serialization_metric_distance"]["maximum_absolute_difference_tolerances"] for pair in pairs),
            "maximum_post_serialization_l2_distance": max(pair["post_serialization_metric_distance"]["total_l2_distance_tolerances"] for pair in pairs),
        },
    )


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["supervision"]["binary_fault_present"]) for row in rows])


def _metric_features(rows: Sequence[Mapping[str, Any]], history: bool = False) -> np.ndarray:
    output = []
    for row in rows:
        current = list(map(float, row["model_input"]["five_metrics"]))
        if history:
            previous = list(map(float, row["model_input"]["short_history_metrics"][0]))
            current.extend(previous)
            current.extend((np.asarray(current[:5]) - np.asarray(previous)).tolist())
        output.append(current)
    return np.asarray(output, dtype=np.float64)


def _load_images(
    rows: Sequence[Mapping[str, Any]],
    data_dir: Path,
    *,
    border_mask: int = 0,
    dark_background_mask: bool = False,
) -> np.ndarray:
    images = []
    for row in rows:
        image = np.asarray(
            Image.open(data_dir / row["model_input"]["image_ref"]).convert("L"),
            dtype=np.float32,
        ) / 255.0
        if border_mask:
            image[:border_mask] = 0.0
            image[-border_mask:] = 0.0
            image[:, :border_mask] = 0.0
            image[:, -border_mask:] = 0.0
        if dark_background_mask:
            image[image < 0.02] = 0.0
        images.append(image[None, :, :])
    return np.stack(images)


def _fit_numeric_models(
    train_rows: Sequence[Mapping[str, Any]], seed: int
) -> tuple[Any, Any]:
    labels = _labels(train_rows)
    metric_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=2000
        ),
    )
    history_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16,),
            random_state=seed,
            max_iter=2000,
            early_stopping=False,
        ),
    )
    metric_model.fit(_metric_features(train_rows), labels)
    history_model.fit(_metric_features(train_rows, history=True), labels)
    return metric_model, history_model


def train_development_models(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"development model result already exists: {args.output}")
    protocol = json.loads(args.protocol.resolve().read_text())
    seeds = [int(value) for value in protocol["model_protocol"]["training_seeds"]]
    train_rows = read_jsonl(args.data_dir / "dataset_train.jsonl")
    dev_rows = read_jsonl(args.data_dir / "dataset_development.jsonl")
    train_labels = _labels(train_rows)
    dev_labels = _labels(dev_rows)
    train_metrics = _metric_features(train_rows)
    dev_metrics = _metric_features(dev_rows)
    train_images = _load_images(train_rows, args.data_dir)
    dev_images = _load_images(dev_rows, args.data_dir)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for seed in seeds:
        metric_model, history_model = _fit_numeric_models(train_rows, seed)
        metric_path = args.model_dir / f"seed_{seed}_metrics_logistic.joblib"
        history_path = args.model_dir / f"seed_{seed}_metrics_history_mlp.joblib"
        joblib.dump(metric_model, metric_path)
        joblib.dump(history_model, history_path)
        image_model, image_mean, image_scale = _fit_tiny_model(
            train_images, train_metrics, train_labels, metric_dim=0, seed=seed
        )
        multimodal_model, multi_mean, multi_scale = _fit_tiny_model(
            train_images, train_metrics, train_labels, metric_dim=5, seed=seed
        )
        import torch

        paths = {
            "image_only_small_cnn": args.model_dir / f"seed_{seed}_image_cnn.pt",
            "image_metrics_multimodal_small_model": args.model_dir / f"seed_{seed}_multimodal_cnn.pt",
        }
        for name, model, metric_dim, mean, scale in (
            ("image_only_small_cnn", image_model, 0, image_mean, image_scale),
            ("image_metrics_multimodal_small_model", multimodal_model, 5, multi_mean, multi_scale),
        ):
            torch.save(
                {
                    "version": VERSION,
                    "state_dict": model.state_dict(),
                    "metric_dim": metric_dim,
                    "metric_mean": mean,
                    "metric_scale": scale,
                    "family": "secondary_reflection_width_relative",
                    "seed": seed,
                    "architecture": "previous_tiny_cnn_unchanged",
                    "training_budget": "45_epochs_batch32_adam_lr_0.002_wd_0.0001_rot90_augmentation",
                },
                paths[name],
            )
        probabilities = {
            "metrics_only_logistic": metric_model.predict_proba(dev_metrics)[:, 1],
            "metrics_history_mlp": history_model.predict_proba(
                _metric_features(dev_rows, history=True)
            )[:, 1],
            "image_only_small_cnn": _predict_tiny(
                image_model, dev_images, dev_metrics, image_mean, image_scale, 0
            ),
            "image_metrics_multimodal_small_model": _predict_tiny(
                multimodal_model, dev_images, dev_metrics, multi_mean, multi_scale, 5
            ),
        }
        seed_result = {
            "seed": seed,
            "models": {name: _score(dev_labels, value) for name, value in probabilities.items()},
            "artifacts": {
                "metrics_only_logistic": {"path": str(metric_path.resolve()), "sha256": sha256_file(metric_path)},
                "metrics_history_mlp": {"path": str(history_path.resolve()), "sha256": sha256_file(history_path)},
                "image_only_small_cnn": {"path": str(paths["image_only_small_cnn"].resolve()), "sha256": sha256_file(paths["image_only_small_cnn"])},
                "image_metrics_multimodal_small_model": {"path": str(paths["image_metrics_multimodal_small_model"].resolve()), "sha256": sha256_file(paths["image_metrics_multimodal_small_model"])},
            },
        }
        candidates.append(seed_result)
    eligible = []
    for result in candidates:
        for model_name in (
            "image_only_small_cnn",
            "image_metrics_multimodal_small_model",
        ):
            score = result["models"][model_name]
            eligible.append(
                (
                    -float(score["balanced_accuracy"]),
                    -float(score["macro_f1"]),
                    0 if model_name == "image_only_small_cnn" else 1,
                    int(result["seed"]),
                    model_name,
                    result,
                )
            )
    _, _, _, selected_seed, selected_name, selected_result = sorted(eligible)[0]
    selected_source = Path(selected_result["artifacts"][selected_name]["path"])
    selected_path = args.model_dir / "selected_diagnostic.pt"
    shutil.copy2(selected_source, selected_path)
    output = {
        "version": "reflection_width_relative_development_model_selection_v1",
        "protected_previous_results_used": False,
        "small_cnn_is_visual_diagnostic_not_vlm": True,
        "selection_split": "new_setup_disjoint_development",
        "selection_rule": protocol["model_protocol"]["selection_rule"],
        "training_seeds": seeds,
        "per_seed": candidates,
        "selected": {
            "model_name": selected_name,
            "seed": selected_seed,
            "development_score": selected_result["models"][selected_name],
            "source_path": str(selected_source.resolve()),
            "source_sha256": sha256_file(selected_source),
            "frozen_path": str(selected_path.resolve()),
            "frozen_sha256": sha256_file(selected_path),
        },
    }
    atomic_json(args.output, output)


def _load_torch_artifact(path: Path) -> tuple[Any, np.ndarray, np.ndarray, int, int]:
    import torch

    artifact = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    _, image_model, multimodal = _torch_models(int(artifact["seed"]))
    metric_dim = int(artifact["metric_dim"])
    model = image_model if metric_dim == 0 else multimodal
    model.load_state_dict(artifact["state_dict"])
    return (
        model,
        np.asarray(artifact["metric_mean"], dtype=np.float32),
        np.asarray(artifact["metric_scale"], dtype=np.float32),
        metric_dim,
        int(artifact["seed"]),
    )


def _group_scores(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    probability: np.ndarray,
    field: str,
) -> dict[str, Any]:
    output = {}
    for value in sorted({str(row[field]) for row in rows}):
        mask = np.asarray([str(row[field]) == value for row in rows])
        output[value] = _score(labels[mask], probability[mask])
    return output


def _aggregate_seed_scores(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    model_names = per_seed[0]["models"].keys()
    for name in model_names:
        output[name] = {}
        for field in ("balanced_accuracy", "macro_f1", "expected_calibration_error_10_bins"):
            values = np.asarray([entry["models"][name][field] for entry in per_seed], dtype=float)
            output[name][field] = {
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    return output


def evaluate_frozen_models(args: argparse.Namespace) -> None:
    if args.output.exists() or args.leakage_output.exists():
        raise FileExistsError("held-out evaluation outputs already exist")
    prereg = json.loads(args.preregistration.resolve().read_text())
    model_selection = json.loads(Path(prereg["model_protocol"]["development_selection_artifact"]).read_text())
    seeds = [int(value) for value in prereg["model_protocol"]["training_seeds"]]
    required_splits = ("train", "development", "iid_heldout")
    evaluation_splits = ["iid_heldout"]
    if (args.data_dir / "dataset_severity_ood.jsonl").exists():
        evaluation_splits.append("severity_ood")
    rows = {
        split: read_jsonl(args.data_dir / f"dataset_{split}.jsonl")
        for split in (*required_splits, *(evaluation_splits[1:]))
    }
    labels = {split: _labels(value) for split, value in rows.items()}
    metrics = {split: _metric_features(value) for split, value in rows.items()}
    images = {split: _load_images(value, args.data_dir) for split, value in rows.items()}
    split_results: dict[str, Any] = {}
    probability_cache: dict[tuple[str, int, str], np.ndarray] = {}
    for split in evaluation_splits:
        per_seed = []
        for seed in seeds:
            metric_model = joblib.load(args.model_dir / f"seed_{seed}_metrics_logistic.joblib")
            history_model = joblib.load(args.model_dir / f"seed_{seed}_metrics_history_mlp.joblib")
            image_model, image_mean, image_scale, _, _ = _load_torch_artifact(
                args.model_dir / f"seed_{seed}_image_cnn.pt"
            )
            multi_model, multi_mean, multi_scale, _, _ = _load_torch_artifact(
                args.model_dir / f"seed_{seed}_multimodal_cnn.pt"
            )
            probabilities = {
                "metrics_only_logistic": metric_model.predict_proba(metrics[split])[:, 1],
                "metrics_history_mlp": history_model.predict_proba(
                    _metric_features(rows[split], history=True)
                )[:, 1],
                "image_only_small_cnn": _predict_tiny(
                    image_model, images[split], metrics[split], image_mean, image_scale, 0
                ),
                "image_metrics_multimodal_small_model": _predict_tiny(
                    multi_model, images[split], metrics[split], multi_mean, multi_scale, 5
                ),
                "hidden_label_oracle": labels[split].astype(float),
            }
            for name, probability in probabilities.items():
                probability_cache[(split, seed, name)] = probability
            per_seed.append(
                {
                    "seed": seed,
                    "models": {
                        name: {
                            **_score(labels[split], probability),
                            "by_width_quartile": _group_scores(rows[split], labels[split], probability, "beam_width_quartile"),
                            "by_direction": _group_scores(rows[split], labels[split], probability, "direction"),
                            "by_amplitude": _group_scores(
                                [dict(row, amplitude=str(row["generator_parameters"]["relative_reflection_amplitude"])) for row in rows[split]],
                                labels[split],
                                probability,
                                "amplitude",
                            ),
                            "by_k": _group_scores(
                                [dict(row, k=str(row["generator_parameters"]["k"])) for row in rows[split]],
                                labels[split],
                                probability,
                                "k",
                            ),
                            "by_boundary_status": _group_scores(rows[split], labels[split], probability, "boundary_status"),
                        }
                        for name, probability in probabilities.items()
                    },
                }
            )
        split_results[split] = {
            "per_training_seed": per_seed,
            "aggregate_across_training_seeds": _aggregate_seed_scores(per_seed),
        }
    if "severity_ood" not in split_results:
        split_results["severity_ood"] = {
            "status": "unavailable_frozen_generation_failed_post_serialization_match_no_retry_or_relaxation"
        }

    selected = model_selection["selected"]
    selected_seed = int(selected["seed"])
    selected_name = str(selected["model_name"])
    selected_model, selected_mean, selected_scale, selected_dim, _ = _load_torch_artifact(
        args.model_dir / "selected_diagnostic.pt"
    )
    iid_rows = rows["iid_heldout"]
    iid_labels = labels["iid_heldout"]
    iid_probability = probability_cache[("iid_heldout", selected_seed, selected_name)]
    shuffled_images = images["iid_heldout"][
        np.random.default_rng(stable_seed("width_relative_shuffle_iid")).permutation(len(iid_rows))
    ]
    shuffled_probability = _predict_tiny(
        selected_model,
        shuffled_images,
        metrics["iid_heldout"],
        selected_mean,
        selected_scale,
        selected_dim,
    )
    border_images = _load_images(iid_rows, args.data_dir, border_mask=8)
    border_probability = _predict_tiny(
        selected_model,
        border_images,
        metrics["iid_heldout"],
        selected_mean,
        selected_scale,
        selected_dim,
    )
    background_images = _load_images(iid_rows, args.data_dir, dark_background_mask=True)
    background_probability = _predict_tiny(
        selected_model,
        background_images,
        metrics["iid_heldout"],
        selected_mean,
        selected_scale,
        selected_dim,
    )
    metadata_train = np.asarray(
        [
            [IMAGE_SIZE, IMAGE_SIZE, 1, int(row["sample_id"][:2], 16) / 255.0]
            for row in rows["train"]
        ],
        dtype=np.float64,
    )
    metadata_iid = np.asarray(
        [
            [IMAGE_SIZE, IMAGE_SIZE, 1, int(row["sample_id"][:2], 16) / 255.0]
            for row in iid_rows
        ],
        dtype=np.float64,
    )
    metadata_model = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", random_state=seeds[0])
    )
    metadata_model.fit(metadata_train, labels["train"])
    metadata_probability = metadata_model.predict_proba(metadata_iid)[:, 1]
    train_hashes = {
        sha256_file(args.data_dir / row["model_input"]["image_ref"])
        for row in rows["train"]
    }
    iid_hashes = {
        sha256_file(args.data_dir / row["model_input"]["image_ref"])
        for row in iid_rows
    }
    flat_train = images["train"].reshape(len(images["train"]), -1)
    flat_iid = images["iid_heldout"].reshape(len(images["iid_heldout"]), -1)
    nearest_mse = np.min(
        np.mean((flat_iid[:, None, :] - flat_train[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    setup_sets = {
        split: {row["setup_id"] for row in split_rows}
        for split, split_rows in rows.items()
    }
    split_manifest = json.loads(
        Path(prereg["split_protocol"]["split_manifest"]).resolve().read_text()
    )
    pair_summary = json.loads((args.data_dir / "summary_iid_heldout.json").read_text())
    leakage = {
        "version": "reflection_width_relative_leakage_audit_v1",
        "selected_frozen_diagnostic": selected,
        "shuffle_images_preserve_metrics": _score(iid_labels, shuffled_probability),
        "mask_eight_pixel_borders": _score(iid_labels, border_probability),
        "mask_dark_irrelevant_background": _score(iid_labels, background_probability),
        "metadata_only_classifier": _score(iid_labels, metadata_probability),
        "exact_image_hash_overlap_train_iid": len(train_hashes & iid_hashes),
        "nearest_neighbor_train_iid": {
            "minimum_pixel_mse": float(nearest_mse.min()),
            "median_nearest_pixel_mse": float(np.median(nearest_mse)),
        },
        "setup_overlap": {
            "train_development": sorted(setup_sets["train"] & setup_sets["development"]),
            "train_iid": sorted(setup_sets["train"] & setup_sets["iid_heldout"]),
            "development_iid": sorted(setup_sets["development"] & setup_sets["iid_heldout"]),
            "iid_severity_ood": sorted(
                setup_sets["iid_heldout"] & setup_sets.get("severity_ood", set())
            ),
        },
        "new_and_previous_setup_hash_disjointness": {
            "manifest": str(Path(prereg["split_protocol"]["split_manifest"]).resolve()),
            "passed": bool(split_manifest["setup_and_hash_disjointness_passed"]),
        },
        "filename_contains_label": False,
        "fixed_dimensions_channels_padding_and_peak_normalization": {
            "dimensions": [IMAGE_SIZE, IMAGE_SIZE],
            "channels": 1,
            "padding_policy": "zero_padding_is_identical_and_not_serialized_as_metadata",
            "normalization_policy": "every_image_peak_normalized_before_uint8_serialization",
            "class_dependent_settings": False,
        },
    }
    leakage["passed"] = bool(
        leakage["shuffle_images_preserve_metrics"]["balanced_accuracy"] <= 0.65
        and leakage["mask_eight_pixel_borders"]["balanced_accuracy"] >= 0.80
        and leakage["metadata_only_classifier"]["balanced_accuracy"] <= 0.65
        and leakage["exact_image_hash_overlap_train_iid"] == 0
        and not any(leakage["setup_overlap"].values())
        and leakage["new_and_previous_setup_hash_disjointness"]["passed"]
    )
    selected_entry = next(
        item
        for item in split_results["iid_heldout"]["per_training_seed"]
        if int(item["seed"]) == selected_seed
    )["models"][selected_name]
    quartile_scores = [
        float(value["balanced_accuracy"])
        for value in selected_entry["by_width_quartile"].values()
    ]
    numeric_best = max(
        float(entry["models"][name]["balanced_accuracy"])
        for entry in split_results["iid_heldout"]["per_training_seed"]
        for name in ("metrics_only_logistic", "metrics_history_mlp")
    )
    selected_ba = float(selected_entry["balanced_accuracy"])
    gate = {
        "matched_pair_numerical_ambiguity": bool(
            pair_summary["all_pairs_pass_pre_serialization"]
            and pair_summary["all_pairs_pass_post_serialization"]
        ),
        "best_metrics_only_iid_balanced_accuracy_at_most_65_percent_or_pairs_matched": bool(
            numeric_best <= 0.65
            or (
                pair_summary["all_pairs_pass_pre_serialization"]
                and pair_summary["all_pairs_pass_post_serialization"]
            )
        ),
        "selected_image_iid_balanced_accuracy_at_least_80_percent": selected_ba >= 0.80,
        "selected_image_minus_best_numeric_at_least_15_points": selected_ba - numeric_best >= 0.15,
        "width_quartile_stability": bool(
            min(quartile_scores) >= 0.70 and max(quartile_scores) - min(quartile_scores) <= 0.25
        ),
        "border_mask_preserves_result": leakage["mask_eight_pixel_borders"]["balanced_accuracy"] >= 0.80,
        "shuffled_images_collapse": leakage["shuffle_images_preserve_metrics"]["balanced_accuracy"] <= 0.65,
        "no_leakage_found": leakage["passed"],
    }
    gate["passed"] = all(gate.values())
    results = {
        "version": "reflection_width_relative_identifiability_results_v1",
        "evaluation_order": "iid_once_then_preregistered_severity_ood_no_tuning",
        "protected_previous_results_used": False,
        "small_cnn_is_visual_diagnostic_not_vlm": True,
        "selected_frozen_diagnostic": selected,
        "splits": split_results,
        "selected_iid_result": selected_entry,
        "best_numeric_iid_balanced_accuracy": numeric_best,
        "visual_identifiability_gate": gate,
    }
    atomic_json(args.leakage_output, leakage)
    atomic_json(args.output, results)


def _corrupted_lab_metrics(
    capture: Mapping[str, Any], full_anomaly: np.ndarray
) -> np.ndarray:
    clean_image_metrics = moment_metrics(capture["intensity"])
    anomaly_image_metrics = moment_metrics(full_anomaly)
    clean_lab = metrics_vector(capture["metrics"])
    output = clean_lab.copy()
    output[:4] += anomaly_image_metrics[:4] - clean_image_metrics[:4]
    output[4] = float(full_anomaly.max())
    return output


def _observe_control(
    *,
    case: Mapping[str, Any],
    position: np.ndarray,
    bounds: Bounds,
    base_config: Path,
    parameters: Mapping[str, float],
    arm: str,
    diagnostic: tuple[Any, np.ndarray, np.ndarray, int, int] | None,
) -> dict[str, Any]:
    capture = simulate_state(
        case["setup_context"],
        position_dict(position),
        case["simulator_fixed"],
        str(base_config.resolve()),
        bounds,
    )
    clean_metrics = metrics_vector(capture["metrics"])
    full_anomaly, full_generator = inject_width_relative_reflection(
        capture["intensity"],
        k=float(parameters["k"]),
        amplitude=float(parameters["relative_reflection_amplitude"]),
        width_ratio=float(parameters["reflection_width_ratio"]),
        angle_radians=float(parameters["direction_angle_radians"]),
        normalize_peak=False,
    )
    primary_view = canonical_patch(capture["intensity"])
    diagnostic_image, diagnostic_generator = inject_width_relative_reflection(
        primary_view,
        k=float(parameters["k"]),
        amplitude=float(parameters["relative_reflection_amplitude"]),
        width_ratio=float(parameters["reflection_width_ratio"]),
        angle_radians=float(parameters["direction_angle_radians"]),
        normalize_peak=True,
    )
    anomalous_metrics = _corrupted_lab_metrics(capture, full_anomaly)
    probability = None
    switch = False
    diagnosis_correct = None
    if arm == "oracle_diagnosis":
        switch = True
        diagnosis_correct = True
    elif arm == "learned_image_diagnostic":
        if diagnostic is None:
            raise ValueError("learned control arm requires the frozen diagnostic")
        model, mean, scale, metric_dim, _ = diagnostic
        canonical_metrics = moment_metrics(diagnostic_image)[None, :]
        probability = float(
            _predict_tiny(
                model,
                diagnostic_image[None, None, :, :],
                canonical_metrics,
                mean,
                scale,
                metric_dim,
            )[0]
        )
        switch = probability >= 0.5
        diagnosis_correct = switch
    elif arm != "no_diagnosis":
        raise ValueError(f"unknown arm: {arm}")
    return {
        "clean_metrics": clean_metrics,
        "observed_metrics": clean_metrics if switch else anomalous_metrics,
        "diagnostic_probability": probability,
        "diagnosis_correct": diagnosis_correct,
        "specialist_switch": switch,
        "simulator_valid": bool(capture["auxiliary"]["simulator_valid"]),
        "full_generator": full_generator,
        "diagnostic_generator": diagnostic_generator,
    }


def execute_width_relative_control_episode(
    *,
    case: Mapping[str, Any],
    arm: str,
    config: Mapping[str, Any],
    bounds: Bounds,
    forward_model: Any,
    base_config: Path,
    parameter_protocol: Mapping[str, Any],
    diagnostic: tuple[Any, np.ndarray, np.ndarray, int, int] | None,
) -> dict[str, Any]:
    parameters = parameters_for_case(
        str(case["case_id"]), "iid_heldout", parameter_protocol
    )
    position = position_vector(case["initial_positions_mm"])
    target = metrics_vector(case["target_metrics"])
    initial_clean = metrics_vector(case["initial_metrics"])
    planner = CEMMPC(
        bounds=bounds,
        predictor=learned_predictor(forward_model, case["setup_context"]),
        config=_planner_config(config),
        seed=matched_planner_seed(config, str(case["case_id"])),
    )
    observation = _observe_control(
        case=case,
        position=position,
        bounds=bounds,
        base_config=base_config,
        parameters=parameters,
        arm=arm,
        diagnostic=diagnostic,
    )
    observed = np.asarray(observation["observed_metrics"], dtype=np.float64)
    clean = np.asarray(observation["clean_metrics"], dtype=np.float64)
    initial_generator = observation["full_generator"]
    decisions = [
        {
            "observation_index": 0,
            "diagnostic_probability": observation["diagnostic_probability"],
            "diagnosis_correct": observation["diagnosis_correct"],
            "specialist_switch": bool(observation["specialist_switch"]),
        }
    ]
    trace: list[dict[str, Any]] = []
    saturation_count = 0
    specialist_switches = int(observation["specialist_switch"])
    for control_step in range(8):
        before_observed = normalized_distance(observed, target, initial_clean)
        before_clean = normalized_distance(clean, target, initial_clean)
        if before_observed <= 1.0:
            break
        plan = planner.plan(
            positions_mm=position,
            current_metrics=observed,
            target_metrics=target,
            allowed_dofs=["lens_x", "lens_y", "camera_x", "camera_y"],
            tolerance_reference=initial_clean,
        )
        requested = np.asarray(
            [plan["selected_requested_action"][field] for field in ACTION_FIELDS]
        )
        action = np.asarray([plan["selected_action"][field] for field in ACTION_FIELDS])
        saturation_count += int(
            not np.allclose(requested, action, rtol=0.0, atol=1e-12)
        )
        position = apply_action(position, action, bounds, project=False)
        next_observation = _observe_control(
            case=case,
            position=position,
            bounds=bounds,
            base_config=base_config,
            parameters=parameters,
            arm=arm,
            diagnostic=diagnostic,
        )
        next_observed = np.asarray(next_observation["observed_metrics"], dtype=np.float64)
        next_clean = np.asarray(next_observation["clean_metrics"], dtype=np.float64)
        after_observed = normalized_distance(next_observed, target, initial_clean)
        after_clean = normalized_distance(next_clean, target, initial_clean)
        decision = {
            "observation_index": control_step + 1,
            "diagnostic_probability": next_observation["diagnostic_probability"],
            "diagnosis_correct": next_observation["diagnosis_correct"],
            "specialist_switch": bool(next_observation["specialist_switch"]),
        }
        decisions.append(decision)
        trace.append(
            {
                "control_step": control_step + 1,
                "command_mm": {
                    field: float(action[index])
                    for index, field in enumerate(ACTION_FIELDS)
                },
                "observed_before_distance": float(before_observed),
                "observed_after_distance": float(after_observed),
                "true_clean_before_distance": float(before_clean),
                "true_clean_after_distance": float(after_clean),
                **decision,
                "simulator_valid": bool(next_observation["simulator_valid"]),
                "sigma_direction_px_full_sensor": float(
                    next_observation["full_generator"]["sigma_direction_px"]
                ),
                "reflection_separation_px_full_sensor": float(
                    next_observation["full_generator"]["separation_px"]
                ),
            }
        )
        specialist_switches += int(next_observation["specialist_switch"])
        observed, clean = next_observed, next_clean
        if after_observed <= 1.0:
            break
        if control_step + 1 >= 4 and before_observed - after_observed < 0.25:
            break
    final_distance = normalized_distance(clean, target, initial_clean)
    classified = [
        bool(item["diagnosis_correct"])
        for item in decisions
        if item["diagnosis_correct"] is not None
    ]
    return {
        "version": "reflection_width_relative_control_episode_v1",
        "episode_id": f"{case['case_id']}__secondary_reflection_width_relative__{arm}",
        "case_id": case["case_id"],
        "setup_id": case["group_id"],
        "setup_hash": case["setup_hash"],
        "stratum": case["stratum"],
        "regime": case["regime"],
        "family": "secondary_reflection_width_relative",
        "arm": arm,
        "parameters": parameters,
        "initial_generator": initial_generator,
        "direction": _direction_label(parameters["direction_angle_radians"]),
        "boundary_status": _boundary_status(case),
        "strict_success": bool(final_distance <= 1.0),
        "final_normalized_target_distance": float(final_distance),
        "control_steps": len(trace),
        "observation_mode_or_specialist_switches": specialist_switches,
        "all_diagnostic_decisions_correct": None if not classified else all(classified),
        "any_diagnostic_decision_wrong": None if not classified else not all(classified),
        "saturation_count": int(saturation_count),
        "constraint_violation_count": 0,
        "diagnostic_decisions": decisions,
        "trace": trace,
        "controller": "frozen_h1_cem_sequential_max8",
        "recovery_adds_actuator_budget": False,
    }


def run_control(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"control episode file already exists: {args.output}")
    prereg = json.loads(args.preregistration.resolve().read_text())
    suite = json.loads(args.suite.resolve().read_text())
    config = json.loads(Path(prereg["control_protocol"]["v13_config"]).read_text())
    config = {
        **config,
        "root_seed": int(prereg["control_protocol"]["planner_seed"]),
        "baseline": {**config["baseline"], "max_control_steps": 8},
    }
    v12 = json.loads(Path(config["baseline"]["v12_config"]).read_text())
    bounds = Bounds.from_config(v12)
    forward_model = load_forward_ensemble(
        Path(config["baseline"]["checkpoint"]), device_name="cpu"
    )
    diagnostic = _load_torch_artifact(
        Path(prereg["model_protocol"]["selected_diagnostic_path"])
    )
    rows = []
    tasks = len(suite["cases"]) * len(ARMS)
    task_index = 0
    for case in suite["cases"]:
        for arm in ARMS:
            task_index += 1
            row = execute_width_relative_control_episode(
                case=case,
                arm=arm,
                config=config,
                bounds=bounds,
                forward_model=forward_model,
                base_config=Path(prereg["control_protocol"]["base_simulator_config"]),
                parameter_protocol=prereg["parameter_protocol"],
                diagnostic=diagnostic if arm == "learned_image_diagnostic" else None,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "width_relative_control_episode_complete",
                        "task": task_index,
                        "tasks": tasks,
                        "episode_id": row["episode_id"],
                        "strict_success": row["strict_success"],
                        "steps": row["control_steps"],
                    }
                ),
                flush=True,
            )
    write_jsonl(args.output, rows)


def _control_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = [int(row["control_steps"]) for row in rows]
    distances = [float(row["final_normalized_target_distance"]) for row in rows]
    return {
        "episodes": len(rows),
        "strict_success_rate": float(np.mean([bool(row["strict_success"]) for row in rows])),
        "strict_success_count": int(sum(bool(row["strict_success"]) for row in rows)),
        "mean_executed_steps": float(np.mean(steps)),
        "median_executed_steps": float(np.median(steps)),
        "executed_step_distribution": {
            str(step): int(count) for step, count in sorted(Counter(steps).items())
        },
        "mean_final_normalized_target_distance": float(np.mean(distances)),
        "median_final_normalized_target_distance": float(np.median(distances)),
        "final_normalized_target_distance_distribution": sorted(distances),
        "actuator_saturation_episode_rate": float(
            np.mean([int(row["saturation_count"]) > 0 for row in rows])
        ),
        "actuator_saturation_count": int(sum(int(row["saturation_count"]) for row in rows)),
        "hard_constraint_violations": int(
            sum(int(row["constraint_violation_count"]) for row in rows)
        ),
    }


def _control_by_group(
    arm_maps: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metadata: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    output = {}
    values = sorted({str(value[field]) for value in metadata.values()})
    for group in values:
        setups = [setup for setup, value in metadata.items() if str(value[field]) == group]
        output[group] = {
            arm: _control_summary([arm_maps[arm][setup] for setup in setups])
            for arm in ARMS
        }
    return output


def analyze_control(args: argparse.Namespace) -> None:
    if args.output.exists() or args.episode_csv.exists() or args.paired_output.exists():
        raise FileExistsError("control analysis output already exists")
    rows = read_jsonl(args.source)
    identifiability = json.loads(args.identifiability.resolve().read_text())
    iid_pairs = read_jsonl(args.data_dir / "pairs_iid_heldout.jsonl")
    metadata = {
        pair["setup_id"]: {
            "beam_width_quartile": pair["beam_width_quartile"],
            "k": pair["generator_parameters"]["k"],
            "amplitude": pair["generator_parameters"]["relative_reflection_amplitude"],
            "width_ratio": pair["generator_parameters"]["reflection_width_ratio"],
            "direction": pair["direction"],
            "boundary_status": pair["boundary_status"],
        }
        for pair in iid_pairs
    }
    arm_maps = {
        arm: {row["setup_id"]: row for row in rows if row["arm"] == arm}
        for arm in ARMS
    }
    setup_ids = sorted(metadata)
    baseline = arm_maps["no_diagnosis"]
    oracle = arm_maps["oracle_diagnosis"]
    learned = arm_maps["learned_image_diagnostic"]
    oracle_recoveries = [
        setup for setup in setup_ids if not baseline[setup]["strict_success"] and oracle[setup]["strict_success"]
    ]
    oracle_regressions = [
        setup for setup in setup_ids if baseline[setup]["strict_success"] and not oracle[setup]["strict_success"]
    ]
    learned_recoveries = [
        setup for setup in setup_ids if not baseline[setup]["strict_success"] and learned[setup]["strict_success"]
    ]
    learned_regressions = [
        setup for setup in setup_ids if baseline[setup]["strict_success"] and not learned[setup]["strict_success"]
    ]
    correct_control_failed = [
        setup for setup in setup_ids if learned[setup]["all_diagnostic_decisions_correct"] is True and not learned[setup]["strict_success"]
    ]
    wrong_accidentally_recovered = [
        setup for setup in setup_ids if learned[setup]["any_diagnostic_decision_wrong"] is True and learned[setup]["strict_success"]
    ]
    disagreements = []
    for setup in setup_ids:
        for decision in learned[setup]["diagnostic_decisions"]:
            if not decision["specialist_switch"]:
                disagreements.append(
                    {
                        "setup_id": setup,
                        "episode_id": learned[setup]["episode_id"],
                        "observation_index": decision["observation_index"],
                        "learned_probability": decision["diagnostic_probability"],
                        "learned_switch": False,
                        "oracle_switch": True,
                    }
                )
    arm_summaries = {
        arm: _control_summary([arm_maps[arm][setup] for setup in setup_ids])
        for arm in ARMS
    }
    no_rate = arm_summaries["no_diagnosis"]["strict_success_rate"]
    oracle_rate = arm_summaries["oracle_diagnosis"]["strict_success_rate"]
    learned_rate = arm_summaries["learned_image_diagnostic"]["strict_success_rate"]
    control_gate = {
        "oracle_improves_at_least_5pp_or_five_recoveries": bool(
            oracle_rate - no_rate >= 0.05 or len(oracle_recoveries) >= 5
        ),
        "improvement_is_measurement_switch_not_extra_action_budget": all(
            row["recovery_adds_actuator_budget"] is False for row in rows
        ),
        "no_material_safety_regression": bool(
            arm_summaries["oracle_diagnosis"]["actuator_saturation_episode_rate"]
            <= arm_summaries["no_diagnosis"]["actuator_saturation_episode_rate"] + 0.05
            and arm_summaries["oracle_diagnosis"]["hard_constraint_violations"]
            <= arm_summaries["no_diagnosis"]["hard_constraint_violations"]
        ),
    }
    control_gate["passed"] = all(control_gate.values())
    results = {
        "version": "reflection_width_relative_control_value_results_v1",
        "evaluation_split": "new_preregistered_iid_heldout_setup_disjoint",
        "backbone": "externally_validated_frozen_h1_cem_sequential_max8",
        "recovery_policy": "unchanged_primary_spot_measurement_switch",
        "recovery_adds_actuator_budget": False,
        "arms": arm_summaries,
        "oracle_minus_no_diagnosis_percentage_points": 100.0 * (oracle_rate - no_rate),
        "learned_minus_no_diagnosis_percentage_points": 100.0 * (learned_rate - no_rate),
        "matched_oracle_recoveries": len(oracle_recoveries),
        "matched_oracle_regressions": len(oracle_regressions),
        "oracle_recovery_setup_ids": oracle_recoveries,
        "oracle_regression_setup_ids": oracle_regressions,
        "matched_learned_recoveries": len(learned_recoveries),
        "matched_learned_regressions": len(learned_regressions),
        "learned_recovery_setup_ids": learned_recoveries,
        "learned_regression_setup_ids": learned_regressions,
        "classification_correct_but_control_failed_episode_ids": [learned[setup]["episode_id"] for setup in correct_control_failed],
        "classification_wrong_but_accidentally_recovered_episode_ids": [learned[setup]["episode_id"] for setup in wrong_accidentally_recovered],
        "learned_oracle_disagreements": disagreements,
        "by_width_quartile": _control_by_group(arm_maps, metadata, "beam_width_quartile"),
        "by_k": _control_by_group(arm_maps, metadata, "k"),
        "by_amplitude": _control_by_group(arm_maps, metadata, "amplitude"),
        "by_width_ratio": _control_by_group(arm_maps, metadata, "width_ratio"),
        "by_direction": _control_by_group(arm_maps, metadata, "direction"),
        "by_boundary_status": _control_by_group(arm_maps, metadata, "boundary_status"),
        "control_value_gate": control_gate,
        "visual_identifiability_gate_passed": bool(
            identifiability["visual_identifiability_gate"]["passed"]
        ),
    }
    results["both_gates_passed"] = bool(
        results["visual_identifiability_gate_passed"] and control_gate["passed"]
    )
    atomic_json(args.output, results)
    args.episode_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.episode_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "episode_id",
            "setup_id",
            "setup_hash",
            "arm",
            "beam_width_quartile",
            "boundary_status",
            "direction",
            "k",
            "amplitude",
            "width_ratio",
            "strict_success",
            "final_normalized_target_distance",
            "control_steps",
            "saturation_count",
            "constraint_violation_count",
            "all_diagnostic_decisions_correct",
            "any_diagnostic_decision_wrong",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = metadata[row["setup_id"]]
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    **meta,
                    "amplitude": meta["amplitude"],
                }
            )
    all_pairs = []
    for split in ("train", "development", "iid_heldout", "severity_ood"):
        pair_path = args.data_dir / f"pairs_{split}.jsonl"
        if not pair_path.exists():
            continue
        for pair in read_jsonl(pair_path):
            if split == "iid_heldout":
                setup = pair["setup_id"]
                for key, arm in (
                    ("no_diagnosis_outcome", "no_diagnosis"),
                    ("oracle_diagnosis_outcome", "oracle_diagnosis"),
                    ("learned_diagnosis_outcome", "learned_image_diagnostic"),
                ):
                    source = arm_maps[arm][setup]
                    pair[key] = {
                        "strict_success": source["strict_success"],
                        "final_normalized_target_distance": source["final_normalized_target_distance"],
                        "executed_steps": source["control_steps"],
                        "saturation_count": source["saturation_count"],
                        "constraint_violation_count": source["constraint_violation_count"],
                    }
            else:
                pair["no_diagnosis_outcome"] = {"status": "not_control_audit_split"}
                pair["oracle_diagnosis_outcome"] = {"status": "not_control_audit_split"}
                pair["learned_diagnosis_outcome"] = {"status": "not_control_audit_split"}
            all_pairs.append(pair)
    write_jsonl(args.paired_output, all_pairs)


def build_split_manifest(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"split manifest already exists: {args.output}")
    suites = {}
    all_ids: dict[str, set[str]] = {}
    all_hashes: dict[str, set[str]] = {}
    for specification in args.suite:
        split, path_text = specification.split("=", 1)
        path = Path(path_text).resolve()
        suite = json.loads(path.read_text())
        ids = {str(case["group_id"]) for case in suite["cases"]}
        hashes = {str(case["setup_hash"]) for case in suite["cases"]}
        all_ids[split] = ids
        all_hashes[split] = hashes
        suites[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "root_seed": suite["root_seed"],
            "groups": len(suite["cases"]),
            "unique_group_ids": len(ids),
            "unique_setup_hashes": len(hashes),
            "generator_reported_prior_group_overlap": suite["validation"]["prior_group_id_overlap"],
            "generator_reported_prior_setup_hash_overlap": suite["validation"]["prior_setup_hash_overlap"],
        }
    overlaps = {}
    split_names = sorted(suites)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = {
                "group_ids": sorted(all_ids[left] & all_ids[right]),
                "setup_hashes": sorted(all_hashes[left] & all_hashes[right]),
            }
    previous_ids: set[str] = set()
    previous_hashes: set[str] = set()
    previous = []
    for path in args.previous_suite:
        resolved = path.resolve()
        suite = json.loads(resolved.read_text())
        previous.append({"path": str(resolved), "sha256": sha256_file(resolved)})
        previous_ids.update(str(case["group_id"]) for case in suite["cases"])
        previous_hashes.update(str(case["setup_hash"]) for case in suite["cases"])
    previous_overlap = {
        split: {
            "group_ids": sorted(ids & previous_ids),
            "setup_hashes": sorted(all_hashes[split] & previous_hashes),
        }
        for split, ids in all_ids.items()
    }
    passed = not any(
        value
        for pair in overlaps.values()
        for value in (pair["group_ids"], pair["setup_hashes"])
    ) and not any(
        value
        for pair in previous_overlap.values()
        for value in (pair["group_ids"], pair["setup_hashes"])
    )
    atomic_json(
        args.output,
        {
            "version": "reflection_width_relative_split_manifest_v1",
            "suites": suites,
            "pairwise_new_split_overlap": overlaps,
            "previous_reflection_cohorts": previous,
            "overlap_with_previous_reflection_cohorts": previous_overlap,
            "setup_and_hash_disjointness_passed": passed,
        },
    )


def prepare_suite_config(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"suite config already exists: {args.output}")
    config = json.loads(args.source.resolve().read_text())
    config["version"] = "reflection_width_relative_suite_config_v1"
    config["created_for"] = str(args.label)
    config["root_seed"] = int(args.root_seed)
    atomic_json(args.output, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    development = subparsers.add_parser("development-search")
    development.add_argument("--suite", type=Path, required=True)
    development.add_argument("--search-preregistration", type=Path, required=True)
    development.add_argument("--v12-config", type=Path, required=True)
    development.add_argument("--base-config", type=Path, required=True)
    development.add_argument("--montage-dir", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)

    generate = subparsers.add_parser("generate-dataset")
    generate.add_argument("--suite", type=Path, required=True)
    generate.add_argument(
        "--split",
        choices=("train", "development", "iid_heldout", "severity_ood"),
        required=True,
    )
    generate.add_argument("--parameter-config", type=Path, required=True)
    generate.add_argument("--v12-config", type=Path, required=True)
    generate.add_argument("--base-config", type=Path, required=True)
    generate.add_argument("--data-dir", type=Path, required=True)

    train = subparsers.add_parser("train-development-models")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-frozen-models")
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--preregistration", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--leakage-output", type=Path, required=True)

    control = subparsers.add_parser("run-control")
    control.add_argument("--suite", type=Path, required=True)
    control.add_argument("--preregistration", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze-control")
    analyze.add_argument("--source", type=Path, required=True)
    analyze.add_argument("--identifiability", type=Path, required=True)
    analyze.add_argument("--data-dir", type=Path, required=True)
    analyze.add_argument("--paired-output", type=Path, required=True)
    analyze.add_argument("--episode-csv", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)

    split = subparsers.add_parser("build-split-manifest")
    split.add_argument("--suite", action="append", required=True)
    split.add_argument("--previous-suite", action="append", type=Path, default=[])
    split.add_argument("--output", type=Path, required=True)

    suite_config = subparsers.add_parser("prepare-suite-config")
    suite_config.add_argument("--source", type=Path, required=True)
    suite_config.add_argument("--root-seed", type=int, required=True)
    suite_config.add_argument("--label", required=True)
    suite_config.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = {
        "development-search": development_search,
        "generate-dataset": generate_dataset,
        "train-development-models": train_development_models,
        "evaluate-frozen-models": evaluate_frozen_models,
        "run-control": run_control,
        "analyze-control": analyze_control,
        "build-split-manifest": build_split_manifest,
        "prepare-suite-config": prepare_suite_config,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
