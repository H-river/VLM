#!/usr/bin/env python3
"""Generate only preregistered saturation pairs for the v1.1 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continuous_control_v12.contracts import Bounds
from continuous_control_v12.simulator import simulate_state
from vlm_optics_benchmark.visual_anomalies import (
    MATCH_TOLERANCE,
    _candidate_options,
    canonical_patch,
    inject_anomaly,
    matched_clean_counterfactual,
    moment_metrics,
    normalized_metric_distance,
    sha256_file,
    stable_seed,
    write_jsonl,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_png(image: np.ndarray, path: Path) -> np.ndarray:
    quantized = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantized, mode="L").save(path)
    return quantized.astype(np.float32) / 255.0


def generate(*, suite_path: Path, split: str, output_dir: Path, v12_config: Path, base_config: Path) -> None:
    dataset_path = output_dir / f"dataset_{split}.jsonl"
    pairs_path = output_dir / f"pairs_{split}.jsonl"
    suite = json.loads(suite_path.resolve().read_text(encoding="utf-8"))
    summary_path = output_dir / f"summary_{split}.json"
    if dataset_path.exists() or pairs_path.exists():
        if not (dataset_path.is_file() and pairs_path.is_file()) or summary_path.exists():
            raise FileExistsError(f"refusing to overwrite incomplete or finalized saturation output for {split}")
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line]
        pairs = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 2 * len(suite["cases"]) or len(pairs) != len(suite["cases"]):
            raise RuntimeError("existing saturation output does not match the preregistered setup count")
        if not all((output_dir / row["model_input"]["image_ref"]).is_file() for row in rows):
            raise RuntimeError("existing saturation output is missing an image")
        if not all(pair["pre_serialization_metric_distance"]["passes_frozen_match"] and pair["post_serialization_metric_distance"]["passes_frozen_match"] for pair in pairs):
            raise RuntimeError("existing saturation output contains a failed counterfactual match")
        atomic_json(summary_path, {
            "version": "supervisor_v1_1_candidate_saturation_v1",
            "split": split,
            "suite": str(suite_path.resolve()),
            "suite_sha256": sha256_file(suite_path),
            "setups": len(suite["cases"]),
            "records": len(rows),
            "pairs": len(pairs),
            "class_distribution": dict(sorted(Counter(row["supervision"]["fault_type"] for row in rows).items())),
            "clip_level_range_preregistered": [0.35, 0.55],
            "observed_clip_level_min": min(pair["severity"]["clip_level_fraction_of_peak"] for pair in pairs),
            "observed_clip_level_max": max(pair["severity"]["clip_level_fraction_of_peak"] for pair in pairs),
            "all_pairs_pass_pre_serialization": True,
            "all_pairs_pass_post_serialization": True,
            "old_fixed_pixel_reflection_generated": False,
            "completion_note": "summary finalized from complete deterministic outputs after a report-only boolean literal error; no sample was regenerated",
        })
        return
    bounds = Bounds.from_config(json.loads(v12_config.resolve().read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for case in suite["cases"]:
        capture = simulate_state(
            case["setup_context"], case["initial_positions_mm"], case["simulator_fixed"],
            str(base_config.resolve()), bounds,
        )
        primary = canonical_patch(capture["intensity"])
        history = moment_metrics(primary).tolist()
        rng = np.random.default_rng(stable_seed("severity", case["case_id"], "sensor_saturation", split))
        clip_level = float(rng.uniform(0.35, 0.55))
        severity = {"clip_level_fraction_of_peak": clip_level}
        anomaly = inject_anomaly(primary, "sensor_saturation", severity)
        clean = matched_clean_counterfactual(primary, moment_metrics(anomaly))
        clean_serialized = np.rint(np.clip(clean, 0.0, 1.0) * 255).astype(np.uint8).astype(np.float32) / 255.0
        anomaly_serialized = np.rint(np.clip(anomaly, 0.0, 1.0) * 255).astype(np.uint8).astype(np.float32) / 255.0
        pre = normalized_metric_distance(moment_metrics(clean), moment_metrics(anomaly))
        post = normalized_metric_distance(moment_metrics(clean_serialized), moment_metrics(anomaly_serialized))
        if not pre["passes_frozen_match"] or not post["passes_frozen_match"]:
            raise RuntimeError(f"counterfactual match failed for {case['case_id']}: pre={pre}, post={post}")
        pair_id = hashlib.sha256(f"v11candidate:{split}:{case['group_id']}:sensor_saturation".encode()).hexdigest()[:20]
        ids: dict[str, str] = {}
        refs: dict[str, str] = {}
        metrics_by_role: dict[str, list[float]] = {}
        for label, image in (("clean", clean), ("sensor_saturation", anomaly)):
            sample_id = hashlib.sha256(f"{pair_id}:{label}:sample".encode()).hexdigest()[:24]
            relative = Path("images") / split / f"img_{sample_id}.png"
            serialized = save_png(image, output_dir / relative)
            metrics = moment_metrics(serialized).tolist()
            ids[label] = sample_id
            refs[label] = relative.as_posix()
            metrics_by_role[label] = metrics
            rows.append({
                "sample_id": sample_id,
                "pair_id": pair_id,
                "setup_id": case["group_id"],
                "setup_hash": case["setup_hash"],
                "split": split,
                "family_audit": "sensor_saturation",
                "severity_scalar": clip_level,
                "model_input": {
                    "image_ref": relative.as_posix(),
                    "five_metrics": metrics,
                    "five_metric_uncertainties": MATCH_TOLERANCE.tolist(),
                    "short_history_metrics": [history],
                    "target": case["target_metrics"],
                    "candidate_options": _candidate_options(sample_id),
                },
                "supervision": {
                    "fault_type": label,
                    "binary_fault_present": label != "clean",
                    "oracle_recovery_decision": "standard_metrics" if label == "clean" else "reduce_exposure_reacquire",
                },
            })
        pairs.append({
            "pair_id": pair_id,
            "setup_id": case["group_id"],
            "setup_hash": case["setup_hash"],
            "split": split,
            "family": "sensor_saturation",
            "clean_sample_id": ids["clean"],
            "anomalous_sample_id": ids["sensor_saturation"],
            "clean_image_ref": refs["clean"],
            "anomalous_image_ref": refs["sensor_saturation"],
            "clean_five_metrics": metrics_by_role["clean"],
            "anomalous_five_metrics": metrics_by_role["sensor_saturation"],
            "target": case["target_metrics"],
            "anomaly_type": "sensor_saturation",
            "severity": severity,
            "pre_serialization_metric_distance": pre,
            "post_serialization_metric_distance": post,
            "normalized_metric_distance": post,
            "oracle_recovery_decision": "reduce_exposure_reacquire",
        })
    rng = np.random.default_rng(stable_seed("v11candidate_saturation_serialization", split))
    rng.shuffle(rows)
    rng.shuffle(pairs)
    write_jsonl(dataset_path, rows)
    write_jsonl(pairs_path, pairs)
    atomic_json(summary_path, {
        "version": "supervisor_v1_1_candidate_saturation_v1",
        "split": split,
        "suite": str(suite_path.resolve()),
        "suite_sha256": sha256_file(suite_path),
        "setups": len(suite["cases"]),
        "records": len(rows),
        "pairs": len(pairs),
        "class_distribution": dict(sorted(Counter(row["supervision"]["fault_type"] for row in rows).items())),
        "clip_level_range_preregistered": [0.35, 0.55],
        "observed_clip_level_min": min(pair["severity"]["clip_level_fraction_of_peak"] for pair in pairs),
        "observed_clip_level_max": max(pair["severity"]["clip_level_fraction_of_peak"] for pair in pairs),
        "all_pairs_pass_pre_serialization": all(pair["pre_serialization_metric_distance"]["passes_frozen_match"] for pair in pairs),
        "all_pairs_pass_post_serialization": all(pair["post_serialization_metric_distance"]["passes_frozen_match"] for pair in pairs),
        "old_fixed_pixel_reflection_generated": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v12-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    args = parser.parse_args()
    generate(suite_path=args.suite, split=args.split, output_dir=args.output_dir, v12_config=args.v12_config, base_config=args.base_config)


if __name__ == "__main__":
    main()
