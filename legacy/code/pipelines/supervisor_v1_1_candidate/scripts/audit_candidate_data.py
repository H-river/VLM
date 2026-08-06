#!/usr/bin/env python3
"""Leakage, distribution, nearest-neighbor, and prompt-visible candidate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_vl_supervisor_v1.export_sft import validate_export_row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "setup_hash": row["setup_hash"],
        "pair_id": row["counterfactual_pair_id"],
        "episode_hash": row.get("episode_hash"),
        "image_sha256": row["assets"]["current_image_sha256"],
        "augmented_base_hash": row["augmented_base_hash"],
    }


def set_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]], key: str) -> int:
    a = {row[key] for row in left if row.get(key) is not None}
    b = {row[key] for row in right if row.get(key) is not None}
    return len(a & b)


def load_image(row: dict[str, Any]) -> np.ndarray:
    with Image.open(ROOT / row["assets"]["current_image_path"]) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def phash(image: np.ndarray) -> np.ndarray:
    small = np.asarray(Image.fromarray(np.rint(image * 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    coeff = dctn(small, norm="ortho")[:8, :8]
    median = float(np.median(coeff[1:, :]))
    return coeff >= median


def nearest_cross_split(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, Any]:
    train_images = np.stack([load_image(row) for row in train])
    dev_images = np.stack([load_image(row) for row in dev])
    train_hashes = np.stack([phash(image) for image in train_images])
    dev_hashes = np.stack([phash(image) for image in dev_images])
    mse_min, phash_min = [], []
    perceptual_duplicates = 0
    for image, bits in zip(dev_images, dev_hashes, strict=True):
        mse = np.mean((train_images - image[None, :, :]) ** 2, axis=(1, 2))
        hamming = np.count_nonzero(train_hashes != bits[None, :, :], axis=(1, 2))
        mse_min.append(float(mse.min()))
        phash_min.append(int(hamming.min()))
        perceptual_duplicates += int(np.any((mse <= 1e-4) & (hamming <= 2)))
    return {
        "scope": "legal candidate train versus legal candidate dev images only; protected image content was not opened",
        "dev_records": len(dev),
        "nearest_pixel_mse": {
            "minimum": min(mse_min), "median": float(np.median(mse_min)), "maximum": max(mse_min),
        },
        "nearest_phash_hamming_64bit": {
            "minimum": min(phash_min), "median": float(np.median(phash_min)), "maximum": max(phash_min),
        },
        "perceptual_duplicate_rule": "pixel MSE <= 1e-4 and pHash Hamming <= 2",
        "perceptual_duplicate_dev_records": perceptual_duplicates,
    }


def distribution(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_class[row["target"]["diagnosis"]].append(row)
    metric_names = ("centroid_x", "centroid_y", "width_x", "width_y", "peak_intensity")
    return {
        "records": len(manifest_rows),
        "pairs": len({row["counterfactual_pair_id"] for row in manifest_rows}),
        "setups": len({row["setup_hash"] for row in manifest_rows}),
        "classes": dict(sorted(Counter(row["target"]["diagnosis"] for row in manifest_rows).items())),
        "anomaly_families": dict(sorted(Counter(row["provenance"]["anomaly_family"] for row in manifest_rows).items())),
        "width_quartiles": dict(sorted(Counter(str(row["provenance"]["width_quartile"]) for row in manifest_rows if row["provenance"]["anomaly_family"] == "secondary_reflection").items())),
        "boundary_status": dict(sorted(Counter(str(row["provenance"]["boundary_status"]) for row in manifest_rows).items())),
        "current_metric_ranges": {
            name: {
                "minimum": min(float(row["model_input"]["current_metrics"][name]) for row in manifest_rows),
                "median": float(np.median([float(row["model_input"]["current_metrics"][name]) for row in manifest_rows])),
                "maximum": max(float(row["model_input"]["current_metrics"][name]) for row in manifest_rows),
            }
            for name in metric_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "train": ROOT / "supervisor_v1_1_candidate/manifests/manifest_train.jsonl",
        "dev": ROOT / "supervisor_v1_1_candidate/manifests/manifest_dev.jsonl",
        "frozen_iid": ROOT / "qwen_vl_supervisor_v1/manifests/manifest_frozen_iid.jsonl",
        "frozen_ood": ROOT / "qwen_vl_supervisor_v1/manifests/manifest_frozen_ood.jsonl",
    }
    legal = {name: rows(path) for name, path in paths.items() if name in {"train", "dev"}}
    # Protected manifests are reduced immediately to identity-only fields. No protected image is opened.
    protected_identity = {name: [identity(row) for row in rows(path)] for name, path in paths.items() if name.startswith("frozen_")}
    legal_identity = {name: [identity(row) for row in values] for name, values in legal.items()}
    overlap: dict[str, Any] = {}
    cohorts = {**legal_identity, **protected_identity}
    for left_index, left in enumerate(cohorts):
        for right in list(cohorts)[left_index + 1:]:
            overlap[f"{left}__{right}"] = {
                key: set_overlap(cohorts[left], cohorts[right], key)
                for key in ("sample_id", "setup_hash", "pair_id", "episode_hash", "image_sha256", "augmented_base_hash")
            }
    candidate_new = {
        split: [row for row in legal[split] if "candidate" in row["provenance"]["source_cohort"]]
        for split in ("train", "dev")
    }
    new_counts = {
        split: {
            "records": len(values),
            "pairs": len({row["counterfactual_pair_id"] for row in values}),
            "setups": len({row["setup_hash"] for row in values}),
            "class_counts": dict(sorted(Counter(row["target"]["diagnosis"] for row in values).items())),
            "source_cohorts": dict(sorted(Counter(row["provenance"]["source_cohort"] for row in values).items())),
            "all_pair_metric_matches_pass": all(row["provenance"]["counterfactual_metric_distances"]["passes_frozen_match"] for row in values),
        }
        for split, values in candidate_new.items()
    }
    sft_paths = {split: ROOT / f"supervisor_v1_1_candidate/sft/sft_{split}.jsonl" for split in ("train", "dev")}
    prompt_audit = {}
    for split, path in sft_paths.items():
        exported = rows(path)
        for row in exported:
            validate_export_row(row, repository_root=ROOT)
        visible = "\n".join(
            item["text"]
            for row in exported
            for message in row["prompt"]
            for item in message["content"]
            if item["type"] == "text"
        ).lower()
        forbidden = [token for token in ("setup_hash", "source_cohort", "source_manifest", "pair_id", "current_image_path", ".png") if token in visible]
        prompt_audit[split] = {"records": len(exported), "validator_passed": True, "forbidden_visible_tokens": forbidden, "image_slots_each": 1}
    raw_paths = sorted((ROOT / "supervisor_v1_1_candidate/raw").glob("**/*.json*"))
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in raw_paths if path.is_file()}
    manifest_hashes = {name: sha256(path) for name, path in paths.items()}
    export_hashes = {split: sha256(path) for split, path in sft_paths.items()}
    red_overlap = any(
        value
        for pair, fields in overlap.items()
        for key, value in fields.items()
        if (pair == "train__dev" or "frozen" in pair) and key in {"setup_hash", "pair_id", "episode_hash", "image_sha256", "augmented_base_hash"}
    )
    red = any((
        new_counts["train"]["records"] != 96,
        new_counts["dev"]["records"] != 24,
        new_counts["train"]["pairs"] != 48,
        new_counts["dev"]["pairs"] != 12,
        not new_counts["train"]["all_pair_metric_matches_pass"],
        not new_counts["dev"]["all_pair_metric_matches_pass"],
        red_overlap,
        any(value["forbidden_visible_tokens"] for value in prompt_audit.values()),
    ))
    report = {
        "version": "supervisor_v1_1_candidate_data_audit_v1",
        "status": "RED_STOP" if red else "PASSED_FOR_DEVELOPMENT_TRAINING",
        "protected_policy": {
            "protected_manifest_use": "identity fields only for leakage comparison",
            "protected_image_content_opened": False,
            "protected_predictions_generated": False,
            "perceptual_audit_scope": "train/dev only because protected image content access is forbidden",
        },
        "new_candidate_counts": new_counts,
        "candidate_distributions": {split: distribution(legal[split]) for split in ("train", "dev")},
        "cross_cohort_identity_overlap": overlap,
        "train_dev_perceptual_nearest_neighbor": nearest_cross_split(legal["train"], legal["dev"]),
        "prompt_visible_audit": prompt_audit,
        "hashes": {"raw_sources": source_hashes, "manifests": manifest_hashes, "exports": export_hashes},
        "old_fixed_pixel_reflection_records": sum(
            not row["provenance"]["source_cohort"].startswith("width_relative_")
            for split in ("train", "dev")
            for row in legal[split]
            if row["provenance"]["anomaly_family"] == "secondary_reflection"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "new_candidate_counts": new_counts, "perceptual": report["train_dev_perceptual_nearest_neighbor"]}, indent=2, sort_keys=True))
    if red:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
