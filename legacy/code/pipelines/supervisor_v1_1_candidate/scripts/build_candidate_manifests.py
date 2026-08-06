#!/usr/bin/env python3
"""Combine sealed legal v1 train/dev rows with new candidate-only pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_vl_supervisor_v1 import build_manifests as v1
from qwen_vl_supervisor_v1.validate_manifest import validate_manifest_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def convert_source(*, source_name: str, dataset_path: Path, pairs_path: Path, split: str, family: str, generator_path: Path) -> list[dict[str, Any]]:
    dataset_rel = dataset_path.resolve().relative_to(ROOT).as_posix()
    pairs_rel = pairs_path.resolve().relative_to(ROOT).as_posix()
    dataset_hash = sha256(dataset_path)
    pair_hash = sha256(pairs_path)
    family_audit = "sensor_saturation" if family == "sensor_saturation" else "secondary_reflection_width_relative"
    v1.SOURCE_SPECS[source_name] = {
        "path": dataset_rel,
        "sha256": dataset_hash,
        "family_audit": family_audit,
        "anomaly_family": family,
        "pair_path": pairs_rel,
        "pair_sha256": pair_hash,
    }
    v1.SOURCE_SPLIT_NAME[source_name] = "train" if split == "train" else "development"
    pair_index = v1.load_pair_index(ROOT, pairs_rel, pair_hash, family)
    module_hash = sha256(generator_path)
    output = []
    for row in read_jsonl(dataset_path):
        if row.get("family_audit") != family_audit:
            raise RuntimeError(f"unexpected family in {dataset_path}: {row.get('family_audit')}")
        pair = pair_index.get(row["pair_id"])
        if pair is None:
            raise RuntimeError(f"missing pair metadata for {source_name}:{row['pair_id']}")
        record = v1.supervisor_record(root=ROOT, source_name=source_name, row=row, pair=pair, split=split)
        candidate_pair_id = "pair_" + v1.stable_hash("supervisor_v1_1_candidate", family, row["pair_id"])[:24]
        record["counterfactual_pair_id"] = candidate_pair_id
        record["sample_id"] = "qvlsup1_" + v1.stable_hash("supervisor_v1_1_candidate", dataset_hash, row["sample_id"], candidate_pair_id)[:24]
        record["provenance"]["source_cohort"] = source_name
        record["provenance"]["source_manifest_path"] = dataset_rel
        record["provenance"]["source_manifest_sha256"] = dataset_hash
        record["provenance"]["source_split"] = "train" if split == "train" else "development"
        record["provenance"]["generator_sha256"] = module_hash
        if family == "sensor_saturation":
            record["provenance"]["generator_version"] = "supervisor_v1_1_candidate_saturation_only_existing_behavior_v1"
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = {
        "train": ROOT / "qwen_vl_supervisor_v1/manifests/manifest_train.jsonl",
        "dev": ROOT / "qwen_vl_supervisor_v1/manifests/manifest_dev.jsonl",
    }
    expected_base = {
        "train": "17718c58516d1ae68b22b3d4e46e1d19758a851a9befbc83a0834c71792c3084",
        "dev": "8a9020195e37894915ff3f600c96bf5193b368a25d32a6186e3900086fb4b669",
    }
    for split in base:
        if sha256(base[split]) != expected_base[split]:
            raise RuntimeError(f"sealed legal v1 {split} manifest drift")
    rows = {split: read_jsonl(path) for split, path in base.items()}
    visual_generator = ROOT / "vlm_optics_benchmark/visual_anomalies.py"
    reflection_generator = ROOT / "vlm_optics_benchmark/reflection_width_relative.py"
    for split, raw_split in (("train", "train"), ("dev", "development")):
        rows[split].extend(convert_source(
            source_name=f"candidate_saturation_{raw_split}",
            dataset_path=ROOT / f"supervisor_v1_1_candidate/raw/saturation/dataset_{raw_split}.jsonl",
            pairs_path=ROOT / f"supervisor_v1_1_candidate/raw/saturation/pairs_{raw_split}.jsonl",
            split=split, family="sensor_saturation", generator_path=visual_generator,
        ))
        rows[split].extend(convert_source(
            source_name=f"width_relative_candidate_{raw_split}",
            dataset_path=ROOT / f"supervisor_v1_1_candidate/raw/reflection/dataset_{raw_split}.jsonl",
            pairs_path=ROOT / f"supervisor_v1_1_candidate/raw/reflection/pairs_{raw_split}.jsonl",
            split=split, family="secondary_reflection", generator_path=reflection_generator,
        ))
    output_dir = args.output_dir.resolve()
    paths = {}
    for split in ("train", "dev"):
        rows[split].sort(key=lambda row: row["sample_id"])
        paths[split] = output_dir / f"manifest_{split}.jsonl"
        write_jsonl(paths[split], rows[split])
    validation = validate_manifest_files([paths["train"], paths["dev"]], repository_root=ROOT)
    base_index = json.loads((ROOT / "qwen_vl_supervisor_v1/manifests/manifest_index.json").read_text(encoding="utf-8"))
    index = {
        "version": "supervisor_v1_1_candidate_manifest_index_v1",
        "status": "NOT SEALED - FROZEN EVALUATION DISABLED",
        "builder": "supervisor_v1_1_candidate.scripts.build_candidate_manifests",
        "manifests": {
            split: {
                "path": paths[split].relative_to(ROOT).as_posix(),
                "sha256": sha256(paths[split]),
                "records": len(rows[split]),
                "pairs": len({row["counterfactual_pair_id"] for row in rows[split]}),
                "setups": len({row["setup_hash"] for row in rows[split]}),
                "class_counts": dict(sorted(Counter(row["target"]["diagnosis"] for row in rows[split]).items())),
                "candidate_added_records": len(rows[split]) - len(read_jsonl(base[split])),
            }
            for split in ("train", "dev")
        },
        "sealed_legal_base": {split: {"path": base[split].relative_to(ROOT).as_posix(), "sha256": expected_base[split]} for split in base},
        "frozen_registry_identity_only": {
            split: base_index["manifests"][split]
            for split in ("frozen_iid", "frozen_ood")
        },
        "validation": validation,
        "frozen_or_protected_image_content_opened": False,
        "frozen_or_protected_predictions_generated": False
    }
    index_path = output_dir / "manifest_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(index["manifests"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
