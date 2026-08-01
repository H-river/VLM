#!/usr/bin/env python3
"""Build an auditable development-only constant-gain classifier control arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier


VERSION = "active_diagnosis_v13_constant_gain_control_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(root: Path) -> list[dict[str, Any]]:
    result = {}
    for path in sorted((root / "probes").glob("records*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                row = json.loads(line)
                result[str(row["record_id"])] = row
    return list(result.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.gate_dir.resolve()
    rows = [
        row
        for row in _rows(root)
        if str(row["design"]) == args.design
        and float(row["fraction"]) == args.fraction
    ]
    if len(rows) != 150:
        raise ValueError(f"expected 150 development probe rows, found {len(rows)}")
    if any(int(str(row["case_id"]).rsplit("_", 1)[1]) >= 10 for row in rows):
        raise ValueError("protected case present in constant-gain model source")
    X = np.asarray(
        [row["policy_record"]["feature_vector"] for row in rows], dtype=np.float64
    )
    label = f"{args.gain:g}"
    classifier = DummyClassifier(strategy="constant", constant=label).fit(
        X, np.full(len(X), label)
    )
    case_to_fold = {str(row["case_id"]): 0 for row in rows}
    bundle = {
        "version": VERSION,
        "ablation": "constant_gain",
        "constant_gain": float(args.gain),
        "feature_names": list(rows[0]["policy_record"]["feature_names"]),
        "retained_feature_indices": list(range(X.shape[1])),
        "fold_models": [classifier],
        "case_to_fold": case_to_fold,
        "full_model": classifier,
    }
    output = args.output_model.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = output.with_suffix(f".joblib.tmp.{os.getpid()}")
    joblib.dump(bundle, temporary_model)
    os.replace(temporary_model, output)
    manifest = {
        "version": VERSION,
        "split": "development_only",
        "protected_set_used": False,
        "design": args.design,
        "fraction": args.fraction,
        "constant_gain": float(args.gain),
        "source_records": len(rows),
        "feature_count": X.shape[1],
        "classifier_bundle": str(output),
        "classifier_bundle_sha256": _sha256(output),
        "role": "control_ablation_not_model_selection",
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
