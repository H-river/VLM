#!/usr/bin/env python3
"""Apply the preregistered within-stratum 8/2 train/dev setup split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.train_output, args.dev_output, args.report)):
        raise FileExistsError("refusing to overwrite candidate split artifacts")
    source = json.loads(args.suite.resolve().read_text(encoding="utf-8"))
    routed = {"train": [], "dev": []}
    for case in source["cases"]:
        suffix = int(str(case["case_id"]).rsplit("_", 1)[1])
        routed["train" if suffix <= 7 else "dev"].append(case)
    if len(routed["train"]) != 24 or len(routed["dev"]) != 6:
        raise RuntimeError(f"preregistered 24/6 split failed: { {k:len(v) for k,v in routed.items()} }")
    if {c["setup_hash"] for c in routed["train"]} & {c["setup_hash"] for c in routed["dev"]}:
        raise RuntimeError("train/dev setup overlap")
    for split, output in (("train", args.train_output), ("dev", args.dev_output)):
        value = dict(source)
        value["version"] = "supervisor_v1_1_candidate_setup_suite_view_v1"
        value["parent_suite"] = str(args.suite.resolve())
        value["parent_suite_sha256"] = sha256(args.suite)
        value["candidate_split"] = split
        value["cases"] = routed[split]
        value["validation"] = dict(source["validation"])
        value["validation"]["groups"] = len(routed[split])
        value["validation"]["unique_group_ids"] = len({c["group_id"] for c in routed[split]})
        value["validation"]["unique_setup_hashes"] = len({c["setup_hash"] for c in routed[split]})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "version": "supervisor_v1_1_candidate_setup_split_report_v1",
        "source": str(args.suite.resolve()),
        "source_sha256": sha256(args.suite),
        "algorithm": "case suffix 0000-0007 train; 0008-0009 dev within each stratum",
        "train": {"path": str(args.train_output.resolve()), "sha256": sha256(args.train_output), "setups": 24},
        "dev": {"path": str(args.dev_output.resolve()), "sha256": sha256(args.dev_output), "setups": 6},
        "setup_overlap": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
