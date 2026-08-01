#!/usr/bin/env python3
"""Select a direction LLM checkpoint strictly by frozen validation macro-F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint numbers")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); rows = []
    for checkpoint in [int(value) for value in args.checkpoints.split(",")]:
        path = args.panel_dir / f"checkpoint-{checkpoint}" / "val_score/summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"checkpoint": checkpoint,
                     "macro_f1": summary["metrics"]["equal_field_macro_f1"],
                     "joint_exact": summary["metrics"]["joint_exact"],
                     "schema_valid_rate": summary["schema_valid_rate"],
                     "maximum_field_collapse_fraction": max(
                         item["majority_fraction"] for item in summary["collapse"].values())})
    selected = max(rows, key=lambda row: (row["macro_f1"], row["joint_exact"], -row["checkpoint"]))
    output = {"selection_metric": "validation equal-field macro-F1", "panel": rows,
              "selected_checkpoint": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
