#!/usr/bin/env python3
"""Assemble validation-selected direction-LLM IID/OOD scores for certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--iid-summary", type=Path, required=True)
    parser.add_argument("--ood-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(
    panel: dict[str, Any], iid: dict[str, Any], ood: dict[str, Any]
) -> dict[str, Any]:
    if iid.get("split") != "eval_iid":
        raise ValueError("IID summary must identify split=eval_iid")
    if ood.get("split") != "eval_ood":
        raise ValueError("OOD summary must identify split=eval_ood")
    selected = panel.get("selected_checkpoint", {})
    if "checkpoint" not in selected:
        raise ValueError("panel summary has no selected checkpoint")
    return {
        "selection_policy": "checkpoint selected only by frozen validation equal-field macro-F1",
        "selected_checkpoint": selected,
        "validation_panel": panel.get("panel", []),
        "eval_iid": iid,
        "eval_ood": ood,
    }


def main() -> None:
    args = parse_args()
    output = assemble(load(args.panel_summary), load(args.iid_summary), load(args.ood_summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
