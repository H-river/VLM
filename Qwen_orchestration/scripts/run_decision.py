#!/usr/bin/env python3
"""Validate and execute one orchestration decision JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.runtime import OrchestrationRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", type=Path)
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="IMAGE_N=PATH",
        help="Bind an image reference from the decision to a local file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.decision.open(encoding="utf-8") as stream:
        decision = json.load(stream)
    bindings = {}
    for item in args.image:
        reference, separator, value = item.partition("=")
        if not separator or not reference or not value:
            raise SystemExit(f"invalid --image binding: {item}")
        bindings[reference] = value
    result = OrchestrationRuntime().dispatch(decision, bindings)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
