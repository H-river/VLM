#!/usr/bin/env python3
"""Execute a registered optics decision tool from a JSON request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .decision_tools import TOOL_CATALOG, run_tool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, help="JSON file with tool_name and arguments")
    parser.add_argument("--list-tools", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tools:
        print(json.dumps(TOOL_CATALOG, indent=2, sort_keys=True))
        return
    if args.request_json:
        request: Any = json.loads(args.request_json.read_text(encoding="utf-8"))
    else:
        request = json.load(sys.stdin)
    if not isinstance(request, dict) or set(request) != {"tool_name", "arguments"}:
        raise SystemExit("request must contain exactly tool_name and arguments")
    result = run_tool(str(request["tool_name"]), request["arguments"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
