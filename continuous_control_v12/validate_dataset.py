#!/usr/bin/env python3
"""Validate a generated v12 dataset and its group-level split hashes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_control_v12.schema import validate_dataset

DEFAULT_CONFIG = Path(__file__).with_name("config_v12.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    result = validate_dataset(args.data_dir, config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

