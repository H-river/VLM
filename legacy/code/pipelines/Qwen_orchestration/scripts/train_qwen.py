#!/usr/bin/env python3
"""Run the frozen shared Qwen trainer without modifying its source."""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "optics_sft/scripts/train_qwen25vl_qlora.py"
EXPECTED_SHA256 = "38feb3cfb257ad2b5661d5174e8deec56c2dd63245478408efa3ec5c5eb4c6da"


def main() -> None:
    digest = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"frozen trainer digest changed: expected {EXPECTED_SHA256}, got {digest}"
        )
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
