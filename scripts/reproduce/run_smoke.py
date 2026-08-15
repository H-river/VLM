#!/usr/bin/env python3
"""Run the canonical CPU smoke suite from a source checkout."""

from optics_vla.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["all-smoke"]))

