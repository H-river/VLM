"""
io_utils.py
Save simulation outputs: images, arrays, metadata JSON, and global summary.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from .optical_elements import OpticalSetup
from .metrics import BeamMetrics


def _numpy_serialise(obj):
    """JSON encoder helper for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_run(output_dir: str,
             run_name: str,
             setup: OpticalSetup,
             intensity: np.ndarray,
             metrics: BeamMetrics) -> dict:
    """
    Save all outputs for a single simulation run.

    Creates:
        <output_dir>/<run_name>/
            beam_profile.png
            intensity.npy
            metadata.json

    Returns the metadata dict (for appending to the global summary).
    """
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- intensity array ---
    np.save(run_dir / "intensity.npy", intensity)

    # --- beam profile image (simple greyscale PNG via matplotlib) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(intensity, cmap="inferno", origin="lower")
        ax.set_title(run_name, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(run_dir / "beam_profile.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass  # matplotlib optional; array is always saved

    # --- metadata ---
    meta = {
        "run_name": run_name,
        "setup": {
            "source": asdict(setup.source),
            "lens": asdict(setup.lens),
            "sensor": {
                "resolution": list(setup.sensor.resolution),
                "pixel_pitch": setup.sensor.pixel_pitch,
                "sensor_size": list(setup.sensor.sensor_size),
            },
            "alignment": asdict(setup.alignment),
            "geometry": {
                "laser_to_lens": setup.laser_to_lens,
                "lens_to_camera": setup.lens_to_camera,
                "effective_camera_distance": setup.effective_camera_distance,
            },
            "simulation": {
                "grid_size": setup.grid_size,
                "grid_extent": setup.grid_extent,
                "propagation_backend": setup.propagation_backend,
            },
        },
        "metrics": metrics.to_dict(),
    }

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=_numpy_serialise)

    return meta


def save_summary(output_dir: str, records: List[dict]) -> None:
    """
    Write a global summary as both JSONL and CSV.

    Files:
        <output_dir>/summary.jsonl
        <output_dir>/summary.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSONL
    with open(out / "summary.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_numpy_serialise) + "\n")

    # Flat CSV — flatten metrics only (setup is in per-run JSON)
    import csv
    if not records:
        return
    fieldnames = ["run_name"] + list(records[0]["metrics"].keys())
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {"run_name": rec["run_name"], **rec["metrics"]}
            writer.writerow(row)
