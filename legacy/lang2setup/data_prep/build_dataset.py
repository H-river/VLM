"""
build_dataset.py
Walk the simulation output directory and produce a text-supervision JSONL dataset.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

from .extract_features import extract_features
from .discretize import discretize_sample, load_bins_config
from .describe import generate_descriptions, load_templates_config


def build_dataset(sim_output_dir: str | Path,
                  output_path: str | Path,
                  bins_config_path: str | Path | None = None,
                  templates_config_path: str | Path | None = None,
                  seed: int = 42) -> List[Dict[str, Any]]:
    """
    Scan all sample directories, generate text–target pairs, write JSONL.

    Parameters
    ----------
    sim_output_dir : path to e.g. outputs/random_5k/
    output_path : where to write the JSONL file
    """
    sim_output_dir = Path(sim_output_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bins_cfg = load_bins_config(bins_config_path) if bins_config_path else load_bins_config()
    tmpl_cfg = load_templates_config(templates_config_path) if templates_config_path else load_templates_config()
    rng = random.Random(seed)

    records = []
    sample_dirs = sorted(sim_output_dir.iterdir())

    for sample_dir in sample_dirs:
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            continue

        features = extract_features(meta_path)
        target = discretize_sample(features, bins_cfg)
        target_continuous = {
            "x_offset": features["x_offset"],
            "y_offset": features["y_offset"],
            "tilt_x": features["tilt_x"],
        }

        descriptions = generate_descriptions(features, tmpl_cfg, rng=rng)

        for text in descriptions:
            record = {
                "text": text,
                "target": target,
                "target_continuous": target_continuous,
                "source_run": features["run_name"],
            }
            records.append(record)

    # Write JSONL
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"✓ Wrote {len(records)} records from {len(sample_dirs)} samples → {output_path}")
    return records
