"""
split.py
Stratified train/val/test split of the text-supervision dataset.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

_DEFAULT_SPLIT_CFG = Path(__file__).parent.parent / "configs" / "split.yaml"


def load_split_config(path: str | Path = _DEFAULT_SPLIT_CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def split_dataset(input_jsonl: str | Path,
                  output_dir: str | Path,
                  split_config_path: str | Path | None = None) -> Dict[str, int]:
    """
    Read the full JSONL dataset, assign stratified splits, write per-split files.

    Returns dict of split_name → record count.
    """
    cfg = load_split_config(split_config_path) if split_config_path else load_split_config()
    scfg = cfg["split"]
    seed = cfg["random_seed"]
    min_group = cfg["min_group_size_for_split"]

    input_jsonl = Path(input_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all records
    with open(input_jsonl) as f:
        records = [json.loads(line) for line in f]

    # Group by source_run (all text variants of one sample share the same split)
    run_groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        run_groups[rec["source_run"]].append(rec)

    # Group runs by (x_bin, angle_bin) for stratification
    strat_groups: Dict[tuple, List[str]] = defaultdict(list)
    for run_name, recs in run_groups.items():
        key = (recs[0]["target"]["x_bin"], recs[0]["target"]["angle_bin"])
        strat_groups[key].append(run_name)

    rng = random.Random(seed)
    split_assignment: Dict[str, str] = {}  # run_name → split

    for key, run_names in strat_groups.items():
        rng.shuffle(run_names)
        n = len(run_names)
        if n < min_group:
            # Too few — all go to train
            for rn in run_names:
                split_assignment[rn] = "train"
        else:
            n_val = max(1, int(n * scfg["val"]))
            n_test = max(1, int(n * scfg["test"]))
            n_train = n - n_val - n_test
            for rn in run_names[:n_train]:
                split_assignment[rn] = "train"
            for rn in run_names[n_train:n_train + n_val]:
                split_assignment[rn] = "val"
            for rn in run_names[n_train + n_val:]:
                split_assignment[rn] = "test"

    # Write split files
    writers = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for split_name in ["train", "val", "test"]:
        writers[split_name] = open(output_dir / f"lang2setup_{split_name}.jsonl", "w")

    for rec in records:
        split_name = split_assignment[rec["source_run"]]
        rec["split"] = split_name
        writers[split_name].write(json.dumps(rec) + "\n")
        counts[split_name] += 1

    for w in writers.values():
        w.close()

    print(f"✓ Split: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return counts
