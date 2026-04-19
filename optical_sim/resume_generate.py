"""Resume random dataset generation from a given index."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.experiment_generator import load_yaml, generate_random_experiments
from src.simulator import run_simulation
from src.metrics import compute_metrics
from src.io_utils import save_run, save_summary
import time

output_dir = "outputs/random_5k"
config_path = "configs/random_config.yaml"
start_from = 3224  # first missing index

experiments = generate_random_experiments(config_path)
experiments = experiments[start_from:]  # skip already completed

print(f"▶ Resuming from index {start_from}, {len(experiments)} remaining")

records = []
t0 = time.time()
for i, (name, setup) in enumerate(experiments):
    t1 = time.time()
    result = run_simulation(setup)
    metrics = compute_metrics(result["intensity"], result["sensor_X"], result["sensor_Y"])
    meta = save_run(output_dir, name, setup, result["intensity"], metrics)
    records.append(meta)
    elapsed = time.time() - t1
    idx = start_from + i + 1
    if idx % 100 == 0 or i == len(experiments) - 1:
        print(f"  [{idx}/5000] {name}  ({elapsed:.2f}s)")

# Rebuild full summary from all metadata files
import json
from pathlib import Path
all_records = []
for d in sorted(Path(output_dir).iterdir()):
    mf = d / "metadata.json"
    if mf.exists():
        with open(mf) as f:
            all_records.append(json.load(f))

save_summary(output_dir, all_records)
total = time.time() - t0
print(f"✓ Done — {len(all_records)} total runs. Resumed {len(records)} in {total:.1f}s")
