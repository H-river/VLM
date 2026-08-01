# V11 legacy-grid baseline parity

V11 is an isolated experiment package. It does not modify or retrain v9/v10
and refuses any v10 dataset path. It preserves the canonical 81-action order,
uses independent JSONL groups as the split/statistical unit, and reports a
parity gate before conceptual ablations can be interpreted.

First generate a new-version system-aligned smoke dataset:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m physics_structured_rebuild_v11.generate_system_aligned \
  --smoke \
  --output-dir /home/jiamo/VLM_data/physics_structured_rebuild_v11_smoke/system_aligned
```

Then run the matched baseline smoke:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m physics_structured_rebuild_v11.train \
  --smoke --ablation baseline --device cpu \
  --training-source /home/jiamo/VLM_data/physics_structured_rebuild_v11_smoke/system_aligned/grids/train.jsonl \
  --development-source /home/jiamo/VLM_data/physics_structured_rebuild_v11_smoke/system_aligned/grids/development.jsonl \
  --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v11_smoke
```

Deterministic 32-group overfit diagnostic:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m physics_structured_rebuild_v11.train \
  --overfit-groups 32 --ablation baseline --device cuda \
  --training-source /home/jiamo/VLM_data/physics_structured_rebuild_v11_smoke/system_aligned/grids/train.jsonl
```

Later learning-curve point (not launched automatically):

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python \
  -m physics_structured_rebuild_v11.train \
  --group-count 1200 --development-groups 600 \
  --ablation baseline --device cuda
```

Run the same group count and seed with `ordinary_loss`, `opaque_action`, and
`legacy_distribution` only after the baseline parity gate is acceptable.
