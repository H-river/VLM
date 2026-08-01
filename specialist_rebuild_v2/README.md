# Specialist rebuild v2

This folder is isolated from the frozen Qwen orchestrator and version-1
specialists. It builds and trains five replacement specialists with one fixed
seed.

## Numerical dataset

- 3,700 independent optical setups:
  2,500 train, 300 validation, 300 IID test, 300 physics-OOD test, and
  300 visual-stress test.
- Each setup has one current state and the full 81-action Cartesian grid.
- Total simulator transitions: 3,700 x 81 = 299,700.
- Numerical inverse train pairs: 24,000 unique, 24,000 ambiguous, and
  12,000 infeasible.
- Every split is separated by `group_id`, so no setup can occur in two splits.

Definitions:

- **IID** means independently sampled from the same numerical ranges as train.
- **Physics-OOD** means optical parameters are sampled outside the train ranges.
- **Unique** means exactly one of the 81 allowed actions reaches the requested
  state within the declared tolerance.
- **Ambiguous** means two or more allowed actions reach it.
- **Infeasible** means none of the 81 actions reaches it.

## Visual dataset

Each setup has one 192 x 192 current-beam image and three desired-beam images:
one unique target, one ambiguous target, and one target infeasible within the
81-action limit (falling back to another reachable status only when a setup has
no member of a requested class). This gives 7,500 visual inverse training pairs
and 10,000 measurement-training images. Training contains clean, noise, blur,
dim, saturation, crop, and gamma conditions.
Visual-stress test has 60 setups in each of five held-out stress conditions.

## Safe execution

Generation is resumable: each completed group is stored as a separate shard.
The builder rejects more than four processes. The recommended laptop command
also limits numerical libraries to one CPU thread per process.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 taskset -c 0-7 nice -n 10 \
  python specialist_rebuild_v2/build_dataset.py \
  --output-dir /home/jiamo/VLM_data/specialist_rebuild_v2 --workers 3 \
  --worker-affinities '0;8;1,9'
```

The affinity list spreads the three simulator workers across both CPU
chiplets. On the target Ryzen 9 7945HX laptop this materially reduces package
temperature compared with allowing all workers to settle on cores 0-5.

After generation:

```bash
python specialist_rebuild_v2/verify_dataset.py \
  /home/jiamo/VLM_data/specialist_rebuild_v2
```

Training is sequential, uses one seed, and runs only one specialist at a time:

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 taskset -c 0-7 nice -n 10 \
  python specialist_rebuild_v2/train_models.py \
  /home/jiamo/VLM_data/specialist_rebuild_v2 \
  /home/jiamo/VLM_runs/specialist_rebuild_v2_one_seed
```

`run_pipeline.py` combines the resumable builder, strict verifier, and
conditional training step. It records the current stage and exact timestamps in
`pipeline_state.json`; training starts only when the verified dataset finishes
before the supplied objective deadline. After training, `audit_artifacts.py`
requires all five checkpoints, finite tensors, exact parameter counts, the
combined training summary, and the verified full-dataset counts before the
pipeline records `stage: complete`.

Image datasets cache decoded grayscale arrays inside their single persistent
loader worker after the first epoch. This avoids repeatedly decoding the same
PNG files while keeping the cache bounded to the current specialist.
