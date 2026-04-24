# profile2setup v2 Workflow

## Overview

`profile2setup` v2 maps:

```text
beam profile + prompt + optional target profile + optional current setup
-> 7-variable optical setup prediction
```

The model outputs:

- `delta`
- `absolute`
- `change_logits`

The routed final setup is:

- if `setup_present = 1`: `final_setup = current_setup + predicted_delta`
- if `setup_present = 0`: `final_setup = predicted_absolute`

## Canonical Variables

The canonical v2 variable order is exactly:

```text
source_to_lens
lens_to_camera
focal_length
lens_x
lens_y
camera_x
camera_y
```

`camera_x` means camera x offset. `camera_y` means camera y offset. v2 does not use `alignment_x` or `alignment_y`.

## Input Modes

### absolute

Input:

- `target_profile_path`
- `prompt`
- no current setup

Tensor profile channels:

- channel 0: zeros
- channel 1: target profile
- channel 2: zeros
- channel 3: target mask ones

Supervision:

- absolute head if `target_setup` exists
- no delta supervision

### edit

Input:

- `current_profile_path`
- `target_profile_path`
- `current_setup`
- `prompt`

Tensor profile channels:

- current
- target
- target - current
- target mask ones

Supervision:

- delta
- absolute
- change_logits

### current_only

Input:

- `current_profile_path`
- `prompt`
- no target profile
- no current setup unless available

Tensor profile channels:

- current
- zeros
- zeros
- target mask zeros

Supervision:

- absolute only if `target_setup` exists
- no delta unless `current_setup` and `target_delta` are explicitly available

### paired_no_setup

Input:

- `current_profile_path`
- `target_profile_path`
- `prompt`
- no current setup

Tensor profile channels:

- current
- target
- target - current
- target mask ones

Supervision:

- absolute if `target_setup` exists
- no delta because no current setup anchor exists

## Dataset Creation

```bash
python -m profile2setup.scripts.build_absolute_dataset_cli \
  --sim-dir optical_sim/outputs/random_v2 \
  --out profile2setup/data/absolute.jsonl \
  --strict
```

```bash
python -m profile2setup.scripts.build_edit_dataset_cli \
  --sim-dir optical_sim/outputs/random_v2 \
  --out profile2setup/data/edit.jsonl \
  --num-pairs 50000 \
  --seed 42 \
  --strict
```

```bash
python -m profile2setup.scripts.split_dataset_cli \
  --input profile2setup/data/edit.jsonl \
  --out-dir profile2setup/data/all_modes \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --test-frac 0.1 \
  --seed 42
```

Adapt paths to match the actual local dataset. `intensity.npy` is the profile input. `beam_profile.png` should not be used as training input.

## Dataset Smoke Test

```bash
python -m profile2setup.scripts.dataset_smoke_test_cli \
  --jsonl profile2setup/data/all_modes/train.jsonl \
  --variables-config profile2setup/configs/variables.yaml \
  --input-size 128 \
  --max-text-len 32 \
  --limit 4
```

## Model Smoke Test

```bash
python -m profile2setup.scripts.model_smoke_test_cli \
  --vocab-size 100 \
  --batch-size 2 \
  --input-size 128 \
  --text-len 32
```

## Training

Smoke training:

```bash
python -m profile2setup.scripts.train_cli \
  --config profile2setup/configs/train.yaml \
  --smoke-test
```

Full training:

```bash
python -m profile2setup.scripts.train_cli \
  --config profile2setup/configs/train.yaml
```

`setup_present` gates `current_setup`. Loss masks decide which heads are supervised. Mixed-mode batches are supported.

## Offline Evaluation

```bash
python -m profile2setup.scripts.run_baselines_cli \
  --train profile2setup/data/all_modes/train.jsonl \
  --test profile2setup/data/all_modes/test.jsonl \
  --variables-config profile2setup/configs/variables.yaml \
  --out profile2setup/results/baselines.json
```

```bash
python -m profile2setup.scripts.evaluate_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_baseline/best.pt \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/model_eval.json
```

Offline evaluation measures setup prediction accuracy. It does not run the optical simulator.

## Closed-Loop Simulation Evaluation

```bash
python -m profile2setup.scripts.closed_loop_eval_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_baseline/best.pt \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/closed_loop.json \
  --simulation-policy target_base \
  --max-examples 100
```

Closed-loop evaluation predicts setup, denormalizes setup, runs the optical simulator, and compares the simulated predicted profile against target `intensity.npy`.

Simulation policies:

- `target_base`: use the target record as the simulator base context.
- `current_base`: use the current record as the simulator base context.
- `auto`: use the evaluator's automatic policy if implemented.

## Integrity Checks

```bash
python -m profile2setup.scripts.check_v2_integrity_cli \
  --root profile2setup \
  --data-dir profile2setup/data \
  --results-dir profile2setup/results
```

The integrity check verifies the canonical variable order, checks for forbidden v2 fields, validates JSONL records, and checks result JSON files.
