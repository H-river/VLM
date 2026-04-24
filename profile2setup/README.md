# profile2setup (v2)

`profile2setup` is the v2 pipeline package.

`lang2setup` is preserved as v1.

v2 predicts 7-variable setup deltas from profile + prompt (with optional target profile and/or current setup context).

Canonical v2 variables are:
- `source_to_lens`
- `lens_to_camera`
- `focal_length`
- `lens_x`
- `lens_y`
- `camera_x`
- `camera_y`

`camera_x` and `camera_y` mean camera offsets.

v2 uses camera offset terminology only. Legacy simulator internals are not v2 dataset, model, result, or config fields.

## Quick Start

Full workflow documentation is in [WORKFLOW.md](WORKFLOW.md). Recommended experiment tracking is in [EXPERIMENTS.md](EXPERIMENTS.md).

Run the integrity check:

```bash
python -m profile2setup.scripts.check_v2_integrity_cli \
  --root profile2setup \
  --data-dir profile2setup/data \
  --results-dir profile2setup/results
```

Run the v2 smoke pipeline:

```bash
python -m profile2setup.scripts.run_v2_smoke_pipeline_cli \
  --train-jsonl profile2setup/data/all_modes/train.jsonl \
  --val-jsonl profile2setup/data/all_modes/val.jsonl \
  --test-jsonl profile2setup/data/all_modes/test.jsonl \
  --config profile2setup/configs/train.yaml \
  --max-closed-loop-examples 5
```

## Stage 3 Dataset Loading (Smoke Test)

Run:

```bash
python -m profile2setup.scripts.dataset_smoke_test_cli \
  --jsonl profile2setup/data/all_modes/train.jsonl \
  --variables-config profile2setup/configs/variables.yaml \
  --input-size 128 \
  --max-text-len 32 \
  --limit 4
```

Notes:
- Profiles are loaded from `intensity.npy` (not `beam_profile.png`).
- The profile tensor has 4 channels: current, target, target-minus-current, and target mask.
- Setup vectors use the canonical v2 variable order:
  `source_to_lens`, `lens_to_camera`, `focal_length`, `lens_x`, `lens_y`, `camera_x`, `camera_y`.
- `target_profile_path` is preserved for later profile-loss / closed-loop evaluation.
- This Stage 3 path does not implement differentiable profile loss, model architecture, training loop, or evaluation.

## Stage 4 Model (Smoke Test)

The first Stage 4 model is:
- profile CNN encoder
- simple text encoder
- setup encoder
- fusion MLP
- multi-head outputs

Inputs:
- `profile`: `[B, 4, H, W]`
- `prompt_tokens`: `[B, T]`
- `current_setup`: `[B, 7]`

Outputs:
- `delta`: `[B, 7]`
- `absolute`: `[B, 7]`
- `change_logits`: `[B, 7]`

The primary training target in the next stage will be `delta`.

Run model smoke test:

```bash
python -m profile2setup.scripts.model_smoke_test_cli \
  --vocab-size 100 \
  --batch-size 2 \
  --input-size 128 \
  --text-len 32
```

## Stage 5 Training

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

The Stage 5 trainer uses the mixed all-modes dataset. The dataset supplies
`setup_present`; the model uses it to gate `current_setup`. Loss masks decide
which heads are supervised for each record. Routed setup validation uses the
delta head when a current setup exists and the absolute head when the current
setup is missing.

## Stage 6 Offline Evaluation

Model evaluation:

```bash
python -m profile2setup.scripts.evaluate_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_baseline/best.pt \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/model_eval.json
```

Baselines:

```bash
python -m profile2setup.scripts.run_baselines_cli \
  --train profile2setup/data/all_modes/train.jsonl \
  --test profile2setup/data/all_modes/test.jsonl \
  --variables-config profile2setup/configs/variables.yaml \
  --out profile2setup/results/baselines.json
```

Evaluation reports absolute, delta, and routed setup metrics. Routed setup is
the final predicted setup: `current_setup + delta` when current setup is present,
and the absolute prediction when current setup is missing. This stage does not
simulate optical profiles; closed-loop simulation evaluation will come later.

## Stage 7 Closed-Loop Simulation Evaluation

Closed-loop evaluation:

```bash
python -m profile2setup.scripts.closed_loop_eval_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_baseline/best.pt \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/closed_loop.json \
  --simulation-policy target_base \
  --max-examples 100
```

The model predicts setup, the routed prediction is denormalized to physical
units, and the optical simulator generates a predicted beam profile from that
setup. The predicted beam profile is compared with the target `intensity.npy`.

`target_base` is the default because arbitrary paired records may not share
non-controlled simulator context. `current_base` is closer to real control, but
it is only fair when the current and target non-controlled context matches.
