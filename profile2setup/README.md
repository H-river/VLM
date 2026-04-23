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

Old alignment field names are not v2 output names. Any legacy alignment fields are only backward-compatible input fallback.

## Stage 3 Dataset Loading (Smoke Test)

Run:

```bash
python -m profile2setup.scripts.dataset_smoke_test_cli \
  --jsonl profile2setup/data/train.jsonl \
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
