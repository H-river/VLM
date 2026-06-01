# Forward-Transition Target Modes

Forward-transition rows support two supervised target modes:

- `centroid_only`: current default for control-relevant forward experiments.
- `full_state`: legacy/full optics target for experiments where the prompt provides enough information to infer beam width and raw intensity.

## Why `centroid_only` Exists

The first forward-only adapter learned centroid transition reasonably, but had large `sigma_x_px`, `sigma_y_px`, and `peak_intensity` errors. The audit found no evaluator field bug and no target/private-eval mismatch. The issue was the data contract:

- Numeric before-state sigma is not prompt-visible.
- Rendered PNG width is not a reliable proxy for simulator second-moment sigma.
- Rendered PNGs are independently normalized/clipped/gamma-transformed/noised/blurred.
- Raw simulator `peak_intensity` is not recoverable from those rendered images.

For current control work, centroid movement is the primary useful signal, so `centroid_only` removes weakly observable fields from the supervised target.

## `centroid_only` Target

The model sees:

- `prompt_inputs.images.before_image_path`
- `prompt_inputs.safe_setup_metadata`
- `prompt_inputs.action`

The model is supervised only on:

```json
{
  "task": "forward_centroid_transition",
  "target_mode": "centroid_only",
  "predicted_after_state": {
    "centroid_x_px": 0.0,
    "centroid_y_px": 0.0
  },
  "predicted_change": {
    "delta_centroid_x_px": 0.0,
    "delta_centroid_y_px": 0.0
  }
}
```

Do not include these in the `centroid_only` supervised target:

- `sigma_x_px`
- `sigma_y_px`
- `peak_intensity`

They remain in `private_eval.before_state` and `private_eval.after_state` for diagnostics and future analysis.

## `full_state` Target

`full_state` preserves the older task shape for experiments that can justify it:

```json
{
  "task": "forward_optics_prediction",
  "target_mode": "full_state",
  "predicted_after_state": {
    "centroid_x_px": 0.0,
    "centroid_y_px": 0.0,
    "sigma_x_px": 0.0,
    "sigma_y_px": 0.0,
    "peak_intensity": 0.0
  },
  "predicted_change": {
    "delta_centroid_x_px": 0.0,
    "delta_centroid_y_px": 0.0,
    "sigma_change_px": {"x": 0.0, "y": 0.0},
    "peak_intensity_change": 0.0
  }
}
```

Use `full_state` only if the prompt includes sufficient numeric context or if rendering preserves the physical quantities being predicted.

## Loss Behavior

The SFT trainer serializes `row["target"]` as the assistant JSON completion. In `centroid_only` mode, cross-entropy is applied only to the centroid-only JSON fields because sigma and peak are not present in the target text.

## Generate A Smoke Dataset

```bash
python optics_sft/scripts/generate_physics_transition_dataset.py \
  --output-dir ../VLM_data/physics_sft_forward_centroid_smoke \
  --num-samples 8 \
  --seed 11 \
  --target-mode centroid_only
```

## Inspect Prompt And Target

```bash
python optics_sft/scripts/inspect_physics_sft_examples.py \
  --jsonl ../VLM_data/physics_sft_forward_centroid_smoke/val.jsonl \
  --image-root ../VLM_data/physics_sft_forward_centroid_smoke/images \
  --max-examples 2
```

## Evaluate Centroid-Only Predictions

```bash
python optics_sft/scripts/eval_forward_physics.py \
  --test-jsonl ../VLM_data/physics_sft_forward_centroid_smoke/val.jsonl \
  --predictions-jsonl ../VLM_data/physics_sft_forward_centroid_smoke/fake_gt_predictions.jsonl \
  --output-json ../VLM_data/physics_sft_forward_centroid_smoke/forward_eval_centroid_only.json \
  --metrics-mode centroid_only
```

The evaluator reports centroid and delta-centroid metrics as main metrics in `centroid_only` mode. Sigma and peak can still be used later as private diagnostics, but they are not primary errors for this mode.
