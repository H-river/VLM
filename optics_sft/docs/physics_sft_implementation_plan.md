# Physics-Aware Optics SFT Implementation Plan

This note is the current implementation baseline for physics-aware optics SFT.
It documents what exists today and what should be added next without changing
training behavior.

## Current SFT Workflow

`optics_sft/` is the repo-local SFT area for Qwen2.5-VL optics control
experiments. It is separate from `optical_sim/`, `profile2setup/`, and legacy
`lang2setup` code.

The current workflow is:

1. Prepare raw or simulator-derived examples outside the repo, normally under
   `../VLM_data/optics_sft.../`.
2. Convert already-shaped JSON samples into train/val JSONL with
   `optics_sft/scripts/build_optics_sft_dataset.py`.
3. Train a QLoRA adapter with
   `optics_sft/scripts/train_qwen25vl_qlora.py`.
4. Run single or batch adapter inference with
   `optics_sft/scripts/infer_qwen25vl_adapter.py` or
   `optics_sft/scripts/batch_infer_qwen25vl_adapter.py`.
5. Evaluate continuous JSON control outputs with
   `optics_sft/scripts/eval_sft_outputs.py`, or coarse direction outputs with
   `optics_sft/scripts/eval_direction_classification.py`.
6. Use `optics_sft/scripts/audit_data_leakage.py` to check simple train/val
   overlap, answer-like metadata keys, direct feature metadata, and suspicious
   exact prediction/label matches.

The dataset builder is still conservative. It accepts JSON files that already
look like the SFT row schema, skips incompatible files, splits deterministically,
and writes JSONL. It does not yet convert `optical_sim/outputs/...` records,
render `.npy` intensity arrays into SFT image pairs, or apply JSON Schema
validation.

`optics_sft/scripts/make_tiny_synthetic_dataset.py` creates a small smoke
dataset with synthetic Gaussian beam PNGs and labels. This is useful for
pipeline execution checks, but it is not a physics-faithful training source.

## Current QLoRA Training Script Behavior

`optics_sft/scripts/train_qwen25vl_qlora.py` implements the executable QLoRA
path:

- Loads a YAML config, JSONL train/val rows, and current/target images with PIL.
- Builds each example as two images plus a chat-style user prompt and an
  assistant-only JSON completion.
- Supports prompt modes:
  - `image_only`: no metadata is placed in the prompt.
  - `setup_only`: only setup metadata keys are prompt-visible.
  - `metadata_assisted`: all row metadata is prompt-visible.
- Supports label modes:
  - `continuous_control`: trains on the row `label` object.
  - `direction_classification`: derives a coarse response from
    `label.control_plan`.
- Loads Qwen2.5-VL with 4-bit NF4 quantization, prepares it for k-bit training,
  attaches LoRA to the configured projection/MLP modules, and trains with TRL
  `SFTTrainer`.
- Preserves VLM token integrity by allowing `max_length: null` and disabling
  packing.
- Provides `--smoke-test`, which caps samples and optimizer steps and writes to
  a `_smoke` output directory.

The script does not currently add physics features, simulator feedback,
post-action simulation metrics, or any closed-loop training behavior.

## Current Dataset Schema

`optics_sft/data_schema/sft_sample_schema.json` defines a raw training sample
with:

- `current_image_path`
- `target_image_path`
- `metadata`
- `label`

The required prompt-side metadata fields are wavelength, beam waist, lens focal
length, source-to-lens distance, and lens-to-camera distance. The label contains
a `beam_alignment` task, visual diagnosis, numeric lens/camera actuator deltas,
and confidence.

The schema allows additional metadata fields, which is useful for private
evaluation and debugging but risky for prompt construction. Prompt-visible
metadata must be filtered so it cannot include labels, control deltas, direct
centroid/target features, sample IDs that reveal splits, or other answer-leaking
fields.

There is no separate JSON schema yet for the `direction_classification` response
shape. That response is produced at training time from the continuous-control
label.

## Current Config Limitations

`optics_sft/configs/qwen25vl_3b_qlora.yaml` currently points at the v1 1000-row
dataset and uses:

- `data.prompt_mode: image_only`
- `data.label_mode: direction_classification`
- local base model path `../HF_models/Qwen2.5-VL-3B-Instruct`
- output/run paths under `../VLM_runs/`

This pairing is useful for a low-leakage visual baseline because no metadata is
shown in the prompt and the target is a coarse direction/magnitude task. Its
limitations are:

- It does not teach numeric continuous control deltas.
- It drops all physical setup metadata from the prompt, so the model cannot use
  wavelength, beam waist, focal length, or distances at inference time.
- It only classifies lens x/y intent; camera deltas are ignored by the derived
  direction target.
- Direction and magnitude classes are derived from `label.control_plan` using
  fixed thresholds in the training script, not from a separate schema or
  simulator-verified policy.
- There is no config field for a physics-aware prompt policy, private metadata
  allowlist, simulator-backed labels, or post-action evaluation.

`optics_sft/configs/qwen25vl_3b_qlora_tiny.yaml` is a smoke config and does not
set `prompt_mode` or `label_mode`, so the training script defaults to
`metadata_assisted` and `continuous_control`.

## Current Simulator Capabilities

`optical_sim/` provides the current physics source for laser-lens-camera
examples:

- `optical_sim/src/optical_elements.py` defines `GaussianSource`, `ThinLens`,
  `Sensor`, `Camera`, `Alignment`, and `OpticalSetup`.
- `optical_sim/src/simulator.py` runs Gaussian source generation, propagation
  from source to lens, thin-lens phase/aperture, propagation to camera, and
  sensor-region extraction with camera x/y offsets.
- Propagation backends are `fresnel_numpy`, `angular_spectrum`, and a
  `waveprop` placeholder that falls back to NumPy Fresnel.
- `optical_sim/src/metrics.py` computes centroid, beam widths, ellipticity,
  rotation angle, FWHM estimates, and peak intensity.
- `optical_sim/src/main_generate_dataset.py` supports single, sweep, and random
  generation and writes per-run `metadata.json`, `intensity.npy`, optional
  `beam_profile.png`, plus summary JSONL/CSV.
- `optical_sim/configs/random_config_v2.yaml` samples wavelength, beam waist,
  focal length, aperture, lens x/y, source/lens/camera distances, and camera x/y.
- `optical_sim/scripts/smoke_v2_offsets.py` checks that lens/camera offsets
  affect intensity and that v2 metadata writes camera/lens offsets without an
  `alignment` block.

Important simulator boundaries for SFT:

- `intensity.npy` is the durable numeric output; PNGs are derived renderings.
- v2 metadata exposes lens and camera offsets in setup metadata, but those
  should not automatically become prompt-visible fields.
- Tilt and defocus exist in the legacy alignment object, but the active v2
  random config focuses on canonical source/lens/camera variables.

## Planned Next Files/Scripts

The next physics-aware SFT work should add files in small, reviewable steps:

- `optics_sft/scripts/build_physics_sft_dataset.py`: convert
  `optical_sim/outputs/...` samples into SFT image pairs and JSONL rows, using
  `.npy` intensity arrays as the source artifact and writing large outputs under
  `../VLM_data/`.
- `optics_sft/scripts/render_intensity_images.py`: deterministic renderer from
  `intensity.npy` to prompt-visible images, with no label metadata embedded in
  filenames or pixels.
- `optics_sft/data_schema/physics_sft_sample_schema.json`: schema that
  separates prompt-visible metadata, private evaluation metadata, image paths,
  and labels.
- `optics_sft/configs/qwen25vl_3b_qlora_physics.yaml`: config for
  physics-aware prompt experiments with explicit metadata allowlists and output
  paths under `../VLM_runs/`.
- `optics_sft/scripts/eval_physics_sft_outputs.py`: evaluator that reports JSON
  validity, direction/continuous-control metrics, and optional simulator
  post-action error when a safe control-loop policy is available.
- `optics_sft/scripts/audit_physics_sft_dataset.py`: stronger leakage audit for
  private-vs-prompt metadata, split overlap, label fields, filenames, and exact
  target/control-plan exposure.

Do not modify the QLoRA training behavior until the dataset schema and prompt
metadata boundary are explicit.

## Guardrails

- Do not commit generated datasets, rendered images, checkpoints, adapters, or
  run logs.
- Keep big data under `../VLM_data/`.
- Keep training runs, checkpoints, adapters, and evaluation outputs under
  `../VLM_runs/`.
- Do not expose answer-leaking metadata in prompts.
- Keep private evaluation metadata separate from prompt-visible metadata.
- Use `.npy` intensity arrays as source artifacts and treat PNGs as derived
  prompt/rendering artifacts.
- Keep live training/inference optional; smoke paths should stay lightweight and
  reproducible.
