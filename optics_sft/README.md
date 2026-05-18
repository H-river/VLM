# Optics SFT / QLoRA

This module is the repository-local home for supervised fine-tuning and QLoRA experiments on optics VLM tasks. It is intentionally separate from `optical_sim/`, `profile2setup/`, and `lang2setup/` so those existing pipelines can keep their current imports, scripts, and data contracts.

## Purpose

The current target is to fine-tune `Qwen/Qwen2.5-VL-3B-Instruct` on optical beam-control examples using 4-bit QLoRA with LoRA adapters. Each example should teach the model to compare a current beam image against a target beam image, use optics metadata as context, and emit a strict JSON control plan.

## Expected Workflow

1. Generate or collect raw optics examples from simulator outputs or measured data.
2. Convert those examples into SFT JSONL rows with `scripts/build_optics_sft_dataset.py`.
3. Validate each row against `data_schema/sft_sample_schema.json`.
4. Train a QLoRA adapter with `scripts/train_qwen25vl_qlora.py`.
5. Run adapter inference with `scripts/infer_qwen25vl_adapter.py`.
6. Evaluate predicted JSON plans with `scripts/eval_sft_outputs.py`.

## Data Location

Large datasets should stay outside this Git repo. The default config assumes paths like:

```text
../VLM_data/optics_sft/train.jsonl
../VLM_data/optics_sft/val.jsonl
../VLM_data/optics_sft/images/
```

Small local smoke files can be placed under `data/optics_sft/`, but generated images and JSONL datasets are ignored by Git.

## Output Location

Training outputs, checkpoints, adapter weights, evaluation reports, and logs should also stay outside Git. The default config uses:

```text
../VLM_runs/qwen25vl_3b_qlora/
```

The top-level `outputs/` directory is available for tiny manually curated artifacts only. Generated runs, checkpoints, and model files are ignored.

## Why Large Files Stay Outside Git

SFT datasets, rendered image folders, model checkpoints, adapter weights, and experiment logs grow quickly and can exceed GitHub file limits. Keeping those artifacts in `../VLM_data/` and `../VLM_runs/` keeps the source repo lightweight, reviewable, and safe to push.

## Input Format

Each raw training row should include:

- current beam image path
- target beam image path
- optics metadata such as wavelength, beam waist, focal length, and distances
- label containing diagnosis, actuator deltas, and confidence

See `data_schema/sft_sample_schema.json` for the exact expected structure.

## Output Format

The model completion should be strict JSON. The intended output is a control plan with a diagnosis, actuator deltas, and confidence. Do not train on free-form prose completions when the downstream evaluator expects JSON.

## Current Status

This module is a scaffold. The scripts include safe CLI entry points and TODO sections for connecting the existing simulator outputs and final Qwen2.5-VL training/inference details.
