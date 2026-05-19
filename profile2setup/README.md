# profile2setup

`profile2setup` is centered on the multimodal LLM/API + SFT optical setup
understanding workflow.

The main question is whether a multimodal LLM API, after supervised fine-tuning
(SFT), can read a user prompt plus beam profile images and predict a strict JSON
optical setup response. The existing local PyTorch `profile2setup` model remains
in this repo as a baseline for comparison.

Full LLM/API workflow documentation is in [LLM_API_WORKFLOW.md](LLM_API_WORKFLOW.md).
The older local model workflow is still available in [WORKFLOW.md](WORKFLOW.md)
and experiment tracking notes are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Canonical Variables

All dataset records, model outputs, API prompts, and evaluation files must use
exactly these variables in this order:

1. `source_to_lens`
2. `lens_to_camera`
3. `focal_length`
4. `lens_x`
5. `lens_y`
6. `camera_x`
7. `camera_y`

Do not introduce or accept legacy setup names such as `alignment`,
`alignment_x`, or `alignment_y`.

## Current Main Workflow

The main workflow is:

1. Check repository/data integrity.
2. Render LLM/API profile images from `intensity.npy`.
3. Build multimodal SFT JSONL data.
4. Audit the SFT data.
5. Create and check an SFT job.
6. Run LLM/API inference.
7. Evaluate predictions.

The core current code lives in:

- `profile2setup/llm_api/`
- `profile2setup/data_prep/build_llm_api_sft_dataset.py`
- `profile2setup/evaluation/llm_api_eval.py`
- `profile2setup/evaluation/llm_api_visualization.py`
- `profile2setup/experiments/llm_api_setup_understanding.py`
- the LLM/API CLI modules in `profile2setup/scripts/`

## LLM/API Pipeline

High-level flow:

```text
existing optical_sim/profile2setup records
-> render intensity.npy into current/target/difference/composite PNGs
-> build multimodal SFT JSONL
-> run base multimodal LLM API inference
-> fine-tune multimodal LLM API
-> run fine-tuned model inference
-> evaluate JSON validity, setup understanding, numerical setup prediction, and simulator agreement
-> compare against the local profile2setup PyTorch baseline
```

The API dataset input contains:

- user prompt
- current profile image when available
- target profile image when available
- target-current difference and composite image when both profiles are available
- current setup table when available

Active input modes are `absolute`, `edit`, and `paired_no_setup`. The old
`current_only` mode is merged into `absolute` because both are profile-only to
setup-prediction tasks.

The assistant target is a JSON string with:

- `valid`
- `task_type`
- `observed_profile_change`
- `setup_understanding`
- `predicted_delta`
- `predicted_setup`
- `confidence`
- `reasoning_summary`
- `rejection_reason`

## Quick Checks

Run the v2 integrity check:

```bash
python -m profile2setup.scripts.check_v2_integrity_cli \
  --root profile2setup \
  --data-dir profile2setup/data \
  --results-dir profile2setup/results
```

Check LLM/API imports:

```bash
python - <<'PY'
from profile2setup.llm_api import schema, validator, image_rendering, sft_records
print("llm_api imports OK")
PY
```

Audit an SFT JSONL file:

```bash
python -m profile2setup.scripts.audit_llm_sft_data_cli \
  --jsonl profile2setup/data/llm_api_sft/train.jsonl
```

## Build LLM/API SFT Data

Small train build:

```bash
python -m profile2setup.scripts.build_llm_api_sft_dataset_cli \
  --input profile2setup/data/all_modes/train.jsonl \
  --out profile2setup/data/llm_api_sft/train.jsonl \
  --image-out-dir profile2setup/data/llm_api_sft/images/train \
  --limit 100 \
  --image-detail low \
  --image-mode base64 \
  --include-composite \
  --strict
```

Validation split:

```bash
python -m profile2setup.scripts.build_llm_api_sft_dataset_cli \
  --input profile2setup/data/all_modes/val.jsonl \
  --out profile2setup/data/llm_api_sft/val.jsonl \
  --image-out-dir profile2setup/data/llm_api_sft/images/val \
  --image-detail low \
  --image-mode base64 \
  --include-composite \
  --strict
```

Test split:

```bash
python -m profile2setup.scripts.build_llm_api_sft_dataset_cli \
  --input profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/data/llm_api_sft/test.jsonl \
  --image-out-dir profile2setup/data/llm_api_sft/images/test \
  --image-detail low \
  --image-mode base64 \
  --include-composite \
  --strict
```

Render one current/target pair directly:

```bash
python -m profile2setup.scripts.render_llm_api_images_cli \
  --current-profile optical_sim/outputs/random_v2/rand_01846/intensity.npy \
  --target-profile optical_sim/outputs/random_v2/rand_02006/intensity.npy \
  --out-dir profile2setup/results/llm_api_render_smoke
```

## API Dry Runs

Build API request payloads without calling the provider:

```bash
python -m profile2setup.scripts.run_llm_api_inference_cli \
  --model gpt-4.1-mini \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/llm_api_predictions/base_dry_run.jsonl \
  --image-out-dir profile2setup/results/llm_api_predictions/images/base \
  --limit 5 \
  --dry-run
```

Create an SFT job dry run:

```bash
python -m profile2setup.scripts.create_llm_api_sft_job_cli \
  --base-model gpt-4.1-mini \
  --train-jsonl profile2setup/data/llm_api_sft/train.jsonl \
  --val-jsonl profile2setup/data/llm_api_sft/val.jsonl \
  --out profile2setup/results/llm_api_sft_jobs/job.json \
  --dry-run
```

Check saved SFT job metadata:

```bash
python -m profile2setup.scripts.check_llm_api_sft_job_cli \
  --job-metadata profile2setup/results/llm_api_sft_jobs/job.json
```

Evaluate predictions:

```bash
python -m profile2setup.scripts.evaluate_llm_api_predictions_cli \
  --predictions profile2setup/results/llm_api_predictions/base.jsonl \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/llm_api_eval/base_eval.json \
  --variables-config profile2setup/configs/variables.yaml
```

API keys must come from environment variables such as `OPENAI_API_KEY`. Do not
write API keys into source files, JSONL data, job metadata, notebooks, or docs.

## Local Baseline

The local PyTorch model is retained as a baseline/local-model tool, not the
current main method:

```bash
python -m profile2setup.scripts.evaluate_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/model_eval.json
```

Profiles for local training and LLM/API rendering are loaded from
`intensity.npy`; do not use `beam_profile.png` as the training image source.

Legacy text-to-discrete-bin, reasoning VLM, one-off stage/debug, and
physics-understanding side-experiment files are under `legacy/`.
