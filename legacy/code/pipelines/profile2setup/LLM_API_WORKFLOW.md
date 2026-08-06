# LLM/API Optical Setup Understanding Workflow

This workflow tests whether a multimodal LLM API can understand optical setup
changes from profile images and prompts after supervised fine-tuning.

The local PyTorch `profile2setup` model is retained as a baseline. It is not the
main experimental path for this workflow.

## Repository Layout

The LLM/API experiment lives under:

```text
profile2setup/
  llm_api/
    schema.py
    validator.py
    image_rendering.py
    sft_records.py
    client.py
    prompts.py
    inference.py
    sft_jobs.py
  data_prep/build_llm_api_sft_dataset.py
  evaluation/llm_api_eval.py
  experiments/llm_api_setup_understanding.py
  scripts/
    check_v2_integrity_cli.py
    render_llm_api_images_cli.py
    build_llm_api_sft_dataset_cli.py
    audit_llm_sft_data_cli.py
    run_llm_api_inference_cli.py
    create_llm_api_sft_job_cli.py
    check_llm_api_sft_job_cli.py
    evaluate_llm_api_predictions_cli.py
    run_llm_api_setup_experiment_cli.py
```

Older/local reasoning experiments are preserved under `legacy/reasoning_vlm/`.
Current development should stay on `profile2setup/llm_api/` unless deliberately
working on legacy reproduction.

## Workflow Order

Run the pieces in this order for the current main path:

1. `check_v2_integrity_cli`
2. `render_llm_api_images_cli`
3. `build_llm_api_sft_dataset_cli`
4. `audit_llm_sft_data_cli`
5. `create_llm_api_sft_job_cli`
6. `check_llm_api_sft_job_cli`
7. `run_llm_api_inference_cli`
8. `evaluate_llm_api_predictions_cli`

The experiment orchestrator `run_llm_api_setup_experiment_cli` wraps the same
workflow when you want a single command for a controlled run.

## Canonical Output Contract

Use exactly these variables, in order:

```text
source_to_lens
lens_to_camera
focal_length
lens_x
lens_y
camera_x
camera_y
```

The forbidden legacy names are `alignment`, `alignment_x`, and `alignment_y`.
They must not appear in generated assistant JSON, result JSON, checkpoints, or
new model-facing records.

## Dataset Format

Each SFT JSONL line is one object with `messages`.

The active task types are `absolute`, `edit`, and `paired_no_setup`.
`current_only` is not a separate mode; it is merged into `absolute` because both
are profile-only to setup-prediction tasks.

The system message instructs the model to output strict JSON using only the
canonical variables.

The user message contains a text block plus image parts:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "Task: edit\nPrompt: ...\nCanonical variables: ...\nCurrent setup: {...}"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,...",
        "detail": "low"
      }
    }
  ]
}
```

Image inputs are rendered from `intensity.npy`:

- `current_profile.png` when a current profile is available
- `target_profile.png` when a target profile is available
- `difference_profile.png` when both profiles are available
- `composite_profile.png` when both profiles are available and
  `--include-composite` is used

The assistant message content is a JSON string, not a Python dict. It must parse
as JSON and pass `profile2setup.llm_api.validator.validate_llm_output`.

Required assistant fields:

- `valid`
- `task_type`
- `observed_profile_change`
- `setup_understanding.current_setup`
- `setup_understanding.target_setup`
- `setup_understanding.changed_variables`
- `setup_understanding.change_direction`
- `predicted_delta`
- `predicted_setup`
- `confidence`
- `reasoning_summary`
- `rejection_reason`

For records with `current_setup` and `target_delta`, `predicted_delta` is the
physical-unit `target_delta`, and `setup_understanding.changed_variables` comes
from nonzero target-delta values. `change_direction` is `increase`, `decrease`,
or `unchanged` for every canonical variable.

For single-profile `absolute` records, `predicted_setup` is the target setup and
`predicted_delta` is `null`.

## Build Data

Smoke build:

```bash
python -m profile2setup.scripts.build_llm_api_sft_dataset_cli \
  --input profile2setup/data/all_modes/train.jsonl \
  --out profile2setup/data/llm_api_sft/smoke_train.jsonl \
  --image-out-dir profile2setup/data/llm_api_sft/images/smoke_train \
  --limit 5 \
  --image-detail low \
  --image-mode base64 \
  --include-composite \
  --strict
```

Audit a built SFT file before upload:

```bash
python -m profile2setup.scripts.audit_llm_sft_data_cli \
  --jsonl profile2setup/data/llm_api_sft/train.jsonl
```

Train subset:

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

Validation:

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

Test:

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

## Dry-Run API Requests

```bash
python -m profile2setup.scripts.run_llm_api_inference_cli \
  --model gpt-4.1-mini \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/llm_api_predictions/base_dry_run.jsonl \
  --image-out-dir profile2setup/results/llm_api_predictions/images/base \
  --limit 5 \
  --dry-run
```

`--dry-run` builds request payloads but does not call the API.

## SFT Job Dry Run

```bash
python -m profile2setup.scripts.create_llm_api_sft_job_cli \
  --base-model gpt-4.1-mini \
  --train-jsonl profile2setup/data/llm_api_sft/train.jsonl \
  --val-jsonl profile2setup/data/llm_api_sft/val.jsonl \
  --out profile2setup/results/llm_api_sft_jobs/job.json \
  --dry-run
```

For real API use, configure credentials through environment variables such as
`OPENAI_API_KEY`. Never commit API keys.

Check a saved job metadata file:

```bash
python -m profile2setup.scripts.check_llm_api_sft_job_cli \
  --job-metadata profile2setup/results/llm_api_sft_jobs/job.json
```

## Evaluate Predictions

```bash
python -m profile2setup.scripts.evaluate_llm_api_predictions_cli \
  --predictions profile2setup/results/llm_api_predictions/base.jsonl \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/llm_api_eval/base_eval.json \
  --variables-config profile2setup/configs/variables.yaml
```

Add `--run-simulator` only when simulator-backed profile agreement is intended.

## Local Baseline

The local PyTorch path is retained only as a baseline/local-model comparison
surface. It is not the current main method.

```bash
python -m profile2setup.scripts.evaluate_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt \
  --data profile2setup/data/all_modes/test.jsonl \
  --out profile2setup/results/profile2setup_baseline_eval.json
```

Compare the API model and local baseline on routed setup error, per-variable
MAE, JSON validity, setup-understanding metrics, and optional simulator
agreement.
