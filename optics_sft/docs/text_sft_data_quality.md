# Text-First SFT: Data Quality and Benchmark Workflow

This document explains **what good SFT data means**, **how to judge generated data**, and **how to run the text-first pipeline** in `optics_sft/`.

## Why start with text?

Images add rendering noise, I/O cost, and VLM-specific training complexity. Text rows expose the same physics task using:

- prompt-visible setup metadata
- prompt-visible beam measurements (`text_observations`)
- assistant JSON target (`control_plan`, etc.)

That makes it easier to answer: **did SFT improve the language model on the optics control task?**

## What counts as good data?

Good SFT data is not just “valid JSON”. Use these layers:

| Layer | Question | Tool |
|------|----------|------|
| Schema | Does every row have the required fields? | `audit_sft_dataset_quality.py` |
| Leakage | Does the prompt expose labels or hidden simulator answers? | same report (`leakage_audit`) |
| Split hygiene | Do train/val share sample IDs or near-duplicate setups? | `split_overlap`, `duplicate_setup_stats` |
| Label sanity | Does the labeled action actually improve simulator error? | `label_sanity.label_improvement_rate` |
| Bounds | Are control deltas inside actuator limits? | `label_sanity.control_within_bounds_rate` |
| Coverage | Do errors/actions span small/medium/large regimes? | `distribution_summary` |
| OOD | Can you test outside the training parameter range? | `make_physics_ood_splits.py` via pipeline |

### Default quality gates

The report marks a dataset as passing when:

- schema errors = 0
- leakage failures = 0
- train/val sample_id overlap = 0
- duplicate setup hash rate ≤ 2%
- label improvement rate ≥ 90%
- control within bounds rate ≥ 98%
- confidence in [0, 1] for ≥ 99% of rows

Inspect `reports/text_quality_report.json` for the full breakdown.

## End-to-end pipeline

From repo root:

```bash
python optics_sft/scripts/run_text_sft_pipeline.py \
  --dataset-version text_inverse_v1 \
  --num-samples 300 \
  --seed 42
```

This will:

1. generate inverse-control physics rows with `optical_sim`
2. write PNGs + physics JSONL under `../VLM_data/text_inverse_v1/physics_inverse/`
3. audit physics data quality
4. build ID/OOD splits on focal length
5. export text JSONL to `../VLM_data/text_inverse_v1/text/`
6. audit text data quality
7. write `../VLM_data/text_inverse_v1/manifest.json`

Smoke run:

```bash
python optics_sft/scripts/run_text_sft_pipeline.py \
  --dataset-version text_inverse_smoke \
  --num-samples 30
```

## Text row format

Each text row contains:

```json
{
  "modality": "text",
  "prompt_inputs": {
    "safe_setup_metadata": { "...": "..." },
    "text_observations": {
      "current": {"x_px": 535.4, "y_px": 500.2, "width_x_px": 32.1, "width_y_px": 29.8},
      "target": {"x_px": 512.0, "y_px": 512.0, "width_x_px": 30.0, "width_y_px": 30.5}
    }
  },
  "target_format": "compact",
  "target": { "... control_plan ...": "..." },
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{...json...}"}
  ]
}
```

`text_observations` is the text substitute for image-derived beam measurements. It must not include hidden labels such as `control_plan` or `true_control_plan`.

## Systematic SFT comparison

### 1. Benchmark base model (0-shot LLM)

```bash
python optics_sft/scripts/eval_text_sft_benchmark.py \
  --eval-jsonl ../VLM_data/text_inverse_v1/text/val.jsonl \
  --model-name ../HF_models/Qwen2.5-3B-Instruct \
  --local-files-only \
  --output-report ../VLM_runs/text_inverse_v1/base_benchmark.json
```

### 1b. CatBoost tabular baseline (same val metrics)

Uses `text_observations` + numeric setup metadata → four `CatBoostRegressor`s
(one per `control_plan` field). Install deps once:

```bash
pip install -r optics_sft/requirements-baselines.txt
```

```bash
python optics_sft/scripts/train_eval_catboost_baseline.py \
  --train-jsonl ../VLM_data/text_inverse_v1/text/train.jsonl \
  --eval-jsonl ../VLM_data/text_inverse_v1/text/val.jsonl \
  --model-dir ../VLM_runs/catboost_text_inverse_v1 \
  --output-report ../VLM_runs/text_inverse_v1/catboost_benchmark.json
```

Compare to the LLM base report:

```bash
python optics_sft/scripts/compare_sft_benchmark.py \
  --base-report ../VLM_runs/text_inverse_v1/base_benchmark.json \
  --sft-report ../VLM_runs/text_inverse_v1/catboost_benchmark.json \
  --output-report ../VLM_runs/text_inverse_v1/llm_base_vs_catboost.json
```

(`compare_sft_benchmark.py` treats the second report as the “candidate”; name is historical.)

### 2. Re-export text JSONL (after prompt / target changes)

```bash
for split in train val test; do
  python optics_sft/scripts/build_text_sft_dataset.py \
    --input-jsonl ../VLM_data/text_inverse_v1/physics_inverse/${split}.jsonl \
    --output-jsonl ../VLM_data/text_inverse_v1/text/${split}.jsonl \
    --training-target compact
done
```

Use `--training-target control_plan_only` to supervise only `task` + `control_plan` (shorter assistant JSON).

### 3. Train text QLoRA adapter

Aligned compact prompt + labels (recommended):

```bash
python optics_sft/scripts/train_text_qlora.py \
  --config optics_sft/configs/qwen25_3b_text_qlora_inverse_v3_aligned.yaml
```

Control-plan-only supervision:

```bash
python optics_sft/scripts/train_text_qlora.py \
  --config optics_sft/configs/qwen25_3b_text_qlora_inverse_v3_control_plan.yaml
```

### 4. Benchmark fine-tuned adapter

Match the training target at eval time (`compact` or `control_plan_only`):

```bash
python optics_sft/scripts/eval_text_sft_benchmark.py \
  --eval-jsonl ../VLM_data/text_inverse_v1/text/val.jsonl \
  --model-name ../HF_models/Qwen2.5-3B-Instruct \
  --adapter-path ../VLM_runs/qwen25_3b_text_qlora_inverse_v3_aligned \
  --training-target compact \
  --local-files-only \
  --output-report ../VLM_runs/text_inverse_v1/sft_benchmark_v3_aligned.json
```

### 5. Compare base vs SFT

```bash
python optics_sft/scripts/compare_sft_benchmark.py \
  --base-report ../VLM_runs/text_inverse_v1/base_benchmark.json \
  --sft-report ../VLM_runs/text_inverse_v1/sft_benchmark.json \
  --output-report ../VLM_runs/text_inverse_v1/sft_comparison.json
```

Primary metrics:

- `json_valid_rate`
- `mean_lens_action_mae`
- `overall_lens_sign_accuracy`

For generalization, repeat step 1/3 on `test_id.jsonl` and `test_ood.jsonl`.

## Manual quality audit

```bash
python optics_sft/scripts/audit_sft_dataset_quality.py \
  --train-jsonl ../VLM_data/text_inverse_v1/text/train.jsonl \
  --val-jsonl ../VLM_data/text_inverse_v1/text/val.jsonl \
  --dataset-name text_inverse_v1 \
  --output-report ../VLM_data/text_inverse_v1/reports/manual_quality_report.json \
  --fail-on-gate
```

## Recommended experiment order

1. Run pipeline with `--num-samples 300` and confirm quality gates pass.
2. Benchmark base model on `val` and `test_ood`.
3. Train text QLoRA.
4. Benchmark adapter on the same splits.
5. Compare reports.
6. Only after text SFT shows clear gains, move to image/VLM SFT.
