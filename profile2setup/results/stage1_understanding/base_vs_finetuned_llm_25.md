# Stage 1D Base vs Fine-Tuned LLM Comparison

## Purpose

This compares base `gpt-4o-2024-08-06`, the fine-tuned profile2setup LLM, and the local PyTorch checkpoint on the 25-record Stage 1D probe.

## Inputs

- Labels: `profile2setup/data/stage1_understanding/stage1_base_llm_probe_25_labels.jsonl`
- Base predictions: `profile2setup/results/stage1_understanding/base_llm_predictions_25.jsonl`
- Fine-tuned predictions: `profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl`
- Local eval: `profile2setup/results/stage1_understanding/local_model_eval_25.json`

## Loaded Rows

- Probe labels: `25`
- Base prediction rows: `25`
- Fine-tuned prediction rows: `25`
- Local examples: `25`

## Metrics

| Metric | Base LLM | Fine-tuned LLM | Local PyTorch |
|---|---:|---:|---:|
| valid_json_rate | 1 | 1 | not available |
| schema_valid_rate | 0.88 | 0.88 | not available |
| canonical_variable_rate | 1 | 1 | not available |
| changed_variable_precision | 0.585366 | 0.446541 | 0.424837 |
| changed_variable_recall | 0.338028 | 1 | 0.915493 |
| changed_variable_f1 | 0.428571 | 0.617391 | 0.580357 |
| change_direction_accuracy | 0.645714 | 0.4 | 0.422857 |
| invalid_rejection_accuracy | 0 | 0 | not applicable |
| fixed_variable_violation_rate | 0 | 0.705882 | 0.705882 |
| contradiction_detection_accuracy | 0 | 0 | not applicable |
| invalid_forced_prediction_rate | not applicable | not applicable | 1 |
| ambiguous_forced_prediction_rate | not applicable | not applicable | 1 |

## Interpretation Notes

- LLM metrics are unavailable until the corresponding prediction JSONL exists and contains rows for the 25 probe record IDs.
- The local PyTorch model has no native rejection head, so invalid rejection is not applicable. Forced-prediction rates are reported separately.
- No paid API calls are made by this comparison script.

## Reproduction Commands

Set `OPENAI_API_KEY` in your shell before running the inference commands. These commands use low image detail and temperature 0.0.

Run base LLM inference:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.run_llm_api_inference_cli \
  --model gpt-4o-2024-08-06 \
  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \
  --out profile2setup/results/stage1_understanding/base_llm_predictions_25.jsonl \
  --image-out-dir profile2setup/results/stage1_understanding/images/base_llm_25 \
  --image-detail low \
  --temperature 0.0
```

Run fine-tuned LLM inference on the same 25 records:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.run_llm_api_inference_cli \
  --model ft:gpt-4o-2024-08-06:personal:profile2setup-mm-sft:DcGtpEep \
  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \
  --out profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl \
  --image-out-dir profile2setup/results/stage1_understanding/images/finetuned_llm_25 \
  --image-detail low \
  --temperature 0.0
```

Run the standard LLM/API evaluators where possible:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_llm_api_predictions_cli \
  --predictions profile2setup/results/stage1_understanding/base_llm_predictions_25.jsonl \
  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \
  --out profile2setup/results/stage1_understanding/base_llm_eval_25.json \
  --variables-config profile2setup/configs/variables.yaml \
  --max-examples 25

/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_llm_api_predictions_cli \
  --predictions profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl \
  --data profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl \
  --out profile2setup/results/stage1_understanding/finetuned_llm_eval_25.json \
  --variables-config profile2setup/configs/variables.yaml \
  --max-examples 25
```

Re-run this comparison:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.stage1_base_vs_finetuned_eval_cli \
  --labels profile2setup/data/stage1_understanding/stage1_base_llm_probe_25_labels.jsonl \
  --base-predictions profile2setup/results/stage1_understanding/base_llm_predictions_25.jsonl \
  --finetuned-predictions profile2setup/results/stage1_understanding/finetuned_llm_predictions_25.jsonl \
  --local-model-eval profile2setup/results/stage1_understanding/local_model_eval_25.json \
  --variables-config profile2setup/configs/variables.yaml \
  --out profile2setup/results/stage1_understanding/base_vs_finetuned_llm_25.json \
  --summary-out profile2setup/results/stage1_understanding/base_vs_finetuned_llm_25.md
```

## Warnings

- No loader warnings.
