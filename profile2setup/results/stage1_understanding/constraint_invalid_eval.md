# Stage 1C Constraint and Invalid-Request Evaluation

## Purpose

This evaluates constraint following, invalid-request rejection, and ambiguous/contradictory prompt handling on the Stage-1 understanding benchmark.

## Inputs

- Labels: `profile2setup/data/stage1_understanding/stage1_understanding_labels.jsonl`
- LLM predictions: `profile2setup/results/stage1_understanding/finetuned_llm_predictions.jsonl`
- Local model eval: `profile2setup/results/stage1_understanding/local_model_eval.json`
- Variables config: `profile2setup/configs/variables.yaml`

## Counts

- Constraint records: `10`
- Invalid records: `10`
- Ambiguous/multi-intent records: `10`
- LLM prediction rows loaded: `0`
- Local examples loaded: `80`

## Metrics

| Metric | LLM/API | Local PyTorch |
|---|---:|---:|
| constraint_following_accuracy | not available | 0 |
| fixed_variable_violation_rate | not available | 0.787879 |
| invalid_rejection_accuracy | not available | not applicable |
| invalid_forced_prediction_rate | not applicable | 1 |
| contradiction_detection_accuracy | not available | not applicable |
| ambiguous_forced_prediction_rate | not applicable | 1 |

## Interpretation

- The local PyTorch model has no native rejection mechanism by design, so invalid rejection is reported as not applicable rather than a failure.
- Local forced-prediction rates indicate whether the numerical model produced setup deltas for requests that are labeled invalid or contradictory.
- LLM rejection metrics are only available when the LLM predictions JSONL exists and contains rows for these Stage-1 benchmark record IDs.

## Warnings

- LLM predictions file not found: profile2setup/results/stage1_understanding/finetuned_llm_predictions.jsonl
