# Stage 1A Local Understanding Proxy

## Purpose

This summarizes non-LLM understanding proxy metrics for the local PyTorch `profile2setup` model. The model does not emit explicit understanding JSON, so changed variables and change directions are derived from its predicted delta head.

## Inputs

- Model eval JSON: `profile2setup/results/model_eval_same50.json`
- Data JSONL: `profile2setup/data/all_modes/test_first50.jsonl`
- Variables config: `profile2setup/configs/variables.yaml`
- Variable order: `source_to_lens, lens_to_camera, focal_length, lens_x, lens_y, camera_x, camera_y`

## Field Mapping

- `predicted_delta`: model_eval.examples[*].predicted_delta_norm, denormalized with variables config
- `routed_prediction`: model_eval.examples[*].predicted_routed_setup_physical
- `target_delta`: data JSONL target_delta when present, else target_setup - current_setup when both are present
- `task_type`: model_eval.examples[*].task_type and data JSONL task_type
- `record_id`: model_eval.examples[*].record_id matched to data JSONL id

## Coverage

- Data records: `50`
- Detailed model examples: `50`
- Records with predicted delta: `50`
- Records with target delta: `20`
- Records evaluated for changed-variable metrics: `20`
- Records evaluated for fixed-constraint metrics: `0`

## Main Metrics

| Metric | Value |
|---|---:|
| changed_variable_precision | 0.928571 |
| changed_variable_recall | 0.936 |
| changed_variable_f1 | 0.932271 |
| change_direction_accuracy | 0.842857 |
| fixed_variable_violation_rate | not available |
| fixed_variable_accuracy | not available |

## Per-Variable Metrics

| Variable | Precision | Recall | F1 | Direction Accuracy |
|---|---:|---:|---:|---:|
| `source_to_lens` | 1 | 0.9 | 0.947368 | 0.8 |
| `lens_to_camera` | 0.9 | 1 | 0.947368 | 0.85 |
| `focal_length` | 1 | 1 | 1 | 1 |
| `lens_x` | 1 | 0.941176 | 0.969697 | 0.95 |
| `lens_y` | 0.722222 | 0.866667 | 0.787879 | 0.55 |
| `camera_x` | 0.944444 | 0.894737 | 0.918919 | 0.85 |
| `camera_y` | 0.9375 | 0.9375 | 0.9375 | 0.9 |

## Notes

- No missing required proxy fields were found for evaluated rows.
- Records without target deltas are excluded from changed-variable and direction metrics because there is no ground-truth change to compare.
- No prompt-level fixed-variable constraints were detected in this subset.
