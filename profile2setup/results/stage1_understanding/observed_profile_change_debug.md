# Observed Profile Change Debug

## Purpose

This report debugs why `observed_profile_change_accuracy` was 0.0 for the Same-50 fine-tuned LLM/API evaluation.

## Inputs

- Predictions: `profile2setup/results/llm_api_predictions/finetuned_50.jsonl`
- Data: `profile2setup/data/all_modes/test_first50.jsonl`
- Existing eval JSON: `profile2setup/results/llm_api_eval/finetuned_50_eval.json`

## Metric Implementation

The current evaluator derives target observed-profile labels from `current_metrics` and `target_metrics`, then compares each field with exact string equality. The new evaluator flag `--normalize-observed-profile-labels` preserves old behavior by default and only normalizes labels when explicitly requested.

## Summary

| Metric | Value |
|---|---:|
| existing eval observed_profile_change_accuracy | 0 |
| raw debug accuracy | 0 |
| normalized debug accuracy | 0.863248 |
| raw correct / total | 0 / 234 |
| normalized correct / total | 202 / 234 |
| parseable-row raw correct / total | 0 / 240 |
| parseable-row normalized correct / total | 207 / 240 |
| evaluator-counted rows | 49 |

The headline debug counts mirror the evaluator by requiring schema-valid predictions. The parseable-row counts are included only to show what changes if the single schema-invalid prediction is inspected instead of excluded.

## Cause Counts

| Cause | Count |
|---|---:|
| coordinate_or_direction_mismatch | 10 |
| missing_target_or_prediction_observed_profile_change | 10 |
| semantic_or_model_mismatch | 22 |
| vocabulary_mismatch | 202 |

## Diagnosis

The zero raw accuracy is primarily caused by vocabulary mismatch and exact-string comparison.

observed_profile_change_accuracy should not be used as a headline Stage-1 metric until labels/vocabulary are fixed.

## Value Vocabulary

### Predicted Values

- `centroid_x`: `{'decrease': 23, 'increase': 17}`
- `centroid_y`: `{'increase': 21, 'decrease': 19}`
- `beam_width_x`: `{'increase': 19, 'decrease': 21}`
- `beam_width_y`: `{'decrease': 26, 'increase': 14}`
- `peak_intensity`: `{'decrease': 20, 'increase': 20}`
- `total_intensity`: `{'decrease': 25, 'increase': 15}`

### Target Values

- `centroid_x`: `{'moves_left': 23, 'moves_right': 17}`
- `centroid_y`: `{'moves_up': 21, 'moves_down': 19}`
- `beam_width_x`: `{'increases': 20, 'decreases': 20}`
- `beam_width_y`: `{'decreases': 23, 'increases': 17}`
- `peak_intensity`: `{'decreases': 22, 'increases': 17, 'approximately_unchanged': 1}`
- `total_intensity`: `{'decreases': 25, 'increases': 14, 'approximately_unchanged': 1}`

## Per-Row Debug Table

### abs_rand_01667

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_00148__rand_02659__2749

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_00489__rand_02912__1084

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### edit_rand_02921__rand_02313__2200

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |

### paired_no_setup_rand_01500__rand_02817__5947

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | approximately_unchanged | False | decreases | approximately_unchanged | False | semantic_or_model_mismatch |

### abs_rand_00203

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_03233__rand_01113__9246

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### paired_no_setup_rand_00035__rand_02102__6916

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_left | False | moves_right | moves_left | False | coordinate_or_direction_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### edit_rand_01035__rand_00279__2536

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_02565__rand_00965__588

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### abs_rand_03285

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_02812__rand_02942__9778

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_02310__rand_02037__6658

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |

### edit_rand_00659__rand_03765__1341

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_left | False | moves_right | moves_left | False | coordinate_or_direction_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### paired_no_setup_rand_04068__rand_02147__7800

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### abs_rand_04931

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_01318__rand_04784__893

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |

### paired_no_setup_rand_04614__rand_04468__5590

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_right | False | moves_left | moves_right | False | coordinate_or_direction_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |

### edit_rand_04373__rand_00872__8022

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |

### paired_no_setup_rand_01193__rand_00992__4088

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |

### abs_rand_04816

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_01621__rand_02188__8890

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_04831__rand_02448__2851

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |
| total_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |

### edit_rand_00421__rand_03366__5617

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_right | False | moves_left | moves_right | False | coordinate_or_direction_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |
| total_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |

### paired_no_setup_rand_02363__rand_04807__7633

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### abs_rand_03434

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_00710__rand_00823__634

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |

### paired_no_setup_rand_03584__rand_02343__2442

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |

### edit_rand_04848__rand_00501__3130

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### paired_no_setup_rand_04092__rand_02304__8728

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### abs_rand_00848

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_00680__rand_02343__5728

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_right | False | moves_left | moves_right | False | coordinate_or_direction_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_04725__rand_01250__2870

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### edit_rand_00449__rand_02584__7149

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_01466__rand_04152__9611

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### abs_rand_00759

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_03527__rand_02649__1113

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_left | False | moves_right | moves_left | False | coordinate_or_direction_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_04204__rand_01015__7619

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_right | False | moves_left | moves_right | False | coordinate_or_direction_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |
| peak_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| total_intensity | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |

### edit_rand_01373__rand_03317__5419

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_04835__rand_01654__1188

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### abs_rand_00474

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_01784__rand_00016__4781

- task_type: `edit`
- valid_json: `False`
- schema_valid: `False`
- counted_by_evaluator: `False`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |
| total_intensity | increase | increases | False | increases | increases | True | vocabulary_mismatch |

### paired_no_setup_rand_04381__rand_00048__916

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_left | False | moves_right | moves_left | False | coordinate_or_direction_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | approximately_unchanged | False | increases | approximately_unchanged | False | semantic_or_model_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### edit_rand_02775__rand_03889__5559

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_right | False | moves_left | moves_right | False | coordinate_or_direction_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_04939__rand_01338__3728

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### abs_rand_00819

- task_type: `absolute`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| not available | not available | not available | not available | not available | not available | not available | missing_target_or_prediction_observed_profile_change |

### edit_rand_02290__rand_04434__3811

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_left | False | moves_right | moves_left | False | coordinate_or_direction_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_02520__rand_03608__6284

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | increase | increases | False | increases | increases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### edit_rand_01741__rand_01593__7532

- task_type: `edit`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | decrease | moves_left | False | moves_left | moves_left | True | vocabulary_mismatch |
| centroid_y | decrease | moves_down | False | moves_down | moves_down | True | vocabulary_mismatch |
| beam_width_x | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |

### paired_no_setup_rand_03322__rand_04066__3908

- task_type: `paired_no_setup`
- valid_json: `True`
- schema_valid: `True`
- counted_by_evaluator: `True`

| Field | Predicted | Target | Raw Equal | Normalized Predicted | Normalized Target | Normalized Equal | Cause |
|---|---|---|---:|---|---|---:|---|
| centroid_x | increase | moves_right | False | moves_right | moves_right | True | vocabulary_mismatch |
| centroid_y | increase | moves_up | False | moves_up | moves_up | True | vocabulary_mismatch |
| beam_width_x | decrease | increases | False | decreases | increases | False | semantic_or_model_mismatch |
| beam_width_y | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
| peak_intensity | increase | decreases | False | increases | decreases | False | semantic_or_model_mismatch |
| total_intensity | decrease | decreases | False | decreases | decreases | True | vocabulary_mismatch |
