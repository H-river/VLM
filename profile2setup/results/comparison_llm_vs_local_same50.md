# LLM/API SFT vs Local PyTorch profile2setup: Same-50 Evaluation

## 1. Purpose

This report compares the fine-tuned multimodal LLM/API model against the LLM-free local PyTorch `profile2setup` checkpoint on the same fixed 50 test records. The subset is `profile2setup/data/all_modes/test_first50.jsonl`, created from the first 50 lines of `profile2setup/data/all_modes/test.jsonl`.

The LLM/API inference completed for all 50 records after running with an API key in the user's shell. The evaluator then re-ran with simulator metrics enabled, so the LLM/API metrics below reflect real fine-tuned model predictions.

## 2. Experimental setup

- Dataset path: `profile2setup/data/all_modes/test_first50.jsonl`
- Number of records: `50`
- Fine-tuned LLM/API model id: `ft:gpt-4o-2024-08-06:personal:profile2setup-mm-sft:DcGtpEep`
- Local checkpoint path: `profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt`
- Variables config: `profile2setup/configs/variables.yaml`
- LLM/API predictions: `profile2setup/results/llm_api_predictions/finetuned_50.jsonl`
- LLM/API evaluation JSON: `profile2setup/results/llm_api_eval/finetuned_50_eval.json`
- Local setup evaluation JSON: `profile2setup/results/model_eval_same50.json`
- Local closed-loop JSON: `profile2setup/results/closed_loop_same50.json`
- Simulator policy: `target_base`
- Variable order: `source_to_lens, lens_to_camera, focal_length, lens_x, lens_y, camera_x, camera_y`
- Test subset task counts: `{'absolute': 10, 'edit': 20, 'paired_no_setup': 20}`

## 3. Main result summary

| Metric | LLM/API SFT | Local PyTorch |
|---|---:|---:|
| Routed setup MAE, mean over 7 variables | 0.0399054 | 0.0224405 |
| Predicted setup MAE, mean over 7 variables | 0.0399054 | 0.0239355 |
| Predicted delta MAE, mean over 7 variables | 0.0439495 | 0.0213986 |
| Normalized profile MSE | 48.5623 | 25.0059 |
| Centroid error px, mean x/y | 88.2707 | 88.0754 |
| Sigma error px, mean x/y | 48.4761 | 39.0197 |
| Simulator success count | 49 | 50 |
| Simulator non-success count | 1 | 0 |

The local checkpoint has lower setup MAE and better mean normalized profile MSE on this same-50 subset. The LLM/API model produced valid JSON for every row and canonical variables for every row, but one row failed strict schema validation and one row did not produce simulator metrics.

## 4. LLM/API format and reasoning metrics

| Metric | Value |
|---|---:|
| valid_json_rate | 1.000 |
| schema_valid_rate | 0.980 |
| canonical_variable_rate | 1.000 |
| legacy_variable_output_count | 0 |
| skipped_rows | 0 |
| prediction_rows | 50 |
| API-error rows in prediction JSONL | 0 |
| changed_variable_precision | 0.893 |
| changed_variable_recall | 0.908 |
| changed_variable_f1 | 0.900 |
| change_direction_accuracy | 0.756 |
| observed_profile_change_accuracy | 0.000 |
| fixed_variable_accuracy | not available |

The fine-tuned model is strong on structured output format in this run: all 50 rows are parseable JSON, all 50 use canonical variables, and there is no legacy-variable leakage. The single schema failure is recorded in `profile2setup/results/llm_api_eval/finetuned_50_eval.json`.

## 5. Per-variable setup MAE

| Variable | LLM/API routed setup MAE | Local routed setup MAE |
|---|---:|---:|
| `source_to_lens` | 0.144002 | 0.0910885 |
| `lens_to_camera` | 0.0791610 | 0.0421293 |
| `focal_length` | 0.0389985 | 0.0201249 |
| `lens_x` | 0.00791968 | 0.00131907 |
| `lens_y` | 0.00796251 | 0.00143438 |
| `camera_x` | 0.000746082 | 0.000500192 |
| `camera_y` | 0.000548160 | 0.000487065 |

For both models, the largest absolute setup errors are on `source_to_lens`, `lens_to_camera`, and `focal_length`. The local model is better on all seven routed setup variables in this run. The LLM/API errors for `lens_x` and `lens_y` are especially large relative to their millimeter-scale ranges.

## 6. Simulator/profile comparison

| Metric | LLM/API SFT | Local PyTorch |
|---|---:|---:|
| normalized_profile_mse | 48.5623 | 25.0059 |
| centroid_error_px | 88.2707 | 88.0754 |
| sigma_error_px | 48.4761 | 39.0197 |
| simulator successes | 49 | 50 |
| simulator failures/errors | 1 non-success row; simulator error list length 0 | 0 |

The local model's lower setup MAE does translate into better mean normalized profile MSE and better sigma error. Centroid error is nearly tied: LLM/API has 88.2707 px and local PyTorch has 88.0754 px. The profile metrics show that small or moderate setup differences can still produce large beam-profile differences, especially for difficult records.

## 7. Interpretation

- SFT appears to have improved structured output strongly: valid JSON rate is 1.0, canonical variable rate is 1.0, and schema valid rate is 0.98.
- The fine-tuned LLM/API model does not beat the local PyTorch model as a direct numerical regressor on this same-50 run. Its routed setup MAE is 0.0399054 versus 0.0224405 for the local model.
- The geometry variables `source_to_lens`, `lens_to_camera`, and `focal_length` remain the main large-scale bottleneck for both models, but the LLM/API model also has large transverse lens-offset errors.
- The LLM/API understanding metrics are useful: changed-variable F1 is 0.900 and change-direction accuracy is 0.756. Observed-profile-change accuracy is 0.0 in the evaluator output, so that part is not currently useful as measured.
- The result supports using the LLM as a reasoning or intent module rather than the final numerical regressor. The local PyTorch model remains the stronger numerical baseline on this subset.

## 8. Recommendation

- Do not spend more on direct LLM numerical regression yet; the same-50 result is worse than the local PyTorch baseline on setup MAE and normalized profile MSE.
- Use the fine-tuned LLM primarily for intent extraction, changed-variable reasoning, and structured task understanding.
- Use the local PyTorch model or simulator optimization for numerical prediction and profile matching.
- Consider a hybrid pipeline: `prompt + profiles -> LLM intent JSON -> local model with intent_features -> simulator refinement`.
- Inspect the single schema-invalid LLM/API row before doing larger paid runs, because the format is almost clean but not perfect.

## 9. Reproduction commands

Create the fixed subset:

```bash
head -n 50 profile2setup/data/all_modes/test.jsonl > profile2setup/data/all_modes/test_first50.jsonl
wc -l profile2setup/data/all_modes/test_first50.jsonl
cmp -s <(head -n 50 profile2setup/data/all_modes/test.jsonl) profile2setup/data/all_modes/test_first50.jsonl && echo subset_matches_first50
```

Run LLM/API inference:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.run_llm_api_inference_cli \
  --model ft:gpt-4o-2024-08-06:personal:profile2setup-mm-sft:DcGtpEep \
  --data profile2setup/data/all_modes/test_first50.jsonl \
  --out profile2setup/results/llm_api_predictions/finetuned_50.jsonl \
  --image-out-dir profile2setup/results/llm_api_predictions/images/finetuned_50 \
  --image-detail low \
  --temperature 0.0
```

Evaluate LLM/API predictions:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_llm_api_predictions_cli \
  --predictions profile2setup/results/llm_api_predictions/finetuned_50.jsonl \
  --data profile2setup/data/all_modes/test_first50.jsonl \
  --out profile2setup/results/llm_api_eval/finetuned_50_eval.json \
  --variables-config profile2setup/configs/variables.yaml \
  --run-simulator \
  --max-examples 50
```

Evaluate the local PyTorch checkpoint:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt \
  --data profile2setup/data/all_modes/test_first50.jsonl \
  --out profile2setup/results/model_eval_same50.json \
  --variables-config profile2setup/configs/variables.yaml \
  --device auto
```

Run local closed-loop evaluation:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.closed_loop_eval_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt \
  --data profile2setup/data/all_modes/test_first50.jsonl \
  --out profile2setup/results/closed_loop_same50.json \
  --variables-config profile2setup/configs/variables.yaml \
  --max-examples 50 \
  --simulation-policy target_base \
  --device auto \
  --no-strict
```

## 10. Generated files

- `profile2setup/data/all_modes/test_first50.jsonl`
- `profile2setup/results/llm_api_predictions/finetuned_50.jsonl`
- `profile2setup/results/llm_api_predictions/images/finetuned_50/`
- `profile2setup/results/llm_api_eval/finetuned_50_eval.json`
- `profile2setup/results/model_eval_same50.json`
- `profile2setup/results/closed_loop_same50.json`
- `profile2setup/results/comparison_llm_vs_local_same50.md`
