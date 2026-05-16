# Next-Stage Planning Summary

Generated for senior review after the Stage-1 LLM/API-vs-local PyTorch comparison. All metrics below are read from saved JSON/Markdown artifacts in this checkout; missing metrics are marked `not found` rather than inferred.

## 1. Executive Summary

- Completed the Same-50 comparison between the fine-tuned multimodal LLM/API model and the LLM-free local PyTorch `profile2setup` checkpoint on identical first-50 test records.
- Completed Stage-1 understanding work: local non-LLM proxy metrics, an 80-row stratified understanding benchmark, constraint/invalid evaluation, base-vs-fine-tuned LLM probe, qualitative case studies, and observed-profile-change metric debugging.
- Main Same-50 result: the local PyTorch model has lower routed setup MAE (`0.0224405`) than the fine-tuned LLM/API model (`0.0399054`), and better normalized profile MSE (`25.0059` vs `48.5623`).
- Direct LLM numerical regression should not be the next primary path. The stronger direction is to use the LLM as a reasoning/intent module and keep numerical prediction/refinement in local PyTorch plus simulator optimization.
- The local PyTorch model is good at direct numerical setup regression on the current distribution, especially relative to the LLM on geometry and profile metrics.
- The fine-tuned LLM/API model is good at producing parseable, canonical structured output and at changed-variable reasoning on natural prompts, but it still struggles with direct numeric setup values and constraint/invalid handling in the Stage-1 probe.

## 2. Models Compared

- Fine-tuned LLM/API model id: `ft:gpt-4o-2024-08-06:personal:profile2setup-mm-sft:DcGtpEep`
- Base model used for SFT: `gpt-4o-2024-08-06`
- Local PyTorch checkpoint: `profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt`
- SFT job metadata kept at: `profile2setup/results/llm_api_sft_jobs/job_upload_3000.json`
- The local checkpoint is the baseline `profile2setup` model from `profile2setup/configs/train.yaml`; that config has no `intent_features` setting, so this comparison treats it as not using intent features.
- Base LLM probe results are available for a 25-row stratified subset in `profile2setup/results/stage1_understanding/base_vs_finetuned_llm_25.json` and `.md`.

## 3. Datasets Used

- Same-50 path: `profile2setup/data/all_modes/test_first50.jsonl`; records: `50`; composition: `{'absolute': 10, 'edit': 20, 'paired_no_setup': 20}`.
- Stage-1 understanding benchmark: `profile2setup/data/stage1_understanding/stage1_understanding_80.jsonl`; composition: `{'absolute': 15, 'ambiguous_multi_intent': 10, 'constraint': 10, 'invalid': 10, 'normal_edit': 20, 'paired_no_setup': 15}`.
- Stage-1 labels: `profile2setup/data/stage1_understanding/stage1_understanding_labels.jsonl`.
- Base LLM probe subset: `profile2setup/data/stage1_understanding/stage1_base_llm_probe_25.jsonl`; labels: `stage1_base_llm_probe_25_labels.jsonl`; target composition was 10 normal, 5 constraint, 5 invalid, 5 ambiguous/multi-intent.
- SFT generated JSONL sizes/counts before cleanup: `{'train.jsonl': 20000, 'train_upload_3000.jsonl': 3000, 'val.jsonl': 2500, 'test.jsonl': 2500}` rows; byte sizes: `{'train.jsonl': 3078776770, 'train_upload_3000.jsonl': 458837182, 'val.jsonl': 380810237, 'test.jsonl': 382681070}`. These are generated base64 multimodal artifacts and can be rebuilt from `profile2setup/data/all_modes/*.jsonl`.
- Stage-1 dataset limitation: `Source test data did not contain explicit constraint, invalid, or ambiguous categories; these rows were synthesized by replacing prompts on sampled edit records.`
- Constraint/invalid/ambiguous examples are prompt variants synthesized on sampled edit records while preserving profile/setup fields. They are useful for instruction-understanding probes but are not a fully independent physical distribution.

## 4. Main Quantitative Results

### LLM/API Format Metrics

| Metric | Same-50 fine-tuned LLM/API | Stage-1 base LLM 25 | Stage-1 fine-tuned LLM 25 |
|---|---:|---:|---:|
| valid_json_rate | 1 | 1 | 1 |
| schema_valid_rate | 0.98 | 0.88 | 0.88 |
| canonical_variable_rate | 1 | 1 | 1 |
| legacy_variable_output_count | 0 | not found | not found |

### Understanding Metrics

| Metric | LLM/API SFT | Local PyTorch | Base LLM probe |
|---|---:|---:|---:|
| changed_variable_precision | 0.892562 | 0.928571 | 0.585366 |
| changed_variable_recall | 0.907563 | 0.936 | 0.338028 |
| changed_variable_f1 | 0.9 | 0.932271 | 0.428571 |
| change_direction_accuracy | 0.756303 | 0.842857 | 0.645714 |
| fixed_variable_violation_rate | 0.705882 | 0.705882 | 0 |
| invalid_rejection_accuracy | 0 | not applicable: no rejection head | 0 |
| observed_profile_change_accuracy_raw | 0 | not applicable | not found |
| observed_profile_change_accuracy_normalized_debug | 0.863248 | not applicable | not found |

### Local Model Understanding Proxy Details

- Same-50 local proxy changed-variable precision/recall/F1: `0.928571` / `0.936` / `0.932271`.
- Same-50 local proxy direction accuracy: `0.842857`.
- Same-50 fixed-variable metrics are `not found` because the first-50 subset has no detected fixed-variable prompt constraints.
- Stage-1 local constraint/invalid metrics: `{'ambiguous_forced_prediction_rate': 1.0, 'ambiguous_forced_predictions': 10, 'ambiguous_total': 10, 'constraint_correct': 0, 'constraint_following_accuracy': 0.0, 'constraint_total': 10, 'fixed_variable_total': 33, 'fixed_variable_violation_rate': 0.7878787878787878, 'fixed_variable_violations': 26, 'invalid_forced_prediction_rate': 1.0, 'invalid_forced_predictions': 10, 'invalid_total': 10, 'native_rejection': 'not_applicable_no_rejection_head'}`.

### Numerical Setup Metrics

| Metric | Fine-tuned LLM/API Same-50 | Local PyTorch Same-50 |
|---|---:|---:|
| routed_setup_mae | 0.0399054 | 0.0224405 |
| predicted_setup_mae | 0.0399054 | 0.0239355 |
| predicted_delta_mae | 0.0439495 | 0.0213986 |

### Per-Variable Routed Setup MAE

| Variable | Fine-tuned LLM/API | Local PyTorch |
|---|---:|---:|
| `source_to_lens` | 0.144002 | 0.0910885 |
| `lens_to_camera` | 0.079161 | 0.0421293 |
| `focal_length` | 0.0389985 | 0.0201249 |
| `lens_x` | 0.00791968 | 0.00131907 |
| `lens_y` | 0.00796251 | 0.00143438 |
| `camera_x` | 0.000746082 | 0.000500192 |
| `camera_y` | 0.00054816 | 0.000487065 |

### Simulator/Profile Metrics

| Metric | Fine-tuned LLM/API Same-50 | Local PyTorch Same-50 |
|---|---:|---:|
| normalized_profile_mse | 48.5623 | 25.0059 |
| centroid_error_px mean x/y | 88.2707 | 88.0754 |
| sigma_error_px mean x/y | 48.4761 | 39.0197 |
| simulator_success_count | 49 | 50 |
| simulator_failure_or_skipped_count | 1 | 0 |

## 5. Qualitative Findings

- Qualitative case-study report: `profile2setup/results/stage1_understanding/qualitative_case_studies.md`; JSON: `qualitative_case_studies.json`.
- Selected cases: `12` from `25` candidates.
- Case types: `{'LLM clearly better on understanding': 2, 'Local clearly better numerically': 2, 'Both correct': 2, 'Both fail': 2, 'Constraint example': 1, 'Invalid example': 1, 'Ambiguous example': 2}`.
- Categories represented: `{'constraint': 3, 'normal_edit': 6, 'invalid': 1, 'ambiguous_multi_intent': 2}`.
- Examples where LLM understanding was better were mostly constraint rows where the LLM captured at least part of the intended direction/fixed-variable semantics better than the local delta proxy, though it still often changed too many variables.
- Examples where the local model was better were primarily numerical: local setup MAE was lower even when both models had mixed qualitative direction correctness.
- Both-fail examples show that prompt-level beam movement intent and physical inverse setup deltas are not equivalent; both models can match some directions while missing others.
- Constraint/invalid examples show the local model has no native rejection mechanism by design, and the LLM probe did not reliably reject invalid/out-of-scope requests.
- Ambiguous examples were included to expose contradiction handling; neither base nor fine-tuned LLM detected contradictions on the 25-row probe.

## 6. Observed-Profile-Change Metric Diagnosis

- Original Same-50 evaluator result: `0`.
- Debug report: `profile2setup/results/stage1_understanding/observed_profile_change_debug.md`.
- Raw exact-string debug count: `0 / 234`.
- Normalized debug count: `202 / 234` = `0.863248`.
- Cause: exact-string/vocabulary mismatch. The LLM predicted `increase/decrease`; evaluator targets use field-specific labels such as `moves_left`, `moves_up`, `increases`, and `decreases`.
- Fix implemented: `profile2setup/evaluation/llm_api_eval.py` has an opt-in `--normalize-observed-profile-labels` flag. Old behavior remains default.
- This metric should not be used as a headline Stage-1 metric until the label vocabulary is standardized. Use the debug-normalized score only as a diagnostic.

## 7. Current Interpretation

- SFT improved structured output: Same-50 valid JSON and canonical variable rates are both 1.0, with schema validity 0.98.
- The LLM is useful for structured intent/changed-variable reasoning, but direct numeric setup regression is weaker than the local PyTorch baseline on Same-50.
- Geometry variables are the main bottleneck for both models. `source_to_lens`, `lens_to_camera`, and `focal_length` dominate routed setup MAE.
- The local model is much stronger on transverse variables than the LLM on Same-50, especially `lens_x` and `lens_y`. `camera_x` and `camera_y` are small in absolute units for both models.
- Simulator/profile metrics can remain poor even when setup MAE improves because the inverse optical mapping is nonlinear and ambiguous; small parameter deviations can move the beam centroid/sigma significantly.
- Invalid/ambiguous instruction handling is not solved. The local model should not be penalized as a rejector because it has no rejection head; the LLM needs better labels/training or prompting for invalid requests.

## 8. Recommended Next-Stage Options

A. Hybrid pipeline: `prompt + profiles -> LLM intent JSON -> local PyTorch model with intent_features -> simulator refinement`.
B. Train the local model with intent features enabled, using ground-truth intent labels from target deltas and/or validated LLM-generated intent JSON.
C. Add simulator-based refinement after local model prediction, especially for `source_to_lens`, `lens_to_camera`, and `focal_length`.
D. Improve dataset and label design for inverse-ambiguous geometry variables; consider multiple valid setup solutions or physics-constrained losses rather than single-point labels only.
E. Reduce paid direct LLM numerical regression runs until an intent-only or hybrid design shows stronger evidence.

## 9. Open Questions For Senior Researchers

- Is the hybrid LLM-intent plus local-regression design scientifically justified for this inverse optical setup task?
- Should the LLM output only changed variables, constraints, rejection status, and high-level beam intent instead of physical numbers?
- Should simulator optimization be the next stage, and should it refine all variables or only the geometry bottleneck variables?
- How should we handle inverse ambiguity in `source_to_lens`, `lens_to_camera`, and `focal_length` when multiple setups can produce similar profiles?
- Is the Stage-1 benchmark sufficient, or do we need more examples with real constraints, invalid requests, and ambiguous prompts?
- Should invalid/out-of-scope handling be an LLM-only responsibility, or should the local pipeline grow a rejection/abstention head?

## 10. Reproducibility

### Key Output Files To Keep

- `profile2setup/results/comparison_llm_vs_local_same50.md`
- `profile2setup/results/llm_api_eval/finetuned_50_eval.json`
- `profile2setup/results/llm_api_predictions/finetuned_50.jsonl`
- `profile2setup/results/model_eval_same50.json`
- `profile2setup/results/closed_loop_same50.json`
- `profile2setup/results/stage1_understanding/*.md` final reports
- `profile2setup/results/stage1_understanding/*.json` final metric/debug/case-study JSONs
- `profile2setup/data/stage1_understanding/*.jsonl`, `*.json`, and `*.md`
- `profile2setup/data/all_modes/test_first50.jsonl`
- `profile2setup/results/llm_api_sft_jobs/job_upload_3000.json`
- `profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt` remains required locally for local-model evaluation; checkpoint files are ignored for git.

### Commands

Same-50 and Stage-1 reproduction commands are already listed in:

- `profile2setup/results/comparison_llm_vs_local_same50.md`
- `profile2setup/results/stage1_understanding/base_vs_finetuned_llm_25.md`
- `profile2setup/results/stage1_understanding/observed_profile_change_debug.md`

Core commands used in the final Same-50 path:

```bash
head -n 50 profile2setup/data/all_modes/test.jsonl > profile2setup/data/all_modes/test_first50.jsonl
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.run_llm_api_inference_cli --model ft:gpt-4o-2024-08-06:personal:profile2setup-mm-sft:DcGtpEep --data profile2setup/data/all_modes/test_first50.jsonl --out profile2setup/results/llm_api_predictions/finetuned_50.jsonl --image-out-dir profile2setup/results/llm_api_predictions/images/finetuned_50 --image-detail low --temperature 0.0
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_llm_api_predictions_cli --predictions profile2setup/results/llm_api_predictions/finetuned_50.jsonl --data profile2setup/data/all_modes/test_first50.jsonl --out profile2setup/results/llm_api_eval/finetuned_50_eval.json --variables-config profile2setup/configs/variables.yaml --run-simulator --max-examples 50
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.evaluate_cli --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt --data profile2setup/data/all_modes/test_first50.jsonl --out profile2setup/results/model_eval_same50.json --variables-config profile2setup/configs/variables.yaml --device auto
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.closed_loop_eval_cli --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt --data profile2setup/data/all_modes/test_first50.jsonl --out profile2setup/results/closed_loop_same50.json --variables-config profile2setup/configs/variables.yaml --max-examples 50 --simulation-policy target_base --device auto --no-strict
```

SFT generated data can be rebuilt if needed:

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.build_llm_api_sft_dataset_cli --split train --include-composite --image-mode base64 --image-detail low
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.build_llm_api_sft_dataset_cli --split val --include-composite --image-mode base64 --image-detail low
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.build_llm_api_sft_dataset_cli --split test --include-composite --image-mode base64 --image-detail low
# Upload-safe subset example used for SFT upload:
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.build_llm_api_sft_dataset_cli --input profile2setup/data/all_modes/train.jsonl --out profile2setup/data/llm_api_sft/train_upload_3000.jsonl --image-out-dir profile2setup/data/llm_api_sft/images/train_upload_3000 --limit 3000 --include-composite --image-mode base64 --image-detail low
```

### Generated Files Safe To Delete Or Archive

- Rendered image folders under `profile2setup/results/**/images/` unless linked by the final qualitative case-study report.
- Smoke/dry-run outputs under `profile2setup/results/*smoke*`, `*dry_run*`, and older `prediction_examples*` directories.
- Huge generated base64 SFT artifacts under `profile2setup/data/llm_api_sft/`; these are reproducible from `profile2setup/data/all_modes/*.jsonl` and should not be committed.
- Cache directories such as `__pycache__/` and `.pytest_cache/`.
