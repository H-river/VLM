# Physics-Aware Optics SFT Audit Report

Date: 2026-05-20
Branch: `optics-sft-smoke-test`

## Summary Verdict

PASS WITH MINOR ISSUES.

The physics-aware SFT implementation is structurally complete, runnable in smoke mode, and scientifically meaningful after the small fixes listed below. The main corrected issues were:

- `choose_control_action` could previously return a Jacobian action that made the simulator residual worse. It now scores no-action, Jacobian, grid, and local-refine candidates and returns the lowest simulator residual.
- `build_physics_mixed_sft_dataset.py --image-root-strategy keep_relative` previously produced paths that did not resolve from the mixed dataset `image_root`. It now writes relative paths from the mixed `images/` directory back to source images without copying them.

No real model training was started. No model download was attempted. Only tiny smoke datasets were generated under `../VLM_data/`, and temporary closed-loop images were written under `../VLM_runs/`.

## Commands Run

Static and policy checks:

```bash
git status -sb
find optics_sft ... physics-aware file inventory
python -m compileall optics_sft optical_sim
python -m json.tool optics_sft/data_schema/physics_sft_sample_schema.json
python optics_sft/scripts/smoke_check_metadata_policy.py
python optics_sft/scripts/smoke_check_sim_adapter.py
python optics_sft/scripts/smoke_check_rendering.py --output-dir ../VLM_data/physics_sft_rendering_audit/images
python optics_sft/scripts/smoke_check_physics_prompts.py
```

Tiny dataset generation:

```bash
python optics_sft/scripts/generate_physics_transition_dataset.py --output-dir ../VLM_data/physics_sft_transition_audit --num-samples 8 --seed 1
python optics_sft/scripts/generate_physics_inverse_dataset.py --output-dir ../VLM_data/physics_sft_inverse_audit --num-samples 8 --seed 2
python optics_sft/scripts/generate_physics_counterfactual_dataset.py --output-dir ../VLM_data/physics_sft_counterfactual_audit --num-samples 4 --seed 3
python optics_sft/scripts/generate_physics_trajectory_dataset.py --output-dir ../VLM_data/physics_sft_trajectory_audit --num-trajectories 3 --max-steps 3 --seed 4
```

Mixed/OOD/evaluation checks:

```bash
python optics_sft/scripts/build_physics_mixed_sft_dataset.py --transition-jsonl ../VLM_data/physics_sft_transition_audit/train.jsonl --inverse-jsonl ../VLM_data/physics_sft_inverse_audit/train.jsonl --counterfactual-jsonl ../VLM_data/physics_sft_counterfactual_audit/train.jsonl --trajectory-jsonl ../VLM_data/physics_sft_trajectory_audit/trajectories.jsonl --output-dir ../VLM_data/physics_sft_mixed_audit --image-root-strategy keep_relative --seed 11 --max-train-samples 12
python optics_sft/scripts/make_physics_ood_splits.py --input-jsonl ../VLM_data/physics_sft_transition_audit/train.jsonl --output-dir ../VLM_data/physics_sft_ood_audit --ood-parameter lens_focal_length_mm --train-min 70 --train-max 105 --ood-min 110 --ood-max 135 --seed 13
python optics_sft/scripts/eval_forward_physics.py --test-jsonl ../VLM_data/physics_sft_transition_audit/all.jsonl --predictions-jsonl ../VLM_data/physics_sft_transition_audit/gt_forward_predictions.jsonl --output-json ../VLM_data/physics_sft_transition_audit/eval_forward_gt.json --output-csv ../VLM_data/physics_sft_transition_audit/eval_forward_gt.csv
python optics_sft/scripts/eval_simulator_in_loop.py --test-jsonl ../VLM_data/physics_sft_inverse_audit/all.jsonl --predictions-jsonl ../VLM_data/physics_sft_inverse_audit/gt_sim_predictions.jsonl --output-json ../VLM_data/physics_sft_inverse_audit/eval_sim_gt.json --output-csv ../VLM_data/physics_sft_inverse_audit/eval_sim_gt.csv
python optics_sft/scripts/eval_counterfactual_consistency.py --test-jsonl ../VLM_data/physics_sft_counterfactual_audit/all.jsonl --predictions-jsonl ../VLM_data/physics_sft_counterfactual_audit/gt_counterfactual_predictions.jsonl --output-json ../VLM_data/physics_sft_counterfactual_audit/eval_counterfactual_gt.json --output-csv ../VLM_data/physics_sft_counterfactual_audit/eval_counterfactual_gt.csv
python optics_sft/scripts/eval_closed_loop_control.py --dry-run --test-jsonl ../VLM_data/physics_sft_trajectory_audit/trajectories.jsonl --output-json ../VLM_data/physics_sft_trajectory_audit/eval_closed_loop_dry_run.json --work-dir ../VLM_runs/physics_sft_closed_loop_audit_tmp --max-steps 3 --success-threshold-px 2.0
python optics_sft/scripts/summarize_physics_eval_reports.py --sim-in-loop-report ../VLM_data/physics_sft_inverse_audit/eval_sim_gt.json --forward-report ../VLM_data/physics_sft_transition_audit/eval_forward_gt.json --counterfactual-report ../VLM_data/physics_sft_counterfactual_audit/eval_counterfactual_gt.json --closed-loop-report ../VLM_data/physics_sft_trajectory_audit/eval_closed_loop_dry_run.json --ood-report ../VLM_data/physics_sft_ood_audit/manifest.json --output-dir ../VLM_data/physics_sft_eval_summary_audit
```

Training/inference compatibility checks:

```bash
python - <<'PY'
# Imported train_qwen25vl_qlora row conversion helpers.
# Converted one legacy_pair row and one physics_mixed row without model loading.
# Imported batch_infer_physics_adapter prompt/image helpers and JSON parser without model loading.
PY
```

Final syntax check:

```bash
python -m compileall optics_sft optical_sim
```

`compileall` exited `0`. It prints many lines in this checkout because it traverses local simulator output and environment directories.

## Files Checked

Core schema and policy:

- `optics_sft/data_schema/physics_sft_sample_schema.json`
- `optics_sft/physics/metadata_policy.py`
- `optics_sft/physics/sim_adapter.py`
- `optics_sft/physics/rendering.py`
- `optics_sft/physics/control_search.py`
- `optics_sft/physics/prompt_builder.py`

Prompt templates:

- `optics_sft/prompts/inverse_control.txt`
- `optics_sft/prompts/forward_transition.txt`
- `optics_sft/prompts/counterfactual_pair.txt`
- `optics_sft/prompts/trajectory.txt`

Dataset generation, ingestion, mixing, and OOD tooling:

- `optics_sft/scripts/generate_physics_transition_dataset.py`
- `optics_sft/scripts/generate_physics_inverse_dataset.py`
- `optics_sft/scripts/generate_physics_counterfactual_dataset.py`
- `optics_sft/scripts/generate_physics_trajectory_dataset.py`
- `optics_sft/scripts/ingest_real_transition_dataset.py`
- `optics_sft/scripts/build_physics_mixed_sft_dataset.py`
- `optics_sft/scripts/make_physics_ood_splits.py`

Training, inference, and evaluation:

- `optics_sft/scripts/train_qwen25vl_qlora.py`
- `optics_sft/scripts/batch_infer_physics_adapter.py`
- `optics_sft/scripts/eval_simulator_in_loop.py`
- `optics_sft/scripts/eval_forward_physics.py`
- `optics_sft/scripts/eval_counterfactual_consistency.py`
- `optics_sft/scripts/eval_closed_loop_control.py`
- `optics_sft/scripts/summarize_physics_eval_reports.py`

Configs:

- `optics_sft/configs/qwen25vl_3b_qlora.yaml`
- `optics_sft/configs/qwen25vl_3b_qlora_tiny.yaml`
- `optics_sft/configs/qwen25vl_3b_qlora_physics_mixed.yaml`
- `optics_sft/configs/qwen25vl_3b_qlora_inverse_continuous.yaml`
- `optics_sft/configs/qwen25vl_3b_qlora_forward_only.yaml`
- `optics_sft/configs/qwen25vl_3b_qlora_inverse_forward_mixed.yaml`
- `optics_sft/configs/qwen25vl_3b_qlora_full_physics_mixed.yaml`

## Smoke Datasets Generated

- `../VLM_data/physics_sft_transition_audit`
  - 8 `forward_transition` rows.
  - `train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`, and images exist.
- `../VLM_data/physics_sft_inverse_audit`
  - 8 `inverse_control` rows.
  - `train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`, and images exist.
- `../VLM_data/physics_sft_counterfactual_audit`
  - 4 `counterfactual_pair` rows.
  - `train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`, and images exist.
  - With only 4 rows and default ratios, val/test rounded to zero rows.
- `../VLM_data/physics_sft_trajectory_audit`
  - 3 `trajectory` rows.
  - `trajectories.jsonl`, `manifest.json`, and images exist.
- `../VLM_data/physics_sft_mixed_audit`
  - 12 mixed rows.
  - Type counts: 6 forward, 3 inverse, 2 counterfactual, 1 trajectory.
- `../VLM_data/physics_sft_ood_audit`
  - OOD split smoke for `lens_focal_length_mm`.
  - Counts: 5 train, 0 val, 0 test_id, 1 test_ood.
- `../VLM_data/physics_sft_eval_summary_audit`
  - Combined Markdown/JSON evaluation summary from fake/ground-truth predictions.

## Validation Results

### Schema and JSON

- `physics_sft_sample_schema.json` passes `python -m json.tool`.
- Required top-level keys are present in generated rows: `sample_id`, `sample_type`, `prompt_inputs`, `target`, `private_eval`, `split_tags`.
- The schema enum includes all four sample types: `inverse_control`, `forward_transition`, `counterfactual_pair`, `trajectory`.
- `jsonschema` is not installed in this environment, so embedded example and generated-row schema validation was skipped and documented.

### Metadata Leakage

- `metadata_policy.py` catches forbidden key patterns including `centroid`, `centroid_error`, `residual`, `true_control`, `control_plan`, `label`, `target_state`, `after_state`, `post_action`, `answer`, `ground_truth`, `dx_px`, and `dy_px`.
- Safe prompt rows pass.
- Intentionally leaking rows fail.
- Counterfactual `safe_setup_metadata.scenario_a` and `safe_setup_metadata.scenario_b` wrappers are now supported by the central metadata policy.
- Generated audit rows pass prompt leakage checks.

### Simulator Adapter and Units

- `apply_action_to_setup` deep-copies setup objects and does not mutate the original.
- Lens/camera action deltas are converted from mm to m.
- Safe setup metadata converts m to nm/mm/um as appropriate.
- Pixel conversion uses sensor pitch and resolution.
- `simulate_and_measure` calls `run_simulation` and `compute_metrics`.
- `residual_error_px` computes Euclidean centroid distance in pixels.
- Applying a nonzero action changed the measured centroid state in smoke checks.

### Rendering

- `rendering.py` outputs RGB PIL images/PNGs.
- Normalization, percentile clipping, gamma, background, read noise, blur, saturation, and seed paths ran in smoke mode.
- Smoke images were written outside the repo under `../VLM_data/physics_sft_rendering_audit/images`.

### Prompt Builder

- Prompt builder smoke passed for all four sample types.
- Expected image slots:
  - inverse: current + target
  - forward: before
  - counterfactual: A current + A target + B current + B target
  - trajectory: initial + target, with optional step-image support
- Templates require strict JSON and explicitly avoid hidden reasoning / chain-of-thought.

### Training Compatibility

- Legacy config has no `data.dataset_format`, so the default remains `legacy_pair`.
- Physics configs use `data.dataset_format: physics_mixed`.
- Legacy row conversion still creates two-image user prompts and assistant completions.
- Physics row conversion supports variable image counts and serializes `row["target"]` as strict JSON.
- `completion_only_loss` remains configurable and is enabled in physics configs.
- Physics configs keep `max_length: null`.
- A local model directory exists, but no training smoke was run because the audit explicitly says not to start real model training.

### Inference and Evaluation

- `batch_infer_physics_adapter.py` uses the same prompt builder and expected image-slot logic as training.
- It has robust JSON extraction: exact JSON parse first, then first-object extraction fallback.
- Evaluators ran with ground-truth/fake predictions:
  - Forward GT: all requested MAEs were `0.0`.
  - Simulator-in-loop GT: `mean_error_reduction_ratio=0.8687615887`, `success_rate_under_2px=1.0`, `divergence_rate=0.0`.
  - Counterfactual GT: `should_action_change_accuracy=1.0`, `changed_parameter_accuracy=1.0`, action MAEs `0.0`.
  - Closed-loop dry-run: `success_rate=1.0`, `mean_final_residual_px=0.6431278354`, `invalid_action_rate=0.0`, `divergence_rate=0.0`.

## Issues Found

### Critical

None remaining.

### Major

Resolved:

1. Action search could produce scientifically bad inverse-control labels.
   - Symptom: earlier tiny inverse audit rows had 5/8 searched actions worsen post-action residual.
   - Fix: `choose_control_action` now selects the best simulator-scored candidate from no-action, Jacobian, grid, and local-refine candidates.
   - Post-fix smoke: inverse audit rows improved or held residual in 8/8 cases; simulator-in-loop GT divergence rate was `0.0`.

2. Mixed dataset `keep_relative` paths did not resolve under the configured mixed `image_root`.
   - Symptom: 22 image slots missing when resolving `../VLM_data/physics_sft_mixed_audit/images/<row path>`.
   - Fix: `keep_relative` now rewrites each image path relative to the mixed dataset `images/` directory and creates that directory without copying images.
   - Post-fix smoke: 0 missing image slots; `physics_row_to_sft_example` loaded mixed forward/counterfactual examples successfully.

### Minor

1. `jsonschema` is not installed, so schema validation beyond `json.tool` was skipped.
2. Tiny split sizes can round val/test or test_id to zero rows, especially counterfactual with 4 rows and OOD with 6 input rows. This is expected for tiny audit runs; larger real runs should use enough rows per split.
3. `data/` and `outputs/` still show as untracked placeholder directories because `.gitignore` intentionally allows their `.gitkeep` files. Do not use `git add .`; stage source files explicitly.
4. Real model training and model-backed inference were skipped by instruction.

## Fixes Made

- `.gitignore`
  - Re-added explicit ignore rules for `optical_sim/outputs/` after the root `outputs/` negation.
- `optics_sft/physics/control_search.py`
  - Changed `choose_control_action` to return the best simulator-scored action rather than blindly accepting a well-conditioned Jacobian solve.
- `optics_sft/physics/metadata_policy.py`
  - Added support for counterfactual `scenario_a` / `scenario_b` safe metadata wrappers.
  - De-duplicated leakage error reporting.
- `optics_sft/scripts/smoke_check_metadata_policy.py`
  - Added a counterfactual safe-wrapper smoke check.
- `optics_sft/scripts/generate_physics_counterfactual_dataset.py`
  - Added `--num-samples` as an alias for `--num-pairs` to match audit commands.
- `optics_sft/scripts/build_physics_mixed_sft_dataset.py`
  - Replaced key-only leakage audit with prompt-builder safety audit.
  - Fixed `keep_relative` image path rewriting and ensured the mixed `images/` directory exists.

## Scientific Consistency

The implementation now supports the intended learning objectives:

- `forward_transition`: before image + safe setup metadata + action -> after beam state and state change.
- `inverse_control`: current image + target image + setup metadata -> simulator-searched control plan, not a fixed pixel regression rule.
- `counterfactual_pair`: paired scenarios alter one physical parameter and require comparing resulting control plans.
- `trajectory`: closed-loop observe-act-observe alignment with per-step actions and residuals.

The evaluation stack prioritizes the right metrics:

- simulator post-action error reduction,
- closed-loop success rate,
- OOD split/performance hooks,
- forward-transition prediction error,
- counterfactual consistency,
- action MAE as a secondary diagnostic.

## Remaining TODOs

1. Install `jsonschema` if strict generated-row schema validation is required in this environment.
2. Run larger, statistically meaningful generation only when ready, still under `../VLM_data/`.
3. Build the final mixed dataset with either:
   - `--image-root-strategy keep_relative`, now fixed and no-copy, or
   - `--image-root-strategy symlink` / `copy` if a self-contained image tree is preferred.
4. Run model-backed training/inference only after explicitly deciding to use local model and adapter files.
5. Keep generated datasets, images, adapters, checkpoints, and run outputs out of Git.

## Recommended Next Commands

Stage only source changes:

```bash
git add .gitignore optics_sft
git status --short
git commit -m "Audit and stabilize physics-aware optics SFT tooling"
git push origin optics-sft-smoke-test
```

Optional source-only validation before commit:

```bash
python -m json.tool optics_sft/data_schema/physics_sft_sample_schema.json >/tmp/physics_schema_check.json
python optics_sft/scripts/smoke_check_metadata_policy.py
python optics_sft/scripts/smoke_check_sim_adapter.py
python optics_sft/scripts/smoke_check_physics_prompts.py
python -m compileall optics_sft optical_sim
```
