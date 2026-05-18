# Cleanup Report

## Summary

- Created `profile2setup/NEXT_STAGE_PLANNING_SUMMARY.md` for senior-review planning.
- Created `profile2setup/CLEANUP_MANIFEST.md` and `.json` before moving or deleting generated artifacts.
- Archived `30` generated/smoke/dry-run paths into `profile2setup/local_archive/cleanup_20260509/`.
- Deleted `73` cache or reproducible duplicate paths, including the ignored 7.3 GB `profile2setup/data/llm_api_sft/` tree and smoke checkpoint directory.
- Kept final Same-50 and Stage-1 reports, metric JSONs, small benchmark datasets, and the final qualitative images referenced by `qualitative_case_studies.md`.
- No paid API calls were run and no API keys were read, written, or exposed.

## Files Kept

Important kept artifacts:

- `profile2setup/NEXT_STAGE_PLANNING_SUMMARY.md`
- `profile2setup/CLEANUP_MANIFEST.md`
- `profile2setup/CLEANUP_MANIFEST.json`
- `profile2setup/results/comparison_llm_vs_local_same50.md`
- `profile2setup/results/llm_api_predictions/finetuned_50.jsonl`
- `profile2setup/results/llm_api_eval/finetuned_50_eval.json`
- `profile2setup/results/model_eval_same50.json`
- `profile2setup/results/closed_loop_same50.json`
- `profile2setup/results/stage1_understanding/*.md` and final `*.json` reports/metrics
- `profile2setup/results/stage1_understanding/images/finetuned_llm_25/` because it is linked by the qualitative report
- `profile2setup/data/all_modes/test_first50.jsonl`
- `profile2setup/data/stage1_understanding/` benchmark, labels, manifest, and readme
- `profile2setup/results/llm_api_sft_jobs/job_upload_3000.json`
- `profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt` remains local and ignored for git.

## Files Archived

Archive root: `profile2setup/local_archive/cleanup_20260509/`

| Original Path | Archive Path |
|---|---|
| `profile2setup/results/llm_api_predictions/images` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/images` |
| `profile2setup/results/llm_api_predictions/profile_pair_viewer_smoke` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/profile_pair_viewer_smoke` |
| `profile2setup/results/llm_api_render_smoke` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_render_smoke` |
| `profile2setup/results/llm_api_setup_experiment` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_setup_experiment` |
| `profile2setup/results/vlm_sft_smoke` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/vlm_sft_smoke` |
| `profile2setup/results/prediction_examples` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/prediction_examples` |
| `profile2setup/results/prediction_examples_b128` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/prediction_examples_b128` |
| `profile2setup/results/reasoned_vlm_prediction_smoke` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/reasoned_vlm_prediction_smoke` |
| `profile2setup/results/reasoned_vlm_prediction_smoke_stage5` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/reasoned_vlm_prediction_smoke_stage5` |
| `profile2setup/results/reasoning_vlm_render_smoke` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/reasoning_vlm_render_smoke` |
| `profile2setup/results/llm_api_eval/gpt4o_baseline_viz` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_eval/gpt4o_baseline_viz` |
| `profile2setup/results/stage1_understanding/images/base_llm_25` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/stage1_understanding/images/base_llm_25` |
| `profile2setup/results/llm_api_predictions/base.jsonl` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/base.jsonl` |
| `profile2setup/results/llm_api_predictions/gpt4o_baseline.jsonl` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/gpt4o_baseline.jsonl` |
| `profile2setup/results/llm_api_eval/base_eval.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_eval/base_eval.json` |
| `profile2setup/results/llm_api_eval/gpt4o_baseline_eval.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_eval/gpt4o_baseline_eval.json` |
| `profile2setup/results/closed_loop_same10.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/closed_loop_same10.json` |
| `profile2setup/results/model_eval_same10.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/model_eval_same10.json` |
| `profile2setup/results/llm_api_predictions/finetuned_small.jsonl` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/finetuned_small.jsonl` |
| `profile2setup/results/llm_api_eval/finetuned_small_eval.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_eval/finetuned_small_eval.json` |
| `profile2setup/results/closed_loop_smoke.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/closed_loop_smoke.json` |
| `profile2setup/results/baselines_check.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/baselines_check.json` |
| `profile2setup/results/closed_loop_check.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/closed_loop_check.json` |
| `profile2setup/results/model_eval_check.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/model_eval_check.json` |
| `profile2setup/results/llm_api_predictions/base_dry_run.jsonl` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/base_dry_run.jsonl` |
| `profile2setup/results/llm_api_predictions/schema_dry_run.jsonl` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/schema_dry_run.jsonl` |
| `profile2setup/results/llm_api_predictions/visualization_smoke_prediction.jsonl` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_predictions/visualization_smoke_prediction.jsonl` |
| `profile2setup/results/llm_api_eval/base_dry_run_eval.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_eval/base_dry_run_eval.json` |
| `profile2setup/results/llm_api_sft_jobs/job.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_sft_jobs/job.json` |
| `profile2setup/results/llm_api_sft_jobs/job_dry_run.json` | `profile2setup/local_archive/cleanup_20260509/profile2setup/results/llm_api_sft_jobs/job_dry_run.json` |

## Files Deleted

Deleted paths were either cache directories or ignored/generated duplicates with regeneration commands recorded in the manifest and planning summary.

| Path |
|---|
| `profile2setup/data/llm_api_sft` |
| `profile2setup/checkpoints/profile2setup_v2_smoke_pipeline` |
| `profile2setup/__pycache__` |
| `profile2setup/models/__pycache__` |
| `profile2setup/evaluation/__pycache__` |
| `profile2setup/reasoning_vlm/__pycache__` |
| `profile2setup/training/__pycache__` |
| `profile2setup/scripts/__pycache__` |
| `profile2setup/data_prep/__pycache__` |
| `profile2setup/inference/__pycache__` |
| `profile2setup/llm_api/__pycache__` |
| `profile2setup/experiments/__pycache__` |
| `lang2setup/__pycache__` |
| `lang2setup/evaluation/__pycache__` |
| `lang2setup/scripts/__pycache__` |
| `lang2setup/baselines/__pycache__` |
| `lang2setup/llm_interface/__pycache__` |
| `lang2setup/data_prep/__pycache__` |
| `optical_sim/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/models/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/utils/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/vcs/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/locations/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/distributions/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/cli/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/commands/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/index/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/metadata/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/metadata/importlib/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/network/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/operations/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/operations/build/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/operations/install/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/resolution/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/resolution/legacy/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_internal/req/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/cachecontrol/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/cachecontrol/caches/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/idna/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/distro/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/certifi/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/dependency_groups/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/rich/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/packaging/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/packaging/licenses/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/requests/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/resolvelib/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/resolvelib/resolvers/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pygments/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pygments/lexers/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pygments/styles/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pygments/filters/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pygments/formatters/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/truststore/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/platformdirs/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/tomli/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/msgpack/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/distlib/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/tomli_w/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/urllib3/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/urllib3/util/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/urllib3/contrib/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/urllib3/contrib/_securetransport/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/urllib3/packages/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/urllib3/packages/backports/__pycache__` |
| `optical_sim/venv/lib/python3.13/site-packages/pip/_vendor/pkg_resources/__pycache__` |
| `optical_sim/src/__pycache__` |

## Ignored Via .gitignore

Updated `.gitignore` now covers:

- `__pycache__/` and `.pytest_cache/`
- `profile2setup/results/**/images/` with an exception for final qualitative `finetuned_llm_25` images
- `profile2setup/results/**/*smoke*/`
- `profile2setup/results/**/*dry_run*`
- `profile2setup/results/prediction_examples*/`
- `profile2setup/data/llm_api_sft/`, `profile2setup/data/llm_api_sft/images/`, and upload JSONL files
- `profile2setup/local_archive/`
- `profile2setup/checkpoints/` and checkpoint `*.pt` files

## Validation

| Command | Result | Notes |
|---|---|---|
| `python -m compileall profile2setup` | PASS | Full compile pass completed before cleanup report. A quiet repeat with `python -m compileall -q profile2setup` also passed after the integrity-checker patch. |
| `/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.dataset_smoke_test_cli --jsonl profile2setup/data/all_modes/test_first50.jsonl --variables-config profile2setup/configs/variables.yaml --limit 4` | PASS | Loaded 4 rows; shapes and record ids printed as expected. The generated `profile2setup/data/vocab.json` side effect was restored with `git restore`. |
| `/home/jiamo/miniconda3/envs/optical_sim/bin/python -m profile2setup.scripts.check_v2_integrity_cli --root profile2setup --data-dir profile2setup/data --results-dir profile2setup/results` | PASS after narrow fix | Initial run failed because Stage-1 `*_labels.jsonl` sidecars are labels, not dataset records. `check_v2_integrity_cli.py` now skips label JSONL sidecars and then passed with 0 warnings. |

## Remaining Large Files

No files over 5 MB remain under `profile2setup/results/`.

| Path | Size | Why It Remains |
|---|---:|---|
| `profile2setup/data/all_modes/all.jsonl` | 65.1 MB | Core all-modes dataset, required for training/rebuilding subsets and SFT artifacts. |
| `profile2setup/data/all_modes/train.jsonl` | 52.1 MB | Core all-modes dataset, required for training/rebuilding subsets and SFT artifacts. |

## Final Recommended Repo Structure

- Core package/source: `profile2setup/{configs,data_prep,evaluation,inference,llm_api,models,reasoning_vlm,scripts,training}`
- Final small datasets: `profile2setup/data/all_modes/test_first50.jsonl` and `profile2setup/data/stage1_understanding/`
- Final results: `profile2setup/results/comparison_llm_vs_local_same50.md`, `profile2setup/results/llm_api_eval/finetuned_50_eval.json`, `profile2setup/results/llm_api_predictions/finetuned_50.jsonl`, `profile2setup/results/model_eval_same50.json`, `profile2setup/results/closed_loop_same50.json`, and `profile2setup/results/stage1_understanding/`
- Local-only archived clutter: `profile2setup/local_archive/cleanup_20260509/`
- Local-only checkpoint: `profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt`

## Manual Inspect Before Commit

- Review the large list of tracked deletions in `git status`; they are archived under `profile2setup/local_archive/cleanup_20260509/` and intentionally removed from the committed tree if you accept the cleanup.
- Review `.gitignore` exceptions for `profile2setup/results/stage1_understanding/images/finetuned_llm_25/`; keep this exception only if you want qualitative PNGs committed/shared.
- Review `profile2setup/scripts/check_v2_integrity_cli.py`; it now skips `*_labels.jsonl` sidecar files during dataset JSONL validation.
- Confirm whether you want to commit final Stage-1 images. They are useful for the Markdown report but add about 4 MB.

## Next-Stage Planning Summary

- Planning summary path: `profile2setup/NEXT_STAGE_PLANNING_SUMMARY.md`
