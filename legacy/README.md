# Legacy Code

This folder preserves older work that is no longer the main development path.
Files were moved here instead of deleted so old experiments remain inspectable.

- `legacy/lang2setup/`: old text-to-discrete-bin setup prediction work.
- `legacy/lang2setup/optical_sim_scripts/`: old optical-simulator helpers for
  the lang2setup line of work.
- `legacy/reasoning_vlm/`: older/local reasoning VLM helpers and SFT utilities.
- `legacy/profile2setup_scripts/`: old reasoning, stage/debug, and local
  experiment CLI modules moved out of the main script surface.
- `legacy/physics_understanding/`: physics-understanding side-experiment code,
  configs, data, and retained summary/result artifacts.
- `legacy/cleanup_reports/`: historical cleanup manifests and reports from
  earlier cleanup passes.

Current development uses the root-level plan-reasoning/controller packages
listed in the repository `README.md`.

New archive layout:

- `legacy/code/rebuilds/`: superseded v3-v11 rebuild implementations.
- `legacy/code/pipelines/`: older orchestration, profile-to-setup, optics
  understanding, supervisor, and Qwen candidate pipelines.
- `legacy/experiments/`: compact reports grouped by version/family and date.

Generated datasets, checkpoints, images, predictions, and full run trees are
kept locally via `.gitignore`; the experiment archive intentionally contains
only compact, reviewable reports.
