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

Current development should use `profile2setup/llm_api/` and the LLM/API CLI
scripts in `profile2setup/scripts/`. Old module and script paths were moved out
of the active tree.
