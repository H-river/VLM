# VLA + Optics: active plan-reasoning controller

This branch contains the active Qwen plan-selection and continuous-control
stack. Historical implementations and compact experiment reports are retained
under `legacy/`; datasets, checkpoints, renders, and raw run artifacts stay
local and are excluded by `.gitignore`.

## Active packages

- `qwen_reasoning_plan_selector_candidate/` — current plan/reasoning selector.
- `continuous_control_v12/` — continuous controller and contracts.
- `vlm_optics_benchmark/` — controller and VLM evaluation harness.
- `active_diagnosis_v13/` — active diagnosis and gain inference.
- `qwen_vl_supervisor_v1/` — Qwen supervisor, schema, and SFT utilities.
- `optical_sim/` — simulator source and configuration.
- `optics_sft/` — simulator adapter used by the active controller.
- `control_rebuild_v5/` and `specialist_rebuild_v2/` — retained runtime
  dependencies of `continuous_control_v12`.
- `reflection_width_relative/` — manifest/data-contract source required by the
  active supervisor; its generated `data/` tree remains local-only.

## Repository layout

- `legacy/code/` — superseded packages, grouped by rebuild or pipeline family.
- `legacy/experiments/` — compact reports grouped by version/family and date.
- `artifacts/`, `runs/`, `data/`, `outputs/`, and checkpoint directories —
  local-only generated content; none is intended for GitHub.

The latest retained overnight summary is
`legacy/experiments/qwen-plan-reasoning/2026-08-06/report.md`.
