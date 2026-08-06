# Qwen Orchestration

This directory is the isolated development area for a multimodal Qwen
orchestrator. It does not modify or retrain the frozen specialist models.

The orchestrator has one responsibility:

```text
natural-language request + optional beam images
-> validated registered route + canonical arguments
```

Numerical prediction, image metrology, and inverse action selection remain the
responsibility of the frozen specialist implementations.

## Status

- Baseline freeze: defined in `freeze/baseline_manifest.json`.
- Numerical interfaces: fixed in `NUMERICAL_CONTRACT.md`.
- Route contract: defined in `schemas/orchestration_decision.schema.json`.
- Stage-1 route contract: defined in `schemas/orchestration_route.schema.json`.
- Specialist registry: defined in `configs/model_registry.yaml`.
- Dataset construction: frozen design in `DATASET_PLAN.md`.
- QLoRA training: frozen design in `TRAINING_PLAN.md`.
- Runtime wrappers and strict dispatcher: implemented and regression-tested.
- Dataset: generated and audited at `../VLM_data/qwen_orchestration/v1`.
- Stage 0: frozen starting adapter evaluated and recorded.
- Stage 1: route/status QLoRA training complete.
- Stage 2: canonical-argument QLoRA training and full validation complete.
  Neither of the two full-validation candidates passed every frozen gate, so
  no checkpoint is promoted.
- Stage 3: checkpoint 1000 passed the validation end-to-end task-quality gates,
  but this does not override the failed Stage-2 orchestration gates.
- Sealed tests: not evaluated or used for selection because Stage 2 has no
  promoted checkpoint.
- Final numerical results: see `EVALUATION_REPORT.md`.

## Safety boundary

Qwen proposes a symbolic `route_name`; it never imports a module, supplies a
filesystem path, or executes arbitrary code. A deterministic validator checks
the proposal against the registry before the dispatcher can call a specialist.
Training targets require missing or conflicting inputs to produce
`needs_clarification`. Promotion measures silent execution on missing-input
cases and requires zero observed failures.

The schema and registry validator can prove structural completeness, types,
finite values, image availability, and route compatibility. It cannot prove
that a structurally valid number was copied from the request instead of
hallucinated by Qwen. Exact numerical-copy tests and the missing-input gate are
therefore empirical controls, not a formal grounding guarantee. A
high-assurance deployment would additionally require source-span or
machine-readable input provenance for every numerical field.

## Baseline verification

From `/home/jiamo/VLM`:

```bash
python Qwen_orchestration/scripts/verify_frozen_baseline.py
python Qwen_orchestration/scripts/validate_scaffold.py
```

The verifier reads every frozen file and compares its SHA-256 digest and byte
size with the manifest. The original artifacts remain in their existing
locations and should be treated as read-only.

The local training wrapper verifies the frozen shared trainer before delegating
to it:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python \
  Qwen_orchestration/scripts/train_qwen.py \
  --config Qwen_orchestration/configs/qwen25vl_3b_orchestrator_stage1_v1.yaml
```

Saved adapters can be checked without loading the base model:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python \
  Qwen_orchestration/scripts/audit_adapter.py \
  ../VLM_runs/qwen_orchestrator_v1_stage1_seed20260724/checkpoint-250
```

## Planned execution boundary

```text
Qwen
  -> orchestration decision JSON
  -> schema and registry validator
  -> deterministic dispatcher
  -> frozen specialist or image meter
  -> structured specialist result
  -> deterministic response formatter
```

Natural-language result generation is deliberately deferred until routing,
argument extraction, and specialist execution pass their independent gates.
The current final response is deterministic and copies the structured
specialist result without numerical rounding.

## Runtime request

No checkpoint is currently approved for deployment. The following interface is
an evaluation/development command only until a later checkpoint passes all
frozen Stage-2 and Stage-3 gates:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python \
  Qwen_orchestration/scripts/orchestrate_request.py \
  --config Qwen_orchestration/configs/qwen25vl_3b_orchestrator_stage2_safe_runtime_v1.yaml \
  --adapter-path ../VLM_runs/qwen_orchestrator_v1_stage2_schema_refinement_v2_seed20260724/checkpoint-1000 \
  --request-file request.txt \
  --image current_beam.png
```

Images are assigned stable references in command-line order: the first image is
`image_0`, the second is `image_1`, and so on. The request must state their
roles and include calibration values required by the chosen visual route.
