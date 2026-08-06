# Implementation plan

## Objective

Build one multimodal Qwen orchestration layer that accepts natural language and
zero, one, or two beam images, then emits a validated call to a frozen
specialist.

## Phase 0: frozen baseline

Deliverables:

- checksum manifest for all baseline weights, specialist bundles, datasets,
  evaluation summaries, and critical source files;
- baseline verification script;
- versioned model registry;
- strict orchestration JSON schema.

Exit condition:

```text
verify_frozen_baseline.py reports zero missing, size-mismatched, or
digest-mismatched artifacts.
```

## Phase 1: callable specialist adapters

Implement read-only wrappers with one interface:

```python
run(arguments: dict, image_bindings: dict) -> dict
```

Required wrappers:

1. `measure_beam_profile_v1`
2. `predict_direction_from_state_v1`
3. `predict_direction_from_image_v1`
4. `predict_forward_from_state_v1`
5. `predict_forward_from_image_v1`
6. `select_inverse_action_from_states_v1`
7. `select_inverse_action_from_images_v1`

The image routes call the deterministic image meter before the numerical
specialist. Qwen never generates centroid, width, or peak values for these
routes.

Exit conditions:

- every wrapper validates its input independently;
- identical frozen inputs reproduce the frozen specialist outputs;
- no wrapper imports the optical simulator during inference;
- route unit tests cover valid, missing, extra, and non-finite arguments.

## Phase 2: deterministic oracle orchestrator

Implement the registry validator and dispatcher before using Qwen.

The oracle orchestrator receives a correct decision object and verifies:

- exact schema;
- registered route;
- required and forbidden fields;
- finite numerical values;
- declared units;
- image reference existence and role uniqueness;
- route-specific image count;
- calibration presence for quantitative image routes.

Exit condition:

```text
100% execution success on valid canonical calls and 100% rejection of the
frozen invalid-call suite.
```

## Phase 3: dataset construction

Build the frozen dataset design in `DATASET_PLAN.md`. Generate canonical source
cases first, assign physical groups to splits, and only then create language
variants and image perturbations.

Exit conditions:

- exact requested split counts;
- no physical-group, base-image, or paraphrase-family overlap;
- exact route/status balance;
- deterministic replay reconstructs every target decision;
- all image placeholders resolve and image roles match the target;
- no private simulator result is visible to Qwen.

## Phase 4: Qwen router training

Run the staged protocol in `TRAINING_PLAN.md`:

1. route/status selection;
2. full canonical argument construction;
3. optional specialist-result interpretation after routing promotion.

Exit condition:

The selected checkpoint passes all routing, clarification, image-role, schema,
and end-to-end gates. Teacher-forced loss is diagnostic only and cannot promote
a checkpoint.

## Phase 5: end-to-end runtime

Runtime sequence:

```text
request
-> Qwen decision
-> strict validation
-> at most one schema-repair retry
-> specialist execution or clarification
-> deterministic final response
```

Failure policy:

- malformed decision after one retry: return an orchestration error;
- missing input: return Qwen's validated clarification question;
- unsupported request: return the supported capability list;
- specialist failure: expose the typed failure without selecting another route;
- never default to the most common route.

## Phase 6: controlled evaluation

Evaluate in this order:

1. development subset;
2. complete validation split;
3. IID sealed test;
4. OOD-language test;
5. visual-stress test;
6. real-image evaluation, if real data becomes available.

No test split may be opened to select checkpoints, prompts, thresholds, or
repair rules.
