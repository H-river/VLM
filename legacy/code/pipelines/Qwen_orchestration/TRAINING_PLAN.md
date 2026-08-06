# Frozen training plan: qwen_orchestrator_v1

## 1. Starting checkpoint

Start from the frozen promoted adapter:

```text
../VLM_runs/qwen25vl_3b_qlora_visual_tool_orchestration_v10_2_seed49
```

Do not start from the incomplete direction-all-field step-150 checkpoint.

The base model remains:

```text
../HF_models/Qwen2.5-VL-3B-Instruct
```

The base weights are loaded in 4-bit NF4 representation with bfloat16
calculation. The existing rank-8 LoRA adapter is loaded as trainable. No
specialist bundle is updated during orchestration training.

## 2. Fixed trainable parameter scope

The frozen adapter contains LoRA tensors in both:

- the language model;
- the visual encoder.

Both remain trainable so the model can learn natural-language routing and image
role/quality grounding. The base Qwen weights remain frozen.

Before training, print and record:

```text
total base parameters
trainable parameters
trainable percentage
count of language LoRA tensors
count of visual LoRA tensors
```

Expected frozen adapter tensor counts are 504 language tensors and 192 visual
tensors. Any mismatch blocks training.

## 3. Record format

Use the existing `prebuilt_chat` path. Each row contains:

```text
images: zero, one, or two paths
prompt: one user message with matching image placeholders and natural language
completion: one assistant message containing only decision JSON
```

No specialist result appears during stages 1 or 2.

`max_length` remains null so image tokens cannot be truncated. Packing remains
disabled. Loss is computed only on assistant completion tokens.

## 4. Stage 0: untrained baseline

Run the frozen seed-49 adapter on:

- all 1,600 validation records;
- a fixed 280-record diagnostic subset containing 20 records per executable
  route and 70 each for clarification and unsupported.

Record generated, parsed decisions. Do not use teacher-forced loss as the
baseline routing score.

## 5. Stage 1: route and status selection

### Target

The assistant completion contains only:

```json
{
  "schema_version": "qwen_orchestration_route_v1",
  "status": "ready",
  "task_type": "forward_prediction",
  "route_name": "predict_forward_from_state_v1"
}
```

Clarification and unsupported records retain their corresponding null fields.

### Data

Use all 10,000 training records, but remove arguments, image-role mappings, and
clarification wording from the completion.

### Optimization

```text
seed:                            20260724
micro-batch size:                1
gradient accumulation:           8
effective batch size:            8
training records:                10,000
optimizer steps per epoch:       1,250
epochs:                          1
maximum optimizer steps:         1,250
learning rate:                   0.00001
warm-up steps:                   100
optimizer:                       paged_adamw_8bit
precision:                       bfloat16
gradient checkpointing:          enabled
logging interval:                10 steps
checkpoint interval:             250 steps
teacher-forced eval interval:    250 steps
```

Do not apply the old direction-label span weighting. Stage 1 begins with
ordinary completion cross-entropy. A weighting ablation is allowed only after
the unweighted run is fully evaluated.

### Generated validation

At checkpoints 250, 500, 750, 1,000, and 1,250:

1. run deterministic generation on the frozen 280-record diagnostic subset;
2. parse with the stage-1 schema;
3. report status, task, and route confusion matrices.

Run the complete 1,600-record validation only for the two best diagnostic
checkpoints, selected by route exact accuracy subject to schema and
clarification gates.

### Stage-1 gate

```text
schema-valid rate:                    >= 0.99
status exact accuracy:                >= 0.97
ready-route exact accuracy:           >= 0.95
clarification recall:                 >= 0.95
unsupported recall:                   >= 0.95
ready prediction on unsupported rows: <= 0.01
```

If no checkpoint passes, stop. Do not proceed by choosing the lowest loss.

## 6. Stage 2: full canonical call construction

Initialize from the selected Stage-1 checkpoint.

### Target

Use the complete `qwen_orchestration_decision_v1` object:

- status;
- task type;
- route;
- canonical arguments;
- image roles;
- missing fields;
- clarification question.

### Data

Use the same 10,000 rows with full completions. Shuffle by seed while preserving
the frozen row membership. Visual and text rows remain mixed throughout
training.

### Optimization

```text
seed:                            20260724
micro-batch size:                1
gradient accumulation:           4
effective batch size:            4
training records:                10,000
optimizer steps per epoch:       2,500
epochs:                          1
maximum optimizer steps:         2,500
learning rate:                   0.000005
warm-up steps:                   125
optimizer:                       paged_adamw_8bit
precision:                       bfloat16
gradient checkpointing:          enabled
logging interval:                10 steps
checkpoint interval:             250 steps
teacher-forced eval interval:    250 steps
```

Use ordinary completion cross-entropy for the first run. Do not reuse the
current `decrease/no_change/increase` decision weighting because the new
critical spans are route names, statuses, image references, and copied numeric
arguments.

### Generated validation

At every checkpoint:

- diagnostic generation on the fixed 280-record subset;
- strict JSON-schema validation;
- registry validation;
- route exact accuracy;
- argument exact match after canonical float formatting;
- image-role exact accuracy;
- clarification field and question checks;
- illegal-dispatch rate.

Rank checkpoints that satisfy every diagnostic safety gate using the frozen
selection order. Run full validation for the strongest two diagnostic passers.
The final checkpoint must pass every gate again on full validation; diagnostic
metrics alone cannot promote it.

### Stage-2 promotion gate

```text
JSON-schema-valid rate:                 >= 0.99
registry-valid ready calls:             >= 0.99
ready-route exact accuracy:             >= 0.95
required argument-group exact accuracy: >= 0.97
numeric value/unit exact accuracy:      >= 0.95
image-role exact accuracy:              >= 0.98
clarification recall:                   >= 0.95
unsupported recall:                     >= 0.95
illegal specialist execution rate:      0.00
```

Checkpoint selection is lexicographic:

1. zero illegal executions;
2. highest registry-valid rate;
3. highest route exact accuracy;
4. highest argument exact accuracy;
5. earliest checkpoint.

## 7. Stage 3: end-to-end specialist evaluation

Do not train in this stage.

For every validation record:

1. generate the Qwen decision;
2. validate it;
3. execute the frozen specialist when status is ready;
4. compare with an oracle run using the correct frozen route and arguments.

Report:

- orchestration execution success;
- conditional specialist accuracy when routing is correct;
- end-to-end direction macro-F1;
- end-to-end forward strict-all-five success;
- end-to-end inverse target-reached rate;
- visual measurement strict-all-five success;
- difference from the oracle specialist pipeline.

Promotion requires:

```text
successful valid execution:             >= 0.95
end-to-end loss versus oracle pipeline:  <= 0.02 absolute
silent execution with missing inputs:    0
simulator calls during inference:         0
```

## 8. Stage 4: optional grounded response generation

Initially use a deterministic formatter. Train natural-language response
generation only after Stage 3 passes.

If authorized later, build 4,000 training records:

```text
original request + validated call + immutable specialist result
-> concise response that copies all numerical values exactly
```

Use a separate adapter version and separate promotion gates. Routing promotion
must not depend on response-generation quality.

## 9. Test policy

After the complete validation protocol is frozen:

1. evaluate the selected checkpoint once on the 1,600-record IID test;
2. evaluate once on the 1,000-record OOD-language test;
3. evaluate once on the 1,200-record visual-stress test.

No retraining, prompt editing, registry editing, threshold selection, or repair
rule may use test results. A failed test produces a new versioned experiment,
not an edit to the frozen v1 result.

## 10. Output locations

Datasets:

```text
../VLM_data/qwen_orchestration/v1/
```

Training runs:

```text
../VLM_runs/qwen_orchestrator_v1_stage1_seed20260724/
../VLM_runs/qwen_orchestrator_v1_stage2_seed20260724/
```

Evaluation:

```text
Qwen_orchestration/results/v1/
```

Large weights, optimizer states, generated image corpora, and full datasets
remain outside Git.
