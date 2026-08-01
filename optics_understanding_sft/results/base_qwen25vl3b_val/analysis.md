# Qwen2.5-VL-3B base-model findings

This is the untouched base-model baseline on all 120 validation records after prompt-contract repair. No adapter was loaded, and neither sealed test prompts nor private test labels were used for inference.

## Headline result

- Equal-task macro score: **0.416**
- Strict-JSON validity: **94.2%**
- Full task-schema validity: **58.3%**
- Total generation latency: **389.4 seconds** (3.24 seconds per record on average)
- Peak allocated CUDA memory: **2.41 GiB**

| Task | N | Score |
|---|---:|---:|
| Setup interpretation | 12 | 0.000 |
| Information sufficiency | 12 | 0.450 |
| Causal effects | 18 | 0.644 |
| Forward prediction | 24 | 0.278 |
| Diagnosis | 18 | 0.402 |
| Constrained intervention | 24 | 0.487 |
| Counterfactual reasoning | 12 | 0.648 |

## What failed

- Setup interpretation used plausible natural-language names such as `source`, `lens`, and `camera` instead of the canonical simulator identifiers. It also frequently emitted numbers as strings and made large unit-conversion errors. The zero score therefore reflects both ontology mismatch and arithmetic failure, not an inability to recognize that a source, lens, and camera exist.
- Five forward records and two counterfactual records were invalid JSON. The dominant failure was inserting unresolved arithmetic expressions, such as multiplication or subtraction, where strict JSON numbers were required.
- Forward predictions that parsed still had poor numerical fidelity. The task score was 0.278, and only about 26% of parsed after-state centroid predictions landed within 2 px.
- Diagnosis collapsed all 18 predictions to `ambiguous`, producing status macro-F1 0.205.
- Control had zero full-schema compliance. It often used the unsupported status `feasible_within_limits`, or declared a case feasible without returning a numeric four-actuator plan. Simulator outcome and minimum-motion partial credit still produced a 0.487 task score.

## Visual slice caution

The overall text score was 0.421 and visual score was 0.410, but that comparison is confounded by task mix and only 12 visual records. Within-task visual counts are too small for a conclusion: causal text/visual scores were 0.600/1.000 (2 visual records), diagnosis 0.396/0.417 (5), forward 0.292/0.208 (4), and counterfactual 0.707/0.000 (1). The initial fine-tuning decision should therefore be based on the full macro score and task failures, not a claimed visual advantage or penalty.

## Decision

The baseline leaves clear learning headroom and exposes the intended non-regression weaknesses: sufficiency classification, ambiguity sets, strict unit/ontology interpretation, quantized feasible control, and numerical/counterfactual consistency. The next controlled experiment should fine-tune on training only, select at most one checkpoint using validation macro score, and keep the sealed 240-record test untouched until that choice is frozen.
