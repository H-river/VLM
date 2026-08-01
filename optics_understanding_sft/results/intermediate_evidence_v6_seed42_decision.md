# Intermediate-evidence v6 seed-42 decision

## Decision

**No checkpoint is promoted.** The v5A-derived confirmation split, v4 holdout, and original pilot test remain sealed.

## What was completed

- Implemented two simulator-independent deterministic tools for direction thresholding and exhaustive minimum-motion selection.
- Exposed and supervised `observed_direction_set`, `successful_action_indices`, `selected_index`, conflict indices, and best-residual index.
- Built 3,600 records from 1,200 fresh v5A source records: 2,400 train, 600 development, and 600 target-free confirmation records.
- Certified all records with zero reconstruction, tool-execution, interpretation-consistency, split, export, or target-leakage failures.
- Passed the one-step trainer smoke and all 57 project tests.
- Completed the frozen seed-42 QLoRA run at 120 steps, with checkpoints at 60 and 120.

## Training diagnostics

| Checkpoint | Development loss | Mean token accuracy |
|---|---:|---:|
| Step 60 | 0.4664 | 0.8902 |
| Step 120 | 0.4203 | 0.8986 |

The lower loss at step 120 did correspond to better outer JSON behavior, but not to correct tool arguments.

## Gate-impossibility evaluation

The frozen protocol requires 0.98 schema validity over 600 development records, 0.80 exact tool arguments over 200 call-construction records, and 0.90 executed-result agreement over those calls. Evaluation may stop once observed failures make a gate mathematically impossible, because untested rows cannot reverse that result.

### Checkpoint 60

The first 13 control call-construction outputs were all invalid JSON and ran to the 512-token generation cap. Even if every other development output were valid, the maximum possible full-development schema rate would be:

`(600 - 13) / 600 = 0.9783`

This is below the 0.98 gate, so checkpoint 60 is rejected.

### Checkpoint 120

Checkpoint 120 improved the outer response form: 32/41 targeted control calls were valid JSON. However, 0/41 contained the required exact arguments and 0/41 produced the correct deterministic tool result. Typical valid outputs invented a shortcut such as:

```json
{"tool_name":"select_minimum_motion_action","arguments":{"action_trial_index":3}}
```

The registered tool instead requires ordered residuals, ordered actuator motions, the success tolerance, and the declared allowed-value tie order. With 41 confirmed exact-call failures among 200 total call-construction records, even perfect performance on every remaining call could reach only:

`(200 - 41) / 200 = 0.795`

This is below both the 0.80 exact-argument gate and the 0.90 executed-result gate, so checkpoint 120 is rejected.

## Interpretation

The decomposition succeeded as an experimental diagnostic:

- tool routing is easy for the model;
- tool-call serialization improves with training;
- exact extraction and preservation of ordered numerical evidence is still the bottleneck;
- ordinary token-level SFT encourages a plausible compressed shortcut instead of the registered call contract.

This is more informative than the v5A binary-status failure. The model is no longer being asked to perform hidden Fresnel simulation or numerical thresholding, yet it still fails before the tool can run because its constructed inputs are incorrect.

## Recommended next experiment

Do not train longer on the same long literal-array target. Split input construction into compact, independently checkable operations:

1. The LLM selects the tool.
2. The LLM selects source JSON paths and semantic roles, for example the residual field, active-actuator field, threshold field, and allowed-order field.
3. A deterministic adapter resolves those paths into numeric arrays and validates lengths, units, finiteness, and ordering.
4. The deterministic decision tool executes.
5. The LLM interprets the returned intermediate evidence and final physical implication.

This keeps input construction testable while removing error-prone copying of long arrays. A schema-constrained function-call interface should be used during inference. The next dataset should supervise each path/role mapping separately before testing the complete call.
