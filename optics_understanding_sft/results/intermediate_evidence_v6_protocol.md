# Intermediate-evidence v6 protocol

This protocol was frozen after dataset certification and before the full training run. It tests whether the model can use deterministic numerical tools, rather than requiring it to emulate thresholding or exhaustive search internally.

## Capability decomposition

Every v5A source record becomes three independently scored records:

1. `tool_choice`: choose one registered tool for the requested operation.
2. `tool_call_construction`: extract ordered numeric arrays, thresholds, actuator motions, and declared tie-break order from prompt-visible evidence.
3. `tool_result_interpretation`: consume the deterministic result, reproduce the intermediate evidence object, and return the physical status and answer.

The two registered tools are:

- `threshold_completion_directions`, which performs inclusive direction thresholding and returns `per_trial_directions`, `observed_direction_set`, and `conflicting_pair_indices`;
- `select_minimum_motion_action`, which exhaustively thresholds the candidate grid and returns `successful_action_indices`, `selected_index`, and `best_residual_index` using the declared minimum-motion tie-break.

The tools receive only visible measurements. They do not import the simulator, read private fields, or emit the final experiment answer.

## Data

- Source: certified fresh evidence-grounded v5A records.
- Total: 3,600 records derived from 1,200 source records.
- Train/development/confirmation: 2,400/600/600 records.
- Each stage contains exactly 1,200 records overall.
- The 120-step curriculum contains 480 unique records from 40 training scenarios.
- Curriculum stage balance: 160 records per stage.
- Within every stage, control and sufficiency contribute 80 records each.
- Interpretation labels are exactly balanced across feasible, infeasible, answerable, and insufficient statuses.
- Confirmation prompts are target-free and confirmation remains sealed.

The independent audit reconstructs every record from v5A, executes every target tool call, checks final-answer consistency, checks split disjointness and zero-image exports, and reports zero failures.

## Training

- Model: local Qwen2.5-VL-3B-Instruct.
- Fresh rank-8 NF4 QLoRA adapter.
- Seed 42.
- Ordinary completion loss; no decision-token weighting.
- Learning rate 1e-5 with 10% warmup.
- Four-way gradient accumulation and one pass over 480 records.
- Maximum 120 optimizer steps, checkpoints at steps 60 and 120.
- No paid API calls.

## Development gates

All gates must pass on all 600 development records:

- schema validity at least 0.98;
- tool-choice accuracy at least 0.95;
- exact tool-argument construction at least 0.80;
- executed tool-result match at least 0.90;
- intermediate-evidence exact match at least 0.80;
- interpretation status macro-F1 at least 0.75 for both source tasks;
- interpretation answer exact match at least 0.65;
- paired status joint accuracy at least 0.70;
- three-stage end-to-end exact match at least 0.60.

Confirmation may open once for the selected checkpoint only if every development gate passes. V4 holdout and the original pilot test remain sealed.

## Pre-training checks

- Dataset audit: 3,600/3,600 records certified; zero reconstruction, execution, interpretation, or confirmation-leakage failures.
- Regression suite: 57/57 tests pass in the QLoRA environment.
- Trainer integration: one-step text-only QLoRA smoke passed.
- Base-model 12-record diagnostic: tool choice 1.000, exact call arguments 0.000, intermediate evidence 0.000, and end-to-end exact match 0.000. This diagnostic is too small for model selection; it only verifies that the stages expose the intended bottleneck.

## Completed seed-42 outcome

The run completed at 120 steps. Checkpoint 60 was rejected after 13/13 targeted control-call outputs were invalid JSON, making the 0.98 full-development schema gate mathematically impossible. Checkpoint 120 improved outer JSON validity but produced 0/41 exact control argument objects and 0/41 executable-result matches; 41 failures cap the best possible full call accuracy at 0.795. Neither checkpoint was promoted and confirmation remains sealed. See `intermediate_evidence_v6_seed42_decision.md`.
