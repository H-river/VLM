# Physics Understanding Budget-512 Result Summary

Date: 2026-05-14

This report summarizes the diagnostic physics-understanding evaluation run on
`profile2setup/data/physics_understanding/probes_budget_512.jsonl`.

The goal was to test whether the light-SFT LLM shows optical/physics and
prompt-conditioned reasoning beyond schema formatting or image-to-setup
regression. The comparison includes:

- Base LLM: `gpt-4o-2024-08-06`
- Fine-tuned LLM: `ft:gpt-4o-2024-08-06:personal:profile2setup-mm-sft:DcGtpEep`
- Local PyTorch baseline: `profile2setup/checkpoints/profile2setup_v2_all_modes_b128/best.pt`

## Files

- Raw comparison: `profile2setup/results/physics_understanding/budget_512/comparison.md`
- Raw comparison JSON: `profile2setup/results/physics_understanding/budget_512/comparison.json`
- Valid-only comparison: `profile2setup/results/physics_understanding/budget_512/valid_only_common/comparison_valid_only.md`
- Valid-only comparison JSON: `profile2setup/results/physics_understanding/budget_512/valid_only_common/comparison_valid_only.json`
- Base predictions: `profile2setup/results/physics_understanding/budget_512/predictions_base.jsonl`
- Fine-tuned predictions: `profile2setup/results/physics_understanding/budget_512/predictions_fine_tuned.jsonl`
- Local predictions: `profile2setup/results/physics_understanding/budget_512/predictions_local.jsonl`

## Run Health

The run completed, but the raw LLM outputs include API/network failures and
invalid model outputs. These are part of the raw-model result and should not be
silently removed from the main evaluation.

| Model | Rows | Completed | API/Error Rows | Invalid JSON / Invalid Schema Rows | Not Applicable |
|---|---:|---:|---:|---:|---:|
| Base LLM | 512 | 357 | 102 | 53 | 0 |
| Fine-tuned LLM | 512 | 313 | 107 | 92 | 0 |
| Local PyTorch | 512 | 494 | 0 | 0 | 18 |

Main observed failure modes:

- Base LLM: many `APIConnectionError` / timeout rows, plus invalid valid-output
  responses missing `predicted_delta` or `predicted_setup`.
- Fine-tuned LLM: many `APIConnectionError` / timeout rows, plus more invalid
  valid-output responses than base.
- Local PyTorch: no API errors; prompt-only probes are not applicable because
  the local model requires profile tensor input.

## Raw Evaluation

Raw evaluation keeps API errors, invalid JSON, invalid schema, and valid rows in
the denominator. This is the primary result for raw model behavior.

| Metric | Base LLM | Fine-tuned LLM | Local PyTorch |
|---|---:|---:|---:|
| valid_json_rate | 0.8008 | 0.7910 | 1.0000 |
| schema_valid_rate | 0.6973 | 0.6113 | 1.0000 |
| canonical_variable_rate | 0.8008 | 0.7910 | 1.0000 |
| paraphrase_consistency_score | 0.8889 | 0.7262 | 0.6667 |
| prompt_sensitivity_score | 0.3333 | 0.0000 | 0.0000 |
| fixed_variable_violation_rate | 0.0032 | 0.0015 | 0.7765 |
| allowed_variable_violation_rate | 0.0000 | 0.1545 | 0.9931 |
| contradiction_detection_accuracy | 0.7647 | 0.8353 | 0.0000 |
| prompt_image_conflict_detection_accuracy | 0.8353 | 0.0941 | 0.0000 |
| direction_accuracy | 0.1674 | 0.6554 | 0.6824 |
| forced_prediction_rate | 0.0000 | 0.2588 | 1.0000 |

Raw per-probe highlights:

- Fine-tuned improved `allowed_variable_constraint` success: 0.7176 vs base
  0.2824.
- Fine-tuned improved `fixed_variable_constraint` success: 0.7412 vs base
  0.5412.
- Fine-tuned improved direction accuracy strongly: 0.6554 vs base 0.1674.
- Base was much better at prompt-image conflict detection: 0.8353 vs fine-tuned
  0.0941.
- Base had no forced predictions; fine-tuned forced predictions on 0.2588 of
  invalid/conflict rows.
- Local PyTorch had high direction accuracy but very poor constraint and
  rejection behavior, consistent with an image-regression baseline rather than
  language/prompt understanding.

## Valid-Only Common Evaluation

The valid-only analysis keeps only probe IDs where both base and fine-tuned LLM
completed and passed strict schema plus canonical-variable validation. This is a
conditional diagnostic, not the primary result.

Filtered subset:

- Source probes: 512
- Common valid LLM probes: 222

| Metric | Base LLM | Fine-tuned LLM | Local PyTorch |
|---|---:|---:|---:|
| valid_json_rate | 1.0000 | 1.0000 | 1.0000 |
| schema_valid_rate | 1.0000 | 1.0000 | 1.0000 |
| canonical_variable_rate | 1.0000 | 1.0000 | 1.0000 |
| paraphrase_consistency_score | 1.0000 | 0.8750 | 0.6250 |
| prompt_sensitivity_score | 0.0000 | 0.0000 | 0.0000 |
| fixed_variable_violation_rate | 0.0000 | 0.0000 | 0.7903 |
| allowed_variable_violation_rate | 0.0000 | 0.1250 | 1.0000 |
| contradiction_detection_accuracy | 1.0000 | 1.0000 | 0.0000 |
| prompt_image_conflict_detection_accuracy | 1.0000 | 0.1250 | 0.0000 |
| direction_accuracy | 0.1830 | 0.6429 | 0.6830 |
| forced_prediction_rate | 0.0000 | 0.3824 | 1.0000 |

Valid-only read:

- Fine-tuned still shows much better direction accuracy than base.
- Fine-tuned remains much worse than base on prompt-image conflict rejection.
- Fine-tuned still produces forced predictions on many invalid/conflict rows.
- Local remains direction-strong but constraint/rejection-weak.

## Interpretation

The result is mixed.

Evidence that the fine-tuned LLM learned useful setup/physics patterns:

- Direction prediction improved substantially over the base LLM.
- Constraint-style probe success improved for allowed-variable and
  fixed-variable probes.
- Contradiction detection improved slightly in the raw aggregate.

Evidence against robust prompt-conditioned physics understanding:

- Fine-tuned schema validity is worse than base in the raw evaluation.
- Fine-tuned prompt-image conflict detection is much worse than base.
- Fine-tuned forced-prediction rate is much higher than base.
- Fine-tuned allowed-variable violation rate is higher than base.
- Prompt sensitivity score did not improve.

Evidence that the local PyTorch model is image-dominant:

- Local direction accuracy is high, close to or slightly above fine-tuned.
- Local has near-total failure on rejection, forced-prediction, allowed-variable,
  and fixed-variable constraint metrics.
- Local cannot handle prompt-only probes natively.

## Bottom Line

The light-SFT LLM appears to learn useful setup-direction and constraint
patterns, but it does not yet demonstrate robust prompt-conditioned optical
reasoning. Compared with the base LLM, it is better at predicting directions
and some constrained setup changes, but worse at raw schema reliability,
prompt-image conflict rejection, and avoiding forced predictions.

The current evidence supports:

- Some physics/setup regression improvement from SFT.
- Some constraint-following improvement on valid multimodal rows.
- A serious weakness in conflict/rejection reasoning.
- A meaningful distinction from the local PyTorch baseline: the local model is
  strong on image-driven direction but poor at language constraints and
  rejection.

The safest conclusion is that the fine-tuned model is not merely learning
schema formatting, but it is also not yet a reliable physics/prompt-understanding
model. It likely learned a mixture of setup-direction patterns and structured
response habits, while still over-predicting when it should reject or flag
conflicts.

## Caveats

- API/network failures were common in the raw run and should be reduced or
  retried for a cleaner estimate.
- The valid-only analysis removes important failure modes and should be treated
  as conditional.
- The subset has 512 probes, not the full 2376-probe set.
- `direction_flip_accuracy` was not scored for this subset.
- Post-processing was not used, which is correct for the raw model diagnostic.
