# Valid-Only Common Evaluation

This is a conditional analysis. It keeps only probe IDs where both base and fine-tuned LLM outputs completed and passed strict schema/canonical-variable validation. Use the raw `budget_512/comparison.md` for primary format/error rates.

- Source probes: `512`
- Common valid probes: `222`

| Metric | Base | Fine-tuned | Local |
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

## Short Read

- Fine-tuned improves direction accuracy on valid common rows.
- Fine-tuned still fails prompt-image conflict handling much more often than base.
- Local PyTorch has high direction accuracy but poor constraint/rejection behavior, consistent with an image-regression baseline rather than language understanding.
