# Physics Understanding Evaluation

- Predictions: `profile2setup/results/physics_understanding/predictions_local.jsonl`
- Probes: `profile2setup/data/physics_understanding/probes.jsonl`
- Prediction rows: `2376`
- Probe rows: `2376`
- Matched probe rows: `2376`

## Format

- valid_json_rate: `1.0000`
- schema_valid_rate: `1.0000`
- canonical_variable_rate: `1.0000`
- format denominator: `2300`

Interpretation: format metrics only check parseability and schema compliance. They do not prove prompt-conditioned physics understanding.

## Probe Counts

- `allowed_variable_constraint`: `489` probes, success_rate `0.0000`
- `contradiction_detection`: `400` probes, success_rate `0.0000`
- `fixed_variable_constraint`: `489` probes, success_rate `0.0736`
- `paraphrase_consistency`: `344` probes, success_rate `0.0785`
- `physics_causal_question`: `2` probes, success_rate `n/a`
- `prompt_image_conflict`: `163` probes, success_rate `0.0000`
- `prompt_sensitivity`: `489` probes, success_rate `0.0409`

## Language And Physics Metrics

- paraphrase_consistency_score: `0.5959` (higher is better).
- prompt_sensitivity_score: `0.0000` (higher is better).
- fixed_variable_violation_rate: `0.7873` (lower is better).
- allowed_variable_violation_rate: `0.9877` (lower is better).
- contradiction_detection_accuracy: `0.0000` (higher is better).
- prompt_image_conflict_detection_accuracy: `0.0000` (higher is better).
- direction_accuracy: `0.7081` (higher is better).
- direction_flip_accuracy: not scored for this prediction set.
- forced_prediction_rate: `1.0000` (lower is better).

Interpretation: low fixed/allowed-variable violation rates indicate better constraint following. High contradiction/conflict accuracy indicates the model can reject inconsistent requests rather than forcing a setup. Paraphrase and prompt-sensitivity scores are the main checks for whether wording changes affect outputs in a semantically meaningful way.

## Input Ablation

- `conflict`: rows `163`, success_rate `0.0000`, direction_accuracy `n/a`, forced_prediction_rate `1.0000`
- `images_only`: rows `0`, success_rate `n/a`, direction_accuracy `n/a`, forced_prediction_rate `n/a`
- `prompt_only`: rows `76`, success_rate `n/a`, direction_accuracy `n/a`, forced_prediction_rate `n/a`
- `prompt_plus_images`: rows `2137`, success_rate `0.0388`, direction_accuracy `0.7081`, forced_prediction_rate `1.0000`
- `shuffled_prompt`: rows `0`, success_rate `n/a`, direction_accuracy `n/a`, forced_prediction_rate `n/a`

Interpretation: compare prompt_only, images_only, prompt_plus_images, conflict, and shuffled_prompt to separate language use from image-driven behavior. A model that performs similarly with and without prompts is likely relying mostly on images.
