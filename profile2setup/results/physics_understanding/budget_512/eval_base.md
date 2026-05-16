# Physics Understanding Evaluation

- Predictions: `profile2setup/results/physics_understanding/budget_512/predictions_base.jsonl`
- Probes: `profile2setup/data/physics_understanding/probes_budget_512.jsonl`
- Prediction rows: `512`
- Probe rows: `512`
- Matched probe rows: `512`

## Format

- valid_json_rate: `0.8008`
- schema_valid_rate: `0.6973`
- canonical_variable_rate: `0.8008`
- format denominator: `512`

Interpretation: format metrics only check parseability and schema compliance. They do not prove prompt-conditioned physics understanding.

## Probe Counts

- `allowed_variable_constraint`: `85` probes, success_rate `0.2824`
- `contradiction_detection`: `85` probes, success_rate `0.7647`
- `fixed_variable_constraint`: `85` probes, success_rate `0.5412`
- `paraphrase_consistency`: `85` probes, success_rate `0.0000`
- `physics_causal_question`: `2` probes, success_rate `1.0000`
- `prompt_image_conflict`: `85` probes, success_rate `0.8353`
- `prompt_sensitivity`: `85` probes, success_rate `0.4941`

## Language And Physics Metrics

- paraphrase_consistency_score: `0.8889` (higher is better).
- prompt_sensitivity_score: `0.3333` (higher is better).
- fixed_variable_violation_rate: `0.0032` (lower is better).
- allowed_variable_violation_rate: `0.0000` (lower is better).
- contradiction_detection_accuracy: `0.7647` (higher is better).
- prompt_image_conflict_detection_accuracy: `0.8353` (higher is better).
- direction_accuracy: `0.1674` (higher is better).
- direction_flip_accuracy: not scored for this prediction set.
- forced_prediction_rate: `0.0000` (lower is better).

Interpretation: low fixed/allowed-variable violation rates indicate better constraint following. High contradiction/conflict accuracy indicates the model can reject inconsistent requests rather than forcing a setup. Paraphrase and prompt-sensitivity scores are the main checks for whether wording changes affect outputs in a semantically meaningful way.

## Input Ablation

- `conflict`: rows `85`, success_rate `0.8353`, direction_accuracy `n/a`, forced_prediction_rate `0.0000`
- `images_only`: rows `0`, success_rate `n/a`, direction_accuracy `n/a`, forced_prediction_rate `n/a`
- `prompt_only`: rows `18`, success_rate `0.9444`, direction_accuracy `1.0000`, forced_prediction_rate `0.0000`
- `prompt_plus_images`: rows `409`, success_rate `0.3961`, direction_accuracy `0.1656`, forced_prediction_rate `0.0000`
- `shuffled_prompt`: rows `0`, success_rate `n/a`, direction_accuracy `n/a`, forced_prediction_rate `n/a`

Interpretation: compare prompt_only, images_only, prompt_plus_images, conflict, and shuffled_prompt to separate language use from image-driven behavior. A model that performs similarly with and without prompts is likely relying mostly on images.
