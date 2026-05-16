# Physics Understanding Comparison

## Metric Table

| Metric | base | fine_tuned | local |
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
| direction_flip_accuracy | n/a | n/a | n/a |
| forced_prediction_rate | 0.0000 | 0.2588 | 1.0000 |

## Per-Probe-Type Breakdown

### allowed_variable_constraint

| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |
|---|---:|---:|---:|---:|---:|---:|
| base | 85 | 0.2824 | n/a | n/a | n/a | 0.0000 |
| fine_tuned | 85 | 0.7176 | n/a | n/a | n/a | 0.0882 |
| local | 85 | 0.0000 | n/a | n/a | n/a | 1.0000 |

### contradiction_detection

| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |
|---|---:|---:|---:|---:|---:|---:|
| base | 85 | 0.7647 | n/a | 0.0000 | 0.0000 | n/a |
| fine_tuned | 85 | 0.8353 | n/a | 0.0000 | 0.0000 | n/a |
| local | 85 | 0.0000 | n/a | 1.0000 | 0.7743 | n/a |

### fixed_variable_constraint

| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |
|---|---:|---:|---:|---:|---:|---:|
| base | 85 | 0.5412 | n/a | n/a | 0.0000 | n/a |
| fine_tuned | 85 | 0.7412 | n/a | n/a | 0.0000 | n/a |
| local | 85 | 0.0471 | n/a | n/a | 0.7793 | n/a |

### paraphrase_consistency

| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |
|---|---:|---:|---:|---:|---:|---:|
| base | 85 | 0.0000 | 0.1656 | n/a | n/a | n/a |
| fine_tuned | 85 | 0.0353 | 0.6548 | n/a | n/a | n/a |
| local | 85 | 0.0824 | 0.6824 | n/a | n/a | n/a |

### physics_causal_question

| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |
|---|---:|---:|---:|---:|---:|---:|
| base | 2 | 1.0000 | 1.0000 | n/a | 0.0000 | 0.0000 |
| fine_tuned | 2 | 0.5000 | 1.0000 | n/a | 0.0000 | 0.0000 |
| local | 2 | n/a | n/a | n/a | n/a | n/a |

### prompt_image_conflict

| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |
|---|---:|---:|---:|---:|---:|---:|
| base | 85 | 0.8353 | n/a | 0.0000 | n/a | n/a |
| fine_tuned | 85 | 0.0941 | n/a | 0.5176 | n/a | n/a |
| local | 85 | 0.0000 | n/a | 1.0000 | n/a | n/a |

### prompt_sensitivity

| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |
|---|---:|---:|---:|---:|---:|---:|
| base | 85 | 0.4941 | n/a | n/a | 0.0556 | 0.0000 |
| fine_tuned | 85 | 0.5529 | n/a | n/a | 0.0263 | 0.2750 |
| local | 85 | 0.0353 | n/a | n/a | 0.7885 | 0.9831 |

## Interpretation

### Where SFT Improves

- fixed_variable_violation_rate: base=0.0032, fine_tuned=0.0015, delta=-0.0017
- contradiction_detection_accuracy: base=0.7647, fine_tuned=0.8353, delta=+0.0706
- direction_accuracy: base=0.1674, fine_tuned=0.6554, delta=+0.4881

### Where SFT Worsens

- valid_json_rate: base=0.8008, fine_tuned=0.7910, delta=-0.0098
- schema_valid_rate: base=0.6973, fine_tuned=0.6113, delta=-0.0859
- canonical_variable_rate: base=0.8008, fine_tuned=0.7910, delta=-0.0098
- paraphrase_consistency_score: base=0.8889, fine_tuned=0.7262, delta=-0.1627
- prompt_sensitivity_score: base=0.3333, fine_tuned=0.0000, delta=-0.3333
- allowed_variable_violation_rate: base=0.0000, fine_tuned=0.1545, delta=+0.1545
- prompt_image_conflict_detection_accuracy: base=0.8353, fine_tuned=0.0941, delta=-0.7412
- forced_prediction_rate: base=0.0000, fine_tuned=0.2588, delta=+0.2588

### Understanding Evidence

- Supports prompt-conditioned physics understanding: `True`
- Schema imitation or image-regression signal: `False`
- Evidence supports prompt-conditioned physics understanding only if SFT improves prompt sensitivity, contradiction/conflict detection, and constraint following, not just JSON/schema validity. High schema validity with weak prompt/conflict metrics is more consistent with schema imitation or image-driven regression.
