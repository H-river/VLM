# Evaluation protocol

Evaluator results are versioned. `v1` is retained unchanged so the original pilot baseline and forty-step trial remain reproducible. `v2` is the frozen selection rubric for experiments started after the error audit. Always report the rubric version beside a macro score; scores from different versions are not directly comparable.

## Version 1 (legacy pilot)

The primary score is the unweighted mean of the seven task-family scores. This prevents the more numerous forward/control records from dominating the result. Invalid JSON receives zero for its record. A parseable envelope can still receive partial task credit when a nested field violates the task schema; JSON validity and full schema validity are reported separately.

| Task | Record score |
|---|---|
| Setup interpretation | Mean of exact component order, adjustable-parameter set F1, total-distance error at most 0.1 mm, and focal-length error at most 1e-6 m. |
| Information sufficiency | 60% status accuracy plus 40% mean of missing-field set F1 and the status-specific requested output. |
| Causal effects | Accuracy over five increase/decrease/no-change labels. |
| Forward prediction | Mean of six tolerance checks over after-state and change: centroid Euclidean error at most 2 px, both widths within 2 px, and intensity within 5% of the after-state scale. Raw errors are also reported. |
| Diagnosis | 50% unique/ambiguous/unsupported status accuracy plus 50% plausible-cause set F1. |
| Constrained intervention | 30% status, 20% quantized-grid validity, 30% simulator-grounded outcome success, and 20% minimum-motion optimality. For a feasible prediction, inactive actuators must be zero and the cached state for the selected exhaustive-grid action must land within the declared centroid tolerance. |
| Counterfactual reasoning | Mean of changed-parameter accuracy, direction-preservation accuracy, and the response-difference tolerance score. |

For task families with more than one status, macro-F1 is reported in addition to status accuracy. Results are sliced by modality and IID/OOD distribution. Runtime reporting includes generation latency, input/output tokens, and peak allocated CUDA memory.

## Baseline protocol

- Model: the local untouched `Qwen2.5-VL-3B-Instruct` checkpoint.
- Quantization: NF4 4-bit with double quantization and bfloat16 computation.
- Decoding: greedy (`do_sample: false`), at most 384 new tokens.
- Data: all 120 canonical validation records only. The sealed test prompts and private test labels are not read by inference.
- Images: passed only for the 12 validation records that reference images; text-only records use no dummy image.
- Resume semantics: prediction rows are flushed individually and keyed by unique `example_id`.

The validation baseline is for model and fine-tuning comparison, not final test-set selection. The simulator-derived labels establish synthetic consistency only and do not demonstrate real-laboratory validity.

## Version 2 (current checkpoint selection)

Version 2 keeps the same parsing, schema, sufficiency, causal, forward, and diagnosis rules. It corrects three artifacts found by paired error analysis:

- Setup interpretation scores semantic component order (including common aliases), total distance, and focal length equally. The adjustable-parameter field remains a diagnostic but is excluded because the original pilot prompt did not expose the actuator interface.
- Constrained intervention gates all downstream credit on the correct feasibility branch. A feasible answer is scored on status, grid validity, simulator success, and minimum motion. An infeasible answer must give a null plan and its best-achievable residual must be within 2 px of exhaustive simulator replay.
- Counterfactual reasoning assigns 40% to its two structural decisions and 60% to the simulator-derived numerical response difference, preventing two categorical fields from overwhelming the physical prediction.

Use `--rubric-version v2` for all new checkpoint comparisons. The targeted v1.1 setup prompts now expose an `actuator_interface`, so future dataset versions may restore adjustability scoring after a new rubric is frozen; this pilot's validation labels and prompts are not retroactively changed.
