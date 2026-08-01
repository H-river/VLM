# Quantitative Performance Diagnosis and Improvement Report (v11.3 confirmed)

## Outcome

V11.3 preserves the confirmed 1.000 seven-task routed-system score and improves ordinary-noise image measurement without changing the language checkpoint. A two-pass deterministic smoothing step is applied only after the quality router identifies ordinary read noise.

| Metric | v10.19/v11.2 route | v11.3 | Change |
|---|---:|---:|---:|
| Ordinary-noise state macro-F1 | 0.937 | 0.957 | +0.020 |
| Ordinary-noise state joint exact | 0.813 | 0.868 | +0.055 |
| Ordinary-noise production visual macro-F1 | 0.869 | 0.878 | +0.009 |
| Five-condition average production macro-F1 | 0.892 | 0.894 | +0.002 |
| Worst-condition production macro-F1 | 0.869 | 0.876 | +0.007 |
| Worst conservative 95% lower bound | 0.821 | 0.821 | +0.0005 |
| Robust LLM-to-tool orchestration end-to-end | 1.000 | 1.000 | preserved |

All source-task state slices improve. Ordinary-noise counterfactual state joint exactness rises from 0.817 to 0.858, diagnosis from 0.779 to 0.821, causal from 0.842 to 0.933, and forward prediction from 0.850 to 0.908.

## Diagnosed cause

The ordinary-noise extractor subtracts a robust background and clips negative residual pixels before computing moments. Across a large frame, zero-mean read noise therefore becomes a positive spatial floor, broadening widths and moving centroids near category boundaries. Two separable `[1, 2, 1] / 4` smoothing passes reduce that noise before the existing calibrated floor and moments are applied.

The candidate was selected on the separate train-only corrupted calibration set. Relative to no smoothing, its train accuracies improve from 0.975/0.963/0.963/0.959 to 0.978/0.982/0.976/0.981 for horizontal centroid, vertical centroid, sigma-x, and sigma-y. It was then evaluated once on the independent 600-record noisy state panel.

## Rejected ablations

- Uniform power-2.5 paired widths improved sigma-x and joint exactness but reduced sigma-y macro-F1 and was rejected.
- Axis-specific power 2.5/2.0 made only a small aggregate gain and reduced counterfactual exactness, so it was rejected.
- Absolute-state kNN features failed train-only CV and were not evaluated.
- Projecting signed pixels to 1-D profiles improved imbalanced train accuracy but collapsed rare decreases on independent validation; both accuracy and macro calibrations were rejected.
- Signed-difference geometry features failed to improve task-agnostic paired-width validation. Direct width-contrast thresholds scored only about 0.41 group-CV macro-F1 and were closed.

These experimental estimators remain opt-in calibration features for reproducibility. None is used by the v11.3 promoted calibration.

## Preservation and claim boundary

Clean, blur, dim-plus-noise, and saturation summaries are reused only because the quality router leaves their calibration paths unchanged. The complete five-condition gate passes. A rebuilt 30-source robust orchestration panel changed one tool-result interpretation prompt; the unchanged seed-49 checkpoint interpreted it correctly, leaving all 90 stages and all 30 source chains exact.

This remains a synthetic hybrid-system result. The language model routes registered tools and interprets validated outputs; it is not promoted as a native precision image meter. No sealed pilot test labels were opened and no paid API was used.

## Remaining bottleneck

Dim-plus-noise is now the worst overall condition at production macro-F1 0.876. Paired change measurement remains harder than state measurement: ordinary-noise pair macro-F1 is 0.815, with sigma-x/sigma-y at 0.730/0.751, and ordinary-noise counterfactual pair joint exactness is 0.683. Future paired-width work should use a new scenario-disjoint confirmation set because the current validation panel has now been used to reject several estimator families.

## Reproducible evidence paths

- Selected noise-state calibration: `data/visual_tool_calibration_noise_v10_17/calibrations/state_noise_floor4_denoise2_v11_3.json`
- Noise-state result: `results/visual_state_denoise2_v11_3_noise/summary.json`
- Packaged adaptive calibration: `data/visual_tool_calibration_adaptive_v11_3/`
- Five-condition robustness gate: `results/visual_robustness_v11_3_gate/summary.json`
- Robust orchestration result: `results/visual_robust_orchestration_production_v11_3_seed49/summary.json`
- Confirmed unified gate: `results/unified_system_v11_3_confirmed_seed49_gate/summary.json`
