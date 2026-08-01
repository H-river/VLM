# Quantitative Performance Diagnosis and Improvement Report (v10.21)

## Outcome

The main quantitative weakness was not one single model-capacity problem. It was a combination of an underdetermined direct-regression task, inconsistent visual evidence, lossy rendering, imbalanced evaluation, and inappropriate numerical proxies.

The production architecture now assigns exact numerical simulation and thresholding to deterministic tools. The Qwen checkpoint is responsible for tool selection, exact visible-input mapping, and result interpretation.

| Metric | Previous production result | Expanded full-sensor v10.9 | Change |
|---|---:|---:|---:|
| State equal-field macro-F1 | 0.710 on 54 records | 0.979 on 600 records | +0.270 |
| State joint exact | 0.407 | 0.930 | +0.523 |
| Pair equal-field macro-F1 | 0.628 on 24 records | 0.884 on 300 records | +0.257 |
| Pair joint exact | 0.625 | 0.913 | +0.288 |
| Production visual field macro-F1 | 0.664 | 0.927 | +0.262 |
| Seven-task equal macro | 0.977 | 0.977 | preserved |
| Visual-tool orchestration end-to-end | 1.000 on 78 sources | 0.990 on a 100-source probe | -0.010 |

The v10.9 clean-image gate and v10.19 robustness gate pass. The clean gate requires 150 independent validation groups, complete target-class coverage, state macro-F1 at least 0.95, pair macro-F1 at least 0.85, and both joint-exact rates at least 0.90. The robustness gate additionally covers blur, noise, dim-plus-noise, and saturation conditions.

| Condition | Production visual macro-F1 | Conservative 95% interval | State joint exact | Pair joint exact |
|---|---:|---:|---:|---:|
| Clean | 0.927 | [0.896, 0.950] | 0.930 | 0.913 |
| Blur radius 1 | 0.909 | [0.864, 0.939] | 0.873 | 0.847 |
| Gaussian read noise, std 2 | 0.869 | [0.821, 0.903] | 0.813 | 0.807 |
| 0.7 gain plus noise | 0.876 | [0.821, 0.911] | 0.788 | 0.797 |
| Saturation clipped at 180/255 | 0.881 | [0.844, 0.908] | 0.890 | 0.817 |

The intervals use 1,000 deterministic physical-group bootstrap resamples. The production interval is a conservative weighted combination of the separate state and pair interval endpoints, not a claimed joint bootstrap interval. Noise is now the lowest point estimate at 0.869, with a conservative lower bound of 0.821. Degraded-image performance remains less certain than clean performance.

## Diagnosed causes

### 1. Exact scalar prediction was underdetermined from prompt-visible evidence

Some forward prompts did not expose all latent simulator state needed to reproduce exact numeric outputs. The language model therefore learned a conservative near-zero change prior. A value such as zero could also receive full credit under broad per-field tolerances even when it did not represent a meaningful prediction.

Resolution: exact forward, counterfactual, sufficiency, and exhaustive-control results are computed by registered deterministic tools. The model is evaluated on selecting the tool, constructing inputs, and interpreting the validated result.

### 2. Token cross-entropy did not match numerical distance

Ordinary SFT token loss treats numeric strings as token sequences. It does not directly encode that `0.10` is closer to `0.11` than to `9.50`. Repeated near-zero outputs can therefore be locally easy without being physically accurate.

Resolution: exact numbers are not delegated to unconstrained token generation. Categorical visual measurements are calibrated from pixels, and simulator quantities are replayed deterministically.

### 3. The rendered image contradicted the full-sensor label

The old v10.1 images showed only the central 512 x 512 pixels, while labels were derived from moments over the full 1024 x 1024 sensor. Wide and displaced beams lost substantial light outside the crop. In an audited case, the full-sensor x centroid was 489 px (left of center), while the cropped visible image measured 522 px (right of center).

Resolution: quantitative evidence now renders the full 1024-pixel sensor into the 384 x 384 image. Crop geometry is stored explicitly in every calibration and used by the execution adapter.

### 4. Calibration overlays overwrote beam pixels

Cyan crosshairs and yellow width rings replaced real pixels. The previous extractor converted those colored pixels to zero, leaving artificial lines and rings in the measured beam.

Resolution: colored aids are detected and filled by deterministic local interpolation before moments are measured. This raised independent state macro-F1 from 0.961 to 0.979 and state joint exact from 0.883 to 0.930.

### 5. Total squared energy was a poor peak-intensity proxy

The old pair tool used total squared grayscale energy. That quantity changes with both peak height and width, so it confounded two different physical effects.

Resolution: because pair images share one intensity scale, the mean of the ten brightest non-overlay pixels is used for peak direction. Train-only group CV macro-F1 improved from 0.857 to 0.972. Independent peak-direction macro-F1 improved from 0.751 to 0.922.

### 6. The original evaluation panel was too small and incompletely covered classes

The original visual dev set contained 78 records from 29 groups: 54 state records and 24 pairs. At least one target class was absent for a field. This made estimates volatile and allowed a macro-F1 implementation error to remain hidden.

Resolution: the expanded panel contains 900 records from 300 source questions and 150 physical groups. It covers every required state and pair class. Macro-F1 uses the union of target and predicted classes, so predicted-only errors are penalized.

### 7. Pair labels are strongly imbalanced

Most changes fall below the physical deadbands. For example, expanded validation contains only 4 sigma-x decreases versus 243 no-change labels. Accuracy alone therefore exaggerates performance.

Resolution: promotion reports equal-field macro-F1, per-field macro-F1, and whole-record joint exact. Accuracy is diagnostic only. Macro-optimized thresholds were tested but rejected when their small macro gain reduced joint exact.

### 8. The model can still over-map visible evidence

On one of 100 orchestration probe sources, the model copied an optional signed-difference path into `source_map` even though the registered pair tool accepts only first and second image roles.

Resolution: a 20-step mapping-repair QLoRA trial fixed the mapping error, but checkpoint 10 reduced direct-task preservation macro from 0.947 to 0.936. The repair adapter was rejected; the seed-49 checkpoint remains the promoted language checkpoint.

### 9. Clean full-frame moments were highly sensitive to low-level sensor noise

Positive noise over a large sensor has disproportionate leverage on second moments. The clean state extractor fell to macro-F1 0.523 under noise and 0.503 under dim-plus-noise. A single noise-robust calibration improved degraded states but slightly reduced clean accuracy; it was therefore not used unconditionally.

Resolution: the deterministic quality router measures border noise, maximum channel value, and colored-pixel fraction. V10.17 uses corruption-specific state thresholds fit only on separately corrupted training calibration sets for ordinary noise and dim-plus-noise. It preserves the clean v10.9 calibration on clean images, uses a robust state floor for blur, uses robust pair centroids for noise, and robust pair widths for blur. Dim-plus-noise state macro-F1 improves from 0.848 to 0.928 and state joint exact from 0.647 to 0.788. Ordinary-noise state macro-F1 improves from 0.880 to 0.937 and state joint exact from 0.693 to 0.813. Newly fit pair calibrations were rejected because they reduced rare-class macro-F1. Perturbed tool orchestration scored 1.000 on 30 complete source chains.

### 10. Low-amplitude noise dominated paired width moments

The original noisy pair route measured widths with ordinary intensity-weighted second moments. Background noise spans far more pixels than the beam and therefore perturbs small before/after width differences even after background subtraction. Fully replacing the pair calibration and macro-optimized thresholds both regressed independent validation.

Resolution: v10.18 squares only the signal weights used for noisy width moments, while retaining the existing centroid and peak outputs exactly. Noise pair macro-F1 improves from 0.800 to 0.815, with joint exact decreasing from 0.823 to 0.807 but remaining above the frozen 0.80 gate. Dim-plus-noise pair macro-F1 improves from 0.773 to 0.833 and joint exact from 0.790 to 0.797. Clean and blur predictions are unchanged.

### 11. Saturation removes peak information and re-exposes optional-role copying

Clipping intensities at 180/255 preserves most position and width information but destroys some peak ordering. Saturation state macro-F1 is 0.964, pair macro-F1 is 0.815, and peak-direction macro-F1 is the weakest field at 0.764. No saturation-specific threshold was fit.

The saturation orchestration probe also produced one raw mapping error in five pair sources: the model correctly copied the first and second image roles but additionally copied the prompt-visible signed-difference reference, which is not a registered input role. Raw source-map exactness and raw end-to-end exactness are both 0.900. A deterministic schema validator now removes only a known optional role whose value exactly matches the prompt; it still rejects missing required roles, unknown extras, invented paths, and altered optional paths. Validated tool execution and validated production end-to-end exactness are 1.000. This is a system-level repair, not a claim that the raw LLM mapping became correct.

An explicit negative instruction naming the optional reference was also tested and rejected. It made the distractor more salient: every affected pair mapping copied the forbidden role, reducing overall raw mapping from 1.000 to 0.500 on the earlier 30-source corruption probe and from 0.900 to 0.500 on saturation.

Resolution: the v10.21 production interface omits the signed-difference path because it is not accepted by the registered pair tool. This is an interface-scope correction, not a target or checkpoint change. Raw mapping and raw end-to-end exactness are 1.000 on both the 30-source noise/blur/dim-noise production probe and the 10-source saturation production probe. The distractor-bearing v10.19 result remains a separate stress test and the validator remains defense in depth.

## Current component status

| Component | Status | Evidence |
|---|---|---|
| Adaptive state image tool v10.17 | promoted | clean macro-F1 0.979; worst tested 0.928 |
| Adaptive pair image tool v10.18 | promoted | clean macro-F1 0.884; worst tested 0.815 |
| Qwen seed-49 orchestration | retained | 0.990 end-to-end on 100-source expanded probe |
| Perturbed orchestration | passed | 1.000 end-to-end on 30-source probe |
| Production perturbed orchestration v10.21 | passed | raw mapping and end-to-end 1.000 on 30 sources |
| Production saturation orchestration v10.21 | passed | raw mapping and end-to-end 1.000 on 10 sources |
| Optional-reference stress probe v10.19 | raw strict fail; validated pass | raw mapping/end-to-end 0.900; validated production 1.000 |
| Mapping-repair checkpoint 10 | rejected | fixes mapping but regresses setup interpretation |
| Exact native numerical generation | not promoted | deterministic simulator/tool route remains required |
| Real-laboratory validity | not claimed | all current ground truth is synthetic |

## Remaining weaknesses

1. Counterfactual paired directions remain the hardest visual subset; causal and forward pairs are perfect on the expanded panel, while counterfactual pairs contain most width-direction errors.
2. Rare decrease classes still have wide statistical uncertainty despite complete class coverage.
3. The visual expert proves that pixels affect predictions, but production metrology is deterministic rather than native VLM measurement.
4. Only the single-source, single-lens, camera topology is represented.
5. Synthetic noise, blur, dim-plus-noise, and saturation tests do not establish robustness to real sensor noise, aberrations, occlusion, calibration drift, or real optical hardware.

## Reproducible evidence paths

- Expanded audit: `data/visual_eval_expanded_fullsensor_v10_6/audit_report.json`
- State result: `results/visual_state_tool_interpolated_v10_9_expanded/summary.json`
- Pair result: `results/visual_pair_tool_interpolated_v10_9_expanded/summary.json`
- Strict visual gate: `results/visual_system_v10_9_seed49_strict_gate/summary.json`
- Robustness gate: `results/visual_robustness_v10_19_gate/summary.json`
- Production perturbed orchestration: `results/visual_robust_orchestration_production_v10_21_seed49/summary.json`
- Production saturation orchestration: `results/visual_saturation_orchestration_production_v10_21_seed49/summary.json`
- Optional-reference stress probe: `results/visual_saturation_orchestration_probe_v10_19_seed49/summary.json`
- Unified system gate: `results/unified_system_v10_21_seed49_gate/summary.json`
- Orchestration probe: `results/visual_tool_orchestration_v10_9_seed49_probe300/summary.json`
- Rejected repair preservation: `results/visual_mapping_repair_v10_10_ckpt10_direct_preservation30/summary.json`
