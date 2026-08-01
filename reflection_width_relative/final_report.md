Final decision: REFLECTION STILL NOT READY

# Width-relative secondary-reflection evaluation

Does width-relative secondary-reflection generation provide a second optical anomaly whose identity is unavailable from the five beam metrics, reliably identifiable from beam images across unseen setups, and useful for improving closed-loop control through a diagnosis-dependent measurement decision?

**No.** It satisfies five-metric ambiguity and has very strong diagnosis-dependent control value. The frozen diagnostic also clears the overall image-accuracy and leakage clauses. It does not clear the preregistered setup-invariance clause: the narrowest unseen beam-width quartile achieved 68.75% balanced accuracy and the quartile range was 31.25 points, versus frozen requirements of at least 70% in every quartile and at most a 25-point range. Reflection therefore cannot be promoted as the second benchmark anomaly family, and Qwen-VL training must not start.

## 1. Prior evidence

The verified numerical backbone remains the externally validated H1-CEM sequential controller: fixed-4 was 39/60 (65%) and frozen sequential was 51/60 (85%), a 20-point improvement with 12 recoveries, zero regressions, no actuator saturation, and no hard violations. It was replayed, not retrained or modified.

Sensor saturation remains a validated visual family from the prior stage: metrics-only 50%, image CNN 100% IID, image CNN 93.3% severity-OOD, and learned/oracle diagnosis 90% control success versus 70% without diagnosis.

The prior fixed-pixel reflection family had metrics-only 50%, image CNN 50%, image+metrics 58.3%, but oracle control 90% versus 0% without diagnosis. Those prior held-out outcomes were not used to select any parameter, model, threshold, or policy in this run.

The clean-process reproduction boundary passed before implementation: the old training pair manifest byte-matched, a selected old reflection PNG reproduced the stored SHA-256 exactly, and six representative oracle control traces matched every stored field.

## 2. Development-only parameter selection

The new mode computes `sigma_direction = sqrt(u^T Sigma_primary u)` from the clean primary before injection and uses `d = k * sigma_direction`. The secondary is a centroid-scaled clean-primary copy, shifted with zero fill and no wrap. Its power is added to the primary; diagnostic views are peak-normalized afterward. The old fixed-pixel path is unchanged.

Twelve entirely new development setups covered three physical strata and four beam-width quartiles. The preregistered grid contained 1,728 evaluations: 4 k values x 3 amplitudes x 3 width ratios x 4 directions x 12 setups. Every float and serialized pair passed the frozen metric thresholds. Overall maximum serialized field and L2 distances were 0.07274 and 0.08676, far inside 0.25 and 0.40.

The frozen contiguous IID range was selected using only these new development setups:

- k: `{2.25, 2.75}`;
- relative amplitude: `{0.40, 0.50}`;
- component width ratio: `{0.8, 1.0}`;
- direction: eight 45-degree bins.

The chosen region had 100% serialized match rate and median matched-image RMS difference 0.0573. Development already exposed the main risk: thresholded shoulder consistency in Q1 was only 27.1%. This risk was recorded before IID generation rather than tuned away.

Thirty new train setups and the unchanged previous small-model budget were used. Across three frozen seeds, development balanced accuracy was:

| Seed | Metrics | Metrics + history | Image CNN | Image + metrics |
|---:|---:|---:|---:|---:|
| 2026081202 | 50.0% | 50.0% | 75.0% | **83.3%** |
| 2026081203 | 50.0% | 50.0% | 75.0% | 70.8% |
| 2026081204 | 50.0% | 50.0% | 62.5% | 75.0% |

The preregistered development rule selected the image+metrics diagnostic at seed 2026081202. This small CNN is a visual diagnostic, not a VLM.

## 3. Frozen new IID results

The IID suite was generated once after preregistration: 30 new setup groups, 60 balanced samples, no overlap with development, training, prior reflection cohorts, or earlier controller suites. All 30 pairs passed both matching checks. Achieved maxima were:

| Distance | Float | Serialized PNG |
|---|---:|---:|
| Maximum normalized field difference | 0.00382 | 0.11782 |
| Normalized metric L2 | 0.00433 | 0.15185 |

The five metric values were therefore strongly counterfactually ambiguous.

Frozen IID performance:

| Model | Balanced accuracy | Macro F1 | ECE |
|---|---:|---:|---:|
| Metrics-only logistic | 50.0% | 50.0% | 0.0003 |
| Metrics + short-history MLP | 50.0% | approximately 50.0% | seed-dependent, low |
| Selected image + metrics diagnostic | **86.7%** | **86.7%** | 0.1010 |
| Hidden-label oracle | 100% | 100% | 0.0000 |

The selected diagnostic confusion matrix was `[[27, 3], [5, 25]]`. Image advantage over the best numerical baseline was 36.7 points.

Per-seed IID balanced accuracy was:

| Seed | Image CNN | Image + metrics |
|---:|---:|---:|
| 2026081202 | 88.3% | **86.7% selected** |
| 2026081203 | 85.0% | 78.3% |
| 2026081204 | 70.0% | 71.7% |
| Mean | 81.1% | 78.9% |

The unselected seed-2026081202 image-only result is reported but cannot replace the frozen development-selected diagnostic.

Selected-model subgroup accuracy:

| Group | Balanced accuracy |
|---|---:|
| Width Q1 | **68.75%** |
| Width Q2 | 92.86% |
| Width Q3 | 100% |
| Width Q4 | 85.71% |
| Boundary | 75.0% |
| Non-boundary | 92.5% |
| Amplitude 0.40 | 84.21% |
| Amplitude 0.50 | 90.91% |
| k 2.25 | 90.91% |
| k 2.75 | 84.21% |

Direction accuracy ranged from 75% east to 100% south-east, with small per-direction sample counts. The identifiability gate failed only the frozen width-quartile stability clause: Q1 was below 70% and the Q1-Q3 range was 31.25 points, above 25 points. The overall >=80% image and >=15-point image-advantage clauses passed.

## 4. Severity-OOD results

No valid severity-OOD prediction result is reported. Its parameter range was frozen before IID (`k=3.25`, amplitude `0.60`, width ratio `{0.7, 1.1}`), and its 18 setup groups were generated with zero overlap. During paired-image creation, one boundary case passed the float match but reached a serialized per-field difference of 0.274828, above the frozen 0.25 limit. The command wrote no OOD dataset result.

The threshold was not relaxed, the case was not dropped, the range was not changed, and generation was not retried. Partial OOD PNGs are explicitly non-evidence. Severity-OOD is unavailable and does not substitute for the IID gate.

## 5. Oracle diagnosis

On the new 30-setup IID control distribution, oracle diagnosis used the unchanged primary-spot measurement switch and the same frozen controller/action budget:

- strict success: 27/30 = 90%;
- mean executed steps: 2.60;
- median executed steps: 2;
- mean final normalized target distance: 1.230;
- median final normalized target distance: 0.569;
- 27 matched recoveries and zero regressions versus no diagnosis;
- zero actuator saturations and zero hard-constraint violations.

The three oracle failures were controller/recovery-limit cases, not extra-budget differences. Recovery never added actuator steps.

## 6. Learned diagnosis

Using the frozen selected diagnostic to trigger the same measurement switch:

- strict success: 23/30 = 76.7%;
- mean executed steps: 2.80;
- median executed steps: 2;
- mean final normalized target distance: 3.564;
- median final normalized target distance: 0.621;
- 23 recoveries and zero regressions versus no diagnosis;
- zero actuator saturations and zero hard-constraint violations.

Two episodes were classified correctly at every decision but still failed control:

- `v12_mpcdiag_reflsig_iid_20260801_02_0004__secondary_reflection_width_relative__learned_image_diagnostic`;
- `v12_mpcdiag_reflsig_iid_20260801_02_0008__secondary_reflection_width_relative__learned_image_diagnostic`.

One episode contained wrong diagnostic decisions but accidentally recovered:

- `v12_mpcdiag_reflsig_iid_20260801_02_0005__secondary_reflection_width_relative__learned_image_diagnostic`.

There were 27 observation-level learned/oracle disagreements across six setups. Exact observation IDs and probabilities are stored in `control_value_results.json`; grouped exactly, they were:

- `..._00_0003`: observations 0-2;
- `..._01_0004`: observations 0-4;
- `..._02_0002`: observations 0-8;
- `..._02_0003`: observations 0-4;
- `..._02_0005`: observations 3 and 5;
- `..._02_0006`: observations 0-2.

## 7. Closed-loop control results

| Arm | Strict success | Mean steps | Median final distance | Saturations | Violations |
|---|---:|---:|---:|---:|---:|
| Normal combined-spot metrics | 0/30 (0%) | 4.53 | 26.349 | 0 | 0 |
| Oracle label + primary measurement | 27/30 (90%) | 2.60 | 0.569 | 0 | 0 |
| Learned diagnostic + primary measurement | 23/30 (76.7%) | 2.80 | 0.621 | 0 | 0 |

Oracle improvement was +90 points and 27 matched recoveries, easily passing the >=5-point-or-five-recovery gate. The effect came from the diagnosis-dependent measurement switch, not extra action steps, and there was no safety regression.

Success counts by important subgroup are shown as no diagnosis / oracle / learned:

- width Q1: 0/8, 6/8, 3/8;
- width Q2: 0/7, 6/7, 6/7;
- width Q3: 0/8, 8/8, 8/8;
- width Q4: 0/7, 7/7, 6/7;
- boundary: 0/10, 7/10, 5/10;
- non-boundary: 0/20, 20/20, 18/20;
- k 2.25: 0/11, 11/11, 9/11;
- k 2.75: 0/19, 16/19, 14/19;
- amplitude 0.40: 0/19, 18/19, 15/19;
- amplitude 0.50: 0/11, 9/11, 8/11.

Full distributions by width, k, amplitude, width ratio, direction and boundary status are in `control_value_results.json`; every episode is in `episode_results.csv`.

## 8. Leakage checks

The leakage audit passed:

- shuffled images with metrics preserved: 45.0% balanced accuracy;
- eight-pixel border mask: 88.3%;
- dark irrelevant-background mask: 86.7%;
- metadata-only classifier: 51.7%;
- exact train/IID image hash overlap: zero;
- train/development/IID setup overlap: zero;
- overlap with all previous reflection cohorts: zero by both group ID and setup hash;
- filenames contain no labels;
- image dimensions, channel count, padding policy and peak normalization are class-independent.

The image signal is therefore real morphology, not metadata, filename, border, normalization, duplicate-image, or setup leakage.

## 9. Failure category and remaining unverified hypotheses

Primary failure category: **inconsistency across beam scale**. Boundary sensitivity is a secondary contributor: visual accuracy was 75% at boundary versus 92.5% elsewhere, and learned control was 5/10 at boundary versus 18/20 elsewhere. This run did not find evidence of leakage, numerical counterfactual artifact in IID, or recovery mismatch at the oracle gate.

The strong form of the hypothesis—width-relative separation alone makes morphology setup-invariant under the existing 128x128 preprocessing and previous small-model protocol—is falsified by the preregistered IID gate. The broader physical hypothesis is only partly falsified: images contain strong usable signal overall, and the control consequence is real, but the narrow-beam and boundary regimes remain unreliable.

Severity extrapolation is unverified because the frozen OOD cohort could not satisfy serialized counterfactual matching. A realizable clipping-aware primary measurement specialist is also unverified: the control audit validates the value of switching to clean-primary measurements, not a newly trained or fitted specialist on these width-relative images.

The smallest next experiment is a **diagnosis-given, clipping-aware analytic primary measurement specialist**, not another classifier search:

1. Use only fresh development setups; keep this generator, classifier, controller and failed IID audit closed.
2. Given the reflection diagnosis and the existing valid-sensor mask, fit a constrained two-component Gaussian or robust primary-plus-residual model directly to linear intensity. The fit must account for truncated pixels in its likelihood; it must not receive k, amplitude, direction, setup ID, or clean metrics.
3. Measure only whether the fitted primary centroid, widths and peak recover the clean-primary five metrics within the existing tolerance vector, reported separately for Q1 and boundary cases.
4. Freeze the fitter and success thresholds, then run one fresh setup-disjoint measurement audit. Run closed-loop replay only if measurement recovery passes.

This isolates clipping-aware measurement realizability with no new visual classifier capacity and no Qwen-VL training. It will not by itself repair the failed visual-identifiability gate; that remains a separate prerequisite for benchmark promotion.

No VLM training-ready manifest or Qwen-VL prompt was created because the required READY decision was not reached.
