# VLM-on-optics next-stage report

## Final decision: PARTIALLY READY

The numerical backbone is validated and frozen. One visual anomaly family, sensor saturation, passes both the irreducible-visual-information gate and the diagnosis-dependent control-value gate. Secondary reflection has strong oracle control value but fails the frozen IID identifiability gate. Because fewer than two families pass both gates, this run does not train Qwen-VL and does not release the preparatory schema as a training-ready dataset.

## 1. Previously known evidence

Before this run, v12 provided the numerical five-metric H1 forward ensemble and one-step CEM replanning. v13 showed that symmetric gain probing is useful but that a linear estimator matches or beats structured-text Qwen, and it identified premature four-step truncation as the strongest remaining controller bottleneck. Development and fresh non-protected cohorts suggested approximately +15-21 points from longer sequential control, but the existing v13 report explicitly left a 30-group, two-seed external confirmation undone.

These statements are prior evidence only. They are not counted as results produced here.

## 2. Results produced in this run

- Audited and hashed the checkpoint, embedded normalization, H1-CEM runtime, fixed-four prefix, sequential rule, gain probe, simulator/rendering path, measurement specialist, setup generator, split manifests, reports, and protected manifests.
- Preregistered exact controllers, two unused planner seeds, one setup-generation contract, three strata, 60 paired episodes, analyses, safety margins, and GO/NO-GO criteria before generating the external suite.
- Generated the external suite once, with 30 unique setup groups/hashes and zero overlap against training plus four earlier diagnostic suites.
- Completed the external controller validation once for each frozen seed.
- Built 120 paired counterfactuals across train, IID-held-out, and severity-OOD splits. Every stored PNG pair remains within 0.25 tolerance on each metric and 0.40 normalized L2 after serialization.
- Trained inexpensive metrics, history, image, multimodal-small-model, and oracle baselines. No large Qwen-VL training occurred.
- Ran 180 matched closed-loop control-value episodes across two anomaly families and three diagnosis arms.
- Replayed deterministic anomaly/control episodes in a clean test process.

## 3. Protected results

The old v13 protected evaluation is inventoried in `repository_audit.md` and remains logically separate. It was not opened for anomaly threshold selection, matching, model training, recovery selection, or the new external controller gate. No result in `identifiability_results.json` or `control_value_results.json` is an old protected result.

## 4. Development results

Development-only image inspection rejected generic rendering corruptions and found the first reflection range too subtle. Before any held-out image generation, reflection was frozen at amplitude 0.40-0.55 and separation 24-32 canonical pixels; saturation was frozen at a clip level of 0.35-0.55 peak. The matching mechanism reached maximum 0.0149 tolerance on any metric and L2 0.0196 over 48 development pairs.

The failed initial IID command was a selector bug (`suffix >=8` rather than `{8,9}`) and wrote no IID rows. The narrow correction changed only split membership; anomaly injection and all frozen thresholds remained unchanged. A later JSON-key serialization correction changed no predictions.

## 5. New frozen external evaluation

| Controller | Strict success | Mean steps | Saturation | Hard violations |
|---|---:|---:|---:|---:|
| Fixed-4 H1-CEM | 65.0% (39/60) | 2.85 | 0/60 | 0 |
| Frozen sequential max-8 | 85.0% (51/60) | 3.68 | 0/60 | 0 |

The paired gain is +20.0 points with preregistered 95% bootstrap interval [+10.0, +30.0]. Planner-seed effects are +26.7 and +13.3 points. There are 12 matched recoveries and zero regressions; exact McNemar p=0.000488. Nominal performance stays 100% in both arms, hidden-gain non-boundary improves 60%→85%, and hidden-gain boundary/clipping improves 35%→70%.

All preregistered freeze checks pass. The official backbone for later comparisons is the unchanged Branch-A probe/replan H1-CEM with the visible improvement>=0.25 sequential rule and maximum horizon 8.

## 6. Oracle results

| Anomaly | No diagnosis | Oracle diagnosis | Change | Recoveries / regressions | Control gate |
|---|---:|---:|---:|---:|---|
| Sensor saturation | 70.0% | 90.0% | +20.0 points | 6 / 0 | pass |
| Secondary reflection | 0.0% | 90.0% | +90.0 points | 27 / 0 | pass |

Both recoveries change the measurement decision rather than the actuator budget. Saturation reacquires an unsaturated observation; reflection switches to primary-spot metrics. Neither increases actuator saturation or hard violations. The three remaining oracle failures in each family are the same hard boundary cases, showing that diagnosis cannot overcome every numerical/backbone limitation.

## 7. Learned-model results

The small CNNs are visual diagnostics, not VLMs.

| Anomaly | Split | Metrics logistic | Metrics-history MLP | Image CNN | Image+metrics small model | Oracle |
|---|---|---:|---:|---:|---:|---:|
| Saturation | IID | 50.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| Saturation | Severity-OOD | 50.0% | 50.0% | 93.3% | 95.0% | 100.0% |
| Reflection | IID | 50.0% | 50.0% | 50.0% | 58.3% | 100.0% |
| Reflection | Severity-OOD | 50.0% | 50.0% | 56.7% | 83.3% | 100.0% |

Saturation passes every visual gate and leakage check. Border masking preserves 100% IID balanced accuracy; shuffled images fall to 58.3%; metadata-only reaches 41.7%; there is no setup or exact-image overlap. Its learned diagnostic reproduces the oracle closed-loop result exactly: 90% success, six recoveries, zero regressions.

Reflection fails the IID image threshold and the +15-point advantage threshold; border-masked accuracy is also 50%. Its stronger severity-OOD multimodal score is not a substitute for the frozen IID gate. In control, the family-specific learned diagnostic reaches 73.3% and 22 recoveries, but this does not erase the IID failure. Three classification-correct reflection episodes still fail control; there are no classification-wrong accidental recoveries.

## 8. Unverified hypotheses

- A reflection separation defined in units of the measured primary-beam width, rather than fixed canonical pixels, may provide stable morphology across narrow and broad beams.
- More development setups may stabilize a reflection classifier, but the current IID failure does not establish that data volume alone is sufficient.
- A dedicated primary-component segmentation/specialist may generalize better than the tiny binary CNN.
- Sensor-edge clipping could become a second family only after a clipping-aware fit/recovery is validated without mixing aperture and finite-sensor mechanisms.
- Compositional two-fault cases have not been generated or evaluated.
- No result here establishes multimodal Qwen-VL performance.

## Ranked readiness diagnosis

1. Secondary reflection has genuine control value but insufficient stable IID visual identifiability under the frozen mild range.
2. The failure is not because the five metrics reveal the label: both numerical baselines are at 50% on exact matched pairs.
3. The failure is not primarily the recovery policy: the oracle gains 90 points.
4. The stronger OOD morphology is identifiable, indicating the current difficulty is signal consistency at the frozen IID severity, not an impossible image task.
5. No evidence supports spending a large Qwen-VL training budget before this is fixed.

## Smallest experiment needed for a second anomaly

Generate a new development-only cohort (no reuse of current IID/OOD groups), define secondary-reflection separation in primary-beam sigma units so every example has a resolvable shoulder or second component, retain exact five-metric counterfactual matching, and train the same small image diagnostic. Freeze that mechanism/range before a new setup-disjoint IID audit. The minimum go condition remains >=80% image balanced accuracy, >=15 points over metrics, passed leakage tests, and the already demonstrated diagnosis-dependent recovery. If this fails, implement and validate an edge-clipping-aware measurement specialist before testing sensor-edge clipping as the replacement family.

## Prepared but not executed next training matrix

| Method | Role |
|---|---|
| Metrics-only rule/linear | irreducibility baseline |
| Metrics-history MLP | temporal numeric baseline |
| Text-only Qwen | non-visual language baseline |
| Image-only VLM | visual ablation |
| Multimodal Qwen-VL | intended supervisor |
| Hidden-state oracle | upper bound |

The preparatory `vlm_dataset_schema.json` defines image/history/metrics/goal/options input and structured fault/evidence/decision/candidate/confidence output. It forbids free-form actuator values. Its status is explicitly not training-ready until a second anomaly passes both gates.
