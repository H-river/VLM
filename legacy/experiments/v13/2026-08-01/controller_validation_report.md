# Frozen external controller validation

## Decision

The preregistered external gate passed. The Branch-A probe/replan H1-CEM controller with the unchanged visible sequential continuation rule is promoted as the frozen benchmark backbone. The fixed-four controller remains the matched baseline but is not the promoted backbone.

This is a new frozen external evaluation, not the old v13 protected result and not a development result. The 30 setup groups were generated once after preregistration, were disjoint by both group ID and setup hash from the training set and four prior diagnostic suites, and were evaluated with exactly the two preregistered unused planner seeds.

## Overall result

| Controller | Episodes | Strict all-five success | Mean steps | Median steps | Mean final distance | Saturation | Hard violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-4 H1-CEM | 60 | 65.0% | 2.85 | 3 | 4.140 | 0/60 | 0 |
| Sequential, max 8 | 60 | 85.0% | 3.68 | 3 | 0.647 | 0/60 | 0 |

The paired effect is +20.0 percentage points. The preregistered paired-episode bootstrap interval is [+10.0, +30.0] points (10,000 draws, seed 2026081199). There were 12 recoveries, zero regressions, and the exact two-sided McNemar p-value is 0.000488.

## By planner seed

| Planner seed | Fixed-4 | Sequential | Paired change |
|---|---:|---:|---:|
| 2026081101 | 60.0% | 86.7% | +26.7 points |
| 2026081102 | 70.0% | 83.3% | +13.3 points |

Both seed effects are non-negative as preregistered.

## By stratum

| Stratum | Fixed-4 | Sequential | Paired change |
|---|---:|---:|---:|
| A: nominal, non-boundary | 100.0% | 100.0% | 0.0 points |
| B: hidden gain, non-boundary | 60.0% | 85.0% | +25.0 points |
| C: hidden gain, boundary/clipping | 35.0% | 70.0% | +35.0 points |

The nominal stratum establishes non-regression; the improvement is concentrated where four steps truncate still-improving gain-drift and boundary cases.

## Step distribution and conditional success

Fixed-4 executed 1/2/3/4 control steps in 15/6/12/27 episodes. Sequential executed 1/2/3/4/5/6/7/8 steps in 15/6/12/8/4/6/2/7 episodes. Sequential outcomes conditional on step count were 100% for 1, 2, 3, 5, 6, and 7 steps; 75% for 4 steps; and 0% for the seven hard cases reaching the cap of 8. The cap therefore remains a real residual failure boundary rather than a claim that every episode becomes solvable.

The median final normalized distance fell from 0.574 to 0.488. The 90th percentile fell from 12.020 to 1.364. The complete carry-forward target-distance trajectory for steps 0-8 is stored in `external_validation_results.json`.

## Recoveries and regressions

Exact recoveries (setup, gain, planner seed):

- `v12_mpcdiag_vlmoptics_external_20260801_01_0001`, 0.75, 2026081101 and 2026081102
- `v12_mpcdiag_vlmoptics_external_20260801_01_0004`, 0.5, 2026081101 and 2026081102
- `v12_mpcdiag_vlmoptics_external_20260801_01_0005`, 0.75, 2026081101
- `v12_mpcdiag_vlmoptics_external_20260801_02_0000`, 0.5, 2026081101
- `v12_mpcdiag_vlmoptics_external_20260801_02_0001`, 0.75, 2026081101
- `v12_mpcdiag_vlmoptics_external_20260801_02_0004`, 0.5, 2026081101 and 2026081102
- `v12_mpcdiag_vlmoptics_external_20260801_02_0005`, 0.75, 2026081101 and 2026081102
- `v12_mpcdiag_vlmoptics_external_20260801_02_0007`, 1.5, 2026081101

There were no matched regressions.

## Frozen-gate evaluation

- Overall improvement at least 10 points: pass (+20.0).
- Improvement non-negative for both seeds: pass.
- Matched regressions no more than 2/60: pass (0/60).
- Saturation increase no more than 5 points: pass (0.0-point change).
- No added hard-constraint violations: pass (0 added).

The external result was not used to tune the rule, anomaly severities, recovery policies, or planner. The promoted implementation and source hashes remain those in `frozen_backbone_manifest.json`.
