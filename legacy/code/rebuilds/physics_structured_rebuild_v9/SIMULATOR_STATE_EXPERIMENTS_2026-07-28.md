# Simulator-state experiments

Date: 2026-07-28

## Executive conclusion

Two different problems are present.

1. The visible-input mapping is partly smooth but steep. Focal length,
   wavelength, and lens-to-camera distance have approximately stable local
   gradients over 0.1% to 1% perturbations, but frequently cross the production
   output tolerances at 1%.
2. The mapping is also piecewise or quantized around camera sampling, clipping,
   and lateral offsets. Pixel size, camera offsets, lens offsets, and aperture
   show strong changes in normalized gradient with perturbation size and strong
   regime dependence.
3. Most importantly, the five current measurements are not a sufficient state
   description once physically plausible source shape and phase variation is
   admitted. Among 100 independently matched hidden-state pairs, 99 pairs
   diverged beyond tolerance for at least one of the same 81 future actions.
   Across all actions, the future-output divergence rate was 74.01%.

Therefore the specialist's problem is not only model capacity or dataset size.
The present input representation can assign the same visible input to different
future responses.

## Definitions

- **Context:** one independently sampled optical setup. It is the independent
  statistical unit.
- **Action:** one simultaneous choice of the four lens/camera displacement
  controls. The canonical grid contains 81 actions, including the no-op action.
- **Current state:** the five measurements before an action:
  `centroid_x_px`, `centroid_y_px`, `sigma_x_px`, `sigma_y_px`, and
  `peak_intensity`.
- **Production tolerance:** 1 pixel for each centroid, 2 pixels for each sigma,
  and 5% of the current peak intensity for the peak.
- **Tolerance crossing:** at least one of the five output differences is larger
  than its production tolerance.
- **Resulting-state difference:** perturbed next state minus unperturbed next
  state for the same action.
- **Action-response difference:** the no-op state is subtracted from each
  action result before comparing the perturbed and unperturbed setups. This
  isolates whether the setup feature changes the system's response to an
  action, rather than only shifting its current state.
- **Normalized gradient:** mean absolute output difference divided by its
  production tolerance and divided by the feature perturbation. Values are
  reported as tolerance units per 1% relative feature change.
- **Matched hidden-state pair:** two simulator states with identical visible
  setup values but different omitted source variables, optimized so all five
  current measurements agree within measurement precision.
- **Collision:** two physically different states that are indistinguishable
  under the representation being tested.

## Experiment 1: local sensitivity

### Design

- 100 independent contexts: 25 ordinary, 25 focusing, 25 clipping, and 25
  camera-boundary contexts.
- The focusing group was sampled near the thin-lens image distance.
- The clipping group used small apertures and decentered lenses. Its mean lens
  transmission was 56.16%, compared with 100% for the other groups.
- The camera-boundary group placed one sensor axis near 68% to 103% of the
  sensor half-width.
- Each of the 12 visible setup values was perturbed individually by a positive
  multiplicative 0.1%, 0.5%, and 1%.
- Every base and perturbed setup was evaluated on all 81 actions.
- Total evaluated state transitions:
  `100 × (1 + 12 × 3) × 81 = 299,700`.
- Confidence intervals use 2,000 bootstrap draws. The bootstrap resamples the
  100 contexts, not the 81 actions inside a context.

### Tolerance-crossing rates

Each number is the percentage of the 8,100 context/action cases that crossed at
least one output tolerance.

| Perturbed feature | Result 0.1% | Result 0.5% | Result 1% | Response 0.1% | Response 0.5% | Response 1% |
|---|---:|---:|---:|---:|---:|---:|
| beam waist | 0.00 | 0.00 | 1.89 | 0.00 | 0.00 | 0.00 |
| camera x offset | 3.67 | 8.43 | 10.56 | 5.44 | 10.11 | 12.33 |
| camera y offset | 5.11 | 11.20 | 13.00 | 6.67 | 10.74 | 12.96 |
| lens aperture | 1.00 | 4.56 | 10.51 | 1.48 | 4.78 | 7.02 |
| lens focal length | 0.11 | 17.77 | 59.70 | 0.67 | 7.93 | 23.96 |
| lens-to-camera distance | 0.00 | 13.22 | 47.93 | 0.00 | 1.48 | 7.44 |
| lens x offset | 0.78 | 9.36 | 17.22 | 1.89 | 6.94 | 13.98 |
| lens y offset | 1.11 | 9.84 | 16.17 | 2.11 | 7.10 | 12.90 |
| pixel size | 16.00 | 24.09 | 33.47 | 19.67 | 23.26 | 27.58 |
| power | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| source-to-lens distance | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| wavelength | 0.11 | 10.10 | 28.89 | 0.78 | 7.96 | 22.12 |

The largest 1% action-response crossing rates, with context-bootstrap 95%
confidence intervals, were:

| Feature | Rate | 95% confidence interval |
|---|---:|---:|
| pixel size | 27.58% | 20.95% to 34.78% |
| lens focal length | 23.96% | 18.55% to 29.19% |
| wavelength | 22.12% | 16.84% to 27.85% |
| lens x offset | 13.98% | 9.25% to 19.43% |
| camera y offset | 12.96% | 9.00% to 17.44% |
| lens y offset | 12.90% | 8.20% to 17.83% |
| camera x offset | 12.33% | 8.00% to 16.78% |

### Regime dependence at 1%

These are action-response crossing rates.

| Feature | Ordinary | Focusing | Clipping | Camera boundary |
|---|---:|---:|---:|---:|
| camera x offset | 8.44% | 1.33% | 19.11% | 20.44% |
| camera y offset | 7.56% | 3.11% | 16.30% | 24.89% |
| lens aperture | 0.00% | 0.00% | 28.10% | 0.00% |
| lens focal length | 23.26% | 24.44% | 26.67% | 21.48% |
| lens-to-camera distance | 3.56% | 5.33% | 11.11% | 9.78% |
| lens x offset | 4.00% | 1.33% | 47.90% | 2.67% |
| lens y offset | 2.67% | 2.67% | 42.72% | 3.56% |
| pixel size | 24.15% | 2.67% | 33.38% | 50.12% |
| wavelength | 20.20% | 24.44% | 21.04% | 22.81% |

Beam waist, power, and source-to-lens distance had 0% action-response
crossing in every regime at 1%.

### Smoothness diagnostic

The table reports the mean of the five output-specific action-response
gradients. A roughly constant value across perturbation sizes is evidence for a
locally smooth mapping. A large decrease as the perturbation grows indicates
quantization, a sharp local transition, or another piecewise response.

| Feature | At 0.1% | At 0.5% | At 1% | Interpretation |
|---|---:|---:|---:|---|
| beam waist | 0.0285 | 0.0284 | 0.0282 | smooth and weak |
| source-to-lens distance | 0.000269 | 0.000269 | 0.000269 | smooth and negligible in this range |
| focal length | 0.2928 | 0.2852 | 0.2741 | smooth and steep |
| lens-to-camera distance | 0.1139 | 0.1102 | 0.1056 | mostly smooth |
| wavelength | 0.2851 | 0.2788 | 0.2621 | mostly smooth and steep |
| camera x offset | 0.5508 | 0.2356 | 0.1636 | strongly scale-dependent |
| camera y offset | 0.6270 | 0.2330 | 0.1538 | strongly scale-dependent |
| lens aperture | 0.2364 | 0.1279 | 0.0915 | clipping-boundary behavior |
| lens x offset | 0.3101 | 0.1985 | 0.1674 | clipping-dependent transition |
| lens y offset | 0.3528 | 0.2171 | 0.1718 | clipping-dependent transition |
| pixel size | 2.5157 | 0.6319 | 0.3881 | strongly quantized or piecewise |
| power | 0.0000 | 0.0000 | 0.0000 | unused by the current field calculation |

The pixel-size and camera-offset result is consistent with the simulator's
nearest-neighbour sensor extraction: a small coordinate change can move a
sensor sample to a different simulation-grid index. The aperture and lens
offset results are consistent with crossing a hard aperture edge. Focal length
and wavelength instead retain similar normalized gradients and are genuinely
steep continuous variables.

`power_w` is not merely insensitive: the current Gaussian source implementation
normalizes its peak amplitude to one and does not multiply the field by source
power. Therefore `power_w` is presently a redundant input unless that simulator
behavior is changed.

## Experiment 2: representation sufficiency

### Design

- Visible setup values were held exactly fixed.
- The baseline used the simulator's default circular planar Gaussian source.
- The paired state was forced to have nonzero quadratic source wavefront phase.
  Source x/y offset, x/y waist scale, and amplitude scale were optimized to
  match the baseline's five current measurements.
- A pair was accepted only if all five current measurements matched within:
  0.1 pixel for each centroid, 0.2 pixel for each sigma, and 0.5% for peak
  intensity. These limits are one tenth of the production tolerances.
- 100 matched independent pairs were obtained: 25 per regime.
- 108 optimizer attempts produced 100 pairs: 92.59% match success.
- The largest accepted current-measurement mismatch was 0.9166 measurement-
  precision units, so every accepted pair met all five matching limits.
- Both members of each pair received the same 81 actions.
- The same-full-state repeat control reran 100 identical states.
- Confidence intervals use 2,000 context-level bootstrap draws.

### Five-measurement representation

| Metric | Result | 95% confidence interval |
|---|---:|---:|
| Matched contexts with at least one future crossing | 99/100 = 99.00% | 97.00% to 100.00% |
| Future actions with any output crossing | 5,995/8,100 = 74.01% | 69.63% to 77.73% |
| Future actions with action-response crossing | 74.09% | 69.79% to 78.05% |

Future-output crossing rate by regime:

| Regime | Rate | 95% confidence interval |
|---|---:|---:|
| ordinary | 73.83% | 66.57% to 80.54% |
| focusing | 67.70% | 58.51% to 76.15% |
| clipping | 81.33% | 74.17% to 87.21% |
| camera boundary | 73.19% | 64.05% to 81.09% |

The per-output crossing rates across 8,100 actions were:

| Output | Crossing count | Rate |
|---|---:|---:|
| centroid x | 2,725 | 33.64% |
| centroid y | 2,778 | 34.30% |
| sigma x | 847 | 10.46% |
| sigma y | 807 | 9.96% |
| peak intensity | 3,690 | 45.56% |

Crossing probability increased with the number of nonzero control components:

| Nonzero control components | Rate |
|---:|---:|
| 0 | 0.00% |
| 1 | 42.50% |
| 2 | 67.08% |
| 3 | 82.22% |
| 4 | 88.38% |

### Richer representations

The normalized full current intensity image was considered matched only when
both normalized root-mean-square error was at most 1% of peak and maximum pixel
difference was at most 5% of peak.

| Representation | Remaining collisions |
|---|---:|
| five current measurements | 100/100 by construction |
| full current intensity image | 1/100 = 1.00%; bootstrap interval 0.00% to 3.00% |
| five sensor phase tilt/curvature descriptors | 0/100 observed |
| known source curvature descriptor | 0/100 observed |

The five phase descriptors are x phase tilt, y phase tilt, x curvature, y
curvature, and x-y cross-curvature. They were considered matched when all five
relative differences were at most 1%.

The one full-intensity-image collision was clipping context `clipping_0026`.
Its two current images had normalized root-mean-square error 0.2698%, maximum
pixel difference 4.7147%, and correlation 0.997844. Despite that close image
match, 54/81 future actions, or 66.67%, diverged beyond tolerance. The sensor
phase descriptors distinguished this pair. This is the expected ambiguity:
one intensity image does not directly measure optical phase.

Because zero phase collisions were observed, the empirical bootstrap interval
is also zero. That bootstrap result should not be read as proof that the true
collision probability is exactly zero; a larger validation set is still
needed.

### Numerical repeat control

The same full physical state was repeated for all 100 contexts.

- Contexts with any nonzero repeated output difference: 0/100.
- Maximum repeated output difference: exactly 0 tolerance units.

The simulator is deterministic under these settings. Hidden-state divergence
is therefore not numerical variability.

## Interpretation of conditional variance

Let `x` contain the visible setup and the five current measurements, let `a` be
an action, and let `s'` be the future five-measurement state.

Within the deliberately expanded hidden-source family tested here,
`P(s' | x, a)` has substantial conditional spread: two states with the same
`x` crossed future tolerance on 74.01% of common actions. The present forward
specialist is consequently being asked to learn a one-to-many mapping as if it
were one-to-one.

This is a constructive stress test, not a claim that the current production
simulator is itself stochastic. With its default planar Gaussian source and a
fully fixed simulator setup, it is deterministic. The representation problem
becomes relevant when the real system or a richer simulator permits source
phase, source shape, source alignment, or other omitted state to vary.

## Recommended engineering decisions

1. Add phase/curvature information to the state. If complex field phase is not
   directly measurable, use two or more intensity images at known defocus
   distances or another phase-retrieval measurement.
2. Retain the full current image as an input: it removed 99 of 100 collisions,
   but do not assume that one intensity image alone is sufficient.
3. Replace nearest-neighbour sensor extraction with bilinear/subpixel
   interpolation if it is a simulator artifact rather than intended sensor
   physics. Then repeat the pixel-size and camera-offset sensitivity tests.
4. Preserve explicit clipping and boundary indicators and oversample those
   regimes during specialist training. A hard aperture is inherently
   piecewise, even after sensor interpolation is improved.
5. Measure or control focal length, wavelength, and lens-to-camera distance
   accurately. Their response is genuinely steep rather than merely
   discontinuous.
6. Either implement optical power in the source-field amplitude or remove
   `power_w` from the specialist input contract.
7. After these representation and simulator changes, repeat the matched-state
   experiment at 300 or more contexts. The present 100-context result is already
   decisive for the initial design decision; 300 is appropriate for validating
   the replacement representation and rare residual collisions.

## Artifacts and safety

- Local sensitivity summary:
  `/home/jiamo/VLM_runs/physics_structured_rebuild_v9_state_experiments/local_sensitivity/summary.json`
- Local sensitivity context shards:
  `/home/jiamo/VLM_runs/physics_structured_rebuild_v9_state_experiments/local_sensitivity/shards/`
- Representation-sufficiency summary:
  `/home/jiamo/VLM_runs/physics_structured_rebuild_v9_state_experiments/representation_sufficiency/summary.json`
- Representation-sufficiency context shards:
  `/home/jiamo/VLM_runs/physics_structured_rebuild_v9_state_experiments/representation_sufficiency/shards/`
- Local safety audit:
  `/home/jiamo/VLM_runs/physics_structured_rebuild_v9_state_experiments/local_sensitivity_safety.json`
- Representation safety audit:
  `/home/jiamo/VLM_runs/physics_structured_rebuild_v9_state_experiments/representation_sufficiency_safety.json`

Local sensitivity safety maxima/minima:

- Maximum GPU memory: 967 MiB.
- Maximum GPU temperature: 57°C.
- Minimum available system memory: 7,578.73 MiB.
- Safety stops: none.

Representation-sufficiency safety maxima/minima:

- Maximum GPU memory: 1,101 MiB.
- Maximum GPU temperature: 59°C.
- Minimum available system memory: 7,418 MiB.
- Safety stops: none.
