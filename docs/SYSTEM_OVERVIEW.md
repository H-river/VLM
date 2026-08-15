# System overview

## Scope

This is a simulation research system. It separates optical measurement,
numerical dynamics, continuous control, and multimodal supervision so that a
language/vision model cannot silently become the owner of actuator values.

## Data flow

```text
Optical setup + actuator position
              |
 corrected simulator semantics
              |
 image + y=[cx, cy, wx, wy, peak]
              |
 measurement quality / abnormality diagnosis
              |
   +----------+----------------------------------+
   |                                             |
Learned-H1 transition ensemble           Qwen2.5-VL supervisor
mean + uncertainty + auxiliaries         diagnosis / measurement / action enum
   |                                             |
   +------------------ guarded join -------------+
                          |
                bounded one-step CEM
                          |
              execute first action only
                          |
                  real simulated observation
                          |
            visible continuation rule (max 8)
```

## Canonical components

| Layer | Canonical implementation | Contract |
|---|---|---|
| Physics engine | `optical_sim.src` through `continuous_control_v12.simulator` | Corrected pixel-area irradiance sampling and explicit mm-to-m conversion |
| State | `continuous_control_v12.contracts.OUTPUT_FIELDS` | `[centroid_x_px, centroid_y_px, sigma_x_px, sigma_y_px, peak_intensity]` |
| Numerical dynamics | `continuous_control_v12.world_model.ForwardEnsemble` | One-step residual ensemble; three members in the frozen checkpoint |
| Continuous control | `continuous_control_v12.mpc.CEMMPC` | Bounded CEM; execute first action and replan |
| Branch-A preamble | `active_diagnosis_v13` behavior, centralized in `configs/controller/branch_a.json` | Symmetric 10% gain probe, no-residual-history gain estimator, then replan |
| Continuation | `optics_vla.control.continuation` configured by `branch_a.json` | Four-step initial prefix; continue after real improvement >=0.25; maximum eight |
| Supervisor | `qwen_vl_supervisor_v1.closed_loop_adapter` | Strict high-level JSON; continuous actuator fields rejected |

The exact controller parameters are configuration, not VLM output. Qwen may
select or recommend a high-level recovery operation only through a validated
schema and cannot change gain, bounds, CEM budget, model horizon, success
tolerance, or maximum controller horizon.

## Supervisor actions

The earlier implemented Qwen contract emits diagnosis, measurement policy, and
one of `execute`, `reacquire`, or `switch_measurement`, with conservative stop
handling in the adapter. A later design proposes `CONTINUE`, `REACQUIRE`,
`REMEASURE`, `PROBE`, and `WIDE_FOV`. That later vocabulary is candidate-only;
it has not replaced the implemented frozen schema.

## Version conflict retained explicitly

`qwen_reasoning_plan_selector_candidate.core.PlanCEM` uses a one-step model and
at most four real control steps. It belongs to the candidate plan-bank study.
The externally evaluated Branch-A backbone uses a one-step Learned-H1 model,
replans after real observations, and permits a visible rule through at most
eight control steps. These implementations answer different experiment
questions and are not interchangeable. The candidate runner remains preserved
until a future experiment either ports its plan programs to the canonical
max-eight controller or retires them with a documented comparison.

## Safety boundary

- `q*`, hidden gain, future simulator outcomes, protected results, and evaluator
  labels are prohibited from deployed policy inputs.
- The action vector is projected against per-step and absolute position bounds.
- An invalid supervisor output produces a conservative non-dispatch outcome.
- Protected/frozen status is experiment-specific. The completed v13 protected
  controller confirmation does not imply that Qwen supervisor or plan-selector
  protected evaluation was completed.

