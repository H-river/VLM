# Control rebuild v3

This folder keeps the frozen v2 data and models unchanged and builds three new,
separable control modules.

## Numerical contracts

The forward input is one 12-number optical setup, one 5-number current beam
state, and one of 81 fixed four-actuator actions. The forward output is five
changes:

1. centroid x in pixels;
2. centroid y in pixels;
3. horizontal width in pixels;
4. vertical width in pixels;
5. peak intensity.

The forward artifact blends the frozen v2 per-transition model with a new model
that predicts the complete 81-action grid jointly. The validation-selected blend
weight is 0.35 for the joint model and 0.65 for v2.

The numerical inverse input is the setup, current beam state, and requested beam
state. For every action, the forward model predicts a resulting state. A base
score is the negative normalized distance from that candidate to the request.
A 385,284-parameter ranker learns a correction to this base score and separately
classifies the request as:

- `unique`: exactly one action reaches the tolerance;
- `ambiguous`: more than one action reaches it;
- `infeasible_within_limits`: no allowed action reaches it.

## Modular visual dataflow

```text
current image + desired image
        |
        v
measurement v3: two sensor-frame 5-number beam states
        |
        v
frame adapter:
  current sensor state -> base legacy state
        |
        v
forward ensemble: 81 predicted legacy-frame candidate states
        |
        v
candidate-specific frame adapter:
  candidate legacy state -> candidate sensor state
        |
        v
residual scorer + learned numerical correction
        |
        v
selected actuator action + unique/ambiguous/infeasible status
```

The second frame conversion depends on each candidate's camera x/y displacement.
This is necessary because moving the camera changes the sensor coordinate frame.
The old monolithic visual ranker did not make this dependency explicit.

## Controlled evaluation

`evaluate_controlled.py` evaluates each intermediate boundary and the final
selected action. Validation can be used for a dry run. The default invocation
uses the three held-out splits exactly once and records:

- forward strict five-output success and error;
- numerical inverse target success and status accuracy;
- image measurement accuracy by condition;
- visual inverse with exact beam measurements;
- visual inverse with model-measured images.
