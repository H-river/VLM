# Qwen H1 meta-controller candidate architecture

Status: **CANDIDATE ONLY — NOT SEALED — FROZEN EVALUATION DISABLED**

This additive package gives Qwen only a finite, strictly parsed H1-configuration
role. Qwen never emits continuous actuator values, bounds, means, variances, or
dispatch commands. A deterministic compiler converts the discrete output into
an H1-only objective view, allow-listed DOF mask, bounded categorical initial
mean, non-expanding action bounds, and a standard/conservative uncertainty
setting. The unchanged learned forward ensemble and existing `CEMMPC` still
choose the continuous action.

The repository does not contain a standalone complete pre-dispatch `SafetyGate`
class. Existing safety behavior is distributed across strict contracts,
per-step and remaining-position projection, CEM boundary/uncertainty penalties,
supervisor fail-safe and budget guards, and simulator validity checks. This
candidate retains those mechanisms and adds a reject-only gate before dispatch.
The report must not describe that additive gate as an unchanged pre-existing
standalone component.

## Runtime flow

```text
measurement specialist -> existing anomaly supervisor
  invalid/recovery -> existing supervisor callback boundary, no meta call
  valid nominal -> candidate meta-controller -> strict JSON parser
    parse/compile failure -> unchanged default H1
    reobserve/stop -> non-dispatch
    guided -> default H1 + guided H1 -> canonical arbiter -> reject-only gate
      accepted -> guided continuous action
      rejected -> unchanged default continuous action
```

`off` is the default feature mode and bypasses the meta-controller, compiler,
and guided planner. `shadow` computes audit data but executes the unchanged
default action. `guarded` is candidate-only and always constructs the default
proposal independently.

The 37/48 reference is tied to the locked 48-group H1 configuration
(`population=24`, `elites=6`, `iterations=3`, four control steps), so that is
the preregistered candidate comparison budget. The separate v12 orchestration
runtime currently uses 256/32/5 and is not silently substituted.

## Frames and units

- Numerical current/target state: five canonical v12 fields in
  `lab_frame_legacy_pseudo_pixels`; raw peak intensity retains corrected v12.1
  power causality.
- Current image: camera sensor frame. It is not the supervisor's 128 px
  diagnostic rendering.
- Continuous positions/actions: millimetres. The ±3 mm absolute domain is a
  repository sampling domain, not a hardware claim.
- No setup context, oracle actuator target, IDs, paths, split metadata, future
  state, or unexecuted default action is visible to Qwen.

H3 is absent from the candidate registry and is rejected by every dispatch
path.
