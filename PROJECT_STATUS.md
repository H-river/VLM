# PAUSED — SIMULATION RESEARCH PROTOTYPE

The VLA + Optics project is temporarily paused for application season. The
repository is preserved for review and later continuation.

## Confirmed

- Corrected optical simulator adapter and continuous action interface.
- Five-value beam state: centroid x/y, width x/y, and peak intensity.
- Numerical Learned-H1 ensemble with one-step CEM replanning.
- Branch-A symmetric gain-probe/replan controller.
- Frozen visible continuation rule with four initial controller steps and a
  configured maximum horizon of eight.
- Sensor-saturation visual diagnosis and diagnosis-conditioned control case.
- Qwen2.5-VL QLoRA engineering, strict JSON parsing, resume checks, and guarded
  high-level orchestration boundary.

## Provisional or candidate-only

- Width-relative reflection is provisional; it failed one preregistered IID
  stability clause and has no valid severity-OOD result.
- Qwen transition-prediction comparisons are candidate-only and do not
  establish Qwen as a numerical physics model or closed-loop controller.
- The counterfactual plan bank showed candidate value, but the Qwen selector
  was negative and its frozen/protected evaluation was not run.
- Proposed future supervisor actions are `CONTINUE`, `REACQUIRE`, `REMEASURE`,
  `PROBE`, and `WIDE_FOV`.

## Excluded or not run

- Fixed-pixel reflection is excluded.
- Real-world deployment, hardware validation, production readiness, and general
  VLM optical reasoning are not established.
- Final frozen/protected Qwen plan-selection and supervisor evaluations are not
  complete.

