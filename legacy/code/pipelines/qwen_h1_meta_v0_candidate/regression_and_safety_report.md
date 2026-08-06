# Regression and safety report

CANDIDATE ONLY — NOT SEALED — FROZEN EVALUATION DISABLED

Reporting gate: **RED**.

## Hard-stop assessment

- RED reasons: `information_insufficient_or_data_gate_red, near_visible_conflict_rate_above_0.20, visible_classifier_macro_f1_below_0.50`.
- YELLOW reasons: `none`.
- H3 remains disabled and forbidden; H1 is the only continuous controller.
- Qwen may only emit strict discrete configuration JSON. It cannot emit or dispatch continuous actuator actions, enlarge bounds, change the target, or bypass uncertainty/budget/dispatch checks.
- Safety in the existing repository is distributed across contracts, bounds/projection, uncertainty handling, budget/state-machine logic and simulator validity. The candidate adds a reject-only pre-dispatch gate; this report does not invent an unchanged standalone SafetyGate that did not exist.
- Formal frozen evaluation remains disabled. No candidate report is evidence about frozen IID/OOD/protected generalization or real hardware.
- The candidate closed-loop harness consumes manifest/synthetic supervisor-validity state; it does not execute the existing Qwen anomaly supervisor. Sequential reobserve recovery has no verified repository backend. Therefore these artifacts are not full-stack or end-to-end validation.

## Supplied test evidence

| Check | Status | Passed | Failed | Notes |
|---|---|---:|---:|---|
| final candidate plus existing targeted regression suite | PASS | 170 | 0 | candidate contracts/compiler/controller/integration/data/training/offline/baseline/closed-loop/ablation/reporting tests plus existing H1, supervisor adapter, MPC and corrected simulator tests |
| feature flag off and deterministic default H1 equivalence | PASS | 3 | 0 | off path calls the unchanged CEM, full-result equality is direct, and the canonical trace digest is repeat-stable |
| strict meta authority and security rejection | PASS | not available | not available | unknown fields, prefixes/suffixes, duplicate keys, NaN/Inf, action, mu, sigma, bounds and covariance injection are rejected; H3 is construction-time impossible |
| source and protocol immutability | PASS | not available | not available | candidate tests verify frozen CEM/forward source bytes; final verify-protocol revalidated every frozen protocol/source hash; git diff reports no tracked existing-file changes |
| post-RED fail-closed entry-point hardening | PASS | not available | not available | alternate identity registries are rejected before read; training requires an official PASS audit and exact 48/24 hash-linked export; offline requires complete 24-row dev coverage for all three seeds; closed-loop source/base config is frozen; gain and reasoning claims require paired evidence |
| information sufficiency gate | EXPECTED_HARD_STOP | not available | not available | official candidate data generator returned exit 3 and withheld SFT export after RED_STOP; no downstream QLoRA or evaluation was started |

## Compatibility flags

- Feature flag off reproduces unchanged default H1: `yes`.
- Compiler preserves or shrinks bounds: `yes`.
- Continuous action/mu/sigma/bounds injection rejected: `yes`.
- Dispatch gate enforced: `yes`.
- Existing state-machine tests pass: `yes`.
- Canonical objective and strict-success semantics preserved: `yes`.
- Frozen/protected content untouched: `yes`.
- H3 disabled: `yes`.
- Known pre-existing failures (not silently waived): `{"final_supported_optical_sim_environment": "the corresponding simulator semantics tests pass in the final 170-test suite under Python 3.11/NumPy 2.4/SciPy 1.17", "prechange_system_python_only": "legacy 128-output hash drift was observed under the old Python 3.12/NumPy 1.26/SciPy 1.11 environment before candidate changes"}`.

A failed hard-stop check is RED. Missing evidence is YELLOW; it is never silently treated as a pass.
