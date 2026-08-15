# VLA + Optics

**Status: PAUSED — SIMULATION RESEARCH PROTOTYPE**

This repository studies closed-loop optical beam control in simulation. The
current architecture separates numerical estimation and control from
multimodal supervision: a corrected optical simulator produces measurements,
a learned one-step transition ensemble (Learned-H1) predicts numerical state
changes, CEM selects bounded continuous actuator commands, and Qwen2.5-VL is
limited to high-level diagnosis, measurement policy, and orchestration.

The project is temporarily paused for application season. It is preserved for
review and later continuation, not abandoned. It is not production-ready, has
not demonstrated real-world deployment, and should not be read as evidence
that a VLM is a validated numerical optical-physics model.

## Scientific motivation

The central question is whether visual and structured observations can support
safe recovery decisions while a numerical controller retains ownership of
continuous physics and actuator values. The controlled beam state is

```text
y = [centroid_x_px, centroid_y_px, sigma_x_px, sigma_y_px, peak_intensity]
```

and the action is a bounded four-dimensional displacement of lens and camera
positions in millimetres.

## Current architecture

```text
simulated camera image + five beam metrics
                    |
          measurement / abnormality diagnosis
                    |
     +--------------+----------------+
     |                               |
Learned-H1 ensemble             Qwen2.5-VL
one-step numerical model        high-level supervisor
     |                          (no actuator values)
     +--------------+----------------+
                    |
       one-step CEM + real-observation replanning
                    |
      Branch-A gain probe + visible continuation
          (four-step preamble, maximum eight)
                    |
             simulated actuator step
```

The proposed future high-level recovery actions are `CONTINUE`, `REACQUIRE`,
`REMEASURE`, `PROBE`, and `WIDE_FOV`. They are candidate-only design concepts;
they are not a completed protected-evaluation result and are not identical to
the earlier three-field Qwen supervisor schema.

Canonical ownership is explicit:

- Simulator: `continuous_control_v12.simulator`, using the corrected
  `v12_sensor_power_semantics_v1` adapter over `optical_sim.src`.
- Learned-H1: `continuous_control_v12.world_model.ForwardEnsemble`.
- CEM: `continuous_control_v12.mpc.CEMMPC`.
- Branch-A probe and continuation contract:
  `configs/controller/branch_a.json`.
- Qwen supervisor schema and guard:
  `qwen_vl_supervisor_v1.closed_loop_adapter`.

The older four-step `PlanCEM` under
`qwen_reasoning_plan_selector_candidate/` is preserved as a candidate study;
it is not the canonical controller.

## Installation

Python 3.11 is the best-supported environment in the retained experiments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

For Learned-H1 checkpoint loading, install the PyTorch extra:

```bash
python -m pip install -e '.[test,learned-h1]'
```

Qwen training requires a separately reviewed CUDA environment; see
`qwen_vl_supervisor_v1/server_training_commands.md`. It is not part of the CPU
smoke test.

## Minimal smoke test

```bash
optics-vla all-smoke
```

This checks imports, canonical configuration, corrected simulator semantics, a
small oracle-backed CEM rollout, and the actuator-free Qwen JSON contract. To
also load the hash-pinned local Learned-H1 checkpoint:

```bash
optics-vla all-smoke --include-h1
```

That test reports `BLOCKED`, rather than passing, when the checkpoint or
PyTorch is unavailable.

## Canonical commands

```bash
# Show every supported validation command.
optics-vla --help

# Train the numerical one-step ensemble (CPU/GPU; data required).
python -m continuous_control_v12.train_forward_model --help

# Run the retained numerical MPC entry point (checkpoint and data required).
python -m continuous_control_v12.run_mpc --help

# Validate the Qwen supervisor schema without loading Qwen.
optics-vla qwen-contract

# Verify the compact confirmed-result table against tracked source reports.
python scripts/reproduce/verify_published_table.py

# Run the reviewable test surface.
python -m pytest
```

Exact historical experiment commands and their artifact requirements are
indexed in `docs/EXPERIMENT_STATUS.md` and `docs/REPRODUCIBILITY.md`.

## Confirmed results

Only results classified `CONFIRMED` are summarized here. All are simulation
results.

- On a 60-episode external controller suite, the fixed four-step Learned-H1+CEM
  controller succeeded on 39/60 episodes (65%); the frozen visible sequential
  rule succeeded on 51/60 (85%), a matched +20 percentage-point result with
  zero reported saturation or hard-constraint violations.
- Sensor saturation is the strongest completed visual-abnormality case in the
  retained reports: the metrics-only diagnostic was 50%, the small image CNN
  was 100% on IID and 93.3% on severity-OOD, and diagnosis-conditioned control
  was summarized as 90% versus 70% without diagnosis.

The machine-readable rows and source literals are in
`results/published_tables/confirmed_results.csv`.

## Limitations

- The repository is simulation-only unless a specific manifest proves
  otherwise; no such proof is used for the claims above.
- Qwen is not validated as a numerical transition or forward-physics model.
- The final counterfactual recovery-plan selector is `CANDIDATE_ONLY`; its
  frozen/protected evaluation was not run.
- Some plan labels predate controller-wrapper fixes. The candidate plan runner
  has a four-step maximum while the externally evaluated Branch-A controller
  has a maximum horizon of eight.
- Fixed-pixel reflection is excluded. Width-relative reflection is
  `PROVISIONAL`: its overall IID diagnostic was useful, but it failed the
  preregistered beam-width-quartile stability clause and produced no valid
  severity-OOD conclusion.
- Candidate transition-comparison results were not preregistered for a winner
  decision and did not establish closed-loop Qwen control.
- Dataset, checkpoint, and model redistribution rights have not been resolved.

See `docs/EXPERIMENT_STATUS.md` for the source-of-truth evidence table and
`PROJECT_STATUS.md` for the compact pause state.

## Data and checkpoint policy

Generated datasets, checkpoints, prediction logs, images, and raw run trees
stay outside normal Git tracking. Keep them in private local or external
artifact storage and record filename, version, size, SHA-256, generation
command, config, code commit, redistribution status, and evidence level using
the templates under `data/manifests/` and `artifacts/manifests/`.

## Repository structure

```text
configs/                 canonical pointers and controller/supervisor contracts
src/optics_vla/          small installable facade over hash-pinned modules
scripts/                 reviewable train/evaluate/reproduce entry-point index
tests/                   facade unit, integration, and regression tests
continuous_control_v12/  corrected simulator adapter, Learned-H1, and CEM
active_diagnosis_v13/     Branch-A and sequential-controller evidence code
qwen_vl_supervisor_v1/    QLoRA engineering and strict supervisor schema
vlm_optics_benchmark/     external controller and visual-anomaly evaluation
legacy/                  preserved superseded code and compact reports
data/, artifacts/, results/ metadata policy and compact reviewable artifacts
docs/                    audit, status, reproducibility, and readiness records
```

The root packages were not physically moved into `src/` because historical
manifests and hashes bind their paths and bytes. The facade makes the canonical
mapping clear without invalidating that provenance.

## Citation and license

Citation metadata is in `CITATION.cff`. No repository license has been selected
because ownership and redistribution terms are unresolved; see
`docs/LICENSE_DECISION_REQUIRED.md`. Absence of a license means public reuse is
not granted by this repository.
