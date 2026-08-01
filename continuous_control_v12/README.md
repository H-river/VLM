# V12 continuous closed-loop optical control

V12 is isolated from frozen v9 and immutable v10. It stores continuous actions
and absolute actuator positions in millimetres, uses the existing optical
simulator through explicit mm-to-m conversion, keeps the legacy 81 actions as
an evaluation/probing grid, and never exposes oracle actuator target `q*` to a
deployed model or controller.

The configured absolute domain of ±3 mm comes from the repository variable
ranges. The simulator does not enforce or document physical hardware limits,
so manifests label this
`repository_sampling_domain_not_hardware_limit`.

## Small deterministic pipeline

```bash
PY=/home/jiamo/miniconda3/envs/optical_sim/bin/python
DATA=/home/jiamo/VLM_data/continuous_control_v12_smoke
RUN=/home/jiamo/VLM_runs/continuous_control_v12_smoke

$PY -m continuous_control_v12.generate_dataset \
  --smoke --output-dir "$DATA"

$PY -m continuous_control_v12.validate_dataset \
  --data-dir "$DATA"

$PY -m continuous_control_v12.generate_targets \
  --data-dir "$DATA"

$PY -m continuous_control_v12.train_forward_model \
  --overfit --data-dir "$DATA" --run-dir "$RUN/overfit" --device cpu

$PY -m continuous_control_v12.train_forward_model \
  --smoke --data-dir "$DATA" --run-dir "$RUN/model" --device cpu

$PY -m continuous_control_v12.run_mpc \
  --smoke --mode oracle --data-dir "$DATA" --max-steps 3 \
  --output "$RUN/oracle_mpc.json"

$PY -m continuous_control_v12.run_mpc \
  --smoke --mode learned --data-dir "$DATA" \
  --checkpoint "$RUN/model/continuous_forward_v12_smoke.pt" \
  --max-steps 3 --output "$RUN/learned_mpc.json"

$PY -m continuous_control_v12.evaluate_legacy_grid \
  --data-dir "$DATA" \
  --checkpoint "$RUN/model/continuous_forward_v12_smoke.pt" \
  --split development --max-groups 1 \
  --output "$RUN/legacy_grid.json"
```

Candidate-infeasible labels are deliberately not produced by default.
Verification is an explicit, potentially expensive command:

```bash
$PY -m continuous_control_v12.generate_targets \
  --data-dir "$DATA" \
  --verify-candidate-infeasible
```

The continuous oracle returns `candidate_infeasible` only above 1.2 tolerance
units with at least three agreeing optimizer runs. Otherwise it returns
`ambiguous_boundary`; optimizer failure is never treated as proof.

## Full configurations

`config_v12.json` preregisters 10,500/600/600 independent groups, 64
continuous probes plus two eight-step trajectories per group, a three-member
numerical ensemble, and four-step CEM. No full generation/training command is
run automatically.

To launch full generation later, choose a fresh output directory and run:

```bash
$PY -m continuous_control_v12.generate_dataset \
  --config /home/jiamo/VLM/continuous_control_v12/config_v12.json \
  --output-dir /home/jiamo/VLM_data/continuous_control_v12
```

Add `--store-images` only after planning storage. Without it, image references
are null and the numerical baseline remains valid. The one-step
`--image-conditioning` training ablation requires stored images. It is
intentionally rejected by multi-step learned MPC because V12 does not yet
predict future images.

## Structured VLM boundary

`goal.py` accepts structured target metrics, tolerances, allowed DOFs,
priorities, and constraints. It rejects direct actuator actions and any form
of `q*`. An external VLM can replace only the parser function; schema
validation and MPC dispatch remain deterministic.

