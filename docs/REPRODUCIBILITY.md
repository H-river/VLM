# Reproducibility

## Supported review environment

Python 3.11 is the closest match to the retained controller/Qwen experiments.
The base CPU surface needs NumPy, SciPy, Pillow, PyYAML, joblib, and pytest.
Learned-H1 loading additionally needs PyTorch. Qwen training/inference needs a
reviewed CUDA environment and local model weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

For an existing managed environment, installation without dependency changes
is also valid:

```bash
python -m pip install -e . --no-deps
```

## Quick validation matrix

| Check | Command | Hardware/data requirement | Expected outcome |
|---|---|---|---|
| Environment sanity | `optics-vla sanity` | CPU | Imports canonical packages and reports `PASS` |
| Config parsing | `optics-vla config-check` | CPU | Confirms four-step prefix, max-eight, and local checkpoint/hash state |
| Simulator smoke | `optics-vla simulator-smoke` | CPU | Corrected semantics, finite five-state output |
| Learned-H1 inference | `optics-vla h1-smoke` | CPU + PyTorch + hash-pinned checkpoint | `PASS`, or exact `BLOCKED` reason if dependency/checkpoint missing |
| CEM rollout | `optics-vla cem-smoke` | CPU | One canonical-budget oracle-backed rollout reduces distance legally |
| Qwen schema | `optics-vla qwen-contract` | CPU; no Qwen weights | Three canonical combinations accepted and actuator field rejected |
| Controller regression | `python -m pytest tests/regression/test_controller_config.py` | CPU | Configured prefix/horizon and visible continuation behavior pass |
| Stored result verification | `python scripts/reproduce/verify_published_table.py` | CPU | Every confirmed table row matches a literal in its tracked source report |
| Reviewed test surface | `python -m pytest` | Dependencies plus some local artifacts | Pass locally where artifacts exist; missing artifacts must be reported `BLOCKED` |

`optics-vla all-smoke` runs every CPU check except Learned-H1. Add
`--include-h1` to require checkpoint inference.

## Canonical training and numerical evaluation

The following commands are entry-point templates. Choose external data/run
directories and record them in manifests; do not write large products into Git.

```bash
python -m continuous_control_v12.generate_dataset \
  --config continuous_control_v12/config_v12_semantics_v2.json \
  --output-dir /path/to/external/continuous_v12_data

python -m continuous_control_v12.validate_dataset \
  --data-dir /path/to/external/continuous_v12_data

python -m continuous_control_v12.train_forward_model \
  --data-dir /path/to/external/continuous_v12_data \
  --run-dir /path/to/external/continuous_v12_run \
  --device cpu

python -m continuous_control_v12.run_mpc \
  --mode learned \
  --data-dir /path/to/external/continuous_v12_data \
  --checkpoint /path/to/external/continuous_forward_v12.pt \
  --max-steps 4 \
  --output /path/to/external/mpc_result.json
```

The last command is the retained v12 entry point, not the entire Branch-A
max-eight evaluation. The Branch-A probe and visible continuation contract is
defined in `configs/controller/branch_a.json`; replay of the frozen external
study also requires its exact local suite, estimator, rule artifact, and hashes.

## Qwen contract and training boundary

```bash
# CPU; no model loading.
optics-vla qwen-contract

# GPU-oriented engineering entry point; validate arguments first.
python -m qwen_vl_supervisor_v1.train_qlora --help
```

Do not run an ad hoc frozen evaluation. The retained
`qwen_vl_supervisor_v1/evaluation_protocol.md` requires a complete three-seed
checkpoint selection, baseline selection, concrete formal runner, and final
freeze verification. Those steps were not completed.

## Determinism and provenance

- Canonical numerical helpers use explicit seeds and SHA-256-derived identity.
- The Branch-A controller config pins the Learned-H1 checkpoint SHA-256.
- Historical manifests/configs containing absolute paths are evidence records;
  new commands use CLI paths or repository-relative config.
- The local transition pilot states that Qwen CUDA attention made exact bitwise
  replay unverified. Three training seeds do not cure that limitation.
- A result is reproducible only when its data bytes, manifest, checkpoint,
  config, code commit, and command all match. Similar metrics are not proof.

## Known blocked cases in a fresh clone

- Learned-H1 smoke: blocked without the external checkpoint and PyTorch.
- Full active-package tests: some benchmark tests open ignored `runs/` data.
- Qwen training/inference: blocked without CUDA-compatible dependencies and
  appropriately licensed local Qwen weights.
- Width-relative reflection replay: blocked without ignored images/PyTorch
  checkpoints and exact suite artifacts.
- Candidate transition replay: source is pre-existing untracked work and its
  7.8 GiB local artifact is not part of a GitHub clone.

No blocked check should be replaced by a toy, blank, repaired, or alternate
backend while retaining a `PASS` label.

