# GitHub readiness report

Report date: 2026-08-15

## Decision

**Private GitHub upload: conditionally ready for a code-and-metadata-only
snapshot after selective staging and review.** The two pre-existing untracked
projects must not be captured accidentally, and ignored local artifacts must
remain local.

**Public GitHub release: not ready.** License/ownership, dataset and model
redistribution, dedicated Git-history secret scanning, and several artifact-
free reproduction paths remain unresolved. Scientific candidate/provisional
labels must also remain visible.

## Summary of changes

- Created safety branch `archive/pre-github-cleanup-20260815` at the original
  commit.
- Audited Git state, branches, versions, files, datasets, checkpoints, results,
  large files, paths, dependencies, entry points, and secret patterns.
- Added an installable `optics_vla` facade without moving hash-pinned packages.
- Declared one canonical corrected simulator, Learned-H1 ensemble, CEM, and
  config-driven Branch-A continuation contract.
- Centralized the Branch-A symmetric probe preamble, four-step prefix, and
  maximum horizon eight in `configs/controller/branch_a.json`.
- Preserved the separate max-four candidate `PlanCEM` and documented the
  mismatch rather than silently merging results.
- Repaired two active imports that still pointed to the simulator audit's old
  pre-archive package path.
- Added CPU smoke commands, Learned-H1 checkpoint smoke, Qwen schema test,
  controller regression, and confirmed-result verification.
- Rewrote the root README and added audit, architecture, status,
  reproducibility, dataset, artifact, archive, contribution, citation, and
  license-decision documentation.
- Added manifest templates and a tiny explicitly synthetic no-op fixture.
- Tightened `.gitignore` so reviewable metadata can live under `data/`,
  `artifacts/`, and `results/` without exposing generated contents.

No dataset, checkpoint, result, branch, tag, or non-generated source file was
deleted. No history was rewritten. Nothing was pushed or published.

## Final reviewable tree

```text
.
├── README.md
├── PROJECT_STATUS.md
├── CONTRIBUTING.md
├── CITATION.cff
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── configs/
│   ├── controller/branch_a.json
│   ├── qwen/supervisor_contract.json
│   ├── simulator/canonical.json
│   └── specialists/README.md
├── src/optics_vla/
│   ├── cli.py
│   ├── common/config.py
│   ├── control/continuation.py
│   ├── dynamics/
│   ├── evaluation/
│   ├── measurement/
│   ├── simulator/
│   └── supervisor/
├── scripts/
│   ├── train/README.md
│   ├── evaluate/README.md
│   ├── reproduce/{run_smoke.py,verify_published_table.py}
│   └── utilities/README.md
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── data/
│   ├── README.md
│   ├── manifests/dataset_manifest.template.json
│   ├── schemas/beam_transition.schema.json
│   └── examples/synthetic_transition_v1.json
├── artifacts/
│   ├── README.md
│   └── manifests/checkpoint_manifest.template.json
├── results/
│   ├── README.md
│   └── published_tables/confirmed_results.csv
├── docs/
│   ├── SYSTEM_OVERVIEW.md
│   ├── EXPERIMENT_STATUS.md
│   ├── REPRODUCIBILITY.md
│   ├── DATASETS.md
│   ├── REPOSITORY_AUDIT.md
│   ├── GITHUB_READINESS_REPORT.md
│   ├── LICENSE_DECISION_REQUIRED.md
│   └── archive/README.md
├── archive/README.md
├── continuous_control_v12/       # canonical implementation owner
├── active_diagnosis_v13/         # Branch-A evidence implementation
├── qwen_vl_supervisor_v1/        # guarded supervisor engineering
├── vlm_optics_benchmark/         # external/visual evaluations
├── qwen_reasoning_plan_selector_candidate/ # candidate-only max-four study
├── reflection_width_relative/    # provisional reflection evidence
└── legacy/                       # preserved prior implementations/reports
```

Large ignored `artifacts/`, `runs/`, `outputs/`, and nested experiment products
are omitted from the tree above but remain on disk.

## Canonical entry points

| Purpose | Entry point |
|---|---|
| Environment/config sanity | `optics-vla sanity`; `optics-vla config-check` |
| Simulator smoke | `optics-vla simulator-smoke` |
| Learned-H1 inference smoke | `optics-vla h1-smoke` |
| CEM smoke rollout | `optics-vla cem-smoke` |
| Qwen schema/guard test | `optics-vla qwen-contract` |
| Combined validation | `optics-vla all-smoke --include-h1` |
| Numerical model training | `python -m continuous_control_v12.train_forward_model` |
| Retained numerical MPC | `python -m continuous_control_v12.run_mpc` |
| Stored confirmed-result check | `python scripts/reproduce/verify_published_table.py` |
| Experiment evidence index | `docs/EXPERIMENT_STATUS.md` |

## Validation results

| Validation | Result | Notes |
|---|---|---|
| Editable package install | `PASS` | `python -m pip install -e . --no-deps` in retained Python 3.11 environment |
| Python compilation | `PASS` | New `src/`, `scripts/`, and `tests/` compiled |
| JSON configuration/schema parsing | `PASS` | Reviewable non-artifact JSON parsed |
| CLI `--help` | `PASS` | Canonical CLI plus retained Learned-H1, MPC, and Qwen trainer help |
| Canonical config | `PASS` | Four-step prefix, max-eight, threshold 0.25, checkpoint present and SHA-256 matched |
| Simulator smoke | `PASS` | Corrected semantics and finite five-value state on CPU |
| Learned-H1 smoke | `PASS` | Hash-pinned three-member ensemble loaded on CPU; zero-action state preserved |
| CEM smoke | `PASS` | Population 24, three iterations, one-step model horizon; legal one-step success |
| Qwen contract | `PASS` | Three canonical combinations accepted; continuous actuator field rejected; no model loaded |
| Confirmed-result table | `PASS` | Seven rows matched tracked source literals |
| New facade tests | `PASS` | 3/3 |
| Reviewed active test surface | `PASS` | 230/230 with 24 Pillow deprecation warnings |
| Initial host pytest invocation | `BLOCKED`, then resolved for repo tests | ROS Jazzy `launch_testing` plugin autoload imported missing `lark`; rerun with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` |
| Ruff | `BLOCKED` | `ruff` not installed in the validation environment |
| Canonical absolute-path scan | `PASS` | No `/home`, `/Users`, or Windows drive path in new canonical code/config/docs |
| Current-tree secret pattern scan | `PASS` with stated scope | No high-confidence secret; one strategy-name false positive; dedicated tools unavailable |
| Dedicated Git-history secret scan | `BLOCKED` | `gitleaks`, `trufflehog`, and `detect-secrets` unavailable |

The 24 warnings come from Pillow's deprecation of the `mode` argument in
`vlm_optics_benchmark.visual_anomalies`; Pillow announces removal in version
13. This is a maintenance item, not a changed metric or failed test.

## Remaining broken or blocked items

- A fresh clone will not contain the canonical Learned-H1 checkpoint, raw
  datasets, visual images, or run manifests needed by artifact-bound tests.
- Root pytest may require disabling unrelated globally installed plugins, as in
  this host environment.
- Qwen full training/frozen evaluation remains GPU- and artifact-dependent.
- Ruff/static type checks were not run because the tools were unavailable.
- Historical active and legacy configs still contain personal absolute paths;
  hash-pinned records were preserved, while new canonical files are relative.
- `optical_sim/README.md` retains a stale link to the formerly root-level
  `profile2setup/` project.
- The two pre-existing untracked directories are not part of this change and
  need owner decisions before they can be staged or ignored.

## Large files requiring external storage

- `artifacts/transition_pilot_v2_20260806_201524`: about 7.8 GiB.
- `artifacts/qwen_h1_selector_v1`: about 4.7 GiB.
- `artifacts/qwen_reasoning_plan_selector`: about 4.5 GiB.
- `qwen_vl_supervisor_v1/artifacts`: about 4.2 GiB.
- `optical_sim/outputs/random_v2`: about 40 GiB.
- `runs/overnight_v12_semantics_20260731_002709`: about 1.1 GiB.
- `runs/active_diagnosis_v13_20260801_010051`: about 538 MiB.

The largest single local file is a roughly 3.43 GiB Qwen resume-state file.
These should remain private until manifests and redistribution rights are
complete. Git LFS is only a possible transport for a small intentional subset,
not a default solution for the local artifact tree.

## License status

`BLOCKED`: no license was selected. Ownership, third-party code, generated data,
and model/checkpoint redistribution require human decisions documented in
`docs/LICENSE_DECISION_REQUIRED.md`.

## Secret-scan status

The current tracked and visible-untracked tree passed a careful pattern and
filename audit without exposing candidate values. Dedicated scanners were not
installed, and Git history plus ignored metadata were not exhaustively scanned.
Public release remains blocked until those scans and remediation reviews pass.

## Unresolved scientific ambiguities

- Four-step candidate `PlanCEM` versus canonical Branch-A max-eight control.
- Candidate plan labels/checkpoints created before wrapper repair.
- Width-relative reflection's useful overall image result versus failed
  beam-width-quartile stability clause and invalid severity-OOD generation.
- Candidate transition results without preregistered winner thresholds or
  closed-loop CEM feasibility.
- Qwen supervisor engineering readiness versus uncompleted final frozen
  evaluation.
- Completed v13 protected numerical controller confirmation versus uncompleted
  Qwen protected evaluations.

## Files moved to legacy/archive

This cleanup moved **no files**. That was intentional: the current branch had
already archived superseded code, and further moves would invalidate paths and
hashes without improving scientific traceability.

Existing preserved archive roots are:

- `legacy/code/rebuilds/`;
- `legacy/code/pipelines/`;
- `legacy/experiments/`;
- `legacy/cleanup_reports/`.

## Deletion candidates (not deleted)

- Python/pytest/ruff caches and nested virtual environments.
- Repeated intermediate Qwen checkpoints, optimizer states, and training logs
  after a final artifact is independently verified.
- Duplicate generated images/videos and raw result trees represented by exact
  manifests and hashes.
- Stale systemd units and machine-specific launch scripts after provenance
  review.
- Old candidate runners only after the max-four/max-eight ambiguity is resolved.

No deletion should occur until a human verifies recovery, provenance, and the
replacement artifact.

## Suggested files for the first GitHub release

For the first **private snapshot only**, include source packages, `configs/`,
`src/`, `scripts/`, `tests/`, docs,
manifest templates, the synthetic fixture, compact confirmed table, tracked
legacy code/reports, and citation metadata. Exclude all ignored datasets,
checkpoints, adapters, raw predictions, generated media, environments, and the
two pre-existing untracked directories unless reviewed separately.

Do not call this a public release until license and redistribution are resolved.

## Recommended commit boundaries

1. `Add canonical optics_vla config and validation facade`
   - `pyproject.toml`, `.gitattributes`, `configs/`, `src/`, `scripts/`,
     `tests/`, import repairs, and `.gitignore` metadata exceptions.
2. `Document repository evidence and reproducibility status`
   - root README/status/contributing/citation, `docs/`, policy READMEs,
     manifests, synthetic fixture, and confirmed result table.

Keep pre-existing `optics_data_prepare/` and `transition_pilot/` changes out of
both commits unless the owner reviews and commits them separately.
