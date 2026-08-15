# Repository audit

Audit date: 2026-08-15 (Asia/Singapore)

This audit records the checkout before the current GitHub-readiness edits. It
does not treat local ignored artifacts as public or redistributable.

## Current repository state

| Item | Baseline value |
|---|---|
| Working directory | Repository root |
| Current branch | `Qwen_Reasoning` |
| Current commit | `f84dba797b9002831954a1a4a052de5622fba0f9` |
| Commit subject | `Clean active Qwen reasoning stack and archive legacy experiments` |
| Tracked files | 1,405 |
| Visible untracked files | 60 |
| Ignored files | 80,175 |
| Visible pre-existing user changes | Untracked `optics_data_prepare/` and `transition_pilot/` |
| Local tags | None |
| Git object store | 1.53 GiB packed plus 271.77 MiB loose |

The visible untracked count excludes ignored generated content inside the two
untracked directories. Both directories were preserved and excluded from the
cleanup change set. `optics_data_prepare/AGENTS.md` governs work inside that
directory; no files there were modified.

A safety branch, `archive/pre-github-cleanup-20260815`, points to the baseline
commit. No branch, tag, dataset, checkpoint, result, or source file was deleted,
and Git history was not rewritten.

## Branch and version map

Git history is a short linear chain, so branch ancestry is strong evidence here
but not sufficient evidence for experiment maturity. The package reports,
hashes, manifests, and executed paths were also inspected.

| Branch | Tip | Relationship | Classification |
|---|---|---|---|
| `Qwen_Reasoning` | `f84dba79` | Current branch; active packages plus existing `legacy/` archive | `canonical` |
| `v13` | `e6f18d4d` | Direct parent; continuous controller, active diagnosis, and Qwen supervisor generation | `supporting` |
| `optics-sft-smoke-test` | `aa7e1f66` | Direct ancestor; optics SFT and simulator adapter work | `supporting` |
| `main` | `64ed0cb1` | Earlier profile-to-setup/discrete line | `legacy` |
| `archive/pre-github-cleanup-20260815` | `f84dba79` | Safety pointer created for this cleanup | `obsolete-but-preserved` |

No local tags existed at audit time.

### Major internal generations

| Generation | Relationship | Classification |
|---|---|---|
| `profile2setup` / `lang2setup` | Early discrete setup prediction; now under `legacy/` | `legacy` |
| measurement/direction/forward/inverse rebuilds v3-v7 | Early specialist sequence, mostly archived under `legacy/code/rebuilds/` | `legacy` |
| TABM/physics rebuilds v8-v11 | Numerical and structured-physics experiments; superseded by continuous v12 | `legacy` |
| `specialist_rebuild_v2` | Retained supporting specialist data/model pipeline | `supporting` |
| `continuous_control_v12` | Corrected simulator adapter, Learned-H1 ensemble, and CEM | `canonical` |
| `active_diagnosis_v13` | Gain fault, Branch-A probe/replan, and continuation evidence | `canonical` |
| `vlm_optics_benchmark` | External validation and visual abnormality evaluation | `supporting` |
| `qwen_vl_supervisor_v1` | QLoRA engineering and strict orchestration contract; formal evaluation not completed | `supporting` |
| `qwen_reasoning_plan_selector_candidate` | Four-step plan bank and Qwen selector study | `candidate-only` |
| `reflection_width_relative` | Width-relative reflection study that failed one preregistered IID stability clause | `candidate-only` |
| `transition_pilot/` | Pre-existing untracked transition comparison source | `unknown` in Git; its local report labels results candidate-only |
| `optics_data_prepare/` | Pre-existing untracked general optics dataset project | `unknown` relative to this controller repository |

## Active versus legacy implementations

The canonical mapping established by this cleanup is:

- simulator: `continuous_control_v12.simulator` over `optical_sim.src`, using
  `continuous_control_v12/config_v12_semantics_v2.json`;
- Learned-H1: `continuous_control_v12.world_model.ForwardEnsemble`;
- CEM: `continuous_control_v12.mpc.CEMMPC`;
- Branch-A and continuation behavior: retained v13 behavior, centralized in
  `configs/controller/branch_a.json`;
- supervisor guard: `qwen_vl_supervisor_v1.closed_loop_adapter`.

`legacy/code/` contains superseded rebuilds and pipelines. It remains in its
existing location rather than being renamed to `archive/legacy_code/` because
historical reports and absolute imports refer to the current paths.

## Duplicate or conflicting implementations

1. `qwen_reasoning_plan_selector_candidate.core.PlanCEM` allows at most four
   control steps. The frozen Branch-A controller uses a visible continuation
   rule after a four-step prefix and permits at most eight real actuator steps.
   The four-step runner is candidate-only and was not selected as canonical.
2. `continuous_control_v12/config_v12.json` contains an older full design with
   horizon-four model rollouts and a large CEM budget. The corrected evidence
   uses `config_v12_semantics_v2.json` plus one-step Learned-H1 replanning and
   the v13 controller budget. Both configs are preserved because they describe
   different experiments.
3. Older rebuild packages v3-v11 contain forward, inverse, and controller
   implementations. They are preserved under `legacy/code/rebuilds/`.
4. `optical_sim.src` is the physics engine, while
   `continuous_control_v12.simulator` is the corrected canonical adapter. They
   are layers, not independent canonical simulators.
5. Some candidate plan labels were produced before controller-wrapper fixes.
   The tracked 2026-08-06 report records both pre-fix and post-fix hashes; old
   labels cannot be silently treated as post-fix evidence.

## Dataset inventory

| Dataset/family | Location or manifest | State | Notes |
|---|---|---|---|
| Original simulator sweeps | `optical_sim/outputs/` | Local ignored | `random_v2` is about 40 GiB; not moved |
| Continuous v12 corrected transitions | `runs/overnight_v12_semantics_20260731_002709/data/` | Local ignored | Configured/generated corrected 128/16/16 group evidence is present locally |
| Full v12 design | `continuous_control_v12/config_v12.json` | Not assumed generated | Declares 10,500/600/600 groups; documentation is not proof of completion |
| V13 development/protected controller suites | `runs/active_diagnosis_v13_20260801_010051/` | Local ignored, compact tracked report | Development and protected controller evidence are distinct |
| Visual benchmark | `runs/vlm_optics_benchmark_20260801_154812/` | Local ignored | 60-episode external validation artifacts are present locally |
| Qwen supervisor manifests | `qwen_vl_supervisor_v1/manifests/manifest_index.json` | Tracked index, local JSONL/images | Progress log records 96 train, 36 dev, 72 frozen-IID, and 60 saturation-only frozen-OOD records |
| Width-relative reflection | `reflection_width_relative/split_manifest.json` and suites | Compact metadata tracked, raw data ignored | 30 train setups, 12 development setups, 30 IID setups; severity-OOD invalidated before a result |
| Specialist rebuild v2 | External paths documented in `specialist_rebuild_v2/README.md` | Not verified in repository root | README describes a 3,700-setup target; source alone is not completion proof |
| Candidate transition pilot v2 | `artifacts/transition_pilot_v2_20260806_201524/` | Local ignored | Candidate-only; 1,152-row full candidate evaluation; no closed-loop CEM feasibility result |
| General optics data preparation | `optics_data_prepare/` | Pre-existing untracked | Separate rules and generated data; requires owner review before integration |
| Tiny fixture | `data/examples/synthetic_transition_v1.json` | Tracked, synthetic | Contract/smoke fixture only; not benchmark evidence |

Redistribution/licensing status is unknown for every local generated family
except the newly created synthetic fixture, which contains no external data.

## Checkpoint and artifact inventory

| Artifact root | Approximate size | Contents/status |
|---|---:|---|
| `artifacts/transition_pilot_v2_20260806_201524` | 7.8 GiB | Qwen/H1 candidate transition checkpoints, data, predictions, reports |
| `artifacts/qwen_h1_selector_v1` | 4.7 GiB | Qwen selector cross-validation/checkpoints |
| `artifacts/qwen_reasoning_plan_selector` | 4.5 GiB | Plan-selector QLoRA checkpoints |
| `qwen_vl_supervisor_v1/artifacts` | 4.2 GiB | Qwen supervisor smoke/pilot checkpoints and resume evidence |
| `runs/overnight_v12_semantics_20260731_002709` | 1.1 GiB | Corrected v12 data, Learned-H1 checkpoints, audits, reports |
| `runs/active_diagnosis_v13_20260801_010051` | 538 MiB | Branch-A estimators, Qwen gain adapter, development/protected results |
| `runs/v12_mpc_h1_h3_diagnosis_20260731_111829` | 45 MiB | Controller diagnostic suite and reports |
| `runs/vlm_optics_benchmark_20260801_154812` | 3.8 MiB | External controller and visual benchmark evidence |

The canonical local Learned-H1 checkpoint is 2.8 MiB and has SHA-256
`d9b30627c80817f6ecade1959d8cc9e91e7a9de9cc51485153fdbfaa173aca2e`.
It remains ignored and is referenced through `configs/controller/branch_a.json`.

The largest file is a 3.43 GiB Qwen resume-state file under the transition
pilot. Several additional resume states are about 494 MiB each; Qwen adapter
weights are about 165 MiB each. An untracked nested environment contains a
roughly 434 MiB Torch shared library. None was added to Git.

## Result-directory inventory

- `runs/`: 1.7 GiB local experiment trees.
- `artifacts/`: 18 GiB local checkpoints, predictions, and candidate reports.
- `optical_sim/outputs/`: 40 GiB simulation output.
- `qwen_vl_supervisor_v1/artifacts/`: 4.2 GiB local Qwen engineering output.
- `reflection_width_relative/`: about 8 MiB; compact reports/configs/joblib are
  tracked while images, JSONL, and PyTorch weights are ignored.
- `legacy/`: about 16 GiB locally because ignored generated content remains
  beside 1,007 tracked archival files.

The repository is about 81 GiB apparent outside `.git`, but the largest tracked
file at baseline was about 341 KiB. The size problem is local ignored state,
not the current tracked snapshot.

## Entry points and tests

The repository had many direct `argparse` modules but no root package metadata
or single command index. Major existing entry points include:

- `continuous_control_v12.generate_dataset`, `train_forward_model`, `run_mpc`;
- `active_diagnosis_v13.run_gate_a` and external validation helpers;
- `qwen_vl_supervisor_v1.train_qlora`, `evaluate_offline`, and freeze tooling;
- `qwen_reasoning_plan_selector_candidate.run_rollouts` and selector tooling;
- `vlm_optics_benchmark.external_validation` and visual benchmarks.

Tests existed inside each active package, but default root discovery also risked
collecting archived or untracked projects. `pyproject.toml` now declares the
reviewed active test roots and an `optics-vla` validation CLI.

## Machine-specific assumptions and stale paths

Active machine-specific paths were found in:

- `active_diagnosis_v13/config_v13.json`;
- `active_diagnosis_v13/qwen_gain_qlora.yaml`;
- historical commands in `continuous_control_v12/README.md`;
- `control_rebuild_v5` shell/systemd files;
- Qwen supervisor training configs and command documents;
- reflection suite/config metadata; and
- the pre-existing untracked transition-pilot source.

Many additional paths occur under `legacy/`. Some are provenance (the machine
where an experiment ran), while others are stale executable defaults. Hash-
pinned historical configs were not rewritten because doing so would invalidate
their recorded evidence. New canonical config and commands use relative paths.

The original system Python lacked `joblib`, `pytest`, `torch`, and
`transformers`. A retained Python 3.11 environment had those dependencies.
Tests requiring local run data or checkpoints will be blocked in a fresh
public clone even if imports succeed. `optical_sim/README.md` also contains a
stale link to a root `profile2setup/` path that is now archived.

## Secret and private-information audit

No dedicated scanner (`gitleaks`, `trufflehog`, or `detect-secrets`) was
installed. A careful current-tree filename and content-pattern audit covered
tracked and visible untracked files without printing candidate values.

- No credential-like filenames or private-key blocks were found.
- One high-entropy token pattern was a false positive inside a hyphenated
  strategy name in a report.
- API-key assignment candidates were limited to legacy client code that reads
  keys from its runtime interface/environment and a README instruction.
- No email-address or explicit phone/address/passport/SSN pattern was found in
  the current tracked/visible-untracked content scan.
- Numerous personal absolute paths remain as historical provenance or stale
  configuration, as listed above.

This was not a full Git-history secret scan. Public release remains blocked on
a dedicated history scan and owner review of ignored metadata/job records.

## Safe to archive versus human review

Already safely archived and retained:

- superseded rebuilds v3-v11 under `legacy/code/rebuilds/`;
- older orchestration/profile/Qwen pipelines under `legacy/code/pipelines/`;
- compact historical reports under `legacy/experiments/`.

Deletion candidates, intentionally not deleted:

- bytecode, pytest/ruff caches, nested virtual environments, tensorboard/W&B
  logs, smoke checkpoints, optimizer states, and duplicate generated media;
- repeated Qwen intermediate checkpoints after final adapter verification;
- duplicate local raw result trees already represented by checksummed compact
  reports;
- old systemd units and machine-specific launch scripts after provenance review.

Human review is required before deleting or publishing:

- any raw dataset, checkpoint, adapter, result, or artifact tree;
- the pre-existing untracked `optics_data_prepare/` and `transition_pilot/`;
- the candidate four-step PlanCEM implementation;
- hash-pinned configs containing historical absolute paths;
- tracked joblib models and retained qualitative images;
- all legacy code with unclear third-party licensing;
- any file whose only replacement is an ignored local artifact.
