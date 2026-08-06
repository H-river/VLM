# Qwen H1 meta-controller candidate progress log

## 2026-08-02 Phase 0 — read-only audit

- Preserved initial `git status`: branch `v13`; pre-existing untracked paths are
  `qwen_vl_supervisor_v1/artifacts/pilot/`,
  `qwen_vl_supervisor_v1/artifacts/training/pilot_qwen25vl_3b_full96_dev36_seed_2026080101_200step/`,
  `qwen_vl_supervisor_v1/configs/training_pilot_local_200step.yaml`, and
  `supervisor_v1_1_candidate/`. None were modified or removed.
- No repository `AGENTS.md` exists.
- Canonical state/action contracts:
  `specialist_rebuild_v2/common.py` and `continuous_control_v12/contracts.py`.
- Learned ensemble interface: `continuous_control_v12/world_model.py::ForwardEnsemble.predict`.
- H1 entry points: `continuous_control_v12/mpc.py::CEMMPC.plan`,
  `learned_predictor`, and `run_closed_loop`.
- Measurement conversion: `Qwen_orchestration/v12/adapter.py::V12Adapter.measure`;
  sensor image frame and lab pseudo-pixel numerical frame remain distinct.
- Supervisor boundary: `qwen_vl_supervisor_v1/closed_loop_adapter.py`.
- Registry finding: no controller factory; H1 is constructed inline by
  `Qwen_orchestration/v12/adapter.py::V12Adapter.inverse`.
- Safety finding: no standalone complete `SafetyGate`; safety is distributed.
  The candidate therefore needs a reject-only additive dispatch gate.
- Guidance-hook finding: existing CEM hard-codes initial mean zero and standard
  deviation `0.75 * action_high`. Candidate code must wrap it additively and
  prove off-mode equivalence, not copy or rewrite the controller.
- Default-H1 ambiguity resolved before result generation: the prompt's 37/48
  reference maps to
  `runs/v12_mpc_h1_h3_diagnosis_20260731_111829/configs/locked_primary_config.json`
  (SHA-256 `2a9694eb...`), not the 256/32/5 orchestration runtime.
- Existing data direct-reuse audit: corrected v12 train/dev has numeric
  transitions but no stored image/target/budget/meta label; direct reuse is
  `RED_FOR_DIRECT_REUSE`. Fresh candidate-only corrected simulation is allowed.
- Protected/frozen image content was not opened and no new prediction was run
  on any frozen/protected split. Identity-only registry metadata was used only
  for overlap planning.

### Pre-change regression baseline

- H1/contracts: 12 passed.
- Supervisor adapter/state-machine contract: 31 passed.
- Simulator/MPC: 15 passed, 1 pre-existing failure.
- Pre-existing failure:
  `test_simulator_semantics_v12.py::test_legacy_128_outputs_remain_bitwise_unchanged`;
  repeated legacy intermediate-array SHA drift under Python 3.12.3, NumPy
  1.26.4, SciPy 1.11.4. Corrected-v12 tests passed. This was present before
  candidate changes and is not relaxed.
- Deterministic in-memory H1 trace fixture: H1, population 24, elites 6,
  iterations 2, seed 90210; repeat-equal; 48 backend calls; trace SHA-256
  `8ff4d490beffa84faa60c295f5e58c591eadaa98ba55b8e08d125bb255fe12e4`.
- Key source hashes are frozen in `protocol/meta_controller_protocol.json`.

## 2026-08-02 Phase 1 — preregistration freeze

- Frozen before generating fresh data, labels, dev metrics, training, or
  candidate closed-loop results:
  `protocol/input_schema.json`, `protocol/output_schema.json`,
  `protocol/guidance_codebook.json`, `protocol/compiler_mapping.json`, and
  `protocol/meta_controller_protocol.json`.
- Status remains candidate-only, not sealed, and frozen evaluation disabled.
- Qwen-visible setup context is explicitly forbidden. If real action-response
  history plus the other runtime-visible fields are insufficient, the task
  stops RED rather than adding hidden simulator fields.

## 2026-08-02 Phases 2–4 — additive implementation and fresh data

- Added an independent strict meta JSON contract, deterministic compiler,
  unchanged-default/off path, shadow path, guarded default-versus-guided H1
  arbiter, reject-only dispatch checks, supervisor-first state-machine
  integration, independent measurement/control budgets, and H1-only guards.
- Added candidate-local QLoRA reuse, exact prompt validation, full-dev raw
  generation, offline field/configuration metrics, rule/frequency/random and
  train-only-standardized numeric MLP baselines, paired candidate simulator
  evaluation, and twelve fixed reasoning-ablation surfaces. None of these
  paths changes the existing supervisor adapter, forward checkpoint, CEM
  source, frozen artifacts, or H1 strict-success definition.
- An initial generation process that had imported an earlier in-development
  source revision was interrupted and moved intact to
  `artifacts/interrupted_generation_old_import_20260802T143042`. It is not an
  official artifact, was not resumed, and is excluded from result hashes.
- The final-source CPU generator produced 48 train, 24 dev, and 36
  candidate-eval rows from 16/8/12 disjoint setups and three targets per setup.
  It made 1,722 simulator calls, 3,528 planner calls, and 3,672 physical
  scoring calls in 1,082.734 seconds. `/usr/bin/time -v` measured 18:03.94
  wall time and 933,792 KiB maximum RSS.
- Identity audit: 298 identity-only known records, zero known overlap, zero
  cross-split exact/rounded overlap, and no rejected duplicates.

## 2026-08-02 Information-sufficiency hard stop

- The official train+dev audit covered all 72 rows. Exact and rounded visible
  collision conflicts were both zero, and adding hidden setup features gave
  0.0 macro-F1 gain.
- The preregistered hard gate nevertheless returned `RED_STOP`: 54/54 nearby
  visible pairs had different finite configuration labels (rate 1.0 versus
  maximum 0.20), and the setup-grouped visible-only nearest-centroid macro-F1
  was 0.019166 versus minimum 0.50.
- The generator exited with status 3 and deliberately wrote no
  `prebuilt_chat_train.jsonl`, `prebuilt_chat_dev.jsonl`, or SFT export report.
  Therefore no training config, QLoRA checkpoint, dev prediction, offline
  performance result, candidate closed-loop run, or reasoning-ablation run was
  started. Their required result surfaces explicitly record
  `not_run_due_to_information_sufficiency_gate` and contain no fabricated
  metrics.

## 2026-08-02 Final fail-closed review and regression

- A final adversarial code review found that the first implementation did not
  sufficiently re-check the RED gate at every future entry point. The current
  code now rejects an alternate identity registry before opening it, pins the
  official registry bytes, requires an exact PASS audit and hash-linked 48/24
  SFT bundle before config generation or training, and revalidates every
  frozen QLoRA/model/quantization setting.
- Offline reduction now requires exactly 24 unique dev predictions for each of
  all three preregistered seeds. Candidate-eval ablation diagnostics cannot be
  presented as dev metrics. The closed-loop loader rechecks the canonical base
  config and all frozen CEM/world/forward/simulator hashes.
- The simulator harness explicitly conditions on synthetic candidate-manifest
  supervisor validity; it does not run the actual Qwen anomaly supervisor, and
  no sequential reobserve recovery backend is available. It therefore cannot
  support full-stack, end-to-end, temporal-recovery, or hardware claims.
- The public controller accepts only the exact preregistered default H1 or the
  named 48x3 compute-fair dual-budget profile. Reporting ignores self-declared
  gain/reasoning flags and requires complete paired episodes, all three seeds,
  setup-bootstrap confidence bounds, fixed reasonable baselines, and the
  preregistered ablation evidence.
- Final combined regression under the optical-sim environment: 170 passed,
  zero failed, 4.20 seconds wall time, 152,892 KiB peak RSS. A separate final
  protocol/source verification revalidated every frozen protocol, CEM, world
  model, forward checkpoint, simulator config, base config, and locked H1 hash.
