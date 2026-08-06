# Candidate command ledger

CANDIDATE ONLY — NOT SEALED — FROZEN EVALUATION DISABLED

Only commands explicitly supplied in the candidate command ledger are reproduced here. This renderer does not infer shell history and executes none of these commands.

## 1. Generate and audit preregistered candidate-only data

```bash
PYTHONPATH=/home/jiamo/VLM /usr/bin/time -v /home/jiamo/miniconda3/envs/optical_sim/bin/python -m qwen_h1_meta_v0_candidate.data_pipeline generate --output-root /home/jiamo/VLM/qwen_h1_meta_v0_candidate/data/generated_v1 --identity-registry /home/jiamo/VLM/qwen_h1_meta_v0_candidate/configs/known_identity_blocklist.json --device cpu
```

- Status/exit: `preregistered information-sufficiency RED_STOP`.
- Wall time seconds: `1082.7339`.
- Peak memory bytes: `956203008`.
- Notes: /usr/bin/time wall=18:03.94; CPU=100%; SFT exported=False.

## 2. Preserve and recheck the repository worktree boundary

```bash
git diff --name-only; git status --short
```

- Status/exit: `PASS`.
- Wall time seconds: `not available`.
- Peak memory bytes: `not available`.
- Notes: No tracked existing file changed; pre-existing untracked paths remain present, and the candidate is confined to qwen_h1_meta_v0_candidate..

## 3. Final combined candidate and existing targeted regression

```bash
PYTHONPATH=/home/jiamo/VLM PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/time -v /home/jiamo/miniconda3/envs/optical_sim/bin/python -m pytest -q qwen_h1_meta_v0_candidate/tests continuous_control_v12/tests/test_contracts.py continuous_control_v12/tests/test_mpc_h1_h3_diagnosis.py qwen_vl_supervisor_v1/tests/test_closed_loop_adapter.py continuous_control_v12/tests/test_simulator_mpc.py continuous_control_v12/tests/test_simulator_semantics_v12.py
```

- Status/exit: `PASS`.
- Wall time seconds: `4.2000`.
- Peak memory bytes: `156561408`.
- Notes: 170 passed, zero failed; output captured in artifacts/evidence/logs/final_post_hardening_targeted_pytest.log..

## 4. Revalidate every frozen candidate protocol and source identity

```bash
PYTHONPATH=/home/jiamo/VLM /home/jiamo/miniconda3/envs/optical_sim/bin/python -m qwen_h1_meta_v0_candidate.data_pipeline verify-protocol
```

- Status/exit: `PASS`.
- Wall time seconds: `not available`.
- Peak memory bytes: `not available`.
- Notes: All protocol, prompt, locked H1, CEM, world-model, forward-checkpoint, corrected-simulator and base-config hashes matched..

## 5. Build deterministic RED candidate reports and artifact inventory

```bash
PYTHONPATH=/home/jiamo/VLM /home/jiamo/miniconda3/envs/optical_sim/bin/python -m qwen_h1_meta_v0_candidate.reporting build --data-audit /home/jiamo/VLM/qwen_h1_meta_v0_candidate/data/generated_v1/reports/data_audit.json --offline /home/jiamo/VLM/qwen_h1_meta_v0_candidate/offline_reasoning_results.json --closed-loop /home/jiamo/VLM/qwen_h1_meta_v0_candidate/closed_loop_results.json --ablation /home/jiamo/VLM/qwen_h1_meta_v0_candidate/ablation_results.json --regression /home/jiamo/VLM/qwen_h1_meta_v0_candidate/artifacts/evidence/regression_results.json --execution-evidence /home/jiamo/VLM/qwen_h1_meta_v0_candidate/artifacts/evidence/execution_evidence.json --command-log /home/jiamo/VLM/qwen_h1_meta_v0_candidate/artifacts/evidence/command_log.json --output-root /home/jiamo/VLM/qwen_h1_meta_v0_candidate
```

- Status/exit: `PASS`.
- Wall time seconds: `not available`.
- Peak memory bytes: `not available`.
- Notes: No model, simulator evaluation, or frozen/protected content is opened by the reporter..

## 6. Validate report terminal boundary and every stable artifact hash

```bash
PYTHONPATH=/home/jiamo/VLM /home/jiamo/miniconda3/envs/optical_sim/bin/python -m qwen_h1_meta_v0_candidate.reporting validate --output-root /home/jiamo/VLM/qwen_h1_meta_v0_candidate
```

- Status/exit: `PASS`.
- Wall time seconds: `not available`.
- Peak memory bytes: `not available`.
- Notes: The hash manifest excludes itself, resumable partials, isolated interrupted output, caches and symlinks by explicit policy..

## Evidence ledger

| Role | Candidate-relative path | SHA-256 |
|---|---|---|
| ablation | `ablation_results.json` | `e90a635c3fbcbd5deb06c7387814baa20cc5ce41fcc8f6278e928b339900c690` |
| closed_loop | `closed_loop_results.json` | `76234beb2d0ea3446a846acd153875ad7129338cb6ae51075f791c58154f1038` |
| command_log | `artifacts/evidence/command_log.json` | `5315232bef9b31fe8b2f0416fe6b0b6fd9bb55e108c505ddef4328a5354d0006` |
| data_audit | `data/generated_v1/reports/data_audit.json` | `fd261540e3d94f20d9a6b43179cfd03d81fda75d329c0d22ae8662baf5602d27` |
| execution | `artifacts/evidence/execution_evidence.json` | `120e7cbeef29dda8359733dda14f4451d8b09085ecf519ea5df8baaa5b0eda7d` |
| offline | `offline_reasoning_results.json` | `76eab1b82c3bc9105f80964ea877166c543620dce900fc236388d8ce7c88ca90` |
| regression | `artifacts/evidence/regression_results.json` | `5065077ab820a7220a3cd9093b9c34c9f23f1065ec57792ccf02c351552ee392` |
