# Qwen + continuous-action v12 stopped status report

## Current status

The integration implementation, deterministic adapter, contracts, tests, frozen-manifest builder, evaluator, and same-environment route smoke are complete. No valid full 105-case formal evaluation completed before the user requested a safe stop, so no formal specialist/E2E macro accuracy is claimed.

## Valid evidence

- New v12 routes remain separate from the legacy v1/81-action registry.
- Unit/contract and continuous-control tests: 51 passed.
- Legacy Qwen orchestration regressions: 24 passed.
- Final smoke in the actual `optics_qlora` evaluation environment: accepted, 35 cases, all seven routes exercised five times, all calls executed by v12, no crash/non-finite value/boundary violation, H1 only, and non-trivial inverse episodes exercised.
- Final-smoke strict specialist correctness: 33/35. By route: measurement 5/5, state direction 5/5, image direction 5/5, state forward 5/5, image forward 3/5, state inverse H1 5/5, image inverse H1 5/5.
- The v12 checkpoint was not trained, modified, or replaced. H3 was not registered or used.

## Invalid formal attempts

1. `formal_20260731T201800_SGT`: completed 105 cases but is invalid because the Qwen environment lacked SciPy; all 40 measurement/image correct-route calls raised `ModuleNotFoundError: No module named 'scipy'`.
2. `formal_replacement_20260731T211500_SGT`: SciPy was fixed and image routes recovered, but the process received external `SIGTERM` after 52/105 cases. It was not resumed.
3. `formal_final_20260731T214500_SGT`: safely stopped by user request after 3/105 cases. No process remains running.

Results from these runs must not be combined, resumed, or presented as a formal accuracy result.

## Environment recovery

- Installed SciPy 1.15.3 in `/home/jiamo/miniconda3/envs/optics_qlora`.
- Added evaluator startup preflight for `torch` and `scipy`.
- Added SciPy to recorded package versions.
- Re-ran all 75 tests and the 35-case route smoke after the change.

## Minimal clean restart

Use a new run directory and rerun from case zero:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python -m Qwen_orchestration.scripts.build_v12_evaluation --output-dir /home/jiamo/VLM/artifacts/v12_qwen_e2e/formal_resume_after_user_stop --ready-per-route 10 --clarification 20 --unsupported 15 --seed 2026073112 --v12-split test --exclude-group v12_test_000000 --exclude-group v12_test_000002 --exclude-measurement-group v2_test_iid_000034 --exclude-measurement-group v2_test_iid_000085
/home/jiamo/miniconda3/envs/optics_qlora/bin/python -m Qwen_orchestration.scripts.evaluate_v12_e2e --run-dir /home/jiamo/VLM/artifacts/v12_qwen_e2e/formal_resume_after_user_stop
```

