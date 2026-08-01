# Width-relative reflection reproduction commands

Run from `/home/jiamo/VLM` with `/home/jiamo/miniconda3/envs/optical_sim/bin/python`. Frozen one-shot evidence files must be inspected or re-analyzed, not regenerated.

## Tests and old deterministic replay

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m pytest -q vlm_optics_benchmark/tests
```

The old pair replay used the previous `visual_anomalies generate --split train` command in a fresh temporary directory, then byte-compared `pairs_train.jsonl` and SHA-256-compared `img_f365ee75da91a2177d4ab2dc.png`. The old control replay command was:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m pytest -q vlm_optics_benchmark/tests/test_benchmark_contracts.py::test_deterministic_episode_replay_per_stratum_and_anomaly
```

## New suite configs and development suites

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative prepare-suite-config --source runs/v12_mpc_h1_h3_diagnosis_20260731_111829/configs/locked_primary_config.json --root-seed 2026082101 --label development --output reflection_width_relative/configs/development_suite_config.json

/home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative prepare-suite-config --source runs/v12_mpc_h1_h3_diagnosis_20260731_111829/configs/locked_primary_config.json --root-seed 2026082102 --label train --output reflection_width_relative/configs/train_suite_config.json
```

The suite-generator invocations used suite labels `reflsig_dev_20260801` with 12 groups and `reflsig_train_20260801` with 30 groups. Both excluded the prior primary suite, prior VLM-optics external suite, all three v13 fresh-holdout suites, and—after it existed—every earlier new suite. Exact seeds, paths and suite hashes are in `preregistration.json` and `split_manifest.json`.

## Development search, selected data and models

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative development-search --suite reflection_width_relative/suites/development_suite.json --search-preregistration reflection_width_relative/development_search_preregistration.json --v12-config continuous_control_v12/config_v12_semantics_v2.json --base-config optical_sim/configs/base_config.yaml --montage-dir reflection_width_relative/development_montage --output reflection_width_relative/development_search.json

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative generate-dataset --suite reflection_width_relative/suites/development_suite.json --split development --parameter-config reflection_width_relative/development_protocol.json --v12-config continuous_control_v12/config_v12_semantics_v2.json --base-config optical_sim/configs/base_config.yaml --data-dir reflection_width_relative/data

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative generate-dataset --suite reflection_width_relative/suites/train_suite.json --split train --parameter-config reflection_width_relative/development_protocol.json --v12-config continuous_control_v12/config_v12_semantics_v2.json --base-config optical_sim/configs/base_config.yaml --data-dir reflection_width_relative/data

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative train-development-models --data-dir reflection_width_relative/data --model-dir reflection_width_relative/models --protocol reflection_width_relative/development_protocol.json --output reflection_width_relative/development_model_selection.json
```

`preregistration.json` was written after these commands and before either held-out suite or held-out image generation.

## Frozen held-out generation

The one-shot IID suite used 30 groups, root seed 2026082103 and label `reflsig_iid_20260801`; the optional OOD suite used 18 groups, seed 2026082104 and label `reflsig_ood_20260801`. In addition to all old exclusions, IID excluded new development and train; OOD also excluded IID. Do not rerun either suite.

The IID data command was:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative generate-dataset --suite reflection_width_relative/suites/iid_heldout_suite.json --split iid_heldout --parameter-config reflection_width_relative/preregistration.json --v12-config continuous_control_v12/config_v12_semantics_v2.json --base-config optical_sim/configs/base_config.yaml --data-dir reflection_width_relative/data
```

The analogous `--split severity_ood` command failed its frozen serialized counterfactual threshold and must not be retried. See `postfreeze_corrections.json`.

## Frozen identifiability and control

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative evaluate-frozen-models --data-dir reflection_width_relative/data --model-dir reflection_width_relative/models --preregistration reflection_width_relative/preregistration.json --output reflection_width_relative/identifiability_results.json --leakage-output reflection_width_relative/leakage_audit.json

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative run-control --suite reflection_width_relative/suites/iid_heldout_suite.json --preregistration reflection_width_relative/preregistration.json --output reflection_width_relative/control_value_episodes.jsonl

/home/jiamo/miniconda3/envs/optical_sim/bin/python -m vlm_optics_benchmark.reflection_width_relative analyze-control --source reflection_width_relative/control_value_episodes.jsonl --identifiability reflection_width_relative/identifiability_results.json --data-dir reflection_width_relative/data --paired-output reflection_width_relative/paired_counterfactuals.jsonl --episode-csv reflection_width_relative/episode_results.csv --output reflection_width_relative/control_value_results.json
```

The evaluator and analyzer are deterministic offline reads, but the existing result artifacts should be used for reporting so the one-shot boundary remains explicit.
