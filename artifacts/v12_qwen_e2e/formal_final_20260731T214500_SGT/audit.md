# V12 Qwen integration audit

- Legacy registry remains isolated at `Qwen_orchestration/configs/model_registry.yaml` with seven v1 routes and the unchanged 81-action contract.
- V12 registry contains seven distinct ready routes: one measurement, state/image direction, state/image forward, and state/image inverse H1.
- V12 action order is `['lens_x_delta_mm', 'lens_y_delta_mm', 'camera_x_delta_mm', 'camera_y_delta_mm']` in canonical mm. Bounds are lens +/-0.05 mm and camera +/-0.02 mm per step; all four absolute positions are limited to +/-3 mm by the repository sampling-domain contract.
- V12 model input is 35 canonical features: eight setup values, four positions, five state values, four normalized actions, four action squares, six pairwise action products, and four position-action products. The adapter calls the checkpoint runtime's canonical `structured_features` path; Qwen never emits these engineered features.
- Output order is `['centroid_x_px', 'centroid_y_px', 'sigma_x_px', 'sigma_y_px', 'peak_intensity']`. Targets and decoded deltas use current-state tolerances `[1 px, 1 px, 2 px, 2 px, 5% peak with 1e-6 floor]`.
- Checkpoint: `/home/jiamo/VLM/runs/overnight_v12_semantics_20260731_002709/models/lc_128g_v2/continuous_forward_v12_128g.pt` (`d9b30627c80817f6ecade1959d8cc9e91e7a9de9cc51485153fdbfaa173aca2e`), three MLP members, model config hash `e5bd305245c3a86e82a31a7d1247fcd5a1b9d052bb43388d9cee08965e63e77f`, normalization hash `cda9a13d95345efc2413aadfbb88b732a0b50cab774266bc333eef2f64a57c7b`.
- Auxiliary order confirmed by runtime: log captured power, clipping fraction, camera-boundary logit, actuator-limit logit.
- Measurement backend is guarded measurement v4: measurement v3 plus v4 calibrator inside gamma [0.75, 1.05], otherwise calibrated analytic moments. Metric order and tolerance match v12.
- Learned H1 CEM is frozen to population 256, 32 elites, 5 iterations, horizon 1, max 5 closed-loop steps, strict all-five max normalized error <= 1. Seed is fixed per case. H3 is not registered.
- Protected v12 test data contains 16 groups, has zero train/development overlap, and is hash-pinned. The original manifest did not store images; this freeze deterministically materializes image-route assets from protected test states before any formal result is observed.
- The Qwen checkpoint is an evaluation candidate rather than a formally promoted v12-trained adapter; this limitation is retained for the final report.
- The manifest, Qwen/v12 checkpoints, Qwen base-model shards, schemas, registries, prompts, adapters, measurement code, continuous-control runtime, and evaluation scripts are all SHA-256 pinned in `run_config.json`; the evaluator checks every hash before model loading.
