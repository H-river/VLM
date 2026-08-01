# Qwen + continuous-action v12 evaluation report

Run ID: `qwen_runner_smoke_final_20260731`  
Start: `2026-07-31T12:12:35.869022+00:00`  
End: `2026-07-31T12:16:47.461781+00:00`  
Elapsed: `251.6` seconds  
Cases: `9` (`7` ready)

## Main results

| Task | Modality | N | Specialist accuracy | E2E accuracy | Qwen/contract gap | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| direction_prediction_v12 | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| direction_prediction_v12 | state | 1 | 100.00% | 0.00% | 100.00% | [0.00%, 79.35%] |
| forward_prediction_v12 | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| forward_prediction_v12 | state | 1 | 100.00% | 0.00% | 100.00% | [0.00%, 79.35%] |
| inverse_control_v12 | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| inverse_control_v12 | state | 1 | 100.00% | 0.00% | 100.00% | [0.00%, 79.35%] |
| measurement | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| Four-task macro | Balanced | 7 | 37.50% | 0.00% | 37.50% | [0.00%, 0.00%] |

The v12 correct-route specialist four-task macro accuracy is **37.50%**. Qwen + contract + adapter + v12 E2E macro accuracy is **0.00%**. The measured Qwen/contract gap is **37.50%**.

Ready micro specialist/E2E accuracy is 42.86%/0.00%. Schema validity is 22.22%, routing accuracy 28.57%, argument extraction 14.29% overall and 0.00% conditional on a correct route, and unit conversion 14.29%. Clarification and unsupported rejection accuracy are 0.00%/0.00%.

Direction per-field specialist/E2E accuracy is 50.00%/0.00%; the table uses strict all-five accuracy. Route-balanced state/image E2E accuracy is 0.00%/0.00%, a state-minus-image gap of 0.00%.

Image measurement strict success is 0.00% on the correct-route specialist and 0.00% after Qwen routing. The ground-truth-state v12 downstream counterpart succeeds on 100.00%; per-image normalized measurement errors, the direct-state downstream outcome, and the full image E2E outcome are all retained per case.

Inverse Learned H1 specialist/E2E success is 50.00%/0.00%, specialist mean steps 0.00, and specialist/E2E actuator violations 0/0. Correct-route inverse failure attribution is `{"measurement": 1}`; model-prediction attribution requires recorded planner-exploitation evidence, otherwise a valid H1 miss is assigned to search/planning.

Latency average/P50/P95 is 25.64/29.66/37.18 seconds.

## Failure localization

Primary failure counts: `{"Qwen invalid JSON/schema": 7, "missing/wrong argument": 1, "wrong task/route": 1}`. The largest observed stage is **Qwen invalid JSON/schema**. Route-by-stage equivalent confusion statistics are `{"inverse_control_from_images_v12_h1": {"Qwen invalid JSON/schema": 1}, "inverse_control_from_states_v12_h1": {"Qwen invalid JSON/schema": 1}, "measure_beam_profile_v12": {"Qwen invalid JSON/schema": 1}, "needs_clarification": {"Qwen invalid JSON/schema": 1}, "predict_direction_from_image_v12": {"Qwen invalid JSON/schema": 1}, "predict_direction_from_state_v12": {"wrong task/route": 1}, "predict_forward_from_image_v12": {"Qwen invalid JSON/schema": 1}, "predict_forward_from_state_v12": {"Qwen invalid JSON/schema": 1}, "unsupported": {"missing/wrong argument": 1}}`. Failures are assigned exactly one primary stage; secondary tags are retained in `failure_analysis.csv`.

## Acceptance and interpretation

Integration acceptance passed before this frozen run: schema/registry validation, unit/order/bounds/no-op tests, state/image v12 dispatch, a non-trivial Learned H1 closed-loop episode, checkpoint/config integrity, and old-route regressions. Formal cases were then run without changing code, config, checkpoint, or manifest.

The routing-versus-argument split is given by route accuracy (28.57%) versus conditional argument accuracy (0.00%). State/image and specialist/E2E gaps are quantified above and in the table. Inverse failures tagged `v12 forward prediction` had planner-exploitation evidence; remaining valid-planner misses are attributed to H1 search/planning.

Top priorities from the three largest observed stages are: adapt Qwen explicitly to the v12 JSON contract and exact route argument groups; improve exact field copying and clarification reason extraction; improve v12 task and state/image route classification.

## Known limitations

This is a single-seed diagnostic with 1 ready cases per route. The Qwen adapter was trained on the v1 route contract, not v12, and is not formally promoted. The v12 checkpoint used only 128 training groups; its preregistered H3 accumulation gate failed, so H3 was excluded. Image-route assets were deterministically materialized from protected numerical test states because the protected v12 manifest stored no images.

Exact reproduction:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python -m Qwen_orchestration.scripts.evaluate_v12_e2e --run-dir /home/jiamo/VLM/artifacts/v12_qwen_e2e/qwen_runner_smoke_final_20260731
```
