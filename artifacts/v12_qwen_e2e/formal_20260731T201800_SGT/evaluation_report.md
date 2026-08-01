# Qwen + continuous-action v12 evaluation report

Run ID: `formal_20260731T201800_SGT`  
Start: `2026-07-31T12:18:48.303904+00:00`  
End: `2026-07-31T13:02:40.837403+00:00`  
Elapsed: `2632.5` seconds  
Cases: `105` (`70` ready)

## Main results

| Task | Modality | N | Specialist accuracy | E2E accuracy | Qwen/contract gap | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| direction_prediction_v12 | image | 10 | 0.00% | 0.00% | 0.00% | [0.00%, 27.75%] |
| direction_prediction_v12 | state | 10 | 100.00% | 30.00% | 70.00% | [10.78%, 60.32%] |
| forward_prediction_v12 | image | 10 | 0.00% | 0.00% | 0.00% | [0.00%, 27.75%] |
| forward_prediction_v12 | state | 10 | 100.00% | 50.00% | 50.00% | [23.66%, 76.34%] |
| inverse_control_v12 | image | 10 | 0.00% | 0.00% | 0.00% | [0.00%, 27.75%] |
| inverse_control_v12 | state | 10 | 100.00% | 0.00% | 100.00% | [0.00%, 27.75%] |
| measurement | image | 10 | 0.00% | 0.00% | 0.00% | [0.00%, 27.75%] |
| Four-task macro | Balanced | 70 | 37.50% | 10.00% | 27.50% | [5.00%, 15.00%] |

The v12 correct-route specialist four-task macro accuracy is **37.50%**. Qwen + contract + adapter + v12 E2E macro accuracy is **10.00%**. The measured Qwen/contract gap is **27.50%**.

Ready micro specialist/E2E accuracy is 42.86%/11.43%. Schema validity is 21.90%, routing accuracy 20.00%, argument extraction 15.71% overall and 57.14% conditional on a correct route, and unit conversion 15.71%. Clarification and unsupported rejection accuracy are 0.00%/0.00%.

Direction per-field specialist/E2E accuracy is 50.00%/22.00%; the table uses strict all-five accuracy. Route-balanced state/image E2E accuracy is 26.67%/0.00%, a state-minus-image gap of 26.67%.

Image measurement strict success is 0.00% on the correct-route specialist and 0.00% after Qwen routing. The ground-truth-state v12 downstream counterpart succeeds on 100.00%; per-image normalized measurement errors, the direct-state downstream outcome, and the full image E2E outcome are all retained per case.

Inverse Learned H1 specialist/E2E success is 50.00%/0.00%, specialist mean steps 0.40, and specialist/E2E actuator violations 0/0. Correct-route inverse failure attribution is `{"measurement": 10}`; model-prediction attribution requires recorded planner-exploitation evidence, otherwise a valid H1 miss is assigned to search/planning.

Latency average/P50/P95 is 24.88/29.12/38.37 seconds.

## Failure localization

Primary failure counts: `{"Qwen invalid JSON/schema": 82, "missing/wrong argument": 6, "wrong status": 4, "wrong task/route": 3, "unit/sign/order error": 2}`. The largest observed stage is **Qwen invalid JSON/schema**. Route-by-stage equivalent confusion statistics are `{"inverse_control_from_images_v12_h1": {"Qwen invalid JSON/schema": 10}, "inverse_control_from_states_v12_h1": {"Qwen invalid JSON/schema": 10}, "measure_beam_profile_v12": {"Qwen invalid JSON/schema": 8, "wrong status": 2}, "needs_clarification": {"Qwen invalid JSON/schema": 18, "wrong status": 2}, "predict_direction_from_image_v12": {"Qwen invalid JSON/schema": 10}, "predict_direction_from_state_v12": {"wrong task/route": 3, "Qwen invalid JSON/schema": 2, "unit/sign/order error": 2}, "predict_forward_from_image_v12": {"Qwen invalid JSON/schema": 10}, "predict_forward_from_state_v12": {"Qwen invalid JSON/schema": 5}, "unsupported": {"missing/wrong argument": 6, "Qwen invalid JSON/schema": 9}}`. Failures are assigned exactly one primary stage; secondary tags are retained in `failure_analysis.csv`.

## Acceptance and interpretation

Integration acceptance passed before this frozen run: schema/registry validation, unit/order/bounds/no-op tests, state/image v12 dispatch, a non-trivial Learned H1 closed-loop episode, checkpoint/config integrity, and old-route regressions. Formal cases were then run without changing code, config, checkpoint, or manifest.

The routing-versus-argument split is given by route accuracy (20.00%) versus conditional argument accuracy (57.14%). State/image and specialist/E2E gaps are quantified above and in the table. Inverse failures tagged `v12 forward prediction` had planner-exploitation evidence; remaining valid-planner misses are attributed to H1 search/planning.

Top priorities from the three largest observed stages are: adapt Qwen explicitly to the v12 JSON contract and exact route argument groups; improve exact field copying and clarification reason extraction; strengthen ready versus clarification versus unsupported classification.

## Known limitations

This is a single-seed diagnostic with 10 ready cases per route. The Qwen adapter was trained on the v1 route contract, not v12, and is not formally promoted. The v12 checkpoint used only 128 training groups; its preregistered H3 accumulation gate failed, so H3 was excluded. Image-route assets were deterministically materialized from protected numerical test states because the protected v12 manifest stored no images.

Exact reproduction:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python -m Qwen_orchestration.scripts.evaluate_v12_e2e --run-dir /home/jiamo/VLM/artifacts/v12_qwen_e2e/formal_20260731T201800_SGT
```
