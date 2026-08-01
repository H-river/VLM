# Qwen + continuous-action v12 evaluation report

Run ID: `qwen_runner_smoke_v2_20260731`  
Start: `2026-07-31T12:01:10.607757+00:00`  
End: `2026-07-31T12:05:51.765514+00:00`  
Elapsed: `281.2` seconds  
Cases: `9` (`7` ready)

## Main results

| Task | Modality | N | Specialist accuracy | E2E accuracy | Qwen/contract gap | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| direction_prediction_v12 | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| direction_prediction_v12 | state | 1 | 100.00% | 0.00% | 100.00% | [0.00%, 79.35%] |
| forward_prediction_v12 | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| forward_prediction_v12 | state | 1 | 100.00% | 100.00% | 0.00% | [20.65%, 100.00%] |
| inverse_control_v12 | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| inverse_control_v12 | state | 1 | 100.00% | 0.00% | 100.00% | [0.00%, 79.35%] |
| measurement | image | 1 | 0.00% | 0.00% | 0.00% | [0.00%, 79.35%] |
| Four-task macro | Balanced | 7 | 37.50% | 12.50% | 25.00% | [12.50%, 12.50%] |

The v12 correct-route specialist four-task macro accuracy is **37.50%**. Qwen + contract + adapter + v12 E2E macro accuracy is **12.50%**. The measured Qwen/contract gap is **25.00%**.

Ready micro specialist/E2E accuracy is 42.86%/14.29%. Schema validity is 33.33%, routing accuracy 14.29%, argument extraction 28.57%, and unit conversion 28.57%. Image measurement strict success is 0.00% on the correct-route specialist and 0.00% after Qwen routing. The ground-truth-state v12 downstream counterpart succeeds on 100.00%; this separates measurement error from downstream model/control error.

Inverse Learned H1 specialist/E2E success is 50.00%/0.00%, mean steps 0.00, actuator violations 0.

Latency average/P50/P95 is 28.89/32.72/37.22 seconds.

## Failure localization

Primary failure counts: `{"Qwen invalid JSON/schema": 6, "wrong task/route": 1}`. The largest observed stage is **Qwen invalid JSON/schema**. Failures are assigned exactly one primary stage; secondary tags are retained in `failure_analysis.csv`.

## Acceptance and interpretation

Integration acceptance passed before this frozen run: schema/registry validation, unit/order/bounds/no-op tests, state/image v12 dispatch, a non-trivial Learned H1 closed-loop episode, checkpoint/config integrity, and old-route regressions. Formal cases were then run without changing code, config, checkpoint, or manifest.

The routing-versus-argument split is given by route accuracy (14.29%) versus argument accuracy (28.57%). State/image and specialist/E2E gaps are shown in the table. Inverse failures tagged `v12 forward prediction` had planner-exploitation evidence; remaining valid-planner misses are attributed to H1 search/planning.

Top priorities follow the three largest failure stages: Qwen invalid JSON/schema, wrong task/route.

## Known limitations

This is a single-seed diagnostic with 1 ready cases per route. The Qwen adapter was trained on the v1 route contract, not v12, and is not formally promoted. The v12 checkpoint used only 128 training groups; its preregistered H3 accumulation gate failed, so H3 was excluded. Image-route assets were deterministically materialized from protected numerical test states because the protected v12 manifest stored no images.

Exact reproduction:

```bash
/home/jiamo/miniconda3/envs/optics_qlora/bin/python -m Qwen_orchestration.scripts.evaluate_v12_e2e --run-dir /home/jiamo/VLM/artifacts/v12_qwen_e2e/qwen_runner_smoke_v2_20260731
```
