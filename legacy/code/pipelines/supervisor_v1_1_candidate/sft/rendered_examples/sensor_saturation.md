# Redacted model-visible example

This rendering contains only the messages visible to the model. The image is represented by a placeholder; paths and hidden manifest metadata are omitted.

## System

```text
You are the high-level supervisor for a frozen optical control system.
Use the current beam image and structured numerical state to diagnose the observation, select a measurement/recovery policy, and select a high-level supervisor action.
Never output continuous actuator commands, actuator deltas, positions, or other numeric control actions. The frozen numerical controller alone chooses continuous actions.
Choose exactly one value for each field from these frozen enums:
diagnosis: nominal, sensor_saturation, secondary_reflection
measurement_policy: standard, lower_exposure_reacquire, primary_spot
supervisor_action: reacquire, switch_measurement, execute, continue, stop
Return exactly one JSON object with exactly the keys diagnosis, measurement_policy, and supervisor_action. Return no rationale, confidence, markdown, or extra text.
```

## User

`[CURRENT_BEAM_IMAGE]`

```text
Current supervisor state (all coordinates and units are explicit):
{"actuator_constraints":{"absolute_limit_source":"repository_sampling_domain_not_hardware_limit","absolute_position_limits":{"camera_x":[-3.0,3.0],"camera_y":[-3.0,3.0],"lens_x":[-3.0,3.0],"lens_y":[-3.0,3.0]},"continuous_actions_selected_by":"frozen_h1_one_step_cem","per_step_delta_limits":{"camera_x":[-0.02,0.02],"camera_y":[-0.02,0.02],"lens_x":[-0.05,0.05],"lens_y":[-0.05,0.05]},"units":"mm"},"current_metrics":{"centroid_x":63.66360467992363,"centroid_y":53.631331933865766,"coordinate_frame":"diagnostic_image_128px","peak_intensity":1.0,"width_x":17.146937685242122,"width_y":11.24830265327838},"decision_context":"static_anomaly_policy","goal_metrics":{"centroid_x":595.9074476706386,"centroid_y":476.37209446499037,"coordinate_frame":"lab_sensor_1024px_and_raw_peak","peak_intensity":1434776.625,"width_x":65.64246403300513,"width_y":44.99346032771014},"recent_history":[],"remaining_step_budget":8}
```

## Assistant

```json
{"diagnosis":"sensor_saturation","measurement_policy":"lower_exposure_reacquire","supervisor_action":"reacquire"}
```
