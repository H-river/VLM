from optics_understanding_sft.direction_inverse_v1.repair_visual_sensor_frame import sensor_state


def test_sensor_state_subtracts_final_camera_offset() -> None:
    state = {"centroid_x_px": 510.0, "centroid_y_px": 500.0, "sigma_x_px": 10.0,
             "sigma_y_px": 11.0, "peak_intensity": 2.0}
    setup = {"pixel_size_um": 5.0, "camera_x_offset_mm": 0.01, "camera_y_offset_mm": -0.02}
    action = {"camera_x_delta_mm": 0.005, "camera_y_delta_mm": -0.005}
    output = sensor_state(state, setup, action)
    assert output["centroid_x_px"] == 507.0
    assert output["centroid_y_px"] == 505.0
