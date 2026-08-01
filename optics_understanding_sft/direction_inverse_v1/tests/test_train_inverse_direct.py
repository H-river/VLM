from optics_understanding_sft.direction_inverse_v1.train_inverse_direct import action_from_prediction, action_labels


def test_action_label_roundtrip() -> None:
    action = {"lens_x_delta_mm": -0.05, "lens_y_delta_mm": 0.0,
              "camera_x_delta_mm": 0.02, "camera_y_delta_mm": -0.02}
    assert action_from_prediction(action_labels(action)) == action
