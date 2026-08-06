import numpy as np

from optics_understanding_sft.direction_inverse_v1.train_inverse_joint import joint_labels


def test_joint_labels_cover_axis_tuple_without_simulation(tmp_path) -> None:
    # Exercise the label conversion through a tiny canonical action grid fixture.
    data = tmp_path / "inverse" / "canonical"
    data.mkdir(parents=True)
    actions = []
    for lx in (-0.05, 0.0, 0.05):
        for ly in (-0.05, 0.0, 0.05):
            for cx in (-0.02, 0.0, 0.02):
                for cy in (-0.02, 0.0, 0.02):
                    actions.append({"lens_x_delta_mm": lx, "lens_y_delta_mm": ly,
                                    "camera_x_delta_mm": cx, "camera_y_delta_mm": cy})
    row = {"task_type": "inverse_action_numeric", "prompt_inputs": {"action_grid": actions}}
    (data / "train.jsonl").write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
    labels = joint_labels(tmp_path, np.asarray([[0, 0, 0, 0], [2, 2, 2, 2]]))
    assert labels.tolist() == [0, 80]
