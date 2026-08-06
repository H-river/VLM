from pathlib import Path

import numpy as np
from PIL import Image

from optics_understanding_sft.build_visual_perturbation_eval_v10_13 import transform


def test_saturation_transform_clips_without_changing_shape(tmp_path: Path):
    source = tmp_path / "source.png"
    destination = tmp_path / "destination.png"
    rgb = np.array([[[0, 100, 255], [179, 180, 181]]], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(source)

    transform(source, destination, "saturation_clip_180", "source.png")

    result = np.asarray(Image.open(destination).convert("RGB"))
    assert result.shape == rgb.shape
    assert result.tolist() == [[[0, 100, 180], [179, 180, 180]]]
