"""Small end-to-end smoke check for profile2setup Stage 2 data prep."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from profile2setup.data_prep.build_absolute_dataset import build_absolute_dataset
from profile2setup.data_prep.build_edit_dataset import build_edit_dataset
from profile2setup.data_prep.split import split_jsonl
from profile2setup.schema import contains_forbidden_v2_keys


def _write_sample(root: Path, sample_id: str, peak_col: int, camera_x: float, camera_y: float) -> None:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    intensity = np.zeros((16, 16), dtype=np.float64)
    intensity[8, peak_col] = 1.0
    np.save(sample_dir / "intensity.npy", intensity)

    metadata = {
        "run_name": sample_id,
        "setup": {
            "geometry": {
                "laser_to_lens": 0.2,
                "lens_to_camera": 0.15,
            },
            "lens": {
                "focal_length": 0.1,
                "clear_aperture": 0.025,
                "diameter": 0.0254,
                "x_offset": 1.0e-3 if sample_id.endswith("a") else -1.0e-3,
                "y_offset": 0.5e-3 if sample_id.endswith("a") else -0.5e-3,
            },
            "camera": {
                "x_offset": camera_x,
                "y_offset": camera_y,
            },
        },
        "metrics": {
            "centroid_x": float(peak_col),
            "centroid_y": 8.0,
            "sigma_x": 1.5,
            "sigma_y": 1.2,
            "peak_intensity": 1.0,
            "total_intensity": 1.0,
        },
    }

    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)

        _write_sample(sim_dir, "sample_a", peak_col=5, camera_x=0.001, camera_y=-0.001)
        _write_sample(sim_dir, "sample_b", peak_col=11, camera_x=-0.001, camera_y=0.001)

        abs_out = tmp_path / "absolute.jsonl"
        edit_out = tmp_path / "edit.jsonl"
        split_dir = tmp_path / "splits"

        build_absolute_dataset(
            sim_dir=sim_dir,
            out_path=abs_out,
            strict=True,
            seed=42,
        )
        build_edit_dataset(
            sim_dir=sim_dir,
            out_path=edit_out,
            num_pairs=2,
            strict=True,
            seed=42,
        )
        split_jsonl(
            input_path=edit_out,
            out_dir=split_dir,
            train_frac=0.5,
            val_frac=0.0,
            test_frac=0.5,
            seed=42,
        )

        assert abs_out.exists()
        assert edit_out.exists()
        assert (split_dir / "train.jsonl").exists()
        assert (split_dir / "val.jsonl").exists()
        assert (split_dir / "test.jsonl").exists()

        for path in (abs_out, edit_out):
            for rec in _read_jsonl(path):
                setup = rec["target_setup"]
                assert "camera_x" in setup
                assert "camera_y" in setup
                assert "profile_loss_reference" in rec
                assert not contains_forbidden_v2_keys(rec)

    print("profile2setup Stage 2 smoke: OK")


def main() -> None:
    run_smoke()


if __name__ == "__main__":
    main()
