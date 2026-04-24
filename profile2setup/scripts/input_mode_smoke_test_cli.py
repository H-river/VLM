"""Smoke test all supported profile2setup input modes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "input_mode_smoke_test_cli requires PyTorch. Install torch to run this smoke test."
    ) from exc

from profile2setup.models import Profile2SetupModel
from profile2setup.schema import VARIABLE_ORDER, compute_delta_setup
from profile2setup.training.dataset import Profile2SetupDataset, profile2setup_collate_fn
from profile2setup.training.losses import compute_profile2setup_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="profile2setup input-mode smoke test")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Path to variables YAML config",
    )
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--max-text-len", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _setup(**overrides: float) -> dict[str, float]:
    values = {
        "source_to_lens": 0.20,
        "lens_to_camera": 0.18,
        "focal_length": 0.08,
        "lens_x": 0.0002,
        "lens_y": -0.0001,
        "camera_x": 0.0003,
        "camera_y": -0.0002,
    }
    values.update(overrides)
    return {key: float(values[key]) for key in VARIABLE_ORDER}


def _write_profile(path: Path, seed: int) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.random((24, 24), dtype=np.float32)
    np.save(path, arr)
    return str(path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _build_records(tmp_path: Path) -> list[dict]:
    current_path = _write_profile(tmp_path / "current.npy", seed=1)
    target_path = _write_profile(tmp_path / "target.npy", seed=2)
    target2_path = _write_profile(tmp_path / "target2.npy", seed=3)

    current_setup = _setup()
    target_setup = _setup(source_to_lens=0.24, lens_x=0.0005, camera_y=0.0001)
    target2_setup = _setup(source_to_lens=0.28, lens_to_camera=0.21, camera_x=-0.0004)

    return [
        {
            "id": "absolute_0",
            "task_type": "absolute",
            "prompt": "predict the setup from this target profile",
            "current_profile_path": None,
            "target_profile_path": target_path,
            "current_setup": None,
            "target_setup": target_setup,
            "target_delta": None,
            "profile_loss_reference": {"target_profile_path": target_path},
        },
        {
            "id": "edit_0",
            "task_type": "edit",
            "prompt": "adjust the current profile to match the target",
            "current_profile_path": current_path,
            "target_profile_path": target_path,
            "current_setup": current_setup,
            "target_setup": target_setup,
            "target_delta": compute_delta_setup(current_setup, target_setup),
            "profile_loss_reference": {
                "current_profile_path": current_path,
                "target_profile_path": target_path,
            },
        },
        {
            "id": "current_only_0",
            "task_type": "current_only",
            "prompt": "infer the target setup from this current profile",
            "current_profile_path": current_path,
            "target_profile_path": None,
            "current_setup": None,
            "target_setup": target2_setup,
            "target_delta": None,
            "profile_loss_reference": {"current_profile_path": current_path},
        },
        {
            "id": "paired_no_setup_0",
            "task_type": "paired_no_setup",
            "prompt": "infer the target setup from current and target profiles",
            "current_profile_path": current_path,
            "target_profile_path": target2_path,
            "current_setup": None,
            "target_setup": target2_setup,
            "target_delta": None,
            "profile_loss_reference": {
                "current_profile_path": current_path,
                "target_profile_path": target2_path,
            },
        },
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with tempfile.TemporaryDirectory(prefix="profile2setup_modes_") as tmp:
        tmp_path = Path(tmp)
        jsonl_path = tmp_path / "modes.jsonl"
        _write_jsonl(jsonl_path, _build_records(tmp_path))

        dataset = Profile2SetupDataset(
            jsonl_path=jsonl_path,
            variables_config_path=args.variables_config,
            input_size=args.input_size,
            max_text_len=args.max_text_len,
            strict=True,
        )
        if len(dataset) != 4:
            raise AssertionError(f"expected 4 input-mode records, got {len(dataset)}")

        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=profile2setup_collate_fn)
        batch = next(iter(loader))

        expected_setup_present = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
        expected_abs_mask = torch.ones(4, 1)
        expected_delta_mask = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
        if not torch.equal(batch["setup_present"], expected_setup_present):
            raise AssertionError(f"setup_present mismatch: {batch['setup_present']}")
        if not torch.equal(batch["absolute_loss_mask"], expected_abs_mask):
            raise AssertionError(f"absolute_loss_mask mismatch: {batch['absolute_loss_mask']}")
        if not torch.equal(batch["delta_loss_mask"], expected_delta_mask):
            raise AssertionError(f"delta_loss_mask mismatch: {batch['delta_loss_mask']}")
        if not torch.equal(batch["change_loss_mask"], expected_delta_mask):
            raise AssertionError(f"change_loss_mask mismatch: {batch['change_loss_mask']}")

        model = Profile2SetupModel(vocab_size=len(dataset.tokenizer.vocab), input_channels=4).to(device)
        model.eval()
        tensor_batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            outputs = model(
                profile=tensor_batch["profile"],
                prompt_tokens=tensor_batch["prompt_tokens"],
                current_setup=tensor_batch["current_setup"],
                setup_present=tensor_batch["setup_present"],
            )
            losses = compute_profile2setup_loss(outputs, tensor_batch)

        if not torch.isfinite(losses["loss"]):
            raise RuntimeError(f"masked loss is not finite: {losses['loss'].item()}")

        print("input-mode smoke test passed")
        print(f"task types: {batch['task_type']}")
        print(f"setup_present: {batch['setup_present'].flatten().tolist()}")
        print(f"delta_loss_mask: {batch['delta_loss_mask'].flatten().tolist()}")
        print(f"masked loss: {losses['loss'].item():.6f}")


if __name__ == "__main__":
    main()
