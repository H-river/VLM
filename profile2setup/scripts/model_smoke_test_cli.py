"""Stage 4 smoke-test CLI for profile2setup model architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "model_smoke_test_cli requires PyTorch. Install torch to run Stage 4 model smoke tests."
    ) from exc

from profile2setup.models import Profile2SetupModel, build_model_from_config, count_parameters
from profile2setup.schema import VARIABLE_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="profile2setup Stage 4 model smoke test")
    parser.add_argument("--vocab-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--config", default="profile2setup/configs/train.yaml")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--jsonl", default=None, help="Optional JSONL for real-batch smoke test")
    parser.add_argument(
        "--variables-config",
        default="profile2setup/configs/variables.yaml",
        help="Variables YAML for optional real-batch mode",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocab JSON path for optional real-batch mode",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict at top level, got {type(data).__name__}")
    return data


def _assert_output_shapes(outputs: dict, batch_size: int) -> None:
    expected = (batch_size, len(VARIABLE_ORDER))

    if tuple(outputs["delta"].shape) != expected:
        raise AssertionError(f"delta shape mismatch: got {tuple(outputs['delta'].shape)} expected {expected}")
    if tuple(outputs["absolute"].shape) != expected:
        raise AssertionError(
            f"absolute shape mismatch: got {tuple(outputs['absolute'].shape)} expected {expected}"
        )
    if tuple(outputs["change_logits"].shape) != expected:
        raise AssertionError(
            "change_logits shape mismatch: "
            f"got {tuple(outputs['change_logits'].shape)} expected {expected}"
        )


def _run_random_smoke(
    model: Profile2SetupModel,
    *,
    batch_size: int,
    input_size: int,
    text_len: int,
    vocab_size: int,
    device: torch.device,
) -> None:
    profile = torch.randn(batch_size, 4, input_size, input_size, device=device)
    prompt_tokens = torch.randint(0, vocab_size, (batch_size, text_len), device=device, dtype=torch.long)
    current_setup = torch.randn(batch_size, len(VARIABLE_ORDER), device=device)

    with torch.no_grad():
        outputs = model(profile=profile, prompt_tokens=prompt_tokens, current_setup=current_setup)

    _assert_output_shapes(outputs, batch_size=batch_size)

    print(f"model parameter count: {count_parameters(model):,}")
    print(f"profile input shape: {tuple(profile.shape)}")
    print(f"prompt token shape: {tuple(prompt_tokens.shape)}")
    print(f"current setup shape: {tuple(current_setup.shape)}")
    print(f"output delta shape: {tuple(outputs['delta'].shape)}")
    print(f"output absolute shape: {tuple(outputs['absolute'].shape)}")
    print(f"output change_logits shape: {tuple(outputs['change_logits'].shape)}")
    print("random-input smoke test passed")


def _run_dataset_smoke(
    config: dict,
    *,
    jsonl_path: str,
    variables_config: str,
    vocab_path: str | None,
    input_size: int,
    text_len: int,
    batch_size: int,
    device: torch.device,
) -> None:
    from profile2setup.training.dataset import Profile2SetupDataset, profile2setup_collate_fn
    from profile2setup.training.text import build_vocab_from_jsonl, load_vocab

    if vocab_path is not None:
        vocab = load_vocab(vocab_path)
    else:
        vocab = build_vocab_from_jsonl(jsonl_path, min_freq=1)

    dataset = Profile2SetupDataset(
        jsonl_path=jsonl_path,
        variables_config_path=variables_config,
        vocab=vocab,
        input_size=input_size,
        max_text_len=text_len,
        normalize_mode="max_log",
        limit=max(batch_size, 1),
        strict=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot run optional real-batch smoke test")

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        collate_fn=profile2setup_collate_fn,
    )
    batch = next(iter(loader))

    model = build_model_from_config(config=config, vocab_size=len(vocab)).to(device)
    model.eval()

    profile = batch["profile"].to(device)
    prompt_tokens = batch["prompt_tokens"].to(device)
    current_setup = batch["current_setup"].to(device)

    with torch.no_grad():
        outputs = model(profile=profile, prompt_tokens=prompt_tokens, current_setup=current_setup)

    _assert_output_shapes(outputs, batch_size=profile.shape[0])

    print("real-batch smoke test passed")
    print(f"real batch profile shape: {tuple(profile.shape)}")
    print(f"real batch prompt token shape: {tuple(prompt_tokens.shape)}")
    print(f"real batch current setup shape: {tuple(current_setup.shape)}")
    print(f"real batch output delta shape: {tuple(outputs['delta'].shape)}")
    print(f"real batch output absolute shape: {tuple(outputs['absolute'].shape)}")
    print(f"real batch output change_logits shape: {tuple(outputs['change_logits'].shape)}")


def main() -> None:
    args = parse_args()

    if args.vocab_size <= 1:
        raise ValueError("--vocab-size must be > 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")
    if args.text_len <= 0:
        raise ValueError("--text-len must be positive")

    device = torch.device(args.device)
    config = _load_config(args.config)

    model = build_model_from_config(config=config, vocab_size=args.vocab_size).to(device)
    model.eval()

    _run_random_smoke(
        model,
        batch_size=args.batch_size,
        input_size=args.input_size,
        text_len=args.text_len,
        vocab_size=args.vocab_size,
        device=device,
    )

    if args.jsonl is not None:
        _run_dataset_smoke(
            config,
            jsonl_path=args.jsonl,
            variables_config=args.variables_config,
            vocab_path=args.vocab,
            input_size=args.input_size,
            text_len=args.text_len,
            batch_size=args.batch_size,
            device=device,
        )

    print("Stage 4 model smoke test succeeded")


if __name__ == "__main__":
    main()
