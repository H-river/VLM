"""Resolve repo-relative config paths consistently."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_config_path(path: Path | str) -> Path:
    """Resolve a config path relative to the repo root when not absolute."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def resolve_adapter_path(path: Path | str | None) -> Path | None:
    """Resolve a PEFT adapter directory, or the latest checkpoint-* under it."""
    if path is None:
        return None
    candidate = resolve_config_path(path)
    if (candidate / "adapter_config.json").is_file():
        return candidate

    checkpoints: list[Path] = []
    if candidate.is_dir():
        for child in candidate.iterdir():
            if not child.is_dir() or not child.name.startswith("checkpoint-"):
                continue
            if (child / "adapter_config.json").is_file():
                suffix = child.name.removeprefix("checkpoint-")
                checkpoints.append(child)

    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        print(f"[adapter] Using latest checkpoint: {latest}", flush=True)
        return latest

    hint = (
        f"Adapter not found: {candidate}\n"
        "Common causes:\n"
        "  1) Training is still running — wait until train_text_qlora prints "
        '"Saved text QLoRA adapter to ..."\n'
        "  2) Training failed or was interrupted — re-run train_text_qlora.py\n"
        "  3) Wrong path — config uses ../VLM_runs/<run_name> under the repo parent\n"
        "You can also pass --adapter-path .../checkpoint-400 explicitly."
    )
    raise FileNotFoundError(hint)
