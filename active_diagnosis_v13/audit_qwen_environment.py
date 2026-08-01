#!/usr/bin/env python3
"""Audit local Qwen/QLoRA readiness without loading weights or training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


VERSION = "active_diagnosis_v13_qwen_environment_preflight_v1"


def weight_files(files: list[Path]) -> list[Path]:
    """Return actual model weight files, excluding cache metadata sidecars."""

    return [path for path in files if path.suffix in {".safetensors", ".bin"}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from optics_sft.scripts.train_qwen25vl_qlora import require_training_imports

    deps = require_training_imports()
    model = args.model_dir.resolve()
    config_path = model / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    package_names = (
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "accelerate",
    )
    files = [path for path in model.rglob("*") if path.is_file()]
    weights = weight_files(files)
    cuda_free = cuda_total = None
    if torch.cuda.is_available():
        cuda_free, cuda_total = torch.cuda.mem_get_info()
    report = {
        "version": VERSION,
        "role": "environment_readiness_only_no_model_load_no_training",
        "protected_set_used": False,
        "python_executable": os.sys.executable,
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "training_imports_available": sorted(deps),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "free_bytes_before_model_load": cuda_free,
            "total_bytes": cuda_total,
            "torch_cuda_version": torch.version.cuda,
        },
        "model": {
            "path": str(model),
            "config_present": config_path.exists(),
            "architectures": model_config.get("architectures"),
            "model_type": model_config.get("model_type"),
            "local_file_count": len(files),
            "weight_file_count": len(weights),
            "total_file_bytes": sum(path.stat().st_size for path in files),
        },
    }
    report["passes"] = bool(
        report["cuda"]["available"]
        and report["model"]["config_present"]
        and report["model"]["weight_file_count"] > 0
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
