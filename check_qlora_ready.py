import os
import sys
from pathlib import Path

print("=" * 80)
print("1. Python / path check")
print("=" * 80)
print("Python:", sys.version)
print("Executable:", sys.executable)

ros_paths = [p for p in sys.path if "ros" in p.lower()]
if ros_paths:
    print("WARNING: ROS paths detected in Python path:")
    for p in ros_paths:
        print("  ", p)
else:
    print("OK: no ROS paths detected in sys.path")

print("\n" + "=" * 80)
print("2. Package import/version check")
print("=" * 80)

packages_ok = True

try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    packages_ok = False
    print("FAIL torch:", repr(e))

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    packages_ok = False
    print("FAIL transformers:", repr(e))

try:
    import accelerate
    print("accelerate:", accelerate.__version__)
except Exception as e:
    packages_ok = False
    print("FAIL accelerate:", repr(e))

try:
    import datasets
    print("datasets:", datasets.__version__)
except Exception as e:
    packages_ok = False
    print("FAIL datasets:", repr(e))

try:
    import peft
    print("peft:", peft.__version__)
except Exception as e:
    packages_ok = False
    print("FAIL peft:", repr(e))

try:
    import trl
    print("trl:", trl.__version__)
except Exception as e:
    packages_ok = False
    print("FAIL trl:", repr(e))

try:
    import bitsandbytes as bnb
    print("bitsandbytes: imported OK")
except Exception as e:
    packages_ok = False
    print("FAIL bitsandbytes:", repr(e))

try:
    import PIL
    print("PIL: imported OK")
except Exception as e:
    packages_ok = False
    print("FAIL PIL:", repr(e))

try:
    import yaml
    print("yaml: imported OK")
except Exception as e:
    packages_ok = False
    print("FAIL yaml:", repr(e))

print("\n" + "=" * 80)
print("3. CUDA check")
print("=" * 80)

if "torch" in sys.modules:
    import torch

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version from torch:", torch.version.cuda)
        print("bf16 supported:", torch.cuda.is_bf16_supported())
        print("Allocated GB:", torch.cuda.memory_allocated() / 1024**3)
        print("Reserved GB:", torch.cuda.memory_reserved() / 1024**3)
    else:
        packages_ok = False
        print("FAIL: CUDA not available")

print("\n" + "=" * 80)
print("4. Local Qwen model folder check")
print("=" * 80)

model_dir = Path("../HF_models/Qwen2.5-VL-3B-Instruct").resolve()
print("Model dir:", model_dir)

required_files = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
]

model_files_ok = True

if not model_dir.exists():
    model_files_ok = False
    print("FAIL: model directory does not exist")
else:
    print("Model directory exists")

    for fname in required_files:
        f = model_dir / fname
        if f.exists():
            print(f"OK: {fname}")
        else:
            model_files_ok = False
            print(f"FAIL missing: {fname}")

    safetensors = sorted(model_dir.glob("*.safetensors"))
    index_file = model_dir / "model.safetensors.index.json"

    print("Safetensors count:", len(safetensors))
    if index_file.exists():
        print("OK: model.safetensors.index.json exists")
    else:
        print("WARNING: model.safetensors.index.json missing")

    total_size_gb = sum(f.stat().st_size for f in safetensors) / 1024**3
    print(f"Total .safetensors size: {total_size_gb:.2f} GB")

    if len(safetensors) == 0:
        model_files_ok = False
        print("FAIL: no .safetensors files found. Model is not fully downloaded.")
    elif total_size_gb < 3:
        model_files_ok = False
        print("FAIL/WARNING: .safetensors total size seems too small for Qwen2.5-VL-3B.")
    else:
        print("OK: model weight files seem present")

print("\n" + "=" * 80)
print("5. Processor local loading check")
print("=" * 80)

processor_ok = False
if model_files_ok:
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        processor_ok = True
        print("OK: AutoProcessor loaded locally")
        print("Processor type:", type(processor))
    except Exception as e:
        print("FAIL: AutoProcessor local load failed")
        print(repr(e))
else:
    print("SKIP: model files incomplete")

print("\n" + "=" * 80)
print("6. 4-bit model local loading check")
print("=" * 80)

model_ok = False
if model_files_ok and processor_ok:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("Using dtype:", dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )

        model_ok = True
        print("OK: model loaded in 4-bit")
        print("First parameter device:", next(model.parameters()).device)
        print("Allocated GB:", torch.cuda.memory_allocated() / 1024**3)
        print("Reserved GB:", torch.cuda.memory_reserved() / 1024**3)

    except Exception as e:
        print("FAIL: 4-bit model load failed")
        print(repr(e))
else:
    print("SKIP: processor/model files not ready")

print("\n" + "=" * 80)
print("Final readiness result")
print("=" * 80)

if packages_ok and model_files_ok and processor_ok and model_ok:
    print("READY: Local environment is ready for QLoRA smoke training.")
else:
    print("NOT READY YET.")
    if not packages_ok:
        print("- Package/CUDA import issue exists.")
    if not model_files_ok:
        print("- Local Qwen model folder is incomplete or missing.")
    if not processor_ok:
        print("- Processor cannot load locally.")
    if not model_ok:
        print("- 4-bit model cannot load yet.")
