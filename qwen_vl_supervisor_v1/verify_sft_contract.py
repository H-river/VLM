"""Run the real Qwen processor/collator over exported smoke records."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from .contracts import parse_target_strict
from .export_sft import validate_export_row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def verify(
    *,
    repository_root: Path,
    data_paths: list[Path],
    model_path: Path,
    revision: str,
    min_pixels: int,
    max_pixels: int,
) -> dict[str, Any]:
    try:
        import torch
        import transformers
        import trl
        from transformers import AutoProcessor
        from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

        from optics_sft.scripts.train_qwen25vl_qlora import prebuilt_chat_row_to_sft_example
    except ImportError as exc:
        raise RuntimeError("run this verifier in the optics_qlora environment") from exc

    repository_root = repository_root.resolve()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        revision=revision,
        local_files_only=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=None,
        completion_only_loss=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if not isinstance(image_token_id, int) or image_token_id < 0:
        raise RuntimeError("Qwen image-pad token is unavailable")

    summaries: list[dict[str, Any]] = []
    total_records = 0
    total_supervised = 0
    max_sequence_tokens = 0
    min_image_tokens: int | None = None
    max_image_tokens = 0
    for data_path in data_paths:
        rows = _read_jsonl(data_path)
        for row in rows:
            validate_export_row(row, repository_root=repository_root)
            example = prebuilt_chat_row_to_sft_example(row, repository_root)
            batch = collator([example])
            input_ids = batch["input_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            if len(input_ids) != len(labels):
                raise RuntimeError(f"{row['example_id']}: input/label length mismatch")
            target_text = row["completion"][0]["content"][0]["text"]
            parse_target_strict(target_text)
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            target_start = _last_subsequence(input_ids, list(target_ids))
            if target_start < 0:
                raise RuntimeError(f"{row['example_id']}: target token sequence missing")
            if any(label != -100 for label in labels[:target_start]):
                raise RuntimeError(f"{row['example_id']}: system/user token contributes to loss")
            if labels[target_start : target_start + len(target_ids)] != input_ids[target_start : target_start + len(target_ids)]:
                raise RuntimeError(f"{row['example_id']}: target tokens are not fully supervised")
            supervised = sum(label != -100 for label in labels)
            if supervised < len(target_ids):
                raise RuntimeError(f"{row['example_id']}: insufficient supervised completion tokens")
            image_tokens = input_ids.count(image_token_id)
            if image_tokens <= 0:
                raise RuntimeError(f"{row['example_id']}: image token was removed")
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
            for critical in ("current_metrics", "goal_metrics", "remaining_step_budget", target_text):
                if critical not in decoded:
                    raise RuntimeError(f"{row['example_id']}: critical content {critical!r} was removed")
            total_records += 1
            total_supervised += supervised
            max_sequence_tokens = max(max_sequence_tokens, len(input_ids))
            min_image_tokens = image_tokens if min_image_tokens is None else min(min_image_tokens, image_tokens)
            max_image_tokens = max(max_image_tokens, image_tokens)
        summaries.append({"path": str(data_path), "sha256": _sha256(data_path), "records": len(rows)})

    return {
        "status": "pass",
        "model_path": str(model_path),
        "model_revision": revision,
        "processor_revision": revision,
        "processor_min_pixels": min_pixels,
        "processor_max_pixels": max_pixels,
        "max_length": None,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "records_iterated": total_records,
        "datasets": summaries,
        "all_system_user_labels_are_minus_100": True,
        "all_target_tokens_supervised": True,
        "all_images_present_after_collation": True,
        "critical_fields_not_truncated": True,
        "minimum_image_pad_tokens": min_image_tokens,
        "maximum_image_pad_tokens": max_image_tokens,
        "maximum_sequence_tokens": max_sequence_tokens,
        "mean_supervised_tokens": total_supervised / total_records if total_records else 0.0,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
    }


def main() -> None:
    module_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=module_root.parent)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[module_root / "sft/sft_smoke_train.jsonl", module_root / "sft/sft_smoke_dev.jsonl"],
    )
    parser.add_argument("--model", type=Path, default=Path("/home/jiamo/HF_models/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--revision", default="66285546d2b821cf421d4f5eb2576359d3770cd3")
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--output", type=Path, default=module_root / "reports/sft_contract_verification.json")
    args = parser.parse_args()
    report = verify(
        repository_root=args.repository_root,
        data_paths=args.data,
        model_path=args.model,
        revision=args.revision,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
