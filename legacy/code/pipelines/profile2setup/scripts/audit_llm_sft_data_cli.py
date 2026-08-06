"""Audit multimodal LLM SFT data for physics-understanding diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from profile2setup.evaluation.llm_api_eval import FALLBACK_TOLERANCES
from profile2setup.schema import VARIABLE_ORDER, compute_delta_setup, validate_setup_dict

CANONICAL_VARIABLE_ORDER = list(VARIABLE_ORDER)
FIXED_VARIABLE_TERMS = ("keep", "fixed", "do not change", "only")
CONTRADICTION_TERMS = ("impossible", "cannot", "but do not change")
VARIABLE_TERMS = tuple(CANONICAL_VARIABLE_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LLM/API SFT JSONL prompt and label diversity without API calls."
    )
    parser.add_argument("--input", required=True, help="Input LLM API SFT JSONL")
    parser.add_argument("--original", required=True, help="Original profile2setup JSONL")
    parser.add_argument("--out", required=True, help="Output JSON audit report")
    parser.add_argument(
        "--markdown-out",
        default=None,
        help="Output Markdown summary. Defaults to the --out path with .md suffix.",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} record must be a JSON object")
            yield line_number, record


def _message_content_parts(message: dict) -> list[Any]:
    content = message.get("content")
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _message_text(message: dict) -> str:
    parts = _message_content_parts(message)
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
        elif isinstance(part, str):
            chunks.append(part)
    return "\n".join(chunks)


def _user_messages(record: dict) -> list[dict]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    return [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "user"]


def _assistant_messages(record: dict) -> list[dict]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    return [msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"]


def _extract_line_value(text: str, label: str) -> str | None:
    prefix = label.lower() + ":"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix) :].strip()
    return None


def _extract_sft_prompt(record: dict) -> str:
    for message in _user_messages(record):
        value = _extract_line_value(_message_text(message), "Prompt")
        if value is not None:
            return value
    value = record.get("prompt")
    return value if isinstance(value, str) else ""


def _extract_sft_task_type(record: dict) -> str:
    for message in _user_messages(record):
        value = _extract_line_value(_message_text(message), "Task")
        if value is not None:
            return value
    for message in _assistant_messages(record):
        text = _message_text(message).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        task_type = payload.get("task_type") if isinstance(payload, dict) else None
        if isinstance(task_type, str):
            return task_type
    value = record.get("task_type")
    return value if isinstance(value, str) else ""


def _count_image_url_parts(record: dict) -> int:
    total = 0
    messages = record.get("messages")
    if not isinstance(messages, list):
        return total
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in _message_content_parts(message):
            if isinstance(part, dict) and part.get("type") == "image_url":
                total += 1
    return total


def _contains_term(text: str, term: str) -> bool:
    if "_" in term:
        return term in text
    pattern = r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])"
    return re.search(pattern, text) is not None


def _term_counts(prompts: list[str], terms: Iterable[str]) -> dict[str, dict[str, float]]:
    total = len(prompts)
    lowered = [prompt.lower() for prompt in prompts]
    out: dict[str, dict[str, float]] = {}
    term_list = list(terms)
    any_count = sum(
        1
        for prompt in lowered
        if any(_contains_term(prompt, term) for term in term_list)
    )
    out["any"] = {
        "count": int(any_count),
        "percentage": (float(any_count) / float(total) * 100.0) if total else 0.0,
    }
    for term in term_list:
        count = sum(1 for prompt in lowered if _contains_term(prompt, term))
        out[term] = {
            "count": int(count),
            "percentage": (float(count) / float(total) * 100.0) if total else 0.0,
        }
    return out


def _clean_setup(value: Any) -> dict | None:
    if not isinstance(value, dict) or not validate_setup_dict(value):
        return None
    return {name: float(value[name]) for name in CANONICAL_VARIABLE_ORDER}


def _target_delta(record: dict) -> dict | None:
    explicit = _clean_setup(record.get("target_delta"))
    if explicit is not None:
        return explicit
    current = _clean_setup(record.get("current_setup"))
    target = _clean_setup(record.get("target_setup"))
    if current is not None and target is not None:
        return compute_delta_setup(current, target)
    return None


def _direction(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _pattern_from_delta(delta: dict, tolerances: dict[str, float]) -> tuple[int, ...]:
    return tuple(_direction(float(delta[name]), tolerances[name]) for name in CANONICAL_VARIABLE_ORDER)


def _changed_set_from_pattern(pattern: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        name
        for idx, name in enumerate(CANONICAL_VARIABLE_ORDER)
        if pattern[idx] != 0
    )


def _pattern_to_dict(pattern: tuple[int, ...]) -> dict[str, str]:
    labels = {-1: "decrease", 0: "unchanged", 1: "increase"}
    return {name: labels[int(pattern[idx])] for idx, name in enumerate(CANONICAL_VARIABLE_ORDER)}


def _set_key(changed_set: tuple[str, ...]) -> str:
    return ",".join(changed_set) if changed_set else "none"


def audit_sft_data(input_path: Path, original_path: Path) -> dict[str, Any]:
    prompt_counter: Counter[str] = Counter()
    task_type_counter: Counter[str] = Counter()
    image_url_counts: list[int] = []
    sft_records = 0

    for _, record in _iter_jsonl(input_path):
        sft_records += 1
        prompt = _extract_sft_prompt(record)
        task_type = _extract_sft_task_type(record)
        prompt_counter[prompt] += 1
        task_type_counter[task_type] += 1
        image_url_counts.append(_count_image_url_parts(record))

    prompts = list(prompt_counter.elements())
    tolerances = {name: float(FALLBACK_TOLERANCES[name]) for name in CANONICAL_VARIABLE_ORDER}
    original_records = 0
    original_records_with_delta = 0
    original_records_without_delta = 0
    pattern_counter: Counter[tuple[int, ...]] = Counter()
    prompt_to_sets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    set_to_prompts: dict[tuple[str, ...], set[str]] = defaultdict(set)

    for _, record in _iter_jsonl(original_path):
        original_records += 1
        delta = _target_delta(record)
        if delta is None:
            original_records_without_delta += 1
            continue
        original_records_with_delta += 1
        pattern = _pattern_from_delta(delta, tolerances)
        changed_set = _changed_set_from_pattern(pattern)
        prompt = record.get("prompt")
        prompt_text = prompt if isinstance(prompt, str) else ""
        pattern_counter[pattern] += 1
        prompt_to_sets[prompt_text].add(changed_set)
        set_to_prompts[changed_set].add(prompt_text)

    prompt_frequency_distribution = Counter(prompt_counter.values())
    avg_images = (
        sum(image_url_counts) / float(len(image_url_counts))
        if image_url_counts
        else 0.0
    )

    prompt_to_multiple_sets = {
        prompt: sorted(_set_key(item) for item in sets)
        for prompt, sets in sorted(prompt_to_sets.items())
        if len(sets) > 1
    }
    set_to_multiple_prompts = {
        _set_key(changed_set): sorted(prompts_for_set)
        for changed_set, prompts_for_set in sorted(
            set_to_prompts.items(),
            key=lambda item: (_set_key(item[0]), len(item[1])),
        )
        if len(prompts_for_set) > 1
    }

    return {
        "input_path": str(input_path),
        "original_path": str(original_path),
        "sft": {
            "record_count": int(sft_records),
            "unique_prompt_count": int(len(prompt_counter)),
            "top_30_prompts": [
                {"prompt": prompt, "count": int(count)}
                for prompt, count in prompt_counter.most_common(30)
            ],
            "prompt_frequency_distribution": {
                str(freq): int(count)
                for freq, count in sorted(prompt_frequency_distribution.items())
            },
            "task_type_distribution": dict(sorted(task_type_counter.items())),
            "average_image_url_parts_per_record": float(avg_images),
            "prompt_term_coverage": {
                "fixed_variable_words": _term_counts(prompts, FIXED_VARIABLE_TERMS),
                "contradiction_words": _term_counts(prompts, CONTRADICTION_TERMS),
                "variable_names": _term_counts(prompts, VARIABLE_TERMS),
            },
        },
        "original": {
            "record_count": int(original_records),
            "records_with_target_delta_pattern": int(original_records_with_delta),
            "records_without_target_delta_pattern": int(original_records_without_delta),
            "tolerances": tolerances,
            "unique_target_delta_pattern_count": int(len(pattern_counter)),
            "target_delta_pattern_distribution": [
                {
                    "pattern": _pattern_to_dict(pattern),
                    "count": int(count),
                }
                for pattern, count in pattern_counter.most_common()
            ],
            "prompts_mapping_to_multiple_changed_variable_sets_count": int(
                len(prompt_to_multiple_sets)
            ),
            "changed_variable_sets_mapping_to_multiple_prompt_wordings_count": int(
                len(set_to_multiple_prompts)
            ),
            "prompts_mapping_to_multiple_changed_variable_sets": prompt_to_multiple_sets,
            "changed_variable_sets_mapping_to_multiple_prompt_wordings": set_to_multiple_prompts,
        },
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    sft = report["sft"]
    original = report["original"]
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SFT Data Audit",
        "",
        "Offline diagnostic summary. No OpenAI APIs were called.",
        "",
        "## Inputs",
        "",
        f"- SFT JSONL: `{report['input_path']}`",
        f"- Original JSONL: `{report['original_path']}`",
        "",
        "## SFT Records",
        "",
        f"- Records: `{sft['record_count']}`",
        f"- Unique prompts: `{sft['unique_prompt_count']}`",
        f"- Average `image_url` parts per record: `{sft['average_image_url_parts_per_record']:.4f}`",
        "",
        "### Task Types",
        "",
    ]
    for task_type, count in sft["task_type_distribution"].items():
        label = task_type if task_type else "(missing)"
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(["", "### Top 30 Prompts", ""])
    for idx, item in enumerate(sft["top_30_prompts"], start=1):
        lines.append(f"{idx}. `{item['count']}` - {item['prompt']}")

    lines.extend(["", "### Prompt Frequency Distribution", ""])
    for frequency, count in sft["prompt_frequency_distribution"].items():
        lines.append(f"- Prompts appearing `{frequency}` time(s): `{count}`")

    lines.extend(["", "### Prompt Term Coverage", ""])
    for group_name, group in sft["prompt_term_coverage"].items():
        lines.append(f"#### {group_name}")
        lines.append("")
        for term, stats in group.items():
            lines.append(
                f"- `{term}`: `{stats['count']}` prompts ({_fmt_pct(stats['percentage'])})"
            )
        lines.append("")

    lines.extend(
        [
            "## Original profile2setup Records",
            "",
            f"- Records: `{original['record_count']}`",
            f"- Records with target-delta pattern: `{original['records_with_target_delta_pattern']}`",
            f"- Records without target-delta pattern: `{original['records_without_target_delta_pattern']}`",
            f"- Unique target-delta patterns: `{original['unique_target_delta_pattern_count']}`",
            "- Prompts mapping to multiple changed-variable sets: "
            f"`{original['prompts_mapping_to_multiple_changed_variable_sets_count']}`",
            "- Changed-variable sets mapping to multiple prompt wordings: "
            f"`{original['changed_variable_sets_mapping_to_multiple_prompt_wordings_count']}`",
            "",
            "### Most Frequent Target-Delta Patterns",
            "",
        ]
    )
    for item in original["target_delta_pattern_distribution"][:30]:
        changed = [
            f"{name}:{direction}"
            for name, direction in item["pattern"].items()
            if direction != "unchanged"
        ]
        label = ", ".join(changed) if changed else "none"
        lines.append(f"- `{item['count']}` - {label}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    original_path = Path(args.original)
    out_path = Path(args.out)
    markdown_path = Path(args.markdown_out) if args.markdown_out else out_path.with_suffix(".md")

    report = audit_sft_data(input_path, original_path)
    _write_json(report, out_path)
    _write_markdown(report, markdown_path)
    print(json.dumps({"out": str(out_path), "markdown_out": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
