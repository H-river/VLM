#!/usr/bin/env python3
"""Generate and score Qwen orchestration decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

# Conservative defaults keep evaluation responsive on a laptop. They are set
# before importing the dispatcher, which imports NumPy-backed specialists.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("MAX_JOBS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from jsonschema import Draft202012Validator
from PIL import Image

from Qwen_orchestration.runtime.constrained_decoding import build_constraint
from Qwen_orchestration.runtime.dispatcher import validate_decision
from Qwen_orchestration.runtime.normalization import normalize_nonexecuting_decision
from Qwen_orchestration.runtime.prompt_contract import (
    apply_decision_contract,
    decision_contract_enabled,
)


ROUTES = (
    "measure_beam_profile_v1",
    "predict_direction_from_state_v1",
    "predict_direction_from_image_v1",
    "predict_forward_from_state_v1",
    "predict_forward_from_image_v1",
    "select_inverse_action_from_states_v1",
    "select_inverse_action_from_images_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--subset", choices=("diagnostic", "all"), default="all")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initial-predictions-jsonl",
        type=Path,
        help="Seed this run from an earlier prediction file before optional repair.",
    )
    parser.add_argument(
        "--repair-invalid-once",
        action="store_true",
        help="Give schema/registry-invalid stage-2 decisions one correction attempt.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score existing output without loading the model or generating.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("config must contain an object")
    return value


def diagnostic_ids(rows: list[dict[str, Any]]) -> set[str]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    selected: set[str] = set()
    targets = {route: 20 for route in ROUTES}
    targets.update({"needs_clarification": 70, "unsupported": 70})
    for category, count in targets.items():
        ordered = sorted(
            by_category[category],
            key=lambda row: hashlib.sha256(
                f"diagnostic:{row['example_id']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < count:
            raise RuntimeError(f"not enough {category} records for diagnostic set")
        selected.update(row["example_id"] for row in ordered[:count])
    if len(selected) != 280:
        raise RuntimeError(f"diagnostic set has {len(selected)} records, expected 280")
    return selected


def first_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escape = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_generated(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as direct:
        block = first_json_object_text(text)
        if block is None:
            return None, str(direct)
        try:
            value = json.loads(block)
        except json.JSONDecodeError as nested:
            return None, str(nested)
    if not isinstance(value, dict):
        return None, "generated JSON is not an object"
    return value, None


def require_dependencies() -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def load_model(
    config: Mapping[str, Any], adapter_override: Path | None
) -> tuple[Any, Any, dict[str, Any], Path]:
    deps = require_dependencies()
    model_cfg = config["model"]
    adapter = adapter_override or Path(model_cfg["adapter_path"])
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    processor = deps["AutoProcessor"].from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    constraint_factory, constraint_metadata = build_constraint(
        config, tokenizer
    )
    deps["constraint_factory"] = constraint_factory
    deps["constraint_metadata"] = constraint_metadata
    torch = deps["torch"]
    dtype = getattr(torch, str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")))
    quantization = deps["BitsAndBytesConfig"](
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
    )
    model = deps["AutoModelForImageTextToText"].from_pretrained(
        model_cfg["name"],
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    model = deps["PeftModel"].from_pretrained(
        model, adapter, local_files_only=True
    )
    model.eval()
    return processor, model, deps, adapter.resolve()


def load_images(paths: list[str], root: Path) -> list[Any]:
    images = []
    for value in paths:
        with Image.open(root / value) as image:
            copy = image.convert("RGB")
            copy.load()
        images.append(copy)
    return images


def generate_batch(
    rows: list[Mapping[str, Any]],
    *,
    image_root: Path,
    processor: Any,
    model: Any,
    deps: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    use_contract = decision_contract_enabled(config)
    rendered = [
        processor.apply_chat_template(
            apply_decision_contract(row["prompt"])
            if use_contract
            else row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    images = [
        image
        for row in rows
        for image in load_images(row["images"], image_root)
    ]
    kwargs: dict[str, Any] = {
        "text": rendered,
        "return_tensors": "pt",
        "padding": True,
    }
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs)
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    generation = config.get("generation", {})
    options: dict[str, Any] = {
        "max_new_tokens": int(generation.get("max_new_tokens", 768)),
        "do_sample": bool(generation.get("do_sample", False)),
    }
    if options["do_sample"]:
        options["temperature"] = float(generation.get("temperature", 0.7))
    constraint_factory = deps.get("constraint_factory")
    if constraint_factory is not None:
        options["prefix_allowed_tokens_fn"] = constraint_factory()
    started = time.perf_counter()
    with deps["torch"].inference_mode():
        generated = model.generate(**inputs, **options)
    elapsed = time.perf_counter() - started
    prompt_length = int(inputs["input_ids"].shape[-1])
    raw_values = processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )
    results = []
    for index, (row, raw) in enumerate(zip(rows, raw_values, strict=True)):
        raw = raw.strip()
        parsed, error = parse_generated(raw)
        pre_normalization = None
        normalization_applied = False
        if (
            isinstance(parsed, dict)
            and config.get("orchestration", {}).get(
                "normalize_nonexecuting_decisions", False
            )
        ):
            normalized, normalization_applied = normalize_nonexecuting_decision(
                parsed
            )
            if normalization_applied:
                pre_normalization = parsed
                parsed = normalized
        input_tokens = int(inputs["attention_mask"][index].sum().item())
        result = {
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "images": row["images"],
            "raw_prediction_text": raw,
            "parsed_json": parsed,
            "parse_error": error,
            "input_tokens": input_tokens,
            "output_tokens": int(generated.shape[-1] - prompt_length),
            "latency_seconds": elapsed / len(rows),
        }
        if normalization_applied:
            result["pre_normalization_parsed_json"] = pre_normalization
            result["normalization_applied"] = True
        results.append(result)
    return results


def expected_stage1(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "qwen_orchestration_route_v1",
        "status": decision["status"],
        "task_type": decision["task_type"],
        "route_name": decision["route_name"],
    }


def numeric_leaves(value: Any, path: str = "") -> dict[str, int | float]:
    """Return canonical field paths for numerical values.

    Unit identity is part of each schema field name (for example,
    ``wavelength_nm``), so matching a path and value checks both the
    normalized unit and its numerical value.
    """
    leaves: dict[str, int | float] = {}
    if isinstance(value, bool):
        return leaves
    if isinstance(value, (int, float)):
        leaves[path] = value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            leaves.update(numeric_leaves(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            leaves.update(numeric_leaves(item, child))
    return leaves


def score(
    predictions: list[dict[str, Any]],
    canonical: Mapping[str, Mapping[str, Any]],
    stage: str,
    image_root: Path,
) -> dict[str, Any]:
    schema_name = (
        "orchestration_route.schema.json"
        if stage == "stage1"
        else "orchestration_decision.schema.json"
    )
    schema = json.loads(
        (REPO_ROOT / "Qwen_orchestration/schemas" / schema_name).read_text()
    )
    validator = Draft202012Validator(schema)
    counts = Counter()
    status_confusion: Counter[str] = Counter()
    route_confusion: Counter[str] = Counter()
    task_confusion: Counter[str] = Counter()
    for item in predictions:
        target_row = canonical[item["example_id"]]
        target_full = target_row["target_decision"]
        target = expected_stage1(target_full) if stage == "stage1" else target_full
        target_status = target["status"]
        predicted = item["parsed_json"]
        counts["total"] += 1

        # Establish every target-conditioned denominator before parsing.  A
        # malformed generation must count as a failure, not disappear from the
        # relevant ready/clarification/unsupported metric.
        if target_status == "ready":
            counts["ready_total"] += 1
            if stage == "stage2":
                target_arguments = target["arguments"]
                counts["required_argument_groups_total"] += len(target_arguments)
                counts["numeric_value_unit_total"] += len(
                    numeric_leaves(target_arguments)
                )
                if target["image_roles"]:
                    counts["image_role_records_total"] += 1
        elif target_status == "needs_clarification":
            counts["clarification_total"] += 1
        elif target_status == "unsupported":
            counts["unsupported_total"] += 1

        if predicted is None:
            counts["parse_invalid"] += 1
            continue
        errors = list(validator.iter_errors(predicted))
        if not errors:
            counts["schema_valid"] += 1
        predicted_status = predicted.get("status")
        status_confusion[f"{target_status}->{predicted_status}"] += 1
        route_confusion[f"{target.get('route_name')}->{predicted.get('route_name')}"] += 1
        task_confusion[f"{target.get('task_type')}->{predicted.get('task_type')}"] += 1
        if predicted_status == target_status:
            counts["status_exact"] += 1
        if predicted.get("task_type") == target.get("task_type"):
            counts["task_exact"] += 1
        if predicted.get("route_name") == target.get("route_name"):
            counts["route_exact_all"] += 1
        if target_status == "ready":
            if predicted.get("route_name") == target.get("route_name"):
                counts["ready_route_exact"] += 1
        if target_status == "needs_clarification":
            if predicted_status == "needs_clarification":
                counts["clarification_recalled"] += 1
        if target_status == "unsupported":
            if predicted_status == "unsupported":
                counts["unsupported_recalled"] += 1
            if predicted_status == "ready":
                counts["unsupported_predicted_ready"] += 1
        if stage == "stage2" and not errors:
            images = {
                f"image_{index}": image_root / value
                for index, value in enumerate(item["images"])
            }
            if target_status == "ready":
                try:
                    validate_decision(predicted, images)
                except Exception:
                    counts["ready_registry_invalid"] += 1
                else:
                    counts["ready_registry_valid"] += 1

                target_arguments = target["arguments"]
                predicted_arguments = predicted.get("arguments", {})
                if predicted_arguments == target_arguments:
                    counts["ready_arguments_exact"] += 1
                for group_name, target_group in target_arguments.items():
                    if (
                        isinstance(predicted_arguments, Mapping)
                        and predicted_arguments.get(group_name) == target_group
                    ):
                        counts["required_argument_groups_exact"] += 1

                target_numbers = numeric_leaves(target_arguments)
                predicted_numbers = numeric_leaves(predicted_arguments)
                counts["numeric_value_unit_exact"] += sum(
                    predicted_numbers.get(path) == value
                    for path, value in target_numbers.items()
                )

                if target["image_roles"]:
                    if predicted.get("image_roles") == target["image_roles"]:
                        counts["image_role_records_exact"] += 1

            if target_status == "needs_clarification":
                if (
                    predicted.get("missing_fields") == target.get("missing_fields")
                    and predicted.get("clarification_question")
                    == target.get("clarification_question")
                ):
                    counts["clarification_fields_exact"] += 1

    def ratio(numerator: str, denominator: str = "total") -> float:
        return counts[numerator] / max(counts[denominator], 1)

    metrics: dict[str, Any] = {
        "count": counts["total"],
        "schema_valid_rate": ratio("schema_valid"),
        "status_exact_accuracy": ratio("status_exact"),
        "task_exact_accuracy": ratio("task_exact"),
        "route_exact_accuracy_all": ratio("route_exact_all"),
        "ready_route_exact_accuracy": ratio("ready_route_exact", "ready_total"),
        "clarification_recall": ratio("clarification_recalled", "clarification_total"),
        "unsupported_recall": ratio("unsupported_recalled", "unsupported_total"),
        "ready_prediction_on_unsupported": ratio(
            "unsupported_predicted_ready", "unsupported_total"
        ),
        "status_confusion": dict(sorted(status_confusion.items())),
        "task_confusion": dict(sorted(task_confusion.items())),
        "route_confusion": dict(sorted(route_confusion.items())),
    }
    if stage == "stage2":
        metrics.update(
            {
                "registry_valid_ready_call_rate": ratio(
                    "ready_registry_valid", "ready_total"
                ),
                "registry_invalid_ready_call_rate": ratio(
                    "ready_registry_invalid", "ready_total"
                ),
                "ready_arguments_exact_accuracy": ratio(
                    "ready_arguments_exact", "ready_total"
                ),
                "required_argument_group_exact_accuracy": ratio(
                    "required_argument_groups_exact",
                    "required_argument_groups_total",
                ),
                "numeric_value_unit_exact_accuracy": ratio(
                    "numeric_value_unit_exact", "numeric_value_unit_total"
                ),
                "image_role_exact_accuracy": ratio(
                    "image_role_records_exact", "image_role_records_total"
                ),
                "clarification_fields_exact_accuracy": ratio(
                    "clarification_fields_exact", "clarification_total"
                ),
                "stage2_denominators": {
                    "ready_records": counts["ready_total"],
                    "required_argument_groups": counts[
                        "required_argument_groups_total"
                    ],
                    "numeric_values_with_unit_paths": counts[
                        "numeric_value_unit_total"
                    ],
                    "visual_ready_records": counts["image_role_records_total"],
                    "clarification_records": counts["clarification_total"],
                },
            }
        )
    return metrics


def existing_predictions(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def normalize_predictions(
    predictions: list[dict[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    enabled = bool(
        config.get("orchestration", {}).get(
            "normalize_nonexecuting_decisions", False
        )
    )
    if not enabled:
        return predictions, 0
    changed_count = 0
    normalized_predictions = []
    for prediction in predictions:
        decision = prediction.get("parsed_json")
        if not isinstance(decision, dict):
            normalized_predictions.append(prediction)
            continue
        already_normalized = prediction.get("normalization_applied") is True
        normalized, changed = normalize_nonexecuting_decision(decision)
        if changed:
            prediction = {
                **prediction,
                "pre_normalization_parsed_json": decision,
                "parsed_json": normalized,
                "normalization_applied": True,
            }
        if already_normalized or changed:
            changed_count += 1
        normalized_predictions.append(prediction)
    return normalized_predictions, changed_count


def prediction_contract_error(
    prediction: Mapping[str, Any],
    row: Mapping[str, Any],
    image_root: Path,
    validator: Draft202012Validator,
) -> str | None:
    decision = prediction.get("parsed_json")
    if not isinstance(decision, Mapping):
        return f"JSON parse failure: {prediction.get('parse_error')}"
    errors = sorted(
        validator.iter_errors(decision),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        return f"schema violation at {path}: {error.message}"
    images = {
        f"image_{index}": image_root / value
        for index, value in enumerate(row["images"])
    }
    try:
        validate_decision(decision, images)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def repair_invalid_once(
    done: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    image_root: Path,
    processor: Any,
    model: Any,
    deps: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    schema = json.loads(
        (
            REPO_ROOT
            / "Qwen_orchestration/schemas/orchestration_decision.schema.json"
        ).read_text()
    )
    validator = Draft202012Validator(schema)
    row_by_id = {row["example_id"]: row for row in rows}
    repaired: dict[str, dict[str, Any]] = {}
    attempted = 0
    for initial in done:
        row = row_by_id[initial["example_id"]]
        error = prediction_contract_error(initial, row, image_root, validator)
        if error is None:
            continue
        attempted += 1
        prompt = list(row["prompt"])
        prompt.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": str(initial.get("raw_prediction_text", "")),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "The decision was rejected: "
                                f"{error}. Correct it and return only valid "
                                "qwen_orchestration_decision_v1 JSON. Do not "
                                "invent missing measurements."
                            ),
                        }
                    ],
                },
            ]
        )
        repair_row = {**row, "prompt": prompt}
        final = generate_batch(
            [repair_row],
            image_root=image_root,
            processor=processor,
            model=model,
            deps=deps,
            config=config,
        )[0]
        final["attempt_count"] = 2
        final["initial_parse_error"] = initial.get("parse_error")
        final["initial_raw_prediction_text"] = initial.get("raw_prediction_text")
        final["initial_contract_error"] = error
        repaired[initial["example_id"]] = final
        print(f"repaired {attempted} invalid decision(s)", flush=True)
    final_rows = [repaired.get(item["example_id"], item) for item in done]
    still_invalid = sum(
        prediction_contract_error(item, row_by_id[item["example_id"]], image_root, validator)
        is not None
        for item in final_rows
    )
    return final_rows, {
        "attempted": attempted,
        "corrected": attempted - still_invalid,
        "still_invalid": still_invalid,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = read_jsonl(args.input_jsonl)
    canonical_rows = read_jsonl(args.canonical_jsonl)
    canonical = {row["example_id"]: row for row in canonical_rows}
    if args.subset == "diagnostic":
        keep = diagnostic_ids(canonical_rows)
        rows = [row for row in rows if row["example_id"] in keep]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.initial_predictions_jsonl and args.resume:
        raise ValueError(
            "--initial-predictions-jsonl cannot be combined with --resume"
        )
    if args.initial_predictions_jsonl:
        done = read_jsonl(args.initial_predictions_jsonl)
    else:
        done = (
            existing_predictions(args.output_jsonl)
            if (args.resume or args.score_only)
            else []
        )
    completed = {row["example_id"] for row in done}
    pending = [row for row in rows if row["example_id"] not in completed]
    adapter = args.adapter_path or Path(config["model"]["adapter_path"])
    if args.score_only:
        if pending:
            raise RuntimeError(
                f"score-only output is missing {len(pending)} requested predictions"
            )
        adapter = adapter.resolve()
    else:
        processor, model, deps, adapter = load_model(config, args.adapter_path)
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume and args.output_jsonl.exists() else "w"
        with args.output_jsonl.open(mode, encoding="utf-8") as stream:
            completed_now = 0
            offset = 0
            while offset < len(pending):
                image_count = len(pending[offset]["images"])
                end = offset
                while (
                    end < len(pending)
                    and end - offset < args.batch_size
                    and len(pending[end]["images"]) == image_count
                ):
                    end += 1
                batch = pending[offset:end]
                predictions = generate_batch(
                    batch,
                    image_root=args.image_root,
                    processor=processor,
                    model=model,
                    deps=deps,
                    config=config,
                )
                for prediction in predictions:
                    stream.write(json.dumps(prediction, sort_keys=True) + "\n")
                    done.append(prediction)
                stream.flush()
                completed_now += len(predictions)
                offset = end
                if completed_now % 20 == 0 or offset == len(pending):
                    print(
                        f"generated {completed_now}/{len(pending)} pending records",
                        flush=True,
                    )
    done, normalized_count = normalize_predictions(done, config)
    repair_report = None
    if args.repair_invalid_once:
        if args.stage != "stage2":
            raise ValueError("--repair-invalid-once requires --stage stage2")
        if args.score_only:
            raise ValueError("--repair-invalid-once cannot be used with --score-only")
        done, repair_report = repair_invalid_once(
            done,
            rows,
            image_root=args.image_root,
            processor=processor,
            model=model,
            deps=deps,
            config=config,
        )
    order = {row["example_id"]: index for index, row in enumerate(rows)}
    done = sorted(
        (row for row in done if row["example_id"] in order),
        key=lambda row: order[row["example_id"]],
    )
    if args.initial_predictions_jsonl or args.repair_invalid_once:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as stream:
            for prediction in done:
                stream.write(json.dumps(prediction, sort_keys=True) + "\n")
    metrics = score(done, canonical, args.stage, args.image_root)
    report = {
        "stage": args.stage,
        "subset": args.subset,
        "adapter_path": str(adapter),
        "input_jsonl": str(args.input_jsonl.resolve()),
        "canonical_jsonl": str(args.canonical_jsonl.resolve()),
        "generation": config.get("generation", {}),
        "constrained_decoding": (
            None if args.score_only else deps.get("constraint_metadata")
        ),
        "repair_invalid_once": repair_report,
        "normalized_nonexecuting_decisions": normalized_count,
        "metrics": metrics,
    }
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
