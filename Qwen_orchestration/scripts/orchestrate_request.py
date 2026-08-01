#!/usr/bin/env python3
"""Route one natural-language optics request through Qwen and one specialist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Qwen_orchestration.runtime import OrchestrationRuntime
from Qwen_orchestration.runtime.dispatcher import validate_decision
from Qwen_orchestration.runtime.formatting import format_outcome
from Qwen_orchestration.runtime.prompt_contract import (
    apply_decision_contract,
    decision_contract_enabled,
)
from Qwen_orchestration.scripts.evaluate_qwen import (
    generate_batch,
    load_config,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    request = parser.add_mutually_exclusive_group(required=True)
    request.add_argument("--request")
    request.add_argument("--request-file", type=Path)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable the single schema/registry correction attempt.",
    )
    return parser.parse_args()


def user_message(request: str, image_count: int) -> dict[str, Any]:
    content = [{"type": "image"} for _ in range(image_count)]
    content.append(
        {
            "type": "text",
            "text": (
                request.rstrip()
                + "\nReturn only qwen_orchestration_decision_v1 JSON."
            ),
        }
    )
    return {"role": "user", "content": content}


def decision_error(
    decision: dict[str, Any] | None,
    parse_error: str | None,
    bindings: dict[str, Path],
) -> str | None:
    if decision is None:
        return f"JSON parse failure: {parse_error}"
    try:
        validate_decision(decision, bindings)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def generate_attempt(
    prompt: list[dict[str, Any]],
    image_paths: list[Path],
    *,
    processor: Any,
    model: Any,
    dependencies: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "example_id": "runtime_request",
        "group_id": "runtime_request",
        "prompt": prompt,
        "images": [str(path.resolve()) for path in image_paths],
    }
    return generate_batch(
        [row],
        image_root=Path("/"),
        processor=processor,
        model=model,
        deps=dependencies,
        config=config,
    )[0]


def main() -> None:
    args = parse_args()
    request = (
        args.request
        if args.request is not None
        else args.request_file.read_text(encoding="utf-8")
    )
    image_paths = [path.resolve() for path in args.image]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    bindings = {
        f"image_{index}": path for index, path in enumerate(image_paths)
    }
    config = load_config(args.config)
    processor, model, dependencies, adapter = load_model(
        config, args.adapter_path
    )
    prompt = [user_message(request, len(image_paths))]
    if decision_contract_enabled(config):
        prompt = apply_decision_contract(prompt)
    attempts = []
    generated = generate_attempt(
        prompt,
        image_paths,
        processor=processor,
        model=model,
        dependencies=dependencies,
        config=config,
    )
    attempts.append(generated)
    error = decision_error(
        generated["parsed_json"], generated["parse_error"], bindings
    )
    if error is not None and not args.no_repair:
        prompt.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": generated["raw_prediction_text"]}
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
        generated = generate_attempt(
            prompt,
            image_paths,
            processor=processor,
            model=model,
            dependencies=dependencies,
            config=config,
        )
        attempts.append(generated)
        error = decision_error(
            generated["parsed_json"], generated["parse_error"], bindings
        )
    if error is not None:
        report = {
            "status": "orchestration_error",
            "adapter_path": str(adapter),
            "attempt_count": len(attempts),
            "error": error,
            "attempts": attempts,
        }
    else:
        decision = generated["parsed_json"]
        outcome = OrchestrationRuntime().dispatch(decision, bindings)
        report = {
            "status": outcome["status"],
            "adapter_path": str(adapter),
            "attempt_count": len(attempts),
            "decision": decision,
            "outcome": outcome,
            "response": format_outcome(outcome),
        }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    if report["status"] == "orchestration_error":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
