"""Generation-time JSON constraints shared by evaluation and live inference."""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ENGINE = "lm-format-enforcer"
SUPPORTED_VERSION = "0.11.3"


def _schema_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _regular_tokens(tokenizer: Any, vocab_size: int) -> list[tuple[int, str, bool]]:
    token_zero = tokenizer.encode("0")[-1]
    values = []
    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        decoded_after_zero = tokenizer.decode([token_zero, token_id])[1:]
        decoded_regular = tokenizer.decode([token_id])
        values.append(
            (
                token_id,
                decoded_after_zero,
                len(decoded_after_zero) > len(decoded_regular),
            )
        )
    return values


def _decoder(tokenizer: Any, tokens: list[int]) -> str:
    return tokenizer.decode(tokens).rstrip("\ufffd")


def build_constraint(
    config: Mapping[str, Any], tokenizer: Any
) -> tuple[
    Callable[[], Callable[[int, Any], list[int]]] | None,
    dict[str, Any] | None,
]:
    """Build a factory for batch-scoped Transformers prefix functions."""
    generation = config.get("generation", {})
    constraint = generation.get("json_schema_constraint")
    if not constraint:
        return None, None
    if not isinstance(constraint, Mapping):
        raise TypeError("generation.json_schema_constraint must be an object")
    engine = str(constraint.get("engine", SUPPORTED_ENGINE))
    if engine != SUPPORTED_ENGINE:
        raise ValueError(
            f"unsupported constrained-decoding engine {engine!r}; "
            f"expected {SUPPORTED_ENGINE!r}"
        )
    installed = importlib.metadata.version("lm-format-enforcer")
    requested = str(constraint.get("version", SUPPORTED_VERSION))
    if requested != SUPPORTED_VERSION or installed != requested:
        raise RuntimeError(
            "constrained-decoding version mismatch: "
            f"config={requested}, installed={installed}, supported={SUPPORTED_VERSION}"
        )
    path = _schema_path(constraint["schema"])
    schema_bytes = path.read_bytes()
    schema = json.loads(schema_bytes)

    from lmformatenforcer import (
        JsonSchemaParser,
        TokenEnforcer,
        TokenEnforcerTokenizerData,
    )
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig

    vocab_size = len(tokenizer)
    tokenizer_data = TokenEnforcerTokenizerData(
        _regular_tokens(tokenizer, vocab_size),
        functools.partial(_decoder, tokenizer),
        tokenizer.eos_token_id,
        False,
        vocab_size,
    )
    force_field_order = bool(constraint.get("force_field_order", True))
    def new_batch_constraint() -> Callable[[int, Any], list[int]]:
        # TokenEnforcer caches every encountered prompt and generated prefix.
        # Scope that cache to one generation batch so a long evaluation cannot
        # retain thousands of completed sequences in RAM.
        parser = JsonSchemaParser(
            schema,
            CharacterLevelParserConfig(force_json_field_order=force_field_order),
        )
        enforcer = TokenEnforcer(tokenizer_data, parser)
        enforcer.root_parser.config.force_json_field_order = force_field_order

        def allowed_tokens(_batch_id: int, sent: Any) -> list[int]:
            return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

        return allowed_tokens

    metadata = {
        "enabled": True,
        "engine": engine,
        "version": installed,
        "schema_path": str(path),
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "force_field_order": force_field_order,
        "cache_scope": "generation_batch",
    }
    return new_batch_constraint, metadata
