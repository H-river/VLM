"""Simple text tokenization and vocabulary utilities for Stage 3."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


class SimpleTokenizer:
    """A minimal whitespace tokenizer with fixed-length encoding."""

    def __init__(self, vocab=None, max_len=32):
        if not isinstance(max_len, int) or max_len <= 0:
            raise ValueError(f"max_len must be a positive int, got {max_len!r}")

        if vocab is None:
            vocab = {"<pad>": 0, "<unk>": 1}
        if not isinstance(vocab, dict):
            raise ValueError("vocab must be a dict mapping token->id")

        if vocab.get("<pad>") != 0:
            raise ValueError("vocab must map '<pad>' to id 0")
        if vocab.get("<unk>") != 1:
            raise ValueError("vocab must map '<unk>' to id 1")

        self.vocab = {str(k): int(v) for k, v in vocab.items()}
        self.max_len = max_len
        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]

    def tokenize(self, text: str) -> list[str]:
        """Lowercase text, strip punctuation, and split on whitespace."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        cleaned = text.lower()
        cleaned = _PUNCT_RE.sub(" ", cleaned)
        tokens = cleaned.strip().split()
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into fixed-length token ids."""
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(tok, self.unk_id) for tok in tokens[: self.max_len]]
        if len(token_ids) < self.max_len:
            token_ids.extend([self.pad_id] * (self.max_len - len(token_ids)))
        return token_ids


def _iter_jsonl_prompts(jsonl_path) -> list[str]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    prompts = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}: {exc}") from exc

            prompt = record.get("prompt", "")
            if isinstance(prompt, str):
                prompts.append(prompt)
            else:
                prompts.append(str(prompt))
    return prompts


def _build_vocab_from_prompts(prompts: list[str], min_freq: int = 1) -> dict:
    if not isinstance(min_freq, int) or min_freq <= 0:
        raise ValueError(f"min_freq must be a positive int, got {min_freq!r}")

    tokenizer = SimpleTokenizer(max_len=1)
    counter = Counter()

    for prompt in prompts:
        counter.update(tokenizer.tokenize(prompt))

    vocab = {"<pad>": 0, "<unk>": 1}
    next_id = 2

    # Deterministic order: highest frequency first, then token string.
    for token, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq < min_freq:
            continue
        vocab[token] = next_id
        next_id += 1

    return vocab


def build_vocab_from_jsonl(jsonl_path, min_freq=1) -> dict:
    """Build vocabulary from one JSONL file's prompt fields."""
    prompts = _iter_jsonl_prompts(jsonl_path)
    return _build_vocab_from_prompts(prompts, min_freq=min_freq)


def build_vocab_from_jsonl_files(jsonl_paths, min_freq=1) -> dict:
    """Build vocabulary from multiple JSONL files' prompt fields."""
    all_prompts = []
    for path in jsonl_paths:
        all_prompts.extend(_iter_jsonl_prompts(path))
    return _build_vocab_from_prompts(all_prompts, min_freq=min_freq)


def save_vocab(vocab, path):
    """Save vocabulary to JSON."""
    if not isinstance(vocab, dict):
        raise ValueError("vocab must be a dict")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_vocab = {str(k): int(v) for k, v in vocab.items()}
    if normalized_vocab.get("<pad>") != 0 or normalized_vocab.get("<unk>") != 1:
        raise ValueError("vocab must contain '<pad>':0 and '<unk>':1")

    with open(out_path, "w") as f:
        json.dump(normalized_vocab, f, indent=2, sort_keys=True)


def load_vocab(path) -> dict:
    """Load vocabulary from JSON."""
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    if not isinstance(vocab, dict):
        raise ValueError(f"Invalid vocab file (expected dict): {vocab_path}")

    normalized_vocab = {str(k): int(v) for k, v in vocab.items()}
    if normalized_vocab.get("<pad>") != 0:
        raise ValueError("Loaded vocab must map '<pad>' to id 0")
    if normalized_vocab.get("<unk>") != 1:
        raise ValueError("Loaded vocab must map '<unk>' to id 1")

    return normalized_vocab
