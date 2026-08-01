"""Deterministic byte identity for local Hugging Face model snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_FINGERPRINT_SCHEMA = "qwen_vl_local_snapshot_tree_v1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
VOLATILE_COMPONENTS = frozenset({".cache", ".git", ".locks", "__pycache__"})
VOLATILE_SUFFIXES = (".lock", ".tmp", ".incomplete")
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_model_directory(model_id: str, *, repository_root: Path) -> Path | None:
    """Resolve a local model directory; return ``None`` for a remote Hub ID."""

    candidate = Path(model_id).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (repository_root / candidate).resolve()
    if resolved.is_dir():
        return resolved
    if resolved.exists():
        raise ValueError(f"local model source exists but is not a directory: {resolved}")
    return None


def _volatile(relative: Path) -> bool:
    return any(part in VOLATILE_COMPONENTS for part in relative.parts) or relative.name.endswith(
        VOLATILE_SUFFIXES
    )


def _snapshot_files(root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    included: list[tuple[str, Path]] = []
    excluded: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        if _volatile(relative):
            if path.is_file():
                excluded.append(relative_text)
            continue
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(f"broken symlink in local model snapshot: {path}")
        if path.is_file():
            included.append((relative_text, path))
    included.sort(key=lambda value: value[0])
    excluded.sort()
    return included, excluded


def _referenced_weight_shards(root: Path, included_paths: set[str]) -> list[str]:
    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(
            index_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid safetensors index {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"safetensors index has no nonempty weight_map: {index_path}")
    shards: set[str] = set()
    for tensor_name, shard in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(shard, str) or not shard:
            raise ValueError(f"invalid weight_map entry in {index_path}")
        shard_path = Path(shard)
        if shard_path.is_absolute() or ".." in shard_path.parts:
            raise ValueError(f"unsafe shard path in {index_path}: {shard!r}")
        normalized = shard_path.as_posix()
        if normalized not in included_paths:
            raise FileNotFoundError(f"referenced weight shard is absent: {root / shard_path}")
        if not normalized.endswith(".safetensors"):
            raise ValueError(f"weight-map shard is not safetensors: {normalized}")
        shards.add(normalized)
    return sorted(shards)


def fingerprint_snapshot_tree(root: Path) -> dict[str, Any]:
    """Hash every nonvolatile regular snapshot file and audit indexed shards.

    The aggregate identity depends only on relative paths, byte lengths, and
    per-file SHA-256 values. Absolute staging path, mtimes, permissions, and
    whether a file is materialized or symlinked do not affect the identity.
    """

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local model snapshot directory does not exist: {root}")
    files, excluded = _snapshot_files(root)
    if not files:
        raise ValueError(f"local model snapshot contains no regular files: {root}")
    included_paths = {relative for relative, _ in files}
    missing = [name for name in REQUIRED_SNAPSHOT_FILES if name not in included_paths]
    if missing:
        raise FileNotFoundError(f"local model snapshot lacks required files: {missing}")
    shards = _referenced_weight_shards(root, included_paths)

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for relative, path in files:
        before = path.stat()
        digest = sha256_path(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"local snapshot file changed while hashing: {path}")
        entry = {"path": relative, "bytes": before.st_size, "sha256": digest}
        entries.append(entry)
        total_bytes += before.st_size

    # Detect additions/removals that raced the fingerprint walk.
    final_files, _ = _snapshot_files(root)
    if [relative for relative, _ in final_files] != [entry["path"] for entry in entries]:
        raise RuntimeError(f"local snapshot file set changed while hashing: {root}")

    canonical = json.dumps(
        {"schema": SNAPSHOT_FINGERPRINT_SCHEMA, "files": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": SNAPSHOT_FINGERPRINT_SCHEMA,
        "root": str(root),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
        "required_snapshot_files": list(REQUIRED_SNAPSHOT_FILES),
        "referenced_weight_shards": shards,
        "excluded_volatile_components": sorted(VOLATILE_COMPONENTS),
        "excluded_volatile_suffixes": list(VOLATILE_SUFFIXES),
        "excluded_volatile_regular_files": excluded,
    }


def resolve_model_source_identity(
    *,
    model_id: str,
    source_id: str,
    revision: str,
    processor_revision: str,
    repository_root: Path,
    expected_local_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve remote revision identity or validate a staged local snapshot."""

    if expected_local_tree_sha256 is not None and not HEX_SHA256.fullmatch(
        expected_local_tree_sha256
    ):
        raise ValueError("expected_local_snapshot_tree_sha256 must be a lowercase SHA-256")
    local = local_model_directory(model_id, repository_root=repository_root)
    if local is None:
        return {
            "kind": "huggingface_revision",
            "model_id": model_id,
            "source_id": source_id,
            "revision": revision,
            "processor_revision": processor_revision,
            "revision_enforcement": "from_pretrained revision arguments",
            "expected_local_snapshot_tree_sha256_if_staged": expected_local_tree_sha256,
        }

    fingerprint = fingerprint_snapshot_tree(local)
    actual = fingerprint["tree_sha256"]
    if expected_local_tree_sha256 is not None and actual != expected_local_tree_sha256:
        raise ValueError(
            "local model snapshot tree SHA-256 mismatch: "
            f"expected {expected_local_tree_sha256}, got {actual} for {local}"
        )
    return {
        "kind": "local_snapshot",
        "model_id": str(local),
        "source_id": source_id,
        "declared_revision": revision,
        "declared_processor_revision": processor_revision,
        "revision_enforcement": (
            "local paths ignore Hub revision arguments; byte identity is the snapshot tree SHA-256"
        ),
        "expected_tree_sha256": expected_local_tree_sha256,
        "expected_tree_sha256_verified": (
            actual == expected_local_tree_sha256
            if expected_local_tree_sha256 is not None
            else None
        ),
        "fingerprint": fingerprint,
    }


def pretrained_revision_kwargs(source_identity: Mapping[str, Any], revision: str) -> dict[str, str]:
    """Pass a Hub revision only when it is meaningful for a remote model ID."""

    return {"revision": revision} if source_identity.get("kind") == "huggingface_revision" else {}


def compact_source_identity(source_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable identity fields suitable for every prediction row."""

    kind = source_identity.get("kind")
    if kind == "local_snapshot":
        fingerprint = source_identity.get("fingerprint")
        if not isinstance(fingerprint, Mapping):
            raise ValueError("local model source identity lacks a fingerprint")
        return {
            "kind": kind,
            "schema": fingerprint.get("schema"),
            "tree_sha256": fingerprint.get("tree_sha256"),
            "file_count": fingerprint.get("file_count"),
            "total_bytes": fingerprint.get("total_bytes"),
            "expected_tree_sha256": source_identity.get("expected_tree_sha256"),
            "expected_tree_sha256_verified": source_identity.get(
                "expected_tree_sha256_verified"
            ),
        }
    if kind == "huggingface_revision":
        return {
            "kind": kind,
            "model_id": source_identity.get("model_id"),
            "revision": source_identity.get("revision"),
            "processor_revision": source_identity.get("processor_revision"),
        }
    raise ValueError(f"unsupported model source identity kind: {kind!r}")
