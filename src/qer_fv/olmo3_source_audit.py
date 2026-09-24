"""Hash-bound source audit for the frozen OLMo 3 HF checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


OLMO3_SOURCE_AUDIT_PROTOCOL = "olmo3-source-audit-v1-20260901"
OLMO3_REPO_ID = "allenai/OLMo-3-7B-Instruct"
OLMO3_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Olmo3SourceAuditError(RuntimeError):
    """Raised when the frozen OLMo source identity or inventory drifts."""


def build_olmo3_source_audit(source_path: str | Path) -> dict[str, Any]:
    source = Path(source_path).resolve()
    if not source.is_dir():
        raise Olmo3SourceAuditError("OLMo source directory is missing")
    revision_path = source / "source_revision.txt"
    try:
        revision = revision_path.read_text(encoding="ascii").strip()
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Olmo3SourceAuditError(f"cannot read OLMo source identity: {error}") from error
    if revision != OLMO3_REVISION:
        raise Olmo3SourceAuditError("OLMo source revision drifted")
    if not isinstance(config, Mapping):
        raise Olmo3SourceAuditError("OLMo source config is invalid")
    if config.get("model_type") != "olmo3":
        raise Olmo3SourceAuditError("OLMo source model_type drifted")
    if config.get("architectures") != ["Olmo3ForCausalLM"]:
        raise Olmo3SourceAuditError("OLMo source architecture drifted")
    files = _hash_tree(source)
    record: dict[str, Any] = {
        "protocol": OLMO3_SOURCE_AUDIT_PROTOCOL,
        "repo_id": OLMO3_REPO_ID,
        "revision": OLMO3_REVISION,
        "model_type": "olmo3",
        "architecture": "Olmo3ForCausalLM",
        "source_path": str(source),
        "files": files,
    }
    record["audit_sha256"] = _record_hash(record)
    return record


def write_olmo3_source_audit(
    audit: Mapping[str, Any],
    path: str | Path,
    *,
    sha256_path: str | Path,
) -> None:
    value = _validate_record(dict(audit))
    destination = Path(path)
    payload = _canonical_bytes(value) + b"\n"
    _atomic_write(destination, payload)
    _atomic_write(Path(sha256_path), hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n")


def load_olmo3_source_audit(
    path: str | Path, *, sha256_path: str | Path
) -> dict[str, Any]:
    audit_path = Path(path)
    try:
        raw = audit_path.read_bytes()
        expected = Path(sha256_path).read_text(encoding="ascii").strip()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Olmo3SourceAuditError(f"cannot load OLMo source audit: {error}") from error
    if not _SHA256.fullmatch(expected) or hashlib.sha256(raw).hexdigest() != expected:
        raise Olmo3SourceAuditError("OLMo source audit detached SHA-256 mismatch")
    if not isinstance(value, dict):
        raise Olmo3SourceAuditError("OLMo source audit must be an object")
    return _validate_record(value)


def verify_olmo3_source_against_audit(
    source_path: str | Path,
    audit_path: str | Path,
    *,
    sha256_path: str | Path,
) -> dict[str, Any]:
    expected = load_olmo3_source_audit(audit_path, sha256_path=sha256_path)
    observed = build_olmo3_source_audit(source_path)
    if observed["files"] != expected["files"]:
        raise Olmo3SourceAuditError("OLMo source inventory drifted")
    for key in ("repo_id", "revision", "model_type", "architecture"):
        if observed[key] != expected[key]:
            raise Olmo3SourceAuditError(f"OLMo source {key} drifted")
    return expected


def _validate_record(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("protocol") != OLMO3_SOURCE_AUDIT_PROTOCOL:
        raise Olmo3SourceAuditError("OLMo source audit protocol drifted")
    if value.get("repo_id") != OLMO3_REPO_ID or value.get("revision") != OLMO3_REVISION:
        raise Olmo3SourceAuditError("OLMo source audit identity drifted")
    if value.get("model_type") != "olmo3" or value.get("architecture") != "Olmo3ForCausalLM":
        raise Olmo3SourceAuditError("OLMo source audit config identity drifted")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise Olmo3SourceAuditError("OLMo source audit files are invalid")
    for name, item in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(item, dict)
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 0
            or not isinstance(item.get("sha256"), str)
            or not _SHA256.fullmatch(item["sha256"])
        ):
            raise Olmo3SourceAuditError("OLMo source audit file entry is invalid")
    expected_hash = value.get("audit_sha256")
    clean = {key: item for key, item in value.items() if key != "audit_sha256"}
    if not isinstance(expected_hash, str) or expected_hash != _record_hash(clean):
        raise Olmo3SourceAuditError("OLMo source audit self-hash mismatch")
    return json.loads(json.dumps(value, ensure_ascii=False))


def _hash_tree(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Olmo3SourceAuditError("OLMo source must not contain symlinks")
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        files[name] = {"bytes": path.stat().st_size, "sha256": _file_hash(path)}
    if not files:
        raise Olmo3SourceAuditError("OLMo source inventory is empty")
    return files


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
