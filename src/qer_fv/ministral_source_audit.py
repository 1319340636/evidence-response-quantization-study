"""Fail-closed source audit for the frozen Ministral-3-8B HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .ministral_preflight import (
    MinistralPreflightError,
    verify_detached_sha256,
    write_detached_sha256,
)


MINISTRAL_REPO_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
MINISTRAL_REVISION = "06cc81bfd6e45321d8fc8f816576c5b6ac67ec22"
MINISTRAL_MODEL_TYPES = frozenset({"mistral3"})
SOURCE_AUDIT_PROTOCOL = "ministral-hf-source-audit-v1-20260809"
OFFICIAL_SOURCE_FILES = (
    ".gitattributes",
    "README.md",
    "SYSTEM_PROMPT.txt",
    "chat_template.jinja",
    "config.json",
    "consolidated.safetensors",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "params.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tekken.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
OFFICIAL_SOURCE_FILE_COUNT = len(OFFICIAL_SOURCE_FILES)
MINISTRAL_OFFICIAL_MANIFEST = {
    ".gitattributes": {"kind": "git_blob", "bytes": 1618, "oid": "c684f12ce1b2a358bfc3e8539ba4ca0ef7b5c9cb"},
    "README.md": {"kind": "git_blob", "bytes": 8637, "oid": "4e89628b5e50296bf5764f974b57a9a1f46ac1cd"},
    "SYSTEM_PROMPT.txt": {"kind": "git_blob", "bytes": 2406, "oid": "dfbb4e37c19f555872f3b54b74103a22e488f6c3"},
    "chat_template.jinja": {"kind": "git_blob", "bytes": 7753, "oid": "64026c215540555bb80db74a5c8b2fea875c509d"},
    "config.json": {"kind": "git_blob", "bytes": 1579, "oid": "d726bfccbcc8aaf8180e201d90fc608aa74dfbfa"},
    "generation_config.json": {"kind": "git_blob", "bytes": 131, "oid": "add11cbc06647495098ee6dd5c9cbc96841a445a"},
    "model.safetensors.index.json": {"kind": "git_blob", "bytes": 52675, "oid": "c2b0eca7d7b6eaf9f94917e48f678c947db5f4e1"},
    "params.json": {"kind": "git_blob", "bytes": 1098, "oid": "63b743825c2462f27806a66b972c4a71e7ced7c5"},
    "processor_config.json": {"kind": "git_blob", "bytes": 976, "oid": "a37d728b12fd27ac60a437894bd51de83449bf30"},
    "special_tokens_map.json": {"kind": "git_blob", "bytes": 147094, "oid": "c5ac9bbb2b7c801eec23cb07fee0d07805aebc42"},
    "tokenizer_config.json": {"kind": "git_blob", "bytes": 198094, "oid": "7f9f2f4f941ab1dffda12e9feeff2fd1d18d3e03"},
    "consolidated.safetensors": {"kind": "lfs", "bytes": 17836115976, "oid": "44db184fe2b0b08edea6370ad8403d7b17d604d38127133f50a3da111a874eca"},
    "model-00001-of-00004.safetensors": {"kind": "lfs", "bytes": 4984292952, "oid": "95f4da19c81e6a06d4f0c61cac3dfdd85ef463a7955473dc4c07e570fd2342f9"},
    "model-00002-of-00004.safetensors": {"kind": "lfs", "bytes": 4999804256, "oid": "18275e4c8413d0ed4b0cb8380fe1fd4faeee34189960fdd92ef5b0d3f4ed1b98"},
    "model-00003-of-00004.safetensors": {"kind": "lfs", "bytes": 4915917680, "oid": "980d1d3635ae29a6e629e2f7ad589c882eba270c3f66da74da4f8ccb11845687"},
    "model-00004-of-00004.safetensors": {"kind": "lfs", "bytes": 2936108304, "oid": "fa80e52af465c5644b18b623eeeebb1d50e47dafbee73816bf26f2473f88d369"},
    "tekken.json": {"kind": "lfs", "bytes": 16753784, "oid": "600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843"},
    "tokenizer.json": {"kind": "lfs", "bytes": 17078128, "oid": "d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135"},
}
MINISTRAL_OFFICIAL_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(
        MINISTRAL_OFFICIAL_MANIFEST,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
_FORBIDDEN_SOURCE_PARTS = frozenset({".cache", "cache", "tmp", "temp"})


class MinistralSourceAuditError(RuntimeError):
    """Raised when the downloaded Ministral source identity drifts."""


def build_ministral_source_audit(
    source_path: str | Path, *, repo_id: str, revision: str
) -> dict[str, Any]:
    """Bind the exact original HF source without loading model weights."""
    root = Path(source_path).resolve()
    if repo_id != MINISTRAL_REPO_ID:
        raise MinistralSourceAuditError("Ministral source repo ID drifted")
    if revision != MINISTRAL_REVISION:
        raise MinistralSourceAuditError("Ministral source revision drifted")
    if not root.is_dir():
        raise MinistralSourceAuditError("Ministral source directory is missing")

    regular_files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in regular_files:
        relative = path.relative_to(root)
        lowered_parts = {part.casefold() for part in relative.parts}
        if relative.name.casefold() == "download.log":
            raise MinistralSourceAuditError(
                "Ministral source download.log is transport state, not official"
            )
        if lowered_parts & _FORBIDDEN_SOURCE_PARTS:
            raise MinistralSourceAuditError(
                "Ministral source cache or temporary content is forbidden"
            )
    names = {path.relative_to(root).as_posix() for path in regular_files}
    if "config.json" not in names:
        raise MinistralSourceAuditError("Ministral source config.json is missing")
    if "tokenizer_config.json" not in names or not names.intersection(
        {"tokenizer.json", "tokenizer.model", "tekken.json"}
    ):
        raise MinistralSourceAuditError("Ministral tokenizer files are missing")
    if not any(name.endswith(".safetensors") for name in names):
        raise MinistralSourceAuditError("Ministral safetensors weights are missing")
    if any(name.lower().endswith(".gguf") for name in names):
        raise MinistralSourceAuditError("GGUF files are forbidden in the HF source")
    if names != set(OFFICIAL_SOURCE_FILES):
        raise MinistralSourceAuditError(
            "Ministral official file inventory drifted from the frozen revision"
        )

    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads(
            (root / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralSourceAuditError(
            "cannot read Ministral config or tokenizer metadata"
        ) from error
    if not isinstance(config, dict):
        raise MinistralSourceAuditError("Ministral source config must be an object")
    model_type = config.get("model_type")
    if model_type not in MINISTRAL_MODEL_TYPES:
        raise MinistralSourceAuditError(
            "Ministral source model_type is not the frozen mistral3 architecture"
        )
    if not isinstance(tokenizer_config, dict):
        raise MinistralSourceAuditError("tokenizer config must be an object")

    files = _verify_trusted_source(root, MINISTRAL_OFFICIAL_MANIFEST)
    record: dict[str, Any] = {
        "schema_version": SOURCE_AUDIT_PROTOCOL,
        "model_key": "ministral3_8b",
        "repo_id": repo_id,
        "revision": revision,
        "model_type": model_type,
        "source_path": str(root),
        "chat_template_present": bool(tokenizer_config.get("chat_template")),
        "official_manifest_sha256": MINISTRAL_OFFICIAL_MANIFEST_SHA256,
        "files": files,
        "total_bytes": sum(value["bytes"] for value in files.values()),
    }
    record["audit_sha256"] = _record_hash(record)
    return record


def write_ministral_source_audit(
    record: Mapping[str, Any], path: str | Path
) -> None:
    """Atomically write a canonical, human-readable source audit."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(record),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        write_detached_sha256(destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_ministral_source_audit(
    path: str | Path, *, sha256_path: str | Path | None = None
) -> dict[str, Any]:
    """Load an audit and reject self-hash or identity drift."""
    audit_path = Path(path)
    detached = (
        Path(sha256_path)
        if sha256_path is not None
        else audit_path.with_name(audit_path.name + ".sha256")
    )
    try:
        verify_detached_sha256(audit_path, detached)
    except MinistralPreflightError as error:
        raise MinistralSourceAuditError(
            f"Ministral source audit detached hash failed: {error}"
        ) from error
    try:
        value = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralSourceAuditError("cannot load Ministral source audit") from error
    if not isinstance(value, dict):
        raise MinistralSourceAuditError("Ministral source audit must be an object")
    expected_keys = {
        "schema_version",
        "model_key",
        "repo_id",
        "revision",
        "model_type",
        "source_path",
        "chat_template_present",
        "files",
        "total_bytes",
        "audit_sha256",
        "official_manifest_sha256",
    }
    if set(value) != expected_keys:
        raise MinistralSourceAuditError("Ministral source audit schema drifted")
    if value.get("schema_version") != SOURCE_AUDIT_PROTOCOL:
        raise MinistralSourceAuditError("Ministral source audit protocol drifted")
    if value.get("repo_id") != MINISTRAL_REPO_ID:
        raise MinistralSourceAuditError("Ministral source audit repo drifted")
    if value.get("revision") != MINISTRAL_REVISION:
        raise MinistralSourceAuditError("Ministral source audit revision drifted")
    if value.get("official_manifest_sha256") != MINISTRAL_OFFICIAL_MANIFEST_SHA256:
        raise MinistralSourceAuditError(
            "Ministral official manifest hash drifted"
        )
    if value.get("model_key") != "ministral3_8b":
        raise MinistralSourceAuditError("Ministral source audit model key drifted")
    if value.get("model_type") not in MINISTRAL_MODEL_TYPES:
        raise MinistralSourceAuditError("Ministral source audit model_type drifted")
    if (
        not isinstance(value.get("source_path"), str)
        or not value["source_path"].strip()
    ):
        raise MinistralSourceAuditError("Ministral source audit source path is invalid")
    if not isinstance(value.get("chat_template_present"), bool):
        raise MinistralSourceAuditError(
            "Ministral source audit chat-template flag is invalid"
        )
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != set(OFFICIAL_SOURCE_FILES):
        raise MinistralSourceAuditError(
            "Ministral source audit official file inventory drifted"
        )
    total_bytes = 0
    for name, metadata in files.items():
        if any(
            part.casefold() in _FORBIDDEN_SOURCE_PARTS
            for part in Path(name).parts
        ) or Path(name).name.casefold() == "download.log":
            raise MinistralSourceAuditError(
                "Ministral source audit contains cache or temporary content"
            )
        if not isinstance(metadata, dict) or set(metadata) != {"bytes", "sha256"}:
            raise MinistralSourceAuditError(
                "Ministral source audit file metadata schema drifted"
            )
        byte_count = metadata.get("bytes")
        digest = metadata.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise MinistralSourceAuditError(
                "Ministral source audit file byte count is invalid"
            )
        _validate_audit_official_size(name, byte_count)
        if not _is_sha256(digest):
            raise MinistralSourceAuditError(
                "Ministral source audit file SHA-256 is invalid"
            )
        total_bytes += byte_count
    if (
        isinstance(value.get("total_bytes"), bool)
        or not isinstance(value.get("total_bytes"), int)
        or value["total_bytes"] != total_bytes
    ):
        raise MinistralSourceAuditError("Ministral source audit total bytes drifted")
    if value.get("audit_sha256") != _record_hash(value):
        raise MinistralSourceAuditError("Ministral source audit hash drifted")
    return value


def verify_ministral_source_against_audit(
    source_path: str | Path, audit_path: str | Path
) -> dict[str, Any]:
    """Re-hash the current source directory and require byte-for-byte identity."""
    audit = load_ministral_source_audit(audit_path)
    root = Path(source_path).resolve()
    if not root.is_dir():
        raise MinistralSourceAuditError("Ministral source directory is missing")
    all_files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in all_files:
        relative = path.relative_to(root)
        lowered_parts = {part.casefold() for part in relative.parts}
        if relative.name.casefold() == "download.log" or (
            lowered_parts & _FORBIDDEN_SOURCE_PARTS
        ):
            raise MinistralSourceAuditError(
                "Ministral source cache, temporary, or download content appeared"
            )
    if {
        path.relative_to(root).as_posix() for path in all_files
    } != set(OFFICIAL_SOURCE_FILES):
        raise MinistralSourceAuditError(
            "Ministral source file inventory drifted"
        )
    files = _verify_trusted_source(root, MINISTRAL_OFFICIAL_MANIFEST)
    if files != audit.get("files"):
        raise MinistralSourceAuditError("Ministral source file inventory drifted")
    if str(root) != audit.get("source_path"):
        raise MinistralSourceAuditError("Ministral source path drifted")
    return audit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_trusted_source(
    root: Path, manifest: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Verify frozen Git blob/LFS identities and return audit metadata."""
    if set(manifest) != {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}:
        raise MinistralSourceAuditError("Ministral official file inventory drifted")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(manifest):
        identity = manifest[name]
        if set(identity) != {"kind", "bytes", "oid"}:
            raise MinistralSourceAuditError("trusted source manifest schema drifted")
        path = root / name
        expected_size = identity["bytes"]
        if path.stat().st_size != expected_size:
            raise MinistralSourceAuditError(
                f"Ministral official source size drifted: {name}"
            )
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1()
        if identity["kind"] == "git_blob":
            git_blob.update(f"blob {expected_size}\0".encode("ascii"))
        elif identity["kind"] != "lfs":
            raise MinistralSourceAuditError("trusted source manifest kind drifted")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha256.update(block)
                if identity["kind"] == "git_blob":
                    git_blob.update(block)
        actual_identity = (
            git_blob.hexdigest()
            if identity["kind"] == "git_blob"
            else sha256.hexdigest()
        )
        if actual_identity != identity["oid"]:
            raise MinistralSourceAuditError(
                f"Ministral official source identity drifted: {name}"
            )
        result[name] = {"bytes": expected_size, "sha256": sha256.hexdigest()}
    return result


def _validate_audit_official_size(name: str, byte_count: int) -> None:
    if byte_count != MINISTRAL_OFFICIAL_MANIFEST[name]["bytes"]:
        raise MinistralSourceAuditError(
            "Ministral source audit official file size drifted"
        )


def _record_hash(record: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in record.items() if key != "audit_sha256"}
    payload = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build_ministral_source_audit(
        args.source, repo_id=args.repo_id, revision=args.revision
    )
    write_ministral_source_audit(record, args.output)
    print(record["audit_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
