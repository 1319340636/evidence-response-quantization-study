"""Fail-closed source audit for the frozen Gemma 4 E4B HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


GEMMA_REPO_ID = "google/gemma-4-E4B-it"
GEMMA_MODEL_TYPE = "gemma4"
GEMMA_SOURCE_FILES = frozenset(
    {
        ".gitattributes",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


class GemmaSourceAuditError(RuntimeError):
    """Raised when the downloaded HF source drifts from the frozen identity."""


def build_gemma_source_audit(
    source_path: str | Path, *, repo_id: str, revision: str
) -> dict[str, Any]:
    root = Path(source_path).resolve()
    if repo_id != GEMMA_REPO_ID:
        raise GemmaSourceAuditError("Gemma source repo ID drifted")
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise GemmaSourceAuditError("Gemma source revision is not exact")
    if not root.is_dir():
        raise GemmaSourceAuditError("Gemma source directory is missing")
    names = {path.name for path in root.iterdir() if path.is_file()}
    if names != GEMMA_SOURCE_FILES:
        raise GemmaSourceAuditError("Gemma source top-level file set drifted")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GemmaSourceAuditError("cannot read Gemma source config") from error
    if not isinstance(config, dict) or config.get("model_type") != GEMMA_MODEL_TYPE:
        raise GemmaSourceAuditError("Gemma source model_type must be gemma4")
    files = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": _file_sha256(root / name),
        }
        for name in sorted(GEMMA_SOURCE_FILES)
    }
    record: dict[str, Any] = {
        "schema_version": "gemma-hf-source-audit-v1-20260720",
        "repo_id": repo_id,
        "revision": revision,
        "model_type": GEMMA_MODEL_TYPE,
        "source_path": str(root),
        "files": files,
    }
    record["audit_sha256"] = _record_hash(record)
    return record


def write_source_audit(record: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build_gemma_source_audit(
        args.source, repo_id=args.repo_id, revision=args.revision
    )
    write_source_audit(record, args.output)
    print(record["audit_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

