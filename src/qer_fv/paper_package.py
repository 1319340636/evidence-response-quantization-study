"""Write and verify an immutable paper-ready artifact package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from qer_fv.paper_evidence import build_paper_evidence
from qer_fv.paper_render import render_paper_files


PAPER_PACKAGE_MANIFEST_PROTOCOL = "paper-package-manifest-v1-20260728"


def run_paper_package(
    *,
    project_root: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Build manuscript files from verified evidence and write them once."""
    evidence = build_paper_evidence(project_root)
    files = render_paper_files(evidence)
    return write_paper_package(
        files,
        output_directory,
        source_identity={
            "source_inventory_sha256": evidence[
                "source_inventory_sha256"
            ],
            "source_publication_manifest_sha256": evidence[
                "source_publication_manifest_sha256"
            ],
            "evidence_protocol": evidence["evidence_protocol"],
            "qwen_triplet_report_manifest_sha256": evidence[
                "qwen_awq_formal"
            ]["report_manifest_sha256"],
        },
    )


def write_paper_package(
    files: Mapping[str, str],
    output_directory: str | Path,
    *,
    source_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Write deterministic files and their self-verifying manifest."""
    output = Path(output_directory)
    if output.exists() and any(
        path.is_file() or path.is_symlink()
        for path in output.rglob("*")
    ):
        raise ValueError("paper output directory is not immutable-empty")
    output.mkdir(parents=True, exist_ok=True)

    normalized: dict[str, str] = {}
    for name, content in files.items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or name == "package_manifest.json"
            or not isinstance(content, str)
        ):
            raise ValueError(f"invalid paper package file: {name}")
        normalized[relative.as_posix()] = content

    for name, content in sorted(normalized.items()):
        _atomic_write_text(output / Path(name), content)

    manifest = {
        "manifest_protocol": PAPER_PACKAGE_MANIFEST_PROTOCOL,
        "evidence_protocol": source_identity["evidence_protocol"],
        "source_inventory_sha256": source_identity[
            "source_inventory_sha256"
        ],
        "source_publication_manifest_sha256": source_identity[
            "source_publication_manifest_sha256"
        ],
        "qwen_triplet_report_manifest_sha256": source_identity[
            "qwen_triplet_report_manifest_sha256"
        ],
        "files": {
            name: _file_sha256(output / Path(name))
            for name in sorted(normalized)
        },
    }
    manifest["manifest_sha256"] = _record_hash(
        manifest, excluded="manifest_sha256"
    )
    _atomic_write_text(
        output / "package_manifest.json",
        _json_text(manifest),
    )
    return manifest


def verify_paper_package_manifest(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify the package self-hash and every listed output file."""
    output = Path(output_directory)
    manifest = _load_json_object(output / "package_manifest.json")
    if manifest.get("manifest_protocol") != PAPER_PACKAGE_MANIFEST_PROTOCOL:
        raise ValueError("paper package manifest protocol mismatch")
    expected_self = manifest.get("manifest_sha256")
    actual_self = _record_hash(manifest, excluded="manifest_sha256")
    if expected_self != actual_self:
        raise ValueError("paper package manifest self-hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("paper package manifest files are invalid")
    for name, expected_hash in sorted(files.items()):
        path = output / Path(name)
        if not path.is_file():
            raise ValueError(f"paper package file is missing: {name}")
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"paper package file hash mismatch: {name}")
    return manifest


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_hash(value: Mapping[str, Any], *, excluded: str) -> str:
    payload = {key: item for key, item in value.items() if key != excluded}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
