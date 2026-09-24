"""Frozen input configuration and byte-level provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_freeze_config(path: str | Path) -> dict[str, Any]:
    """Load a freeze configuration and require a JSON object at its root."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("freeze configuration must be a JSON object")
    return config


def verify_frozen_files(
    project_root: str | Path, config: Mapping[str, Any]
) -> dict[str, str]:
    """Verify every raw file whose path and SHA-256 are frozen in config."""
    root = Path(project_root)
    declarations = (
        ("vitaminc.test", "vitaminc", "test_path", "test_sha256"),
        ("vitaminc.dev", "vitaminc", "dev_path", "dev_sha256"),
        (
            "vitaminc.old_evaluation",
            "vitaminc",
            "old_evaluation_path",
            "old_evaluation_sha256",
        ),
        ("tabfact.test", "tabfact", "test_path", "test_sha256"),
        (
            "tabfact.old_evaluation",
            "tabfact",
            "old_evaluation_path",
            "old_evaluation_sha256",
        ),
    )
    verified: dict[str, str] = {}
    for name, section_name, path_key, hash_key in declarations:
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            continue
        relative_path = section.get(path_key)
        expected = section.get(hash_key)
        if relative_path is None and expected is None:
            continue
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise ValueError(f"{name} requires string path and SHA-256 values")
        actual = sha256_file(root / relative_path)
        if actual.casefold() != expected.casefold():
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {expected}, got {actual}"
            )
        verified[name] = actual
    return verified
