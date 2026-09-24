"""Import-safe, hash-bound preflight contracts for OLMo 3 artifacts."""

from __future__ import annotations

import hashlib
import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping


GPTQ_DEPENDENCY_VERSIONS = {
    "gptqmodel": "7.1.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
}
AWQ_DEPENDENCY_VERSIONS = {
    "llmcompressor": "0.12.0",
    "compressed-tensors": "0.17.1",
    "transformers": "5.10.1",
    "torch": "2.12.0",
}
COMPATIBILITY_PROTOCOL = "olmo3-quantizer-compatibility-v1-20260901"
REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Olmo3PreflightError(RuntimeError):
    """Raised before expensive OLMo artifact construction on drift."""


def validate_dependency_versions(
    expected: Mapping[str, str],
    *,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    installed: dict[str, str] = {}
    for name, required in expected.items():
        try:
            actual = str(version_getter(name))
        except Exception as error:
            raise Olmo3PreflightError(
                f"required dependency is missing: {name}=={required}"
            ) from error
        if actual != required:
            raise Olmo3PreflightError(
                f"required dependency mismatch: {name}=={required}; installed {actual}"
            )
        installed[name] = actual
    return installed


def load_olmo3_compatibility_certificate(
    path: str | Path,
    *,
    sha256_path: str | Path,
    route: str,
    source_audit_sha256: str,
) -> dict[str, Any]:
    certificate_path = Path(path)
    try:
        raw = certificate_path.read_bytes()
        detached = Path(sha256_path).read_text(encoding="ascii").strip()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Olmo3PreflightError(f"cannot load compatibility certificate: {error}") from error
    if not _SHA256.fullmatch(detached) or hashlib.sha256(raw).hexdigest() != detached:
        raise Olmo3PreflightError("compatibility certificate detached SHA-256 mismatch")
    if not isinstance(value, dict):
        raise Olmo3PreflightError("compatibility certificate must be an object")
    supplied = value.get("certificate_sha256")
    clean = {key: item for key, item in value.items() if key != "certificate_sha256"}
    if supplied != _canonical_hash(clean):
        raise Olmo3PreflightError("compatibility certificate self-hash mismatch")
    if (
        value.get("protocol") != COMPATIBILITY_PROTOCOL
        or value.get("model_key") != "olmo3_7b"
        or value.get("model_type") != "olmo3"
        or value.get("architecture") != "Olmo3ForCausalLM"
        or value.get("revision") != REVISION
    ):
        raise Olmo3PreflightError("compatibility certificate identity drifted")
    if value.get("route") != route:
        raise Olmo3PreflightError("compatibility certificate route drifted")
    if value.get("source_audit_sha256") != source_audit_sha256:
        raise Olmo3PreflightError("compatibility certificate source audit drifted")
    if not isinstance(value.get("artifact_files"), dict) or not value["artifact_files"]:
        raise Olmo3PreflightError("compatibility artifact inventory is empty")
    return json.loads(json.dumps(value, ensure_ascii=False))


def write_detached_sha256(path: str | Path) -> Path:
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    sidecar.write_text(_file_hash(source) + "\n", encoding="ascii")
    return sidecar


def verify_detached_sha256(
    path: str | Path, *, sha256_path: str | Path
) -> str:
    source = Path(path)
    try:
        expected = Path(sha256_path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise Olmo3PreflightError(f"cannot read detached SHA-256: {error}") from error
    actual = _file_hash(source)
    if not _SHA256.fullmatch(expected) or actual != expected:
        raise Olmo3PreflightError("detached SHA-256 mismatch")
    return actual


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "write"))
    parser.add_argument("path", type=Path)
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args(argv)
    if args.action == "check":
        verify_detached_sha256(args.path, sha256_path=args.sidecar)
    else:
        written = write_detached_sha256(args.path)
        if written != args.sidecar:
            args.sidecar.write_text(written.read_text(encoding="ascii"), encoding="ascii")
            written.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
