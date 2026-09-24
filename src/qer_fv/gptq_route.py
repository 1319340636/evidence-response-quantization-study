"""Frozen configuration and artifact audit for the Stage B GPTQ route."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


GPTQMODEL_VERSION = "7.1.0"
GPTQ_ROUTE_PROTOCOL = "gptq-route-v1-20260717"
MINISTRAL3_8B_REPO_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
MINISTRAL3_8B_REVISION = "06cc81bfd6e45321d8fc8f816576c5b6ac67ec22"
OLMO3_7B_REPO_ID = "allenai/OLMo-3-7B-Instruct"
OLMO3_7B_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"


class GPTQRouteError(RuntimeError):
    """Raised when the frozen GPTQ route or its artifacts drift."""


@dataclass(frozen=True)
class GPTQRouteConfig:
    protocol_version: str
    model_key: str
    source_path: Path
    output_path: Path
    source_repo_id: str | None
    source_revision: str | None
    package_version: str
    algorithm: str
    bits: int
    group_size: int
    sym: bool
    desc_act: bool
    calibration_source_role: str
    calibration_count: int
    batch_size: int

    @classmethod
    def qwen35_9b(
        cls, *, source_path: str | Path, output_path: str | Path
    ) -> "GPTQRouteConfig":
        source = Path(source_path)
        if source.suffix.lower() == ".gguf" or (source.exists() and source.is_file()):
            raise GPTQRouteError(
                "GPTQ source must be original Hugging Face weights, not GGUF"
            )
        return cls(
            protocol_version=GPTQ_ROUTE_PROTOCOL,
            model_key="qwen35_9b",
            source_path=source,
            output_path=Path(output_path),
            source_repo_id=None,
            source_revision=None,
            package_version=GPTQMODEL_VERSION,
            algorithm="GPTQ",
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            calibration_source_role="vitaminc_dev",
            calibration_count=256,
            batch_size=1,
        )

    @classmethod
    def gemma4_e4b(
        cls,
        *,
        source_path: str | Path,
        output_path: str | Path,
        source_revision: str,
    ) -> "GPTQRouteConfig":
        source = Path(source_path)
        if source.suffix.lower() == ".gguf" or (
            source.exists() and source.is_file()
        ):
            raise GPTQRouteError(
                "GPTQ source must be original Hugging Face weights, not GGUF"
            )
        if (
            not isinstance(source_revision, str)
            or len(source_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in source_revision
            )
        ):
            raise GPTQRouteError(
                "Gemma source revision must be an exact lowercase commit SHA"
            )
        return cls(
            protocol_version=GPTQ_ROUTE_PROTOCOL,
            model_key="gemma4_e4b",
            source_path=source,
            output_path=Path(output_path),
            source_repo_id="google/gemma-4-E4B-it",
            source_revision=source_revision,
            package_version=GPTQMODEL_VERSION,
            algorithm="GPTQ",
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            calibration_source_role="vitaminc_dev",
            calibration_count=256,
            batch_size=1,
        )

    @classmethod
    def ministral3_8b(
        cls,
        *,
        source_path: str | Path,
        output_path: str | Path,
        source_revision: str,
    ) -> "GPTQRouteConfig":
        source = Path(source_path)
        if source.suffix.lower() == ".gguf" or (
            source.exists() and source.is_file()
        ):
            raise GPTQRouteError(
                "GPTQ source must be original Hugging Face weights, not GGUF"
            )
        if source_revision != MINISTRAL3_8B_REVISION:
            raise GPTQRouteError(
                "Ministral source revision must match the frozen commit"
            )
        return cls(
            protocol_version=GPTQ_ROUTE_PROTOCOL,
            model_key="ministral3_8b",
            source_path=source,
            output_path=Path(output_path),
            source_repo_id=MINISTRAL3_8B_REPO_ID,
            source_revision=source_revision,
            package_version=GPTQMODEL_VERSION,
            algorithm="GPTQ",
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            calibration_source_role="vitaminc_dev",
            calibration_count=256,
            batch_size=1,
        )

    @classmethod
    def olmo3_7b(
        cls,
        *,
        source_path: str | Path,
        output_path: str | Path,
        source_revision: str,
    ) -> "GPTQRouteConfig":
        source = Path(source_path)
        if source.suffix.lower() == ".gguf" or (
            source.exists() and source.is_file()
        ):
            raise GPTQRouteError(
                "GPTQ source must be original Hugging Face weights, not GGUF"
            )
        if source_revision != OLMO3_7B_REVISION:
            raise GPTQRouteError("OLMo source revision must match the frozen commit")
        return cls(
            protocol_version=GPTQ_ROUTE_PROTOCOL,
            model_key="olmo3_7b",
            source_path=source,
            output_path=Path(output_path),
            source_repo_id=OLMO3_7B_REPO_ID,
            source_revision=source_revision,
            package_version=GPTQMODEL_VERSION,
            algorithm="GPTQ",
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            calibration_source_role="vitaminc_dev",
            calibration_count=256,
            batch_size=1,
        )

    def quantize_kwargs(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "desc_act": self.desc_act,
        }

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["source_path"] = self.source_path.as_posix()
        record["output_path"] = self.output_path.as_posix()
        if self.source_repo_id is None:
            record.pop("source_repo_id")
        if self.source_revision is None:
            record.pop("source_revision")
        return record


def build_gptq_artifact_audit(
    config: GPTQRouteConfig,
    *,
    calibration_manifest: Mapping[str, Any],
    calibration_records_path: str | Path,
    package_versions: Mapping[str, str],
    gpu_inventory: Sequence[Mapping[str, Any]],
    calibration_texts_path: str | Path | None = None,
    calibration_token_ids_path: str | Path | None = None,
    source_audit_path: str | Path | None = None,
    source_audit_sha256_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind source, calibration, output, software, and hardware identities."""
    if not isinstance(config, GPTQRouteConfig):
        raise GPTQRouteError("config must be a GPTQRouteConfig")
    if package_versions.get("gptqmodel") != config.package_version:
        raise GPTQRouteError("GPTQModel package version mismatch")
    calibration = _validate_calibration(
        calibration_manifest,
        Path(calibration_records_path),
        expected_role=config.calibration_source_role,
    )
    source_files = _directory_hashes(config.source_path, "source")
    output_files = _directory_hashes(config.output_path, "output")
    record: dict[str, Any] = {
        "schema_version": "gptq-artifact-audit-v1-20260717",
        **config.to_record(),
        "calibration": calibration,
        "package_versions": dict(sorted(package_versions.items())),
        "gpu_inventory": [dict(item) for item in gpu_inventory],
        "source_files": source_files,
        "output_files": output_files,
    }
    if config.model_key in {"ministral3_8b", "olmo3_7b"}:
        if (
            calibration_texts_path is None
            or calibration_token_ids_path is None
            or source_audit_path is None
        ):
            label = "OLMo" if config.model_key == "olmo3_7b" else "Ministral"
            raise GPTQRouteError(
                f"{label} GPTQ requires source and rendered calibration audits"
            )
        if config.model_key == "ministral3_8b":
            from .ministral_source_audit import verify_ministral_source_against_audit

            source_audit = verify_ministral_source_against_audit(
                config.source_path, source_audit_path
            )
        else:
            if source_audit_sha256_path is None:
                raise GPTQRouteError("OLMo GPTQ requires source audit SHA-256")
            from .olmo3_source_audit import verify_olmo3_source_against_audit

            source_audit = verify_olmo3_source_against_audit(
                config.source_path,
                source_audit_path,
                sha256_path=source_audit_sha256_path,
            )
        expected_source = {
            name: {"bytes": item["bytes"], "sha256": item["sha256"]}
            for name, item in source_audit["files"].items()
        }
        if source_files != expected_source:
            raise GPTQRouteError(
                f"{config.model_key} GPTQ source audit inventory drifted"
            )
        text_path = Path(calibration_texts_path)
        token_path = Path(calibration_token_ids_path)
        text_rows = _load_jsonl(text_path, "rendered calibration texts")
        token_rows = _load_jsonl(token_path, "calibration token IDs")
        ids = calibration_manifest.get("ids")
        if [row.get("unique_id") for row in text_rows] != ids or [
            row.get("unique_id") for row in token_rows
        ] != ids:
            raise GPTQRouteError(
                f"{config.model_key} GPTQ calibration rendering order drifted"
            )
        if any(not isinstance(row.get("text"), str) or not row["text"] for row in text_rows):
            raise GPTQRouteError(
                f"{config.model_key} GPTQ rendered calibration text is invalid"
            )
        if any(
            not isinstance(row.get("input_ids"), list)
            or not row["input_ids"]
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in row["input_ids"])
            for row in token_rows
        ):
            raise GPTQRouteError(
                f"{config.model_key} GPTQ calibration token IDs are invalid"
            )
        record["source_audit_sha256"] = source_audit["audit_sha256"]
        record["source_audit_file_sha256"] = _file_sha256(Path(source_audit_path))
        record["calibration"].update(
            texts_sha256=_file_sha256(text_path),
            token_ids_sha256=_file_sha256(token_path),
            maximum_token_length=max(len(row["input_ids"]) for row in token_rows),
        )
    record["audit_sha256"] = _record_hash(record)
    return record


def _validate_calibration(
    manifest: Mapping[str, Any],
    records_path: Path,
    *,
    expected_role: str,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise GPTQRouteError("calibration manifest must be an object")
    record = dict(manifest)
    if record.get("source_role") != expected_role:
        raise GPTQRouteError("GPTQ calibration source_role must be vitaminc_dev")
    if not records_path.is_file():
        raise GPTQRouteError("GPTQ calibration records are missing")
    if record.get("records_sha256") != _file_sha256(records_path):
        raise GPTQRouteError("GPTQ calibration records SHA-256 mismatch")
    ids = record.get("ids")
    if (
        not isinstance(ids, list)
        or any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
    ):
        raise GPTQRouteError("GPTQ calibration IDs are invalid")
    if record.get("count") != len(ids):
        raise GPTQRouteError("GPTQ calibration count mismatch")
    if record.get("ids_sha256") != _ids_sha256(ids):
        raise GPTQRouteError("GPTQ calibration ID SHA-256 mismatch")
    rows = _load_jsonl(records_path, "calibration records")
    if [row.get("unique_id") for row in rows] != ids:
        raise GPTQRouteError("GPTQ calibration ID order drifted")
    return record


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GPTQRouteError(f"cannot read GPTQ {label}") from error
    if not values or any(not isinstance(value, dict) for value in values):
        raise GPTQRouteError(f"GPTQ {label} must contain JSON objects")
    return values


def _directory_hashes(path: Path, label: str) -> dict[str, dict[str, Any]]:
    if not path.is_dir():
        raise GPTQRouteError(f"GPTQ {label} directory is missing")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise GPTQRouteError(f"GPTQ {label} directory is empty")
    return {
        item.relative_to(path).as_posix(): {
            "bytes": item.stat().st_size,
            "sha256": _file_sha256(item),
        }
        for item in files
    }


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{item}\n" for item in ids).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
