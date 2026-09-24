"""Fail-closed audit certificate for paired Hugging Face inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .awq_route import AWQRouteError, load_awq_artifact_audit
from .hf_model_identity import HFModelIdentityError, hf_model_identity
from .hf_runtime import (
    HF_DIRECT_LOGIT_METHOD,
    HFTokenAudit,
    frozen_hf_load_policy,
)


HF_RUNTIME_AUDIT_PROTOCOL = "hf-runtime-audit-v2-20260718"
HF_RUNTIME_PRECISIONS = frozenset({"FP16", "GPTQ_INT4", "AWQ_INT4"})
HF_RUNTIME_PACKAGE_VERSIONS = MappingProxyType(
    {
        "gptqmodel": "7.1.0",
        "torch": "2.13.0",
        "torchvision": "0.28.0",
        "transformers": "5.14.1",
    }
)
HF_RUNTIME_AWQ_PACKAGE_VERSIONS = MappingProxyType(
    {
        **dict(HF_RUNTIME_PACKAGE_VERSIONS),
        "compressed-tensors": "0.17.1",
    }
)


class HFRuntimeAuditError(RuntimeError):
    """Raised when an HF runtime cannot satisfy the frozen paired protocol."""


@dataclass(frozen=True)
class HFRuntimeAuditCertificate:
    protocol_version: str
    model_key: str
    precision: str
    model_path: str
    model_files: Mapping[str, str]
    package_versions: Mapping[str, str]
    load_policy: Mapping[str, Any]
    prompt_contract_sha256: str
    choice_token_ids: Mapping[str, int]
    generation_input_ids_sha256: str
    generation_prompt_sha256: str
    chat_template_sha256: str
    direct_logit_method: str
    gptq_artifact_audit_sha256: str | None
    gptq_artifact_file_sha256: str | None
    certificate_sha256: str
    awq_artifact_audit_sha256: str | None = None
    awq_artifact_file_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_files", MappingProxyType(dict(self.model_files))
        )
        object.__setattr__(
            self, "package_versions", MappingProxyType(dict(self.package_versions))
        )
        object.__setattr__(
            self, "load_policy", MappingProxyType(dict(self.load_policy))
        )
        object.__setattr__(
            self, "choice_token_ids", MappingProxyType(dict(self.choice_token_ids))
        )

    def to_record(self) -> dict[str, Any]:
        record = {
            "protocol_version": self.protocol_version,
            "model_key": self.model_key,
            "precision": self.precision,
            "model_path": self.model_path,
            "model_files": dict(self.model_files),
            "package_versions": dict(self.package_versions),
            "load_policy": dict(self.load_policy),
            "prompt_contract_sha256": self.prompt_contract_sha256,
            "choice_token_ids": dict(self.choice_token_ids),
            "generation_input_ids_sha256": self.generation_input_ids_sha256,
            "generation_prompt_sha256": self.generation_prompt_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "direct_logit_method": self.direct_logit_method,
            "gptq_artifact_audit_sha256": self.gptq_artifact_audit_sha256,
            "gptq_artifact_file_sha256": self.gptq_artifact_file_sha256,
            "certificate_sha256": self.certificate_sha256,
        }
        if self.awq_artifact_audit_sha256 is not None:
            record["awq_artifact_audit_sha256"] = (
                self.awq_artifact_audit_sha256
            )
            record["awq_artifact_file_sha256"] = self.awq_artifact_file_sha256
        return record

    def rehash(self) -> str:
        return _record_hash(self.to_record())


def build_hf_runtime_audit(
    *,
    model_path: str | Path,
    model_key: str,
    precision: str,
    package_versions: Mapping[str, str],
    load_policy: Mapping[str, Any],
    prompt_contract_path: str | Path,
    token_audit: HFTokenAudit,
    gptq_artifact_audit_path: str | Path | None = None,
    awq_artifact_audit_path: str | Path | None = None,
) -> HFRuntimeAuditCertificate:
    """Bind one paired condition to its checkpoint, software, and prompt."""
    try:
        identity = hf_model_identity(model_key)
    except HFModelIdentityError as error:
        raise HFRuntimeAuditError(str(error)) from error
    if precision not in HF_RUNTIME_PRECISIONS:
        raise HFRuntimeAuditError("HF paired audit precision is not frozen")
    normalized_versions = {
        str(name): str(version) for name, version in package_versions.items()
    }
    expected_versions = (
        dict(HF_RUNTIME_AWQ_PACKAGE_VERSIONS)
        if precision == "AWQ_INT4"
        else dict(HF_RUNTIME_PACKAGE_VERSIONS)
    )
    if normalized_versions != expected_versions:
        raise HFRuntimeAuditError("HF paired audit package versions drifted")
    normalized_policy = dict(load_policy)
    if normalized_policy != frozen_hf_load_policy(
        precision, model_key=model_key
    ):
        raise HFRuntimeAuditError("HF paired audit load policy drifted")

    root = Path(model_path).resolve()
    if not root.is_dir():
        raise HFRuntimeAuditError(f"model directory does not exist: {root}")
    config = _load_json_object(root / "config.json", "model config")
    if config.get("model_type") != identity.model_type:
        raise HFRuntimeAuditError(
            f"model config must declare model_type {identity.model_type}"
        )
    model_files = _hash_model_files(root)

    prompt_path = Path(prompt_contract_path)
    try:
        prompt_contract_sha256 = _file_sha256(prompt_path)
    except OSError as error:
        raise HFRuntimeAuditError("cannot hash prompt contract") from error

    choice_token_ids = _validate_choice_token_ids(token_audit.choice_token_ids)
    if token_audit.direct_logit_method != HF_DIRECT_LOGIT_METHOD:
        raise HFRuntimeAuditError("direct-logit scoring method drifted")
    for field, value in (
        ("generation input IDs", token_audit.generation_input_ids_sha256),
        ("generation prompt", token_audit.generation_prompt_sha256),
        ("chat template", token_audit.chat_template_sha256),
    ):
        _validate_sha256(value, field)

    gptq_audit_sha: str | None = None
    gptq_file_sha: str | None = None
    awq_audit_sha: str | None = None
    awq_file_sha: str | None = None
    if precision == "GPTQ_INT4":
        if awq_artifact_audit_path is not None:
            raise HFRuntimeAuditError(
                "GPTQ_INT4 must not bind an AWQ artifact audit"
            )
        if gptq_artifact_audit_path is None:
            raise HFRuntimeAuditError(
                "GPTQ_INT4 requires the quantization artifact audit"
            )
        artifact_path = Path(gptq_artifact_audit_path)
        artifact = _load_json_object(artifact_path, "GPTQ artifact audit")
        if (
            artifact.get("model_key") != identity.model_key
            or artifact.get("algorithm") != "GPTQ"
            or artifact.get("bits") != 4
            or artifact.get("group_size") != 128
        ):
            raise HFRuntimeAuditError("GPTQ artifact audit protocol drifted")
        gptq_audit_sha = artifact.get("audit_sha256")
        _validate_sha256(gptq_audit_sha, "GPTQ artifact audit")
        try:
            gptq_file_sha = _file_sha256(artifact_path)
        except OSError as error:
            raise HFRuntimeAuditError("cannot hash GPTQ artifact audit") from error
    elif precision == "AWQ_INT4":
        if gptq_artifact_audit_path is not None:
            raise HFRuntimeAuditError(
                "AWQ_INT4 must not bind a GPTQ artifact audit"
            )
        if awq_artifact_audit_path is None:
            raise HFRuntimeAuditError(
                "AWQ_INT4 requires the quantization artifact audit"
            )
        artifact_path = Path(awq_artifact_audit_path)
        try:
            artifact = load_awq_artifact_audit(artifact_path)
        except AWQRouteError as error:
            raise HFRuntimeAuditError("AWQ artifact audit is invalid") from error
        if artifact.get("model_key") != identity.model_key:
            raise HFRuntimeAuditError("AWQ artifact audit identity drifted")
        awq_audit_sha = artifact.get("audit_sha256")
        _validate_sha256(awq_audit_sha, "AWQ artifact audit")
        try:
            awq_file_sha = _file_sha256(artifact_path)
        except OSError as error:
            raise HFRuntimeAuditError("cannot hash AWQ artifact audit") from error
    else:
        if gptq_artifact_audit_path is not None:
            raise HFRuntimeAuditError(
                "FP16 must not bind a GPTQ artifact audit"
            )
        if awq_artifact_audit_path is not None:
            raise HFRuntimeAuditError(
                "FP16 must not bind an AWQ artifact audit"
            )

    record = {
        "protocol_version": HF_RUNTIME_AUDIT_PROTOCOL,
        "model_key": model_key,
        "precision": precision,
        "model_path": str(root),
        "model_files": model_files,
        "package_versions": normalized_versions,
        "load_policy": normalized_policy,
        "prompt_contract_sha256": prompt_contract_sha256,
        "choice_token_ids": choice_token_ids,
        "generation_input_ids_sha256": token_audit.generation_input_ids_sha256,
        "generation_prompt_sha256": token_audit.generation_prompt_sha256,
        "chat_template_sha256": token_audit.chat_template_sha256,
        "direct_logit_method": token_audit.direct_logit_method,
        "gptq_artifact_audit_sha256": gptq_audit_sha,
        "gptq_artifact_file_sha256": gptq_file_sha,
    }
    if precision == "AWQ_INT4":
        record["awq_artifact_audit_sha256"] = awq_audit_sha
        record["awq_artifact_file_sha256"] = awq_file_sha
    return HFRuntimeAuditCertificate(
        **record, certificate_sha256=_record_hash(record)
    )


def write_hf_runtime_audit(
    certificate: HFRuntimeAuditCertificate, path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            certificate.to_record(),
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


def load_hf_runtime_audit(
    path: str | Path,
) -> HFRuntimeAuditCertificate:
    value = _load_json_object(Path(path), "HF runtime certificate")
    try:
        certificate = HFRuntimeAuditCertificate(**value)
    except (TypeError, ValueError) as error:
        raise HFRuntimeAuditError("invalid HF runtime certificate schema") from error
    if certificate.protocol_version != HF_RUNTIME_AUDIT_PROTOCOL:
        raise HFRuntimeAuditError("HF runtime certificate protocol is not current")
    try:
        hf_model_identity(certificate.model_key)
    except HFModelIdentityError as error:
        raise HFRuntimeAuditError(str(error)) from error
    if certificate.precision not in HF_RUNTIME_PRECISIONS:
        raise HFRuntimeAuditError("HF runtime certificate precision is not frozen")
    expected_versions = (
        dict(HF_RUNTIME_AWQ_PACKAGE_VERSIONS)
        if certificate.precision == "AWQ_INT4"
        else dict(HF_RUNTIME_PACKAGE_VERSIONS)
    )
    if dict(certificate.package_versions) != expected_versions:
        raise HFRuntimeAuditError("HF runtime certificate package versions drifted")
    if dict(certificate.load_policy) != frozen_hf_load_policy(
        certificate.precision, model_key=certificate.model_key
    ):
        raise HFRuntimeAuditError("HF runtime certificate load policy drifted")
    if certificate.rehash() != certificate.certificate_sha256:
        raise HFRuntimeAuditError("HF runtime certificate hash mismatch")
    return certificate


def require_matching_hf_runtime_audit(
    certificate: HFRuntimeAuditCertificate,
    passed_smoke_path: str | Path,
) -> None:
    """Require stable runtime identity to equal the passed smoke certificate."""
    expected = load_hf_runtime_audit(passed_smoke_path)
    probe_fields = {
        "certificate_sha256",
        "generation_input_ids_sha256",
        "generation_prompt_sha256",
    }
    current_record = {
        key: value
        for key, value in certificate.to_record().items()
        if key not in probe_fields
    }
    expected_record = {
        key: value
        for key, value in expected.to_record().items()
        if key not in probe_fields
    }
    if current_record != expected_record:
        raise HFRuntimeAuditError(
            "formal runtime certificate does not match the passed smoke"
        )


def _hash_model_files(root: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise HFRuntimeAuditError("model directory contains no files")
    hashes: dict[str, str] = {}
    try:
        for path in files:
            relative = path.relative_to(root).as_posix()
            hashes[relative] = _file_sha256(path)
    except OSError as error:
        raise HFRuntimeAuditError("cannot hash model files") from error
    return hashes


def _validate_choice_token_ids(value: Mapping[str, int]) -> dict[str, int]:
    if set(value) != {"A", "B", "C"}:
        raise HFRuntimeAuditError("choice token IDs must contain exactly A/B/C")
    normalized = dict(value)
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in normalized.values()
    ) or len(set(normalized.values())) != 3:
        raise HFRuntimeAuditError("A/B/C choice token IDs are invalid")
    return normalized


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFRuntimeAuditError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise HFRuntimeAuditError(f"{label} must be a JSON object")
    return value


def _validate_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HFRuntimeAuditError(f"{label} SHA-256 is invalid")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    clean = {
        key: value
        for key, value in record.items()
        if key != "certificate_sha256"
    }
    payload = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
