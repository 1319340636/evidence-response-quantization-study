"""Outcome-blind structural gate for the frozen Ministral HF triplet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .hf_awq_triplet_gate import (
    HFAWQTripletGateError,
    _record_hash,
    _write_atomic_json,
    validate_hf_awq_triplet,
)
from .hf_runtime_audit import HFRuntimeAuditError, load_hf_runtime_audit
from .ministral_triplet_freeze import (
    MINISTRAL_REPO_ID,
    MINISTRAL_REVISION,
    MinistralTripletFreezeError,
    load_ministral_triplet_freeze,
)
from .ministral_source_audit import (
    MinistralSourceAuditError,
    load_ministral_source_audit,
)


MINISTRAL_HF_TRIPLET_GATE_PROTOCOL = (
    "ministral-hf-triplet-gate-v1-20260809"
)
_MODES = {"ministral_smoke16", "ministral_formal4800"}
_FORBIDDEN = {
    "accuracy",
    "balanced_accuracy",
    "effect",
    "prediction",
    "probability",
    "score",
}


class MinistralHFTripletGateError(RuntimeError):
    """Raised when any frozen Ministral structural binding drifts."""


def build_ministral_triplet_gate(
    *,
    fp16_export: str | Path,
    gptq_export: str | Path,
    awq_export: str | Path,
    fp16_audit: str | Path,
    gptq_audit: str | Path,
    awq_audit: str | Path,
    gptq_artifact_audit: str | Path,
    awq_artifact_audit: str | Path,
    freeze_config: str | Path,
    freeze_sha256: str | Path,
    source_audit: str | Path,
    mode: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind all three aligned routes without serializing any outcome."""
    if mode not in _MODES:
        raise MinistralHFTripletGateError("Ministral gate mode is invalid")
    for path in (
        fp16_export,
        gptq_export,
        awq_export,
        fp16_audit,
        gptq_audit,
        awq_audit,
        gptq_artifact_audit,
        awq_artifact_audit,
        freeze_config,
        freeze_sha256,
        source_audit,
    ):
        if any("recovery" in part.casefold() for part in Path(path).parts):
            raise MinistralHFTripletGateError("recovery paths are forbidden")
    try:
        freeze = load_ministral_triplet_freeze(
            freeze_config, sha256_path=freeze_sha256
        )
    except (MinistralTripletFreezeError, OSError, ValueError) as error:
        raise MinistralHFTripletGateError(
            f"Ministral freeze binding is invalid: {error}"
        ) from error
    freeze_hash = Path(freeze_sha256).read_text(encoding="ascii").strip()

    artifact_path = Path(gptq_artifact_audit)
    artifact = _load_gptq_artifact(artifact_path)
    try:
        gptq_certificate = load_hf_runtime_audit(gptq_audit)
    except HFRuntimeAuditError as error:
        raise MinistralHFTripletGateError(
            f"GPTQ runtime certificate is invalid: {error}"
        ) from error
    artifact_file_hash = _file_sha256(artifact_path)
    if (
        gptq_certificate.gptq_artifact_audit_sha256
        != artifact["audit_sha256"]
        or gptq_certificate.gptq_artifact_file_sha256 != artifact_file_hash
    ):
        raise MinistralHFTripletGateError(
            "GPTQ artifact certificate binding drifted"
        )
    try:
        base = validate_hf_awq_triplet(
            fp16_export=fp16_export,
            gptq_export=gptq_export,
            awq_export=awq_export,
            fp16_audit=fp16_audit,
            gptq_audit=gptq_audit,
            awq_audit=awq_audit,
            awq_artifact_audit=awq_artifact_audit,
            model_key="ministral3_8b",
            mode=mode,
        )
    except HFAWQTripletGateError as error:
        raise MinistralHFTripletGateError(str(error)) from error

    if source_audit is not None:
        try:
            source = load_ministral_source_audit(source_audit)
            fp_certificate = load_hf_runtime_audit(fp16_audit)
            awq_certificate = load_hf_runtime_audit(awq_audit)
        except (MinistralSourceAuditError, HFRuntimeAuditError) as error:
            raise MinistralHFTripletGateError(
                f"checkpoint identity chain is invalid: {error}"
            ) from error
        awq_artifact_path = Path(awq_artifact_audit)
        awq_artifact = _load_json_audit(awq_artifact_path, "AWQ")
        source_file_hash = _file_sha256(Path(source_audit))
        expected_source = {
            name: item["sha256"] for name, item in source["files"].items()
        }
        awq_targets = awq_artifact.get("quantized_target_modules")
        if not isinstance(awq_targets, list) or not awq_targets or any(
            not isinstance(name, str) or not name.startswith("model.language_model.")
            for name in awq_targets
        ):
            raise MinistralHFTripletGateError("AWQ text-backbone target binding drifted")
        gptq_source = {
            name: item["sha256"] for name, item in artifact.get("source_files", {}).items()
        }
        if (
            artifact.get("source_audit_sha256") != source["audit_sha256"]
            or artifact.get("source_audit_file_sha256") != source_file_hash
            or awq_artifact.get("source_audit_sha256") != source["audit_sha256"]
            or awq_artifact.get("source_audit_file_sha256") != source_file_hash
            or gptq_source != expected_source
            or awq_artifact.get("source_files") != expected_source
            or dict(fp_certificate.model_files) != expected_source
        ):
            raise MinistralHFTripletGateError("source checkpoint identity chain drifted")
        expected_gptq = {
            name: item["sha256"] for name, item in artifact.get("output_files", {}).items()
        }
        if dict(gptq_certificate.model_files) != expected_gptq:
            raise MinistralHFTripletGateError("GPTQ runtime checkpoint drifted")
        if dict(awq_certificate.model_files) != awq_artifact.get("output_files"):
            raise MinistralHFTripletGateError("AWQ runtime checkpoint drifted")
        if mode == "ministral_formal4800" and base.get("split_sha256") != freeze["split"]["sha256"]:
            raise MinistralHFTripletGateError("formal population does not match frozen split")

    result = {
        **base,
        "protocol_version": MINISTRAL_HF_TRIPLET_GATE_PROTOCOL,
        "ministral_triplet_freeze_sha256": freeze_hash,
        "gptq_artifact_audit_sha256": artifact["audit_sha256"],
        "gptq_artifact_file_sha256": artifact_file_hash,
        "frozen_split_sha256": freeze["split"]["sha256"],
    }
    if source_audit is not None:
        result["source_audit_sha256"] = source["audit_sha256"]
        result["source_audit_file_sha256"] = source_file_hash
    result["gate_sha256"] = _record_hash(result, excluded="gate_sha256")
    _reject_forbidden_keys(result)
    if output_path is not None:
        _write_atomic_json(result, Path(output_path))
    return result


def _load_gptq_artifact(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralHFTripletGateError(
            "GPTQ artifact audit cannot be read"
        ) from error
    if not isinstance(value, dict):
        raise MinistralHFTripletGateError("GPTQ artifact audit must be an object")
    expected_hash = _record_hash(value, excluded="audit_sha256")
    if (
        value.get("audit_sha256") != expected_hash
        or value.get("model_key") != "ministral3_8b"
        or value.get("source_repo_id") != MINISTRAL_REPO_ID
        or value.get("source_revision") != MINISTRAL_REVISION
        or not isinstance(value.get("calibration"), Mapping)
        or value["calibration"].get("count") != 256
    ):
        raise MinistralHFTripletGateError("GPTQ artifact audit drifted")
    return value


def _load_json_audit(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralHFTripletGateError(f"{label} artifact audit cannot be read") from error
    if not isinstance(value, dict) or value.get("audit_sha256") != _record_hash(
        value, excluded="audit_sha256"
    ):
        raise MinistralHFTripletGateError(f"{label} artifact audit drifted")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise MinistralHFTripletGateError(
            "GPTQ artifact audit cannot be hashed"
        ) from error
    return digest.hexdigest()


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN:
                raise MinistralHFTripletGateError(
                    "outcome key is forbidden in the structural gate"
                )
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)
