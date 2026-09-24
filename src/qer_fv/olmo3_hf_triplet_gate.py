"""Outcome-blind structural gate for the frozen OLMo 3 HF triplet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .awq_route import AWQRouteError, load_awq_artifact_audit
from .hf_awq_triplet_gate import (
    HFAWQTripletGateError,
    _record_hash,
    _write_atomic_json,
    validate_hf_awq_triplet,
)
from .hf_runtime_audit import HFRuntimeAuditError, load_hf_runtime_audit
from .olmo3_source_audit import (
    OLMO3_REPO_ID,
    OLMO3_REVISION,
    Olmo3SourceAuditError,
    load_olmo3_source_audit,
)
from .olmo3_triplet_freeze import (
    Olmo3TripletFreezeError,
    load_olmo3_triplet_freeze,
)


OLMO3_HF_TRIPLET_GATE_PROTOCOL = "olmo3-hf-triplet-gate-v1-20260901"
_MODES = {"olmo3_smoke16", "olmo3_formal4800"}


class Olmo3HFTripletGateError(RuntimeError):
    """Raised when any OLMo triplet identity or structure drifts."""


def build_olmo3_triplet_gate(
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
    source_audit_sha256: str | Path,
    mode: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in _MODES:
        raise Olmo3HFTripletGateError("OLMo gate mode is invalid")
    all_paths = (
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
        source_audit_sha256,
    )
    if any(
        "recovery" in part.casefold()
        for value in all_paths
        for part in Path(value).parts
    ):
        raise Olmo3HFTripletGateError("recovery paths are forbidden")
    try:
        freeze = load_olmo3_triplet_freeze(
            freeze_config, sha256_path=freeze_sha256
        )
        source = load_olmo3_source_audit(
            source_audit, sha256_path=source_audit_sha256
        )
        fp_certificate = load_hf_runtime_audit(fp16_audit)
        gptq_certificate = load_hf_runtime_audit(gptq_audit)
        awq_certificate = load_hf_runtime_audit(awq_audit)
        awq_artifact = load_awq_artifact_audit(awq_artifact_audit)
    except (
        Olmo3TripletFreezeError,
        Olmo3SourceAuditError,
        HFRuntimeAuditError,
        AWQRouteError,
        OSError,
        ValueError,
    ) as error:
        raise Olmo3HFTripletGateError(f"OLMo identity chain is invalid: {error}") from error

    gptq_path = Path(gptq_artifact_audit)
    gptq_artifact = _load_gptq_artifact(gptq_path)
    try:
        base = validate_hf_awq_triplet(
            fp16_export=fp16_export,
            gptq_export=gptq_export,
            awq_export=awq_export,
            fp16_audit=fp16_audit,
            gptq_audit=gptq_audit,
            awq_audit=awq_audit,
            awq_artifact_audit=awq_artifact_audit,
            model_key="olmo3_7b",
            mode=mode,
        )
    except HFAWQTripletGateError as error:
        raise Olmo3HFTripletGateError(str(error)) from error

    source_file_hash = _file_sha256(Path(source_audit))
    expected_source = {
        name: item["sha256"] for name, item in source["files"].items()
    }
    gptq_source = {
        name: item["sha256"]
        for name, item in gptq_artifact.get("source_files", {}).items()
    }
    gptq_output = {
        name: item["sha256"]
        for name, item in gptq_artifact.get("output_files", {}).items()
    }
    if (
        gptq_artifact.get("source_audit_sha256") != source["audit_sha256"]
        or gptq_artifact.get("source_audit_file_sha256") != source_file_hash
        or awq_artifact.get("source_audit_sha256") != source["audit_sha256"]
        or awq_artifact.get("source_audit_file_sha256") != source_file_hash
        or gptq_source != expected_source
        or awq_artifact.get("source_files") != expected_source
        or dict(fp_certificate.model_files) != expected_source
    ):
        raise Olmo3HFTripletGateError("OLMo source checkpoint chain drifted")
    if dict(gptq_certificate.model_files) != gptq_output:
        raise Olmo3HFTripletGateError("OLMo GPTQ runtime checkpoint drifted")
    if dict(awq_certificate.model_files) != awq_artifact.get("output_files"):
        raise Olmo3HFTripletGateError("OLMo AWQ runtime checkpoint drifted")
    gptq_file_hash = _file_sha256(gptq_path)
    if (
        gptq_certificate.gptq_artifact_audit_sha256
        != gptq_artifact["audit_sha256"]
        or gptq_certificate.gptq_artifact_file_sha256 != gptq_file_hash
    ):
        raise Olmo3HFTripletGateError("OLMo GPTQ artifact binding drifted")
    if mode == "olmo3_formal4800" and base.get("split_sha256") != freeze["split"]["sha256"]:
        raise Olmo3HFTripletGateError("OLMo formal population drifted")

    result = {
        **base,
        "protocol_version": OLMO3_HF_TRIPLET_GATE_PROTOCOL,
        "olmo3_triplet_freeze_sha256": Path(freeze_sha256)
        .read_text(encoding="ascii")
        .strip(),
        "source_audit_sha256": source["audit_sha256"],
        "source_audit_file_sha256": source_file_hash,
        "gptq_artifact_audit_sha256": gptq_artifact["audit_sha256"],
        "gptq_artifact_file_sha256": gptq_file_hash,
        "frozen_split_sha256": freeze["split"]["sha256"],
    }
    result["gate_sha256"] = _record_hash(result, excluded="gate_sha256")
    if output_path is not None:
        _write_atomic_json(result, Path(output_path))
    return result


def _load_gptq_artifact(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Olmo3HFTripletGateError("OLMo GPTQ artifact cannot be read") from error
    if not isinstance(value, dict):
        raise Olmo3HFTripletGateError("OLMo GPTQ artifact must be an object")
    if (
        value.get("audit_sha256") != _record_hash(value, excluded="audit_sha256")
        or value.get("model_key") != "olmo3_7b"
        or value.get("source_repo_id") != OLMO3_REPO_ID
        or value.get("source_revision") != OLMO3_REVISION
        or not isinstance(value.get("calibration"), Mapping)
        or value["calibration"].get("count") != 256
    ):
        raise Olmo3HFTripletGateError("OLMo GPTQ artifact drifted")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
