"""Fail-closed loader for the prospective Qwen-only HF triplet freeze."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


QWEN_TRIPLET_FREEZE_PROTOCOL = "qwen-hf-triplet-only-v1-20260728"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class QwenTripletFreezeError(ValueError):
    """Raised when the prospective Qwen-only contract is not exact."""


def validate_qwen_triplet_freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached Qwen-only freeze mapping."""
    if value.get("protocol") != QWEN_TRIPLET_FREEZE_PROTOCOL:
        raise QwenTripletFreezeError("Qwen-only freeze protocol is invalid")
    if value.get("active_models") != ["qwen35_9b"]:
        raise QwenTripletFreezeError("active scope must remain Qwen-only")
    if value.get("routes") != ["HF_FP16", "HF_GPTQ_INT4", "HF_AWQ_INT4"]:
        raise QwenTripletFreezeError("Qwen-only HF triplet routes are invalid")

    gate = _object(value, "engineering_gate")
    if gate.get("common_full_evidence_inputs") != 16:
        raise QwenTripletFreezeError(
            "engineering gate requires 16 common full-evidence inputs"
        )
    if gate.get("outcome_access") is not False:
        raise QwenTripletFreezeError("engineering gate forbids outcome access")
    if gate.get("triplet_gate_required") is not True:
        raise QwenTripletFreezeError("structural triplet gate is required")

    formal = _object(value, "formal_run")
    if formal.get("quartets") != 1200 or formal.get("pages") != 1078:
        raise QwenTripletFreezeError("formal population must be 1,200 quartets")
    if formal.get("full_evidence_inputs_per_route") != 4800:
        raise QwenTripletFreezeError("formal route requires 4,800 full inputs")
    if formal.get("new_awq_inputs") != 4800:
        raise QwenTripletFreezeError("formal run requires 4,800 new AWQ inputs")
    if formal.get("reuse_existing_fp16_gptq") is not True:
        raise QwenTripletFreezeError("existing FP16/GPTQ inputs must be reused")

    backend = _object(value, "backend_boundary")
    if backend.get("same_hf_backend") is not True:
        raise QwenTripletFreezeError("all routes require the same HF backend")
    if backend.get("causal_claim") != "within_backend_quantization_method_contrast":
        raise QwenTripletFreezeError("same HF backend causal boundary is invalid")
    if value.get("recovery_access") is not False:
        raise QwenTripletFreezeError("recovery access must remain disabled")
    if value.get("frozen_before_awq_outcome_access") is not True:
        raise QwenTripletFreezeError("freeze must precede AWQ outcome access")

    retired = _object(value, "retired_routes")
    gemma = retired.get("gemma4_e4b")
    expected_gemma = {
        "route": "HF_AWQ_INT4",
        "status": "not_formally_completed",
        "boundary": "not_a_formal_result",
        "resume_allowed": False,
    }
    if gemma != expected_gemma:
        raise QwenTripletFreezeError("Gemma AWQ must remain retired and incomplete")

    sources = _object(value, "source_bindings")
    for name in ("fp16_gptq_report_manifest_sha256", "dose_subset_sha256"):
        if not _SHA256.fullmatch(str(sources.get(name, ""))):
            raise QwenTripletFreezeError(f"source binding is invalid: {name}")
    return json.loads(json.dumps(value, ensure_ascii=False))


def load_qwen_triplet_freeze(
    path: str | Path, *, sha256_path: str | Path
) -> dict[str, Any]:
    """Load a strict UTF-8 freeze and verify its external SHA-256 sidecar."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QwenTripletFreezeError(f"cannot load Qwen-only freeze: {error}") from error
    if not isinstance(value, dict):
        raise QwenTripletFreezeError("Qwen-only freeze must be a JSON object")
    try:
        expected = Path(sha256_path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise QwenTripletFreezeError(f"cannot load freeze SHA-256: {error}") from error
    actual = hashlib.sha256(raw).hexdigest()
    if not _SHA256.fullmatch(expected) or expected != actual:
        raise QwenTripletFreezeError("Qwen-only freeze SHA-256 mismatch")
    return validate_qwen_triplet_freeze(value)


def _object(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise QwenTripletFreezeError(f"{key} must be an object")
    return selected
