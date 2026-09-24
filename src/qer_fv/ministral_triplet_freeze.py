"""Fail-closed loader for the prospective Ministral HF triplet extension."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


MINISTRAL_TRIPLET_FREEZE_PROTOCOL = "ministral-hf-triplet-v1-20260809"
MINISTRAL_REPO_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
MINISTRAL_REVISION = "06cc81bfd6e45321d8fc8f816576c5b6ac67ec22"
DOSE_SUBSET_SHA256 = (
    "bbe3e66dc7e53e0fa08c656716e113f63b8c5232de8b9fa895de6046b861b9f1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MinistralTripletFreezeError(ValueError):
    """Raised when the prospective Ministral extension contract drifts."""


def validate_ministral_triplet_freeze(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach the frozen Ministral triplet mapping."""
    if value.get("protocol") != MINISTRAL_TRIPLET_FREEZE_PROTOCOL:
        raise MinistralTripletFreezeError("Ministral freeze protocol is invalid")
    if value.get("scientific_role") != (
        "prospectively_frozen_post_confirmatory_extension"
    ):
        raise MinistralTripletFreezeError("scientific role is invalid")

    model = _object(value, "model")
    expected_model = {
        "model_key": "ministral3_8b",
        "repo_id": MINISTRAL_REPO_ID,
        "revision": MINISTRAL_REVISION,
    }
    if model != expected_model:
        if model.get("revision") != MINISTRAL_REVISION:
            raise MinistralTripletFreezeError("Ministral source revision drifted")
        raise MinistralTripletFreezeError("Ministral model identity drifted")
    if value.get("routes") != ["HF_FP16", "HF_GPTQ_INT4", "HF_AWQ_INT4"]:
        raise MinistralTripletFreezeError("Ministral triplet routes are invalid")
    qwen_reference = _object(value, "qwen_reference")
    if qwen_reference != {
        "results_path": "reports/qwen_hf_triplet_formal_v1/results.json",
        "results_file_sha256": "d433af237dc2208d60e21bce283232f3c0f53acac3d02a6eeb100bfc14ff544e",
        "report_manifest_path": "reports/qwen_hf_triplet_formal_v1/report_manifest.json",
        "report_manifest_file_sha256": "0a83f6b0e404ec316cf82c6f6bbb0022b99ffeb311ea29dcd53668e78be744e2",
        "awq_minus_fp16_estimate": -1.801813978432282,
        "awq_minus_gptq_estimate": -1.384059021335807,
    }:
        raise MinistralTripletFreezeError("immutable Qwen reference drifted")

    gate = _object(value, "engineering_gate")
    if gate.get("common_full_evidence_inputs") != 16:
        raise MinistralTripletFreezeError(
            "engineering gate requires 16 common full-evidence inputs"
        )
    if gate.get("outcome_access") is not False:
        raise MinistralTripletFreezeError("engineering gate forbids outcome access")
    if gate.get("triplet_gate_required") is not True:
        raise MinistralTripletFreezeError("structural triplet gate is required")

    formal = _object(value, "formal_run")
    if formal.get("quartets") != 1200 or formal.get("pages") != 1078:
        raise MinistralTripletFreezeError("formal population must be 1,200 quartets")
    if formal.get("full_evidence_inputs_per_route") != 4800:
        raise MinistralTripletFreezeError("formal route requires 4,800 full inputs")

    split = _object(value, "split")
    if split != {"name": "dose_subset", "sha256": DOSE_SUBSET_SHA256}:
        raise MinistralTripletFreezeError("dose-subset source binding drifted")

    calibration = _object(value, "calibration")
    if calibration != {
        "source_role": "vitaminc_dev",
        "count": 256,
        "shuffle": False,
        "allow_truncation": False,
    }:
        raise MinistralTripletFreezeError(
            "calibration must remain the frozen 256-row VitaminC set"
        )

    gptq = _object(value, "gptq_recipe")
    if gptq != {
        "algorithm": "GPTQ",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "package": "gptqmodel",
        "package_version": "7.1.0",
    }:
        raise MinistralTripletFreezeError("GPTQ recipe drifted")
    awq = _object(value, "awq_recipe")
    if awq != {
        "algorithm": "AWQ",
        "scheme": "W4A16_ASYM",
        "bits": 4,
        "activation_bits": 16,
        "group_size": 128,
        "zero_point": True,
        "llmcompressor": "0.12.0",
        "compressed_tensors": "0.17.1",
    }:
        raise MinistralTripletFreezeError("AWQ recipe drifted")

    statistics = _object(value, "statistics")
    if (
        statistics.get("cluster") != "page"
        or statistics.get("bootstrap_draws") != 10_000
        or statistics.get("seed") != 20260711
        or statistics.get("confidence_level") != 0.95
        or statistics.get("alternative") != "two_sided"
        or statistics.get("primary_holm_family")
        != ["awq_minus_fp16", "awq_minus_gptq"]
        or statistics.get("secondary_contrast") != "gptq_minus_fp16"
    ):
        raise MinistralTripletFreezeError("formal bootstrap policy drifted")

    if value.get("failure_policy") != (
        "engineering_failure_is_not_a_scientific_result"
    ):
        raise MinistralTripletFreezeError("engineering failure policy drifted")
    if value.get("fallback_model_activation") is not False:
        raise MinistralTripletFreezeError("fallback model activation is forbidden")
    if value.get("recovery_access") is not False:
        raise MinistralTripletFreezeError("recovery access must remain disabled")
    if value.get("frozen_before_ministral_outcome_access") is not True:
        raise MinistralTripletFreezeError("freeze must precede outcome access")
    return json.loads(json.dumps(value, ensure_ascii=False))


def load_ministral_triplet_freeze(
    path: str | Path, *, sha256_path: str | Path
) -> dict[str, Any]:
    """Load the UTF-8 freeze and verify its detached SHA-256 sidecar."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MinistralTripletFreezeError(
            f"cannot load Ministral freeze: {error}"
        ) from error
    if not isinstance(value, dict):
        raise MinistralTripletFreezeError("Ministral freeze must be a JSON object")
    try:
        expected = Path(sha256_path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise MinistralTripletFreezeError(
            f"cannot load freeze SHA-256: {error}"
        ) from error
    actual = hashlib.sha256(raw).hexdigest()
    if not _SHA256.fullmatch(expected) or expected != actual:
        raise MinistralTripletFreezeError("Ministral freeze SHA-256 mismatch")
    return validate_ministral_triplet_freeze(value)


def _object(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise MinistralTripletFreezeError(f"{key} must be an object")
    return selected
