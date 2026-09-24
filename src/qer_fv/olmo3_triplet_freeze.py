"""Fail-closed loader for the prospective OLMo 3 HF triplet extension."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


OLMO3_TRIPLET_FREEZE_PROTOCOL = "olmo3-hf-triplet-v1-20260901"
OLMO3_REPO_ID = "allenai/OLMo-3-7B-Instruct"
OLMO3_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
DOSE_SUBSET_SHA256 = (
    "bbe3e66dc7e53e0fa08c656716e113f63b8c5232de8b9fa895de6046b861b9f1"
)
CALIBRATION_RECORDS_SHA256 = (
    "f58dae1576c76c937026a565b3a899fec75658587e9d84ed39f652c2228503ed"
)
CALIBRATION_MANIFEST_SHA256 = (
    "7b2cc940e0e8bcba84ca6a329f3662d323efaf0c6c7eb2463efbd0065c8f6851"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Olmo3TripletFreezeError(ValueError):
    """Raised when the prospective OLMo extension contract drifts."""


def validate_olmo3_triplet_freeze(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach the frozen OLMo triplet mapping."""
    if value.get("protocol") != OLMO3_TRIPLET_FREEZE_PROTOCOL:
        raise Olmo3TripletFreezeError("OLMo freeze protocol is invalid")
    if value.get("freeze_date") != "2026-09-01":
        raise Olmo3TripletFreezeError("OLMo freeze date drifted")
    if value.get("scientific_role") != (
        "prospectively_frozen_post_confirmatory_extension"
    ):
        raise Olmo3TripletFreezeError("scientific role is invalid")

    model = _object(value, "model")
    expected_model = {
        "model_key": "olmo3_7b",
        "repo_id": OLMO3_REPO_ID,
        "revision": OLMO3_REVISION,
        "model_type": "olmo3",
        "architecture": "Olmo3ForCausalLM",
    }
    if model != expected_model:
        if model.get("revision") != OLMO3_REVISION:
            raise Olmo3TripletFreezeError("OLMo source revision drifted")
        raise Olmo3TripletFreezeError("OLMo model identity drifted")
    if value.get("routes") != ["HF_FP16", "HF_GPTQ_INT4", "HF_AWQ_INT4"]:
        raise Olmo3TripletFreezeError("OLMo triplet routes are invalid")

    gate = _object(value, "engineering_gate")
    if gate.get("common_full_evidence_inputs") != 16:
        raise Olmo3TripletFreezeError(
            "engineering gate requires 16 common full-evidence inputs"
        )
    if gate.get("outcome_access") is not False:
        raise Olmo3TripletFreezeError("engineering gate forbids outcome access")
    if gate.get("triplet_gate_required") is not True:
        raise Olmo3TripletFreezeError("structural triplet gate is required")
    if gate.get("disposable_calibration_records_per_route") != 1:
        raise Olmo3TripletFreezeError("compatibility gate requires one record")
    if gate.get("bounded_compatibility_repairs_per_route") != 1:
        raise Olmo3TripletFreezeError("compatibility repair count drifted")

    formal = _object(value, "formal_run")
    if formal.get("quartets") != 1200 or formal.get("pages") != 1078:
        raise Olmo3TripletFreezeError("formal population must be 1,200 quartets")
    if formal.get("full_evidence_inputs_per_route") != 4800:
        raise Olmo3TripletFreezeError("formal route requires 4,800 full inputs")
    if formal.get("serial_routes") is not True:
        raise Olmo3TripletFreezeError("formal routes must remain serial")

    if _object(value, "split") != {
        "name": "dose_subset",
        "sha256": DOSE_SUBSET_SHA256,
    }:
        raise Olmo3TripletFreezeError("dose-subset source binding drifted")
    if _object(value, "calibration") != {
        "source_role": "vitaminc_dev",
        "count": 256,
        "records_sha256": CALIBRATION_RECORDS_SHA256,
        "manifest_sha256": CALIBRATION_MANIFEST_SHA256,
        "shuffle": False,
        "allow_truncation": False,
    }:
        raise Olmo3TripletFreezeError(
            "calibration must remain the frozen 256-row VitaminC set"
        )

    if _object(value, "gptq_recipe") != {
        "algorithm": "GPTQ",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "static_groups": False,
        "mse": False,
        "package": "gptqmodel",
        "package_version": "7.1.0",
    }:
        raise Olmo3TripletFreezeError("GPTQ recipe drifted")
    if _object(value, "awq_recipe") != {
        "algorithm": "AWQ",
        "scheme": "W4A16_ASYM",
        "bits": 4,
        "activation_bits": 16,
        "group_size": 128,
        "zero_point": True,
        "targets": ["Linear"],
        "ignore": ["lm_head"],
    }:
        raise Olmo3TripletFreezeError("AWQ recipe drifted")

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
        or statistics.get("cross_model_endpoint_policy")
        != "scale_invariant_only"
        or statistics.get("cross_model_multiplicity_control") is not True
    ):
        raise Olmo3TripletFreezeError("formal bootstrap policy drifted")

    if value.get("causal_boundary") != (
        "model_route_associated_not_pure_quantizer_kernel_packing_or_"
        "architecture_effect"
    ):
        raise Olmo3TripletFreezeError("causal boundary drifted")
    if value.get("failure_policy") != (
        "engineering_failure_is_not_a_scientific_result"
    ):
        raise Olmo3TripletFreezeError("engineering failure policy drifted")
    if value.get("fallback_model") != "granite":
        raise Olmo3TripletFreezeError("fallback model drifted")
    if value.get("fallback_model_activation") is not False:
        raise Olmo3TripletFreezeError("fallback activation must remain false")
    if value.get("recovery_access") is not False:
        raise Olmo3TripletFreezeError("recovery access must remain disabled")
    if value.get("frozen_before_olmo_hf_outcome_access") is not True:
        raise Olmo3TripletFreezeError("freeze must precede OLMo outcome access")
    return json.loads(json.dumps(value, ensure_ascii=False))


def load_olmo3_triplet_freeze(
    path: str | Path, *, sha256_path: str | Path
) -> dict[str, Any]:
    """Load the UTF-8 freeze and verify its detached SHA-256 sidecar."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Olmo3TripletFreezeError(
            f"cannot load OLMo freeze: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Olmo3TripletFreezeError("OLMo freeze must be a JSON object")
    try:
        expected = Path(sha256_path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise Olmo3TripletFreezeError(
            f"cannot load freeze SHA-256: {error}"
        ) from error
    actual = hashlib.sha256(raw).hexdigest()
    if not _SHA256.fullmatch(expected) or expected != actual:
        raise Olmo3TripletFreezeError("OLMo freeze SHA-256 mismatch")
    return validate_olmo3_triplet_freeze(value)


def _object(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise Olmo3TripletFreezeError(f"{key} must be an object")
    return selected
