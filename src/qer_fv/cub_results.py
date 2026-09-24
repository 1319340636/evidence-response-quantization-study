"""Validation and pairing of completed CUB condition exports."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .cub_statistics import PairedCubObservation


class CubResultsError(RuntimeError):
    pass


@dataclass(frozen=True)
class CubConditionResult:
    protocol_version: str
    schema_version: str | None
    model_key: str
    quantization: str
    audit_certificate_sha256: str
    source_manifest_sha256: str
    prompt_contract_sha256: str
    contexts: Mapping[str, Mapping[str, Any]]


def load_cub_condition_export(directory: str | Path) -> CubConditionResult:
    root = Path(directory)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CubResultsError("cannot read CUB export manifest") from error
    if not isinstance(manifest, dict):
        raise CubResultsError("CUB export manifest must be an object")
    expected_self_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if expected_self_hash != _record_hash(unhashed):
        raise CubResultsError("CUB export manifest hash mismatch")
    context_path = root / "contexts.jsonl"
    query_path = root / "queries.jsonl"
    if _file_hash(context_path) != manifest.get("contexts_jsonl_sha256"):
        raise CubResultsError("contexts JSONL hash mismatch")
    if _file_hash(query_path) != manifest.get("queries_jsonl_sha256"):
        raise CubResultsError("queries JSONL hash mismatch")
    identity = manifest.get("run_identity")
    if not isinstance(identity, dict):
        raise CubResultsError("missing CUB run identity")
    protocol_version = str(identity.get("protocol_version", ""))
    schema_version = manifest.get("schema_version")
    if protocol_version == "cub-paired-run-v4-20260715":
        if schema_version != "cub-store-v4-20260715":
            raise CubResultsError("CUB v4 schema version mismatch")
        expected_contexts = manifest.get("contexts_registered")
    elif protocol_version == "cub-paired-run-v1-20260714":
        if schema_version is not None:
            raise CubResultsError("CUB v3 export has an unexpected schema version")
        expected_contexts = manifest.get("contexts_completed")
    else:
        raise CubResultsError("unsupported CUB run protocol")
    contexts: dict[str, Mapping[str, Any]] = {}
    for line in context_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise CubResultsError("invalid context export row")
        if row["sample_id"] in contexts:
            raise CubResultsError("duplicate context sample ID")
        contexts[row["sample_id"]] = row
    if len(contexts) != expected_contexts:
        raise CubResultsError("context export count mismatch")
    return CubConditionResult(
        protocol_version=protocol_version,
        schema_version=None if schema_version is None else str(schema_version),
        model_key=str(identity["model_key"]),
        quantization=str(identity["quantization"]),
        audit_certificate_sha256=str(identity["audit_certificate_sha256"]),
        source_manifest_sha256=str(identity["manifest_sha256"]),
        prompt_contract_sha256=str(identity["prompt_contract_sha256"]),
        contexts=contexts,
    )


def pair_cub_conditions(
    f16: CubConditionResult, quantized: CubConditionResult
) -> tuple[PairedCubObservation, ...]:
    if f16.quantization != "F16" or quantized.quantization == "F16":
        raise CubResultsError("CUB pairing requires F16 then quantized")
    if f16.protocol_version != quantized.protocol_version:
        raise CubResultsError("CUB run protocol mismatch")
    if (
        f16.model_key != quantized.model_key
        or f16.source_manifest_sha256 != quantized.source_manifest_sha256
        or f16.prompt_contract_sha256 != quantized.prompt_contract_sha256
    ):
        raise CubResultsError("CUB condition identity mismatch")
    if set(f16.contexts) != set(quantized.contexts):
        raise CubResultsError("CUB sample pairing mismatch")
    paired = []
    for sample_id in sorted(f16.contexts):
        left = f16.contexts[sample_id]
        right = quantized.contexts[sample_id]
        for key in (
            "query_identity_sha256",
            "context_identity_sha256",
            "analysis_context_type",
            "target_new",
        ):
            if _identity_value(left, key) != _identity_value(right, key):
                label = "target mismatch" if key == "target_new" else "pair identity mismatch"
                raise CubResultsError(label)
        context_type = _identity_value(left, "analysis_context_type")
        left_ccu_target = _identity_value(left, "ccu_target_label")
        right_ccu_target = _identity_value(right, "ccu_target_label")
        if context_type != "irrelevant" and left_ccu_target != right_ccu_target:
            raise CubResultsError("target mismatch")
        paired.append(
            PairedCubObservation(
                sample_id=sample_id,
                query_identity_sha256=str(_identity_value(left, "query_identity_sha256")),
                analysis_context_type=str(_identity_value(left, "analysis_context_type")),
                bcu_f16=_bcu(left),
                bcu_quantized=_bcu(right),
                ccu_f16=_ccu(left),
                ccu_quantized=_ccu(right),
                hard_ok_f16=_stage_ok(left, "hard"),
                hard_ok_quantized=_stage_ok(right, "hard"),
                probability_ok_f16=_stage_ok(left, "probability"),
                probability_ok_quantized=_stage_ok(right, "probability"),
            )
        )
    return tuple(paired)


def _bcu(row: Mapping[str, Any]) -> bool:
    if "hard_status" in row:
        payload = _stage_payload(row, "hard")
        value = payload.get("bcu")
        return value if row.get("hard_status") == "ok" and type(value) is bool else False
    return row.get("bcu") if row.get("status") == "ok" and type(row.get("bcu")) is bool else False


def _ccu(row: Mapping[str, Any]) -> float | None:
    if "probability_status" in row:
        value = _stage_payload(row, "probability").get("ccu")
        status = row.get("probability_status")
    else:
        value = row.get("ccu")
        status = row.get("status")
    if status != "ok" or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) and -1.0 <= score <= 1.0 else None


def _stage_ok(row: Mapping[str, Any], stage: str) -> bool:
    if f"{stage}_status" in row:
        return row.get(f"{stage}_status") == "ok"
    return row.get("status") == "ok"


def _stage_payload(row: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    value = row.get(f"{stage}_payload")
    if not isinstance(value, Mapping):
        raise CubResultsError(f"invalid {stage} payload")
    return value


def _identity_value(row: Mapping[str, Any], key: str) -> Any:
    if key == "ccu_target_label" and "probability_payload" in row:
        probability_payload = _stage_payload(row, "probability")
        if key not in probability_payload:
            raise CubResultsError(f"missing pair identity field: {key}")
        return probability_payload[key]
    values = []
    if key in row:
        values.append(row[key])
    for stage in ("hard", "probability"):
        payload = row.get(f"{stage}_payload")
        if isinstance(payload, Mapping) and key in payload:
            values.append(payload[key])
    if not values:
        raise CubResultsError(f"missing pair identity field: {key}")
    if any(value != values[0] for value in values[1:]):
        raise CubResultsError(f"intra-row identity mismatch: {key}")
    return values[0]


def _record_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise CubResultsError(f"cannot read export file: {path.name}") from error
