"""Deterministic blind technical smoke gate for the CUB runner."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .cub_druid import CubRecord
from .cub_queries import build_cub_query_groups


SMOKE_SELECTION_VERSION = "cub-blind-smoke-v1-20260714"
SMOKE_COUNTS = {"gold": 5, "conflicting": 5, "irrelevant": 2}
_TECHNICAL_FIELDS = {
    "model_key",
    "quantization",
    "queries_completed",
    "contexts_completed",
    "query_failures",
    "context_failures",
    "cache_hits",
    "probe_logprob_max_range",
    "elapsed_seconds",
    "error_codes",
}
_OUTCOME_FIELDS = {
    "accuracy",
    "bcu",
    "ccu",
    "delta_bcu",
    "delta_ccu",
    "quantization_contrast",
}


def select_cub_smoke_records(
    records: Sequence[CubRecord],
) -> tuple[CubRecord, ...]:
    groups = build_cub_query_groups(
        records, prompt_contract_version="prompt-v2-20260714"
    )
    query_by_sample = {
        record.sample_id: group.query_identity_sha256
        for group in groups
        for record in group.records
    }
    by_type: dict[str, list[CubRecord]] = defaultdict(list)
    for record in records:
        by_type[record.analysis_context_type].append(record)
    used_queries: set[str] = set()
    selected: list[CubRecord] = []
    for context_type in ("gold", "conflicting", "irrelevant"):
        ranked = sorted(by_type[context_type], key=_smoke_rank)
        accepted = []
        for record in ranked:
            query_identity = query_by_sample[record.sample_id]
            if query_identity in used_queries:
                continue
            accepted.append(record)
            used_queries.add(query_identity)
            if len(accepted) == SMOKE_COUNTS[context_type]:
                break
        if len(accepted) != SMOKE_COUNTS[context_type]:
            raise ValueError(f"insufficient query-disjoint {context_type} records")
        selected.extend(accepted)
    return tuple(selected)


def build_cub_smoke_technical_report(
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError("smoke condition must be a mapping")
        keys = set(condition)
        if keys & _OUTCOME_FIELDS:
            raise ValueError("smoke report cannot contain an outcome metric")
        unexpected = keys - _TECHNICAL_FIELDS
        if unexpected:
            raise ValueError(f"unsupported smoke technical fields: {sorted(unexpected)}")
        if "model_key" not in condition or "quantization" not in condition:
            raise ValueError("smoke condition requires model identity")
        normalized.append(dict(condition))
    normalized.sort(key=lambda item: (item["model_key"], item["quantization"]))
    return {
        "protocol_version": SMOKE_SELECTION_VERSION,
        "conditions": normalized,
    }


def _smoke_rank(record: CubRecord) -> str:
    payload = "|".join(
        (
            SMOKE_SELECTION_VERSION,
            record.analysis_context_type,
            record.sample_id,
            record.context_identity_sha256,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
