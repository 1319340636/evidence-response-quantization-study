"""Deterministic query grouping for paired CUB inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

from .cub_druid import CUB_OUTPUT_LABELS, CubRecord


@dataclass(frozen=True)
class CubQueryGroup:
    query_identity_sha256: str
    claim: str
    claimant: str
    prompt_contract_version: str
    required_labels: tuple[str, ...]
    context_sample_ids: tuple[str, ...]
    records: tuple[CubRecord, ...]


def build_cub_query_groups(
    records: Sequence[CubRecord], *, prompt_contract_version: str
) -> tuple[CubQueryGroup, ...]:
    """Group contexts sharing the exact dataset-native query semantics."""
    if not isinstance(prompt_contract_version, str) or not prompt_contract_version:
        raise ValueError("prompt_contract_version must be a nonempty string")
    if not records:
        raise ValueError("CUB records must be nonempty")

    seen_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[CubRecord]] = {}
    for record in records:
        if not isinstance(record, CubRecord):
            raise ValueError("CUB query grouping requires CubRecord values")
        if record.sample_id in seen_ids:
            raise ValueError(f"duplicate CUB sample ID: {record.sample_id}")
        seen_ids.add(record.sample_id)
        if not record.claim or not record.claimant:
            raise ValueError("CUB claim and claimant must be nonempty")
        if record.relevant:
            if record.target_new not in CUB_OUTPUT_LABELS:
                raise ValueError("relevant CUB record requires a supported target_new")
        elif record.target_new is not None:
            raise ValueError("irrelevant CUB record target_new must be null")
        grouped.setdefault((record.claim, record.claimant), []).append(record)

    result: list[CubQueryGroup] = []
    for claim, claimant in sorted(grouped):
        group_records = tuple(
            sorted(grouped[(claim, claimant)], key=lambda item: item.sample_id)
        )
        labels = {
            record.target_new
            for record in group_records
            if record.relevant and record.target_new is not None
        }
        required_labels = tuple(
            label for label in CUB_OUTPUT_LABELS if label in labels
        )
        payload = json.dumps(
            {
                "claim": claim,
                "claimant": claimant,
                "control": "query_only",
                "dataset": "cub_druid",
                "prompt_contract_version": prompt_contract_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result.append(
            CubQueryGroup(
                query_identity_sha256=hashlib.sha256(payload).hexdigest(),
                claim=claim,
                claimant=claimant,
                prompt_contract_version=prompt_contract_version,
                required_labels=required_labels,
                context_sample_ids=tuple(
                    record.sample_id for record in group_records
                ),
                records=group_records,
            )
        )
    return tuple(result)
