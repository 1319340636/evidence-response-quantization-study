"""Prediction-free materialization of the CUB/DRUID official test split."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONTEXT_TYPE_MAP = {
    "gold": "gold",
    "edited": "conflicting",
    "irrelevant": "irrelevant",
}
CUB_OUTPUT_LABELS = ("True", "False", "None")
TRUE_TARGETS = frozenset({"True", "False", "Half True"})
NEW_TARGETS = frozenset(CUB_OUTPUT_LABELS)


@dataclass(frozen=True)
class CubRecord:
    sample_id: str
    claim: str
    claimant: str
    evidence: str
    relevant: bool
    context_type: str
    target_true: str
    target_new: str | None
    context_identity_sha256: str

    @property
    def analysis_context_type(self) -> str:
        return CONTEXT_TYPE_MAP[self.context_type]

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "analysis_context_type": self.analysis_context_type,
            "claim": self.claim,
            "claimant": self.claimant,
            "context_identity_sha256": self.context_identity_sha256,
            "context_type": self.context_type,
            "evidence": self.evidence,
            "relevant": self.relevant,
            "sample_id": self.sample_id,
            "target_new": self.target_new,
            "target_true": self.target_true,
        }


@dataclass(frozen=True)
class CubTestArtifact:
    records: tuple[CubRecord, ...]
    variant_files: tuple[str, ...]


def cub_binary_context_utilisation(
    *,
    context_type: str,
    prediction_without_context: str,
    prediction_with_context: str,
    target_new: str | None,
) -> bool:
    """Score CUB BCU exactly at the published hard-prediction boundary.

    Gold and edited contexts are successful when the context-conditioned
    prediction follows ``target_new``. Irrelevant contexts are successful when
    the prediction is unchanged by adding the context.
    """
    if context_type not in CONTEXT_TYPE_MAP:
        raise ValueError(f"unsupported CUB context_type: {context_type}")
    if not isinstance(prediction_without_context, str):
        raise ValueError("CUB prediction_without_context must be a string")
    if not isinstance(prediction_with_context, str):
        raise ValueError("CUB prediction_with_context must be a string")

    prediction_without = prediction_without_context.strip().casefold()
    prediction_with = prediction_with_context.strip().casefold()
    if context_type == "irrelevant":
        if target_new is not None:
            raise ValueError("irrelevant CUB target_new must be null")
        return prediction_with == prediction_without

    if target_new not in NEW_TARGETS:
        raise ValueError("relevant CUB target_new must be True, False, or None")
    return prediction_with == target_new.casefold()


def cub_continuous_context_utilisation(
    *,
    probability_without_context: float,
    probability_with_context: float,
) -> float:
    """Compute CUB's published continuous context-utilisation score."""
    p0 = _strict_probability(
        probability_without_context, "probability_without_context"
    )
    p1 = _strict_probability(
        probability_with_context, "probability_with_context"
    )
    score = (p1 - p0) / (1.0 - p0) if p1 >= p0 else (p1 - p0) / p0
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("CUB CCU score is outside [-1, 1]")
    return score


def _strict_probability(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(f"CUB {label} must be a finite probability in (0, 1)")
    return float(value)


def _required_text(row: Mapping[str, Any], key: str, label: str | None = None) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"CUB {label or key} must be a nonempty string")
    return value


def canonicalize_cub_row(row: Mapping[str, Any]) -> CubRecord:
    """Allowlist semantic source fields and discard all model-dependent fields."""
    sample_id = _required_text(row, "id", "sample ID")
    claim = _required_text(row, "claim")
    claimant = _required_text(row, "claimant")
    evidence = _required_text(row, "evidence")
    relevant = row.get("relevant")
    if type(relevant) is not bool:
        raise ValueError("CUB relevant must be a boolean")
    context_type = row.get("context_type")
    if context_type not in CONTEXT_TYPE_MAP:
        raise ValueError(f"unsupported CUB context_type: {context_type}")
    target_true_raw = row.get("target_true")
    if not isinstance(target_true_raw, str):
        raise ValueError("CUB target_true must be a string")
    target_true = target_true_raw.strip()
    if target_true not in TRUE_TARGETS:
        raise ValueError(f"unsupported CUB target_true: {target_true}")
    target_new_raw = row.get("target_new")
    if target_new_raw is None:
        target_new = None
    elif isinstance(target_new_raw, str):
        target_new = target_new_raw.strip()
        if target_new not in NEW_TARGETS:
            raise ValueError(f"unsupported CUB target_new: {target_new}")
    else:
        raise ValueError("CUB target_new must be a string or null")

    if context_type == "gold" and target_new != target_true:
        raise ValueError("gold context must preserve the true target")
    if context_type == "edited" and (target_new is None or target_new == target_true):
        raise ValueError("edited context must conflict with the true target")
    if context_type == "irrelevant" and target_new is not None:
        raise ValueError("irrelevant context must have null target_new")

    semantic = {
        "claim": claim,
        "claimant": claimant,
        "context_type": context_type,
        "evidence": evidence,
        "relevant": relevant,
        "sample_id": sample_id,
        "target_new": target_new,
        "target_true": target_true,
    }
    payload = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return CubRecord(
        sample_id=sample_id,
        claim=claim,
        claimant=claimant,
        evidence=evidence,
        relevant=relevant,
        context_type=context_type,
        target_true=target_true,
        target_new=target_new,
        context_identity_sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_cub_test(
    repository: str | Path,
    *,
    expected_variant_count: int | None = None,
    expected_rows_per_file: int | None = None,
) -> CubTestArtifact:
    """Require semantic agreement across every model-version test file."""
    root = Path(repository)
    paths = sorted(root.glob("*_test.jsonl"))
    if expected_variant_count is not None and len(paths) != expected_variant_count:
        raise ValueError(
            f"CUB test variant count mismatch: expected {expected_variant_count}, "
            f"got {len(paths)}"
        )
    if not paths:
        raise ValueError("no CUB test variants found")

    canonical_by_id: dict[str, CubRecord] = {}
    baseline_ids: set[str] | None = None
    for path in paths:
        file_records: dict[str, CubRecord] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{line_number} must be a JSON object")
                record = canonicalize_cub_row(value)
                if record.sample_id in file_records:
                    raise ValueError(
                        f"{path.name} duplicate sample ID: {record.sample_id}"
                    )
                file_records[record.sample_id] = record
        if expected_rows_per_file is not None and len(file_records) != expected_rows_per_file:
            raise ValueError(
                f"{path.name} row count mismatch: expected {expected_rows_per_file}, "
                f"got {len(file_records)}"
            )
        file_ids = set(file_records)
        if baseline_ids is None:
            baseline_ids = file_ids
            canonical_by_id = file_records
            continue
        if file_ids != baseline_ids:
            raise ValueError(f"{path.name} sample ID set differs from first variant")
        for sample_id, record in file_records.items():
            if (
                record.context_identity_sha256
                != canonical_by_id[sample_id].context_identity_sha256
            ):
                raise ValueError(f"semantic conflict for CUB sample ID: {sample_id}")

    records = tuple(canonical_by_id[sample_id] for sample_id in sorted(canonical_by_id))
    return CubTestArtifact(records=records, variant_files=tuple(path.name for path in paths))
