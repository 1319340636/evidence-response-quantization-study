"""Frozen input selection for the Stage B GPTQ and TabFact smoke routes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .vitaminc_splits import stable_rank


TABFACT_SMOKE_SEED = "experiment-b-tabfact-smoke-v1-20260717"
GPTQ_CALIBRATION_SEED = "experiment-b-gptq-calibration-v1-20260717"
GPTQ_CALIBRATION_SOURCE_ROLE = "vitaminc_dev"

_TABFACT_FIELDS = {
    "sample_id",
    "table_id",
    "claim_index",
    "claim",
    "label",
    "caption",
    "table_path",
}


def select_tabfact_smoke(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: str = TABFACT_SMOKE_SEED,
    per_label: int = 25,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced smoke with one claim per table."""
    if not isinstance(seed, str) or not seed:
        raise ValueError("TabFact smoke seed must be nonempty")
    if type(per_label) is not int or per_label < 1:
        raise ValueError("TabFact smoke per_label must be positive")
    normalized = [_validate_tabfact_row(row) for row in rows]
    sample_ids = [row["sample_id"] for row in normalized]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("TabFact smoke source has duplicate sample IDs")

    selected: list[dict[str, Any]] = []
    used_tables: set[str] = set()
    for label in (0, 1):
        candidates = sorted(
            (row for row in normalized if row["label"] == label),
            key=lambda row: (
                stable_rank(seed, f"tabfact-smoke-label-{label}", row["sample_id"]),
                row["sample_id"],
            ),
        )
        for row in candidates:
            if row["table_id"] in used_tables:
                continue
            selected.append(row)
            used_tables.add(row["table_id"])
            if sum(item["label"] == label for item in selected) == per_label:
                break
        if sum(item["label"] == label for item in selected) != per_label:
            raise ValueError(
                "TabFact smoke cannot satisfy balanced distinct tables"
            )
    return sorted(
        selected,
        key=lambda row: (
            stable_rank(seed, "tabfact-smoke-order", row["sample_id"]),
            row["sample_id"],
        ),
    )


def select_vitaminc_dev_calibration(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: str = GPTQ_CALIBRATION_SEED,
    limit: int = 256,
    source_role: str,
) -> list[dict[str, Any]]:
    """Select calibration text while refusing evaluation-source aliases."""
    if source_role != GPTQ_CALIBRATION_SOURCE_ROLE:
        raise ValueError("GPTQ calibration source_role must be vitaminc_dev")
    if not isinstance(seed, str) or not seed:
        raise ValueError("GPTQ calibration seed must be nonempty")
    if type(limit) is not int or limit < 1:
        raise ValueError("GPTQ calibration limit must be positive")
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("VitaminC calibration row must be an object")
        row = dict(raw)
        for field in ("unique_id", "claim", "evidence"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(
                    f"VitaminC calibration row requires nonempty {field}"
                )
        normalized.append(row)
    if len({row["unique_id"] for row in normalized}) != len(normalized):
        raise ValueError("VitaminC calibration IDs must be unique")
    if limit > len(normalized):
        raise ValueError("VitaminC calibration limit exceeds source rows")
    return sorted(
        normalized,
        key=lambda row: (
            stable_rank(seed, "gptq-calibration", row["unique_id"]),
            row["unique_id"],
        ),
    )[:limit]


def build_calibration_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_path: str | Path,
    seed: str,
    source_role: str,
) -> dict[str, Any]:
    """Bind already-selected calibration membership to its exact source."""
    if source_role != GPTQ_CALIBRATION_SOURCE_ROLE:
        raise ValueError("GPTQ calibration source_role must be vitaminc_dev")
    path = Path(source_path)
    if not path.is_file():
        raise ValueError("GPTQ calibration source file is missing")
    normalized = [dict(row) for row in rows]
    ids: list[str] = []
    for row in normalized:
        value = row.get("unique_id")
        if not isinstance(value, str) or not value:
            raise ValueError("GPTQ calibration manifest requires unique_id")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError("GPTQ calibration manifest IDs must be unique")
    return {
        "schema_version": "gptq-calibration-v1-20260717",
        "source_role": source_role,
        "source_path": path.as_posix(),
        "source_sha256": _file_sha256(path),
        "seed": seed,
        "count": len(ids),
        "ids": ids,
        "ids_sha256": _ids_sha256(ids),
    }


def _validate_tabfact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != _TABFACT_FIELDS:
        raise ValueError("TabFact smoke row schema mismatch")
    normalized = dict(row)
    for field in ("sample_id", "table_id", "claim", "caption", "table_path"):
        if not isinstance(normalized[field], str) or not normalized[field]:
            raise ValueError(f"TabFact smoke requires nonempty {field}")
    if type(normalized["claim_index"]) is not int or normalized["claim_index"] < 0:
        raise ValueError("TabFact smoke claim_index must be nonnegative")
    if type(normalized["label"]) is not int or normalized["label"] not in (0, 1):
        raise ValueError("TabFact smoke label must be binary")
    expected_id = f"{normalized['table_id']}:{normalized['claim_index']}"
    if normalized["sample_id"] != expected_id:
        raise ValueError("TabFact smoke sample_id mismatch")
    return normalized


def _ids_sha256(ids: Iterable[str]) -> str:
    payload = "".join(f"{item}\n" for item in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

