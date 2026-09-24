"""Deterministic reconstruction of strict VitaminC 2x2 quartets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


NEGATIVE_LABELS = frozenset({"REFUTES", "NOT ENOUGH INFO"})


@dataclass(frozen=True)
class VitaminCQuartet:
    """A canonical 2x2 quartet with SUPPORTS on the main diagonal."""

    case_id: str
    page: str
    negative_label: str
    claims: tuple[str, str]
    evidences: tuple[str, str]
    row_ids: tuple[str, str, str, str]
    labels: tuple[str, str, str, str]


@dataclass(frozen=True)
class VitaminCExclusion:
    """One deterministic reason why a candidate case was not a strict quartet."""

    case_id: str | None
    reason: str
    row_count: int


@dataclass(frozen=True)
class VitaminCQuartetAudit:
    """Accepted strict quartets and mutually exclusive exclusion records."""

    quartets: tuple[VitaminCQuartet, ...]
    exclusions: tuple[VitaminCExclusion, ...]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file without modifying replacement characters."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            rows.append(value)
    return rows


def build_strict_quartets(
    rows: Iterable[Mapping[str, Any]],
) -> list[VitaminCQuartet]:
    """Return groups satisfying the preregistered strict-quartet definition."""
    return list(build_strict_quartet_audit(rows).quartets)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def build_strict_quartet_audit(
    rows: Iterable[Mapping[str, Any]],
) -> VitaminCQuartetAudit:
    """Reconstruct strict quartets and retain one exclusion reason per case."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    missing_case_rows = 0
    for row in rows:
        case_id = row.get("case_id")
        if _nonempty_text(case_id):
            grouped[case_id].append(row)
        else:
            missing_case_rows += 1

    quartets: list[VitaminCQuartet] = []
    exclusions: list[VitaminCExclusion] = []
    if missing_case_rows:
        exclusions.append(VitaminCExclusion(None, "missing_case_id", missing_case_rows))

    def exclude(case_id: str, group: list[Mapping[str, Any]], reason: str) -> None:
        exclusions.append(VitaminCExclusion(case_id, reason, len(group)))

    for case_id in sorted(grouped):
        group = grouped[case_id]
        if len(group) != 4:
            exclude(case_id, group, "row_count")
            continue
        if any(row.get("revision_type") != "real" for row in group):
            exclude(case_id, group, "non_real")
            continue
        if any(not _nonempty_text(row.get("claim")) for row in group):
            exclude(case_id, group, "invalid_claim")
            continue
        if any(not _nonempty_text(row.get("evidence")) for row in group):
            exclude(case_id, group, "invalid_evidence")
            continue
        if any(not _nonempty_text(row.get("page")) for row in group):
            exclude(case_id, group, "invalid_page")
            continue
        if any(not _nonempty_text(row.get("unique_id")) for row in group):
            exclude(case_id, group, "invalid_unique_id")
            continue
        if any(not _nonempty_text(row.get("label")) for row in group):
            exclude(case_id, group, "invalid_label")
            continue
        unique_ids = {row.get("unique_id") for row in group}
        if len(unique_ids) != len(group):
            exclude(case_id, group, "duplicate_unique_id")
            continue
        claims = tuple(sorted({row.get("claim") for row in group}))
        evidences = tuple(sorted({row.get("evidence") for row in group}))
        pages = {row.get("page") for row in group}
        if len(claims) != 2 or len(evidences) != 2 or len(pages) != 1:
            exclude(case_id, group, "dimensions")
            continue

        cells = {(row.get("claim"), row.get("evidence")): row for row in group}
        if len(cells) != 4:
            exclude(case_id, group, "duplicate_cartesian_cell")
            continue
        ordered = [cells.get((claim, evidence)) for claim in claims for evidence in evidences]
        if any(row is None for row in ordered):
            exclude(case_id, group, "incomplete_cartesian")
            continue
        labels = tuple(row.get("label") for row in ordered if row is not None)
        if not (
            labels[0] == labels[3]
            and labels[1] == labels[2]
            and labels[0] != labels[1]
        ):
            exclude(case_id, group, "label_pattern")
            continue
        label_set = set(labels)
        if "SUPPORTS" not in label_set or len(label_set) != 2:
            exclude(case_id, group, "label_pattern")
            continue
        negative_label = next(label for label in label_set if label != "SUPPORTS")
        if negative_label not in NEGATIVE_LABELS:
            exclude(case_id, group, "unsupported_label")
            continue

        oriented_evidences = evidences
        if labels[0] != "SUPPORTS":
            oriented_evidences = (evidences[1], evidences[0])
        oriented = [
            cells[(claim, evidence)]
            for claim in claims
            for evidence in oriented_evidences
        ]
        row_ids = tuple(row.get("unique_id") for row in oriented)
        oriented_labels = tuple(row.get("label") for row in oriented)
        quartets.append(
            VitaminCQuartet(
                case_id=case_id,
                page=next(iter(pages)),
                negative_label=negative_label,
                claims=claims,
                evidences=oriented_evidences,
                row_ids=row_ids,
                labels=oriented_labels,
            )
        )
    return VitaminCQuartetAudit(tuple(quartets), tuple(exclusions))


def canonical_id_manifest(ids: Iterable[str]) -> bytes:
    """Serialize sorted identifiers as UTF-8, one per LF-terminated line."""
    return ("\n".join(sorted(ids)) + "\n").encode("utf-8")


def id_manifest_sha256(ids: Iterable[str]) -> str:
    """Hash the canonical identifier manifest."""
    return hashlib.sha256(canonical_id_manifest(ids)).hexdigest()
