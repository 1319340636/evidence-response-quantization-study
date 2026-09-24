"""Strict parsing and deterministic selection for frozen TabFact data."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .vitaminc_splits import stable_rank


@dataclass(frozen=True)
class TabFactClaim:
    table_id: str
    claim_index: int
    claim: str
    label: int
    caption: str

    @property
    def sample_id(self) -> str:
        return f"{self.table_id}:{self.claim_index}"


@dataclass(frozen=True)
class OldTabFactOverlap:
    rows: int
    table_ids: frozenset[str]


def parse_tabfact_examples(
    raw: Mapping[str, Any], tables_dir: str | Path
) -> list[TabFactClaim]:
    """Validate and flatten the official table-to-claims mapping."""
    table_root = Path(tables_dir)
    examples: list[TabFactClaim] = []
    for table_id, value in raw.items():
        if not isinstance(table_id, str) or not table_id:
            raise ValueError("TabFact requires a nonempty table ID")
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{table_id} must contain exactly three fields")
        claims, labels, caption = value
        if not isinstance(claims, list) or not isinstance(labels, list):
            raise ValueError(f"{table_id} claims and labels must be lists")
        if len(claims) != len(labels):
            raise ValueError(f"{table_id} claims and labels must have the same length")
        if not isinstance(caption, str):
            raise ValueError(f"{table_id} caption must be a string")
        if not (table_root / table_id).is_file():
            raise ValueError(f"missing table CSV: {table_id}")
        for claim_index, (claim, label) in enumerate(zip(claims, labels)):
            if not isinstance(claim, str):
                raise ValueError(f"{table_id}:{claim_index} claim must be a string")
            if type(label) is not int or label not in (0, 1):
                raise ValueError(
                    f"{table_id}:{claim_index} label must be a binary integer"
                )
            examples.append(
                TabFactClaim(
                    table_id=table_id,
                    claim_index=claim_index,
                    claim=claim,
                    label=label,
                    caption=caption,
                )
            )
    return examples


def load_tabfact_examples(
    path: str | Path, tables_dir: str | Path
) -> list[TabFactClaim]:
    """Load and validate the official TabFact JSON file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("TabFact test file must contain a JSON object")
    return parse_tabfact_examples(raw, tables_dir)


def extract_old_table_ids(
    old_rows: Iterable[Mapping[str, Any]],
) -> OldTabFactOverlap:
    """Extract all table IDs exposed by the old evaluation."""
    rows = list(old_rows)
    table_ids = frozenset(
        table_id
        for row in rows
        if isinstance((table_id := row.get("group_id")), str) and table_id
    )
    return OldTabFactOverlap(rows=len(rows), table_ids=table_ids)


def fresh_eligible(
    claims: Iterable[TabFactClaim], overlap: OldTabFactOverlap
) -> list[TabFactClaim]:
    """Exclude every claim from any table exposed by the old evaluation."""
    return [claim for claim in claims if claim.table_id not in overlap.table_ids]


def select_fresh_subset(
    claims: Iterable[TabFactClaim], seed: str, limit: int = 2_000
) -> list[TabFactClaim]:
    """Select claims by frozen table-first deterministic round-robin."""
    by_table: dict[str, list[TabFactClaim]] = defaultdict(list)
    for claim in claims:
        by_table[claim.table_id].append(claim)
    if limit > sum(len(rows) for rows in by_table.values()):
        raise ValueError("TabFact selection limit exceeds eligible claims")
    table_order = sorted(
        by_table,
        key=lambda table_id: stable_rank(seed, "tabfact-table", table_id),
    )
    for table_id, rows in by_table.items():
        rows.sort(
            key=lambda row: stable_rank(
                seed, "tabfact-claim", f"{table_id}|{row.claim_index}"
            )
        )
    selected: list[TabFactClaim] = []
    pass_number = 0
    while len(selected) < limit:
        progressed = False
        for table_id in table_order:
            rows = by_table[table_id]
            if pass_number < len(rows):
                selected.append(rows[pass_number])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            raise ValueError("TabFact round-robin exhausted before selection limit")
        pass_number += 1
    return selected
