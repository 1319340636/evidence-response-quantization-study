"""Strict, complete input construction for frozen TabFact inference."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .prompts import PromptInput


@dataclass(frozen=True)
class SerializedTabFactTable:
    text: str
    rows: int
    truncated: bool = False


@dataclass(frozen=True)
class TabFactManifestRow:
    sample_id: str
    table_id: str
    claim_index: int
    claim: str
    label: int
    caption: str
    table_path: Path


@dataclass(frozen=True)
class TabFactInferenceUnit:
    sample_id: str
    table_id: str
    claim_index: int
    gold_label: str
    table_rows: int
    table_truncated: bool
    prompt_input: PromptInput

    @property
    def owner_key(self) -> str:
        return f"{self.sample_id}|full"


def serialize_tabfact_table(path: str | Path) -> SerializedTabFactTable:
    """Serialize every hash-delimited table cell without truncation."""
    table_path = Path(path)
    if not table_path.is_file():
        raise ValueError(f"missing TabFact table: {table_path}")
    with table_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="#"))
    if not rows or any(not row for row in rows):
        raise ValueError("TabFact table must contain nonempty rows")
    text = "\n".join(
        f"Row {index}: " + " | ".join(cell for cell in row)
        for index, row in enumerate(rows)
    )
    return SerializedTabFactTable(text=text, rows=len(rows), truncated=False)


def load_tabfact_manifest(
    path: str | Path, *, project_root: str | Path
) -> list[TabFactManifestRow]:
    """Load a prediction-free TabFact manifest and contain all table paths."""
    manifest_path = Path(path)
    root = Path(project_root).resolve()
    records: list[TabFactManifestRow] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid TabFact JSON at line {line_number}"
                ) from error
            records.append(_parse_manifest_row(raw, root))
    if not records:
        raise ValueError("TabFact manifest must be nonempty")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("TabFact manifest sample IDs must be unique")
    return records


def build_tabfact_prompt_inputs(
    rows: Sequence[TabFactManifestRow],
) -> list[TabFactInferenceUnit]:
    """Build full-evidence binary units with complete table serialization."""
    units: list[TabFactInferenceUnit] = []
    for row in rows:
        table = serialize_tabfact_table(row.table_path)
        units.append(
            TabFactInferenceUnit(
                sample_id=row.sample_id,
                table_id=row.table_id,
                claim_index=row.claim_index,
                gold_label="Entailed" if row.label == 1 else "Refuted",
                table_rows=table.rows,
                table_truncated=table.truncated,
                prompt_input=PromptInput(
                    dataset="tabfact",
                    sample_id=row.sample_id,
                    claim=row.claim,
                    evidence=table.text,
                    metadata={"caption": row.caption},
                ),
            )
        )
    return units


def tabfact_evidence_manifest_sha256(
    rows: Sequence[TabFactManifestRow],
) -> str:
    """Hash every unique referenced table ID and its exact file bytes."""
    if not rows:
        raise ValueError("TabFact evidence manifest must be nonempty")
    paths: dict[str, Path] = {}
    for row in rows:
        path = row.table_path.resolve()
        existing = paths.get(row.table_id)
        if existing is not None and existing != path:
            raise ValueError("TabFact table ID maps to conflicting paths")
        paths[row.table_id] = path
    lines = [
        f"{table_id}\t{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        for table_id, path in sorted(paths.items())
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _parse_manifest_row(
    raw: object, project_root: Path
) -> TabFactManifestRow:
    required = {
        "sample_id",
        "table_id",
        "claim_index",
        "claim",
        "label",
        "caption",
        "table_path",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("TabFact manifest row schema mismatch")
    for field in ("sample_id", "table_id", "claim", "caption", "table_path"):
        if not isinstance(raw[field], str) or not raw[field]:
            raise ValueError(f"TabFact manifest requires nonempty {field}")
    if type(raw["claim_index"]) is not int or raw["claim_index"] < 0:
        raise ValueError("TabFact claim_index must be nonnegative")
    if type(raw["label"]) is not int or raw["label"] not in (0, 1):
        raise ValueError("TabFact label must be binary")
    expected_id = f"{raw['table_id']}:{raw['claim_index']}"
    if raw["sample_id"] != expected_id:
        raise ValueError("TabFact sample_id mismatch")
    table_path = (project_root / raw["table_path"]).resolve()
    try:
        table_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("TabFact table path escapes project root") from error
    if not table_path.is_file():
        raise ValueError(f"missing TabFact table: {raw['table_path']}")
    return TabFactManifestRow(
        sample_id=raw["sample_id"],
        table_id=raw["table_id"],
        claim_index=raw["claim_index"],
        claim=raw["claim"],
        label=raw["label"],
        caption=raw["caption"],
        table_path=table_path,
    )
