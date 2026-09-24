"""Outcome-blind structural gate for the balanced HF mapping experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from .hf_runtime_audit import load_hf_runtime_audit
from .hf_vitaminc_runner import (
    HF_BALANCED_MAPPING_FORMAL_PROTOCOL,
    HF_BALANCED_MAPPING_SMOKE_PROTOCOL,
)
from .vitaminc_balanced_mapping_protocol import (
    load_balanced_mapping_protocol,
)


def _record_hash(record: dict[str, object], *, excluded: str) -> str:
    clean = {key: value for key, value in record.items() if key != excluded}
    payload = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_balanced_cells(
    project_root: str | Path,
) -> tuple[tuple[str, str], ...]:
    protocol = load_balanced_mapping_protocol(project_root)
    return tuple(
        (mapping, condition)
        for mapping in protocol.mappings
        for condition in protocol.conditions
    )


def _validate_cell(
    run_dir: Path,
    *,
    model_key: str,
    precision: str,
    mapping_variant: str,
    expected: int,
    protocol_base: str,
) -> dict[str, object]:
    database = run_dir / "run.sqlite3"
    manifest_path = run_dir / "export/manifest.json"
    records_path = run_dir / "export/records.jsonl"
    audit_path = run_dir / "audit.json"
    for path in (database, manifest_path, records_path, audit_path):
        if not path.is_file():
            raise ValueError(f"balanced mapping cell artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != _record_hash(
        manifest, excluded="manifest_sha256"
    ):
        raise ValueError("balanced mapping manifest self-hash mismatch")
    records_sha = _file_sha256(records_path)
    if manifest.get("records_jsonl_sha256") != records_sha:
        raise ValueError("balanced mapping records SHA-256 mismatch")
    identity = manifest.get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError("balanced mapping run identity is missing")
    expected_protocol = f"{protocol_base}:{mapping_variant}"
    if (
        identity.get("model_key") != model_key
        or identity.get("quantization") != precision
        or identity.get("protocol_version") != expected_protocol
        or identity.get("expected_full") != expected
        or identity.get("expected_no_evidence") != 0
        or identity.get("expected_knowledge_only") != 0
    ):
        raise ValueError("balanced mapping run identity mismatch")
    audit = load_hf_runtime_audit(audit_path)
    if (
        audit.model_key != model_key
        or audit.precision != precision
        or identity.get("audit_certificate_sha256") != audit.certificate_sha256
    ):
        raise ValueError("balanced mapping audit binding mismatch")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        controls = dict(
            connection.execute(
                "SELECT control,COUNT(*) FROM owners GROUP BY control"
            )
        )
        hard = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(status='error'),0) FROM hard_results"
        ).fetchone()
        probability = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(status='error'),0) "
            "FROM probability_results"
        ).fetchone()
    finally:
        connection.close()
    if controls != {"full": expected} or hard != (expected, 0) or probability != (expected, 0):
        raise ValueError("balanced mapping database structure mismatch")
    return {
        "audit_file_sha256": _file_sha256(audit_path),
        "hard_errors": hard[1],
        "hard_results": hard[0],
        "manifest_file_sha256": _file_sha256(manifest_path),
        "owners": expected,
        "probability_errors": probability[1],
        "probability_results": probability[0],
        "records_jsonl_sha256": records_sha,
    }


def build_balanced_mapping_gate(
    project_root: str | Path,
    run_root: str | Path,
    *,
    population: str,
) -> dict[str, object]:
    protocol = load_balanced_mapping_protocol(project_root)
    if population not in {"engineering", "formal"}:
        raise ValueError("balanced mapping gate population is invalid")
    expected = 16 if population == "engineering" else 4800
    protocol_base = (
        HF_BALANCED_MAPPING_SMOKE_PROTOCOL
        if population == "engineering"
        else HF_BALANCED_MAPPING_FORMAL_PROTOCOL
    )
    root = Path(run_root).resolve()
    cells: dict[str, object] = {}
    for mapping, condition in expected_balanced_cells(project_root):
        model_key, precision = condition.rsplit("_", 1)
        if precision == "INT4":
            model_key, route = condition.rsplit("_", 2)[0], condition.rsplit("_", 2)[1]
            precision = f"{route}_INT4"
        cells[f"{mapping}/{condition}"] = _validate_cell(
            root / mapping / condition,
            model_key=model_key,
            precision=precision,
            mapping_variant=mapping,
            expected=expected,
            protocol_base=protocol_base,
        )
    gate: dict[str, object] = {
        "cell_count": len(cells),
        "cells": cells,
        "config_sha256": protocol.config_sha256,
        "expected_inputs_per_cell": expected,
        "gate_passed": True,
        "mapping_count": len(protocol.mappings),
        "model_route_count": len(protocol.conditions),
        "population": population,
        "protocol_version": protocol.protocol_version,
    }
    gate["gate_sha256"] = _record_hash(gate, excluded="gate_sha256")
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--population", choices=("engineering", "formal"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gate = build_balanced_mapping_gate(
        args.project_root, args.run_root, population=args.population
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate_passed": True, "cell_count": gate["cell_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
