"""Blind structural gate for VitaminC D3 engineering cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .prompts import load_prompt_contract, render_prompt
from .vitaminc_mapping_protocol import (
    load_mapping_protocol,
    load_mapping_units,
    require_d3_path,
)
from .vitaminc_mapping_runner import build_mapping_run_protocol
from .vitaminc_store_v4 import VITAMINC_STORE_V4_SCHEMA_VERSION


D3_ENGINEERING_GATE_PROTOCOL = (
    "vitaminc-mapping-engineering-gate-v1-20260719"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    frozen = load_mapping_protocol(args.project_root)
    cell_sources = {
        f"{condition}__{mapping}": {
            "export_directory": (
                args.run_root / f"{condition}__{mapping}" / "export"
            ),
            "audit_certificate": (
                args.run_root / f"{condition}__{mapping}" / "audit.json"
            ),
        }
        for condition in frozen.conditions
        for mapping in ("original", "reversed")
    }
    gate = build_mapping_engineering_gate(
        project_root=args.project_root,
        cell_sources=cell_sources,
        output_path=args.output,
    )
    print(_json_text(gate), end="")
    return 0


def build_mapping_engineering_gate(
    *,
    project_root: str | Path,
    cell_sources: Mapping[str, Mapping[str, str | Path]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate D3 structure and prompt hashes without aggregating outcomes."""
    root = Path(project_root).resolve()
    frozen = load_mapping_protocol(root)
    units, population = load_mapping_units(
        root, population="engineering"
    )
    expected_cells = {
        f"{condition}__{mapping}"
        for condition in frozen.conditions
        for mapping in ("original", "reversed")
    }
    if set(cell_sources) != expected_cells:
        raise ValueError("D3 engineering cell set mismatch")
    contract = load_prompt_contract(
        root / "configs/prompt_contract_v2.json"
    )
    unit_by_owner = {unit.owner_key: unit for unit in units}
    cell_summaries: list[dict[str, Any]] = []

    for condition in frozen.conditions:
        condition_identity = frozen.condition_identities[condition]
        for mapping_variant in ("original", "reversed"):
            cell = f"{condition}__{mapping_variant}"
            supplied = cell_sources[cell]
            if set(supplied) != {
                "export_directory",
                "audit_certificate",
            }:
                raise ValueError("D3 engineering source declaration mismatch")
            export = require_d3_path(supplied["export_directory"])
            audit_path = require_d3_path(supplied["audit_certificate"])
            manifest_path = require_d3_path(export / "manifest.json")
            records_path = require_d3_path(export / "records.jsonl")
            manifest = _read_object(manifest_path)
            unsigned = dict(manifest)
            claimed_manifest_hash = unsigned.pop("manifest_sha256", None)
            if claimed_manifest_hash != _record_hash(unsigned):
                raise ValueError("D3 export manifest self-hash mismatch")
            if (
                manifest.get("schema_version")
                != VITAMINC_STORE_V4_SCHEMA_VERSION
            ):
                raise ValueError("D3 export schema mismatch")
            if manifest.get("records_jsonl_sha256") != _file_sha256(
                records_path
            ):
                raise ValueError("D3 records SHA-256 mismatch")

            audit = _read_object(audit_path)
            certificate_sha256 = audit.get("certificate_sha256")
            identity = manifest.get("run_identity")
            if (
                not isinstance(certificate_sha256, str)
                or len(certificate_sha256) != 64
                or not isinstance(identity, Mapping)
            ):
                raise ValueError("D3 audit or run identity is invalid")
            expected_identity = {
                "protocol_version": build_mapping_run_protocol(
                    backend=condition_identity.backend,
                    mapping_variant=mapping_variant,
                    mapping_config_sha256=frozen.config_sha256,
                ),
                "split": "discovery",
                "split_sha256": population.case_ids_sha256,
                "audit_certificate_sha256": certificate_sha256,
                "prompt_contract_sha256": frozen.prompt_contract_sha256,
                "model_key": condition_identity.model_key,
                "quantization": condition_identity.quantization,
                "expected_full": population.unit_count,
                "expected_no_evidence": 0,
                "expected_knowledge_only": 0,
            }
            for field, expected in expected_identity.items():
                if identity.get(field) != expected:
                    raise ValueError(
                        f"D3 run identity mismatch at {field}"
                    )
            expected_counts = {
                "units_registered": population.unit_count,
                "full_registered": population.unit_count,
                "no_evidence_registered": 0,
                "knowledge_only_registered": 0,
                "hard_completed": population.unit_count,
                "hard_failures": 0,
                "probability_completed": population.unit_count,
                "probability_failures": 0,
            }
            for field, expected in expected_counts.items():
                if manifest.get(field) != expected:
                    label = (
                        "failure"
                        if field.endswith("failures")
                        else "count"
                    )
                    raise ValueError(
                        f"D3 engineering {label} mismatch at {field}"
                    )

            rows = _read_jsonl(records_path)
            if (
                len(rows) != population.unit_count
                or [row.get("owner_key") for row in rows]
                != sorted(unit_by_owner)
            ):
                raise ValueError("D3 engineering owner identity mismatch")
            for row in rows:
                owner = row["owner_key"]
                unit = unit_by_owner[owner]
                metadata = row.get("metadata")
                if (
                    row.get("control") != "full"
                    or row.get("hard_status") != "ok"
                    or row.get("probability_status") != "ok"
                    or not isinstance(metadata, Mapping)
                    or metadata.get("case_id") != unit.case_id
                    or metadata.get("mapping_variant") != mapping_variant
                ):
                    raise ValueError("D3 engineering record structure mismatch")
                expected_prompt = render_prompt(
                    contract,
                    unit.prompt_input,
                    control="full",
                    mapping_variant=mapping_variant,
                ).prompt_sha256
                hard = row.get("hard_payload")
                probability = row.get("probability_payload")
                if (
                    not isinstance(hard, Mapping)
                    or not isinstance(probability, Mapping)
                    or hard.get("prompt_sha256") != expected_prompt
                    or probability.get("prompt_sha256") != expected_prompt
                ):
                    raise ValueError("D3 engineering prompt hash mismatch")

            cell_summaries.append(
                {
                    "condition": condition,
                    "backend": condition_identity.backend,
                    "model_key": condition_identity.model_key,
                    "quantization": condition_identity.quantization,
                    "mapping_variant": mapping_variant,
                    "prompt_hashes_validated": population.unit_count,
                    "audit_certificate_sha256": _file_sha256(audit_path),
                    "export_manifest_sha256": _file_sha256(manifest_path),
                    "records_jsonl_sha256": _file_sha256(records_path),
                }
            )

    gate: dict[str, Any] = {
        "protocol_version": D3_ENGINEERING_GATE_PROTOCOL,
        "gate_passed": True,
        "mapping_config_sha256": frozen.config_sha256,
        "prompt_contract_sha256": frozen.prompt_contract_sha256,
        "population": {
            "quartets": population.case_count,
            "full_units_per_cell": population.unit_count,
        },
        "cells": cell_summaries,
    }
    gate["gate_sha256"] = _record_hash(gate)
    destination = require_d3_path(output_path)
    if destination.exists():
        raise ValueError("D3 engineering gate output already exists")
    _atomic_write(destination, _json_text(gate))
    return gate


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D3 expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("D3 records must contain JSON objects")
        values.append(value)
    return values


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
