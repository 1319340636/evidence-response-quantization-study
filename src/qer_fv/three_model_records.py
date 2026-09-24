"""Text-free analysis release for the three-model HF heterogeneity study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .major_revision_records import (
    _canonical_json,
    _contains_dataset_text_key,
    _derive_quartets,
    _file_sha256,
    _install_idempotently,
    _read_jsonl,
    _read_object,
    _record_hash,
    _released_quartet_interaction,
    _text,
)


PROTOCOL = "three-model-analysis-records-v1-20260902"
CONDITIONS = {
    "qwen_fp16": ("qwen35_9b", "FP16"),
    "qwen_gptq": ("qwen35_9b", "GPTQ_INT4"),
    "qwen_awq": ("qwen35_9b", "AWQ_INT4"),
    "ministral_fp16": ("ministral3_8b", "FP16"),
    "ministral_gptq": ("ministral3_8b", "GPTQ_INT4"),
    "ministral_awq": ("ministral3_8b", "AWQ_INT4"),
    "olmo_fp16": ("olmo3_7b", "FP16"),
    "olmo_gptq": ("olmo3_7b", "GPTQ_INT4"),
    "olmo_awq": ("olmo3_7b", "AWQ_INT4"),
}
FAMILY_KEYS = {
    "qwen": "qwen35_9b",
    "ministral": "ministral3_8b",
    "olmo": "olmo3_7b",
}


def load_text_free_three_model_inputs(
    output_directory: str | Path,
    *,
    formal: bool = True,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, str]]:
    """Verify the release and reconstruct the nine aligned condition arrays."""

    root = Path(output_directory)
    records_path = root / "analysis_records.jsonl"
    manifest_path = root / "analysis_records_manifest.json"
    sidecar_path = root / "analysis_records_manifest.json.sha256"
    manifest = _read_object(manifest_path)
    try:
        sidecar_parts = sidecar_path.read_text(encoding="utf-8").split()
    except OSError as error:
        raise ValueError("three-model record detached hash is missing") from error
    if (
        sidecar_parts != [_file_sha256(manifest_path), manifest_path.name]
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("manifest_sha256")
        != _record_hash(manifest, "manifest_sha256")
    ):
        raise ValueError("three-model record manifest integrity mismatch")
    if (
        manifest.get("dataset_text_included") is not False
        or manifest.get("records_jsonl_sha256") != _file_sha256(records_path)
    ):
        raise ValueError("three-model record release integrity mismatch")

    expected_quartets = 1200 if formal else 1
    expected_counts = {condition: expected_quartets for condition in CONDITIONS}
    if manifest.get("conditions") != expected_counts:
        raise ValueError("three-model record condition counts mismatch")
    rows = _read_jsonl(records_path)
    if len(rows) != 9 * expected_quartets or manifest.get("records") != len(rows):
        raise ValueError("three-model record row count mismatch")

    indexed: dict[str, dict[str, float]] = {condition: {} for condition in CONDITIONS}
    metadata_by_condition: dict[str, dict[str, tuple[str, str]]] = {
        condition: {} for condition in CONDITIONS
    }
    reference_metadata: dict[str, tuple[str, str]] | None = None
    for row in rows:
        if _contains_dataset_text_key(row):
            raise ValueError("three-model record release contains dataset text")
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ValueError("three-model record condition identity mismatch")
        model_key, precision = CONDITIONS[str(condition)]
        if row.get("model_key") != model_key or row.get("precision") != precision:
            raise ValueError("three-model record model or precision identity mismatch")
        case_id = _text(row.get("case_id"), "case identity")
        page = _text(row.get("page"), "page identity")
        negative_label = _text(row.get("negative_label"), "negative label")
        value = _released_quartet_interaction(row, negative_label)
        if case_id in indexed[str(condition)]:
            raise ValueError("three-model record contains duplicate quartets")
        indexed[str(condition)][case_id] = value
        metadata_by_condition[str(condition)][case_id] = (page, negative_label)

    for condition in sorted(CONDITIONS):
        metadata = metadata_by_condition[condition]
        if len(metadata) != expected_quartets:
            raise ValueError("three-model quartet membership mismatch")
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError("nine-condition quartet metadata mismatch")
    assert reference_metadata is not None
    models = {
        family: {
            route: indexed[f"{family}_{route}"]
            for route in ("fp16", "gptq", "awq")
        }
        for family in FAMILY_KEYS
    }
    pages = {case: page for case, (page, _) in reference_metadata.items()}
    return models, pages


def write_text_free_three_model_records(
    *,
    exports: Mapping[str, str | Path],
    output_directory: str | Path,
    formal: bool = True,
) -> dict[str, Path]:
    """Validate nine exports and atomically release analysis-only fields."""

    if set(exports) != set(CONDITIONS):
        raise ValueError("three-model release requires exactly nine conditions")
    all_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    condition_counts: dict[str, int] = {}
    common_identity: tuple[str, str, str] | None = None
    expected_full = 4800 if formal else 4
    expected_quartets = 1200 if formal else 1
    for condition in sorted(CONDITIONS):
        root = Path(exports[condition])
        if any(part.casefold() == "recovery" for part in root.parts):
            raise ValueError("recovery sources are forbidden")
        manifest_path = root / "manifest.json"
        records_path = root / "records.jsonl"
        manifest = _read_object(manifest_path)
        records_hash = _file_sha256(records_path)
        if manifest.get("records_jsonl_sha256") != records_hash:
            raise ValueError(f"{condition} records hash mismatch")
        source_hashes[f"{condition}_manifest"] = _file_sha256(manifest_path)
        source_hashes[f"{condition}_records"] = records_hash
        identity = manifest.get("run_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{condition} run identity is missing")
        model_key, precision = CONDITIONS[condition]
        if identity.get("model_key") != model_key:
            raise ValueError(f"{condition} model identity mismatch")
        if identity.get("quantization", identity.get("precision")) != precision:
            raise ValueError(f"{condition} precision identity mismatch")
        current_identity = (
            str(identity.get("split", "")),
            str(identity.get("split_sha256", "")),
            str(identity.get("prompt_contract_sha256", "")),
        )
        if not all(current_identity):
            raise ValueError(f"{condition} source identity is incomplete")
        if common_identity is None:
            common_identity = current_identity
        elif current_identity != common_identity:
            raise ValueError("nine-condition split or prompt identity mismatch")

        raw_rows = _read_jsonl(records_path)
        if len(raw_rows) != manifest.get("units_registered"):
            raise ValueError(f"{condition} registered unit count mismatch")
        full_rows = [row for row in raw_rows if row.get("control") == "full"]
        if len(full_rows) != expected_full:
            raise ValueError(f"{condition} full-evidence count mismatch")
        owner_keys = [_text(row.get("owner_key"), "owner key") for row in full_rows]
        if len(owner_keys) != len(set(owner_keys)):
            raise ValueError(f"{condition} contains duplicate owner identities")
        derived = _derive_quartets(condition, model_key, precision, full_rows)
        if len(derived) != expected_quartets:
            raise ValueError(f"{condition} quartet count mismatch")
        all_rows.extend(sorted(derived, key=lambda row: row["case_id"]))
        condition_counts[condition] = len(derived)

    assert common_identity is not None
    destination = Path(output_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".three-model-records.", dir=destination.parent)
    )
    try:
        records_path = temporary / "analysis_records.jsonl"
        records_path.write_text(
            "".join(_canonical_json(row) + "\n" for row in all_rows),
            encoding="utf-8",
        )
        manifest: dict[str, Any] = {
            "protocol": PROTOCOL,
            "dataset_text_included": False,
            "models": 3,
            "choice_to_label": {
                "A": "SUPPORTS",
                "B": "REFUTES",
                "C": "NOT ENOUGH INFO",
            },
            "quartet_interaction_formula": (
                "(m0-m1-m2+m3)/2; "
                "m=logp(SUPPORTS)-logp(negative_label)"
            ),
            "records": len(all_rows),
            "conditions": condition_counts,
            "split": common_identity[0],
            "split_sha256": common_identity[1],
            "prompt_contract_sha256": common_identity[2],
            "records_jsonl_sha256": _file_sha256(records_path),
            "source_sha256": source_hashes,
            "reconstructable_estimands": [
                "quartet_interaction_I",
                "within_model_route_contrasts",
                "all_pairwise_reference_standardized_route_effects",
                "all_pairwise_directional_common_language_effects",
            ],
        }
        manifest["manifest_sha256"] = _record_hash(manifest, "manifest_sha256")
        manifest_path = temporary / "analysis_records_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        sidecar = temporary / "analysis_records_manifest.json.sha256"
        sidecar.write_text(
            f"{_file_sha256(manifest_path)}  {manifest_path.name}\n",
            encoding="utf-8",
        )
        _install_idempotently(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "records": destination / "analysis_records.jsonl",
        "manifest": destination / "analysis_records_manifest.json",
        "detached_sha256": destination / "analysis_records_manifest.json.sha256",
    }
