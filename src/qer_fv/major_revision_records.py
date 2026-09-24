"""Text-free analysis-level release for the CI major revision."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


PROTOCOL = "ci-major-revision-analysis-records-v1-20260829"
CONDITIONS = {
    "qwen_fp16": ("qwen35_9b", "FP16"),
    "qwen_gptq": ("qwen35_9b", "GPTQ_INT4"),
    "qwen_awq": ("qwen35_9b", "AWQ_INT4"),
    "ministral_fp16": ("ministral3_8b", "FP16"),
    "ministral_gptq": ("ministral3_8b", "GPTQ_INT4"),
    "ministral_awq": ("ministral3_8b", "AWQ_INT4"),
}


def load_text_free_scale_inputs(
    output_directory: str | Path,
    *,
    formal: bool = True,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, str],
]:
    """Verify a text-free release and reconstruct the six scale-analysis inputs."""

    root = Path(output_directory)
    records_path = root / "analysis_records.jsonl"
    manifest_path = root / "analysis_records_manifest.json"
    sidecar_path = root / "analysis_records_manifest.json.sha256"
    manifest = _read_object(manifest_path)
    try:
        sidecar_parts = sidecar_path.read_text(encoding="utf-8").split()
    except OSError as error:
        raise ValueError("analysis-record detached hash is missing") from error
    if (
        sidecar_parts != [_file_sha256(manifest_path), manifest_path.name]
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("manifest_sha256")
        != _record_hash(manifest, "manifest_sha256")
    ):
        raise ValueError("analysis-record manifest integrity mismatch")
    if (
        manifest.get("dataset_text_included") is not False
        or manifest.get("records_jsonl_sha256") != _file_sha256(records_path)
    ):
        raise ValueError("analysis-record release integrity mismatch")

    expected_quartets = 1200 if formal else 1
    expected_counts = {condition: expected_quartets for condition in CONDITIONS}
    if manifest.get("conditions") != expected_counts:
        raise ValueError("analysis-record condition counts mismatch")
    rows = _read_jsonl(records_path)
    if len(rows) != 6 * expected_quartets or manifest.get("records") != len(rows):
        raise ValueError("analysis-record row count mismatch")

    indexed: dict[str, dict[str, float]] = {condition: {} for condition in CONDITIONS}
    reference_metadata: dict[str, tuple[str, str]] | None = None
    metadata_by_condition: dict[str, dict[str, tuple[str, str]]] = {
        condition: {} for condition in CONDITIONS
    }
    for row in rows:
        if _contains_dataset_text_key(row):
            raise ValueError("analysis-record release contains dataset text")
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ValueError("analysis-record condition identity mismatch")
        model_key, precision = CONDITIONS[str(condition)]
        if row.get("model_key") != model_key or row.get("precision") != precision:
            raise ValueError("analysis-record model or precision identity mismatch")
        case_id = _text(row.get("case_id"), "case identity")
        page = _text(row.get("page"), "page identity")
        negative_label = _text(row.get("negative_label"), "negative label")
        value = _released_quartet_interaction(row, negative_label)
        if case_id in indexed[str(condition)]:
            raise ValueError("analysis-record contains duplicate quartet identities")
        indexed[str(condition)][case_id] = value
        metadata_by_condition[str(condition)][case_id] = (page, negative_label)

    for condition in sorted(CONDITIONS):
        metadata = metadata_by_condition[condition]
        if len(metadata) != expected_quartets:
            raise ValueError("analysis-record quartet membership mismatch")
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError("analysis-record six-condition metadata mismatch")
    assert reference_metadata is not None
    qwen = {route: indexed[f"qwen_{route}"] for route in ("fp16", "gptq", "awq")}
    ministral = {
        route: indexed[f"ministral_{route}"] for route in ("fp16", "gptq", "awq")
    }
    pages = {case_id: page for case_id, (page, _) in reference_metadata.items()}
    return qwen, ministral, pages


def write_text_free_analysis_records(
    *,
    exports: Mapping[str, str | Path],
    output_directory: str | Path,
    formal: bool = True,
) -> dict[str, Path]:
    """Validate six exports and atomically release only analysis fields."""

    if set(exports) != set(CONDITIONS):
        raise ValueError("text-free release requires exactly six conditions")
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
            raise ValueError("six-condition split or prompt identity mismatch")

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
    temporary_parent = Path(output_directory).parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".ci-major-revision-records.", dir=temporary_parent)
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
                "reference_standardized_route_effect",
                "directional_common_language_effect",
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
        _install_idempotently(temporary, Path(output_directory))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    output = Path(output_directory)
    return {
        "records": output / "analysis_records.jsonl",
        "manifest": output / "analysis_records_manifest.json",
        "detached_sha256": output / "analysis_records_manifest.json.sha256",
    }


def _derive_quartets(
    condition: str,
    model_key: str,
    precision: str,
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{condition} metadata is incomplete")
        case_id = _text(metadata.get("case_id"), "case identity")
        grouped.setdefault(case_id, []).append(row)

    output: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        quartet_rows = grouped[case_id]
        if len(quartet_rows) != 4:
            raise ValueError(f"{condition} quartet must contain four full cells")
        pages: set[str] = set()
        negative_labels: set[str] = set()
        cells: dict[int, dict[str, Any]] = {}
        for row in quartet_rows:
            metadata = row["metadata"]
            page = _text(metadata.get("page"), "page identity")
            negative_label = _text(
                metadata.get("negative_label"), "negative label"
            )
            cell = _derive_cell(condition, row)
            cell_index = cell["cell_index"]
            if cell_index in cells:
                raise ValueError(f"{condition} quartet contains duplicate cells")
            pages.add(page)
            negative_labels.add(negative_label)
            cells[cell_index] = cell
        if set(cells) != {0, 1, 2, 3}:
            raise ValueError(f"{condition} quartet cells are incomplete")
        if len(pages) != 1 or len(negative_labels) != 1:
            raise ValueError(f"{condition} quartet metadata is inconsistent")
        negative_label = next(iter(negative_labels))
        negative_choice = {
            "REFUTES": "B",
            "NOT ENOUGH INFO": "C",
        }.get(negative_label)
        if negative_choice is None:
            raise ValueError(f"{condition} negative label is unsupported")
        margins = [
            cells[index]["choice_logprobs"]["A"]
            - cells[index]["choice_logprobs"][negative_choice]
            for index in range(4)
        ]
        output.append({
            "condition": condition,
            "model_key": model_key,
            "precision": precision,
            "case_id": case_id,
            "page": next(iter(pages)),
            "negative_label": negative_label,
            "quartet_interaction": (
                margins[0] - margins[1] - margins[2] + margins[3]
            ) / 2.0,
            "cells": [cells[index] for index in range(4)],
        })
    return output


def _derive_cell(condition: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("hard_status") != "ok" or row.get("probability_status") != "ok":
        raise ValueError(f"{condition} contains an unsuccessful full owner")
    metadata = row.get("metadata")
    probability = row.get("probability_payload")
    hard = row.get("hard_payload")
    if not isinstance(metadata, Mapping) or not isinstance(probability, Mapping):
        raise ValueError(f"{condition} payload is incomplete")
    if not isinstance(hard, Mapping):
        raise ValueError(f"{condition} hard payload is incomplete")
    owner_key = _text(row.get("owner_key"), "owner key")
    choice_logprobs = probability.get("choice_logprobs")
    if not isinstance(choice_logprobs, Mapping) or set(choice_logprobs) != {
        "A",
        "B",
        "C",
    }:
        raise ValueError(f"{condition} choice log-probabilities are incomplete")
    normalized_scores = {
        key: _finite(choice_logprobs[key], "choice log-probability")
        for key in ("A", "B", "C")
    }
    token_ids = probability.get("token_ids")
    if not isinstance(token_ids, Mapping) or set(token_ids) != {"A", "B", "C"}:
        raise ValueError(f"{condition} token identities are incomplete")
    normalized_tokens: dict[str, int] = {}
    for key in ("A", "B", "C"):
        value = token_ids[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"{condition} token identity is invalid")
        normalized_tokens[key] = value
    return {
        "owner_key": owner_key,
        "cell_index": _index(metadata.get("cell_index"), "cell index"),
        "claim_index": _index(metadata.get("claim_index"), "claim index"),
        "evidence_index": _index(metadata.get("evidence_index"), "evidence index"),
        "gold_label": _text(metadata.get("gold_label"), "gold label"),
        "hard_scored_label": _text(hard.get("scored_label"), "hard scored label"),
        "choice_logprobs": normalized_scores,
        "token_ids": normalized_tokens,
        "direct_logit_method": _text(
            probability.get("direct_logit_method"), "direct logit method"
        ),
        "prompt_sha256": _hash_text(
            probability.get("prompt_sha256"), "prompt SHA-256"
        ),
    }


def _released_quartet_interaction(
    row: Mapping[str, Any], negative_label: str
) -> float:
    cells = row.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("analysis-record quartet cells are incomplete")
    by_index: dict[int, Mapping[str, Any]] = {}
    negative_choice = {
        "REFUTES": "B",
        "NOT ENOUGH INFO": "C",
    }.get(negative_label)
    if negative_choice is None:
        raise ValueError("analysis-record negative label is unsupported")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("analysis-record cell is invalid")
        cell_index = _index(cell.get("cell_index"), "cell index")
        if cell_index in by_index:
            raise ValueError("analysis-record quartet contains duplicate cells")
        scores = cell.get("choice_logprobs")
        if not isinstance(scores, Mapping) or set(scores) != {"A", "B", "C"}:
            raise ValueError("analysis-record choice log-probabilities are incomplete")
        for choice in ("A", "B", "C"):
            _finite(scores[choice], "choice log-probability")
        by_index[cell_index] = cell
    if set(by_index) != {0, 1, 2, 3}:
        raise ValueError("analysis-record quartet cells are incomplete")
    margins = [
        float(by_index[index]["choice_logprobs"]["A"])
        - float(by_index[index]["choice_logprobs"][negative_choice])
        for index in range(4)
    ]
    reconstructed = (margins[0] - margins[1] - margins[2] + margins[3]) / 2.0
    recorded = _finite(row.get("quartet_interaction"), "quartet interaction")
    if reconstructed != recorded:
        raise ValueError("analysis-record quartet interaction mismatch")
    return recorded


def _contains_dataset_text_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in {"claim", "evidence", "claim_text", "evidence_text"}:
                return True
            if _contains_dataset_text_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_dataset_text_key(item) for item in value)
    return False


def _install_idempotently(temporary: Path, output: Path) -> None:
    if output.exists():
        names = {path.name for path in temporary.iterdir()}
        if not output.is_dir() or {path.name for path in output.iterdir()} != names:
            raise ValueError("analysis-record output already exists with different files")
        for name in names:
            if (temporary / name).read_bytes() != (output / name).read_bytes():
                raise ValueError("analysis-record output already exists with different content")
        shutil.rmtree(temporary)
        return
    temporary.replace(output)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read analysis-record source: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"analysis-record source must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read analysis-record source: {path}") from error
    return rows


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _hash_text(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise ValueError(f"{label} is invalid")
    return text.lower()


def _index(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _record_hash(value: Mapping[str, Any], excluded: str) -> str:
    return hashlib.sha256(
        _canonical_json({key: item for key, item in value.items() if key != excluded}).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot hash analysis-record source: {path}") from error
    return digest.hexdigest()
