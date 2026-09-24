"""Whitelisted text-free release of the four Fresh TabFact formal conditions."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .major_revision_records import (
    _canonical_json, _file_sha256, _finite, _hash_text, _index,
    _install_idempotently, _read_jsonl, _read_object, _record_hash, _text,
)


CONDITIONS = {
    "qwen_f16": ("qwen35_9b", "F16"),
    "qwen_q4": ("qwen35_9b", "Q4_K_M"),
    "gemma_f16": ("gemma4_e4b", "F16"),
    "gemma_q4": ("gemma4_e4b", "Q4_K_M"),
}
PROTOCOL = "fresh-tabfact-text-free-v1-20260924"


def write_tabfact_text_free(
    exports: Mapping[str, str | Path],
    output_directory: str | Path,
    *,
    formal: bool = True,
) -> dict[str, Path]:
    """Bind source hashes and release paired classification fields, never source text."""

    if set(exports) != set(CONDITIONS):
        raise ValueError("TabFact release requires exactly four conditions")
    expected = 2000 if formal else 1
    all_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    shared_identity: tuple[str, str, str] | None = None
    reference_metadata: dict[str, tuple[str, int, str]] | None = None
    for condition, (model, precision) in CONDITIONS.items():
        root = Path(exports[condition])
        if any(part.casefold() == "recovery" for part in root.parts):
            raise ValueError("recovery source forbidden")
        manifest_path = root / "manifest.json"
        records_path = root / "records.jsonl"
        manifest = _read_object(manifest_path)
        source_hash = _file_sha256(records_path)
        if source_hash != manifest.get("records_jsonl_sha256"):
            raise ValueError(f"{condition} source records hash mismatch")
        source_hashes[f"{condition}_manifest"] = _file_sha256(manifest_path)
        source_hashes[f"{condition}_records"] = source_hash
        identity = manifest.get("run_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{condition} run identity missing")
        if identity.get("model_key") != model or identity.get("quantization") != precision:
            raise ValueError(f"{condition} model or precision mismatch")
        current_identity = tuple(str(identity.get(key, "")) for key in (
            "split_sha256", "evidence_sha256", "prompt_contract_sha256"
        ))
        if not all(current_identity):
            raise ValueError(f"{condition} source identity incomplete")
        if shared_identity is None:
            shared_identity = current_identity
        elif shared_identity != current_identity:
            raise ValueError("TabFact split, evidence, or prompt identity mismatch")
        if (manifest.get("owners") != expected
            or manifest.get("hard_completed") != expected
            or manifest.get("probability_completed") != expected
            or manifest.get("hard_failures") != 0
            or manifest.get("probability_failures") != 0):
            raise ValueError(f"{condition} source structural gate failed")
        raw_rows = _read_jsonl(records_path)
        if len(raw_rows) != expected:
            raise ValueError(f"{condition} source row count mismatch")
        derived: dict[str, dict[str, Any]] = {}
        metadata_by_id: dict[str, tuple[str, int, str]] = {}
        for raw in raw_rows:
            if raw.get("hard_status") != "ok" or raw.get("probability_status") != "ok":
                raise ValueError(f"{condition} unsuccessful row")
            metadata = raw.get("metadata")
            hard = raw.get("hard_payload")
            probability = raw.get("probability_payload")
            if not all(isinstance(x, Mapping) for x in (metadata, hard, probability)):
                raise ValueError(f"{condition} missing payload")
            sample_id = _text(metadata.get("sample_id"), "sample identity")
            if sample_id in derived:
                raise ValueError(f"{condition} duplicate sample")
            gold = _text(metadata.get("gold_label"), "gold label")
            predicted = _text(hard.get("scored_label"), "hard scored label")
            if gold not in {"Entailed", "Refuted"} or predicted not in {"Entailed", "Refuted"}:
                raise ValueError(f"{condition} unsupported label")
            scores = probability.get("choice_logprobs")
            tokens = probability.get("token_ids")
            if not isinstance(scores, Mapping) or set(scores) != {"A", "B"}:
                raise ValueError(f"{condition} choice scores incomplete")
            if not isinstance(tokens, Mapping) or set(tokens) != {"A", "B"}:
                raise ValueError(f"{condition} token IDs incomplete")
            normalized_tokens = {}
            for choice in ("A", "B"):
                value = tokens[choice]
                if type(value) is not int or value < 0:
                    raise ValueError(f"{condition} token ID invalid")
                normalized_tokens[choice] = value
            table_id = _text(metadata.get("table_id"), "table identity")
            claim_index = _index(metadata.get("claim_index"), "claim index")
            truncated = metadata.get("table_truncated")
            if type(truncated) is not bool:
                raise ValueError(f"{condition} truncation flag invalid")
            metadata_by_id[sample_id] = (table_id, claim_index, gold)
            derived[sample_id] = {
                "condition": condition, "model_key": model, "precision": precision,
                "sample_id": sample_id, "table_id": table_id,
                "claim_index": claim_index, "gold_label": gold,
                "hard_scored_label": predicted, "table_truncated": truncated,
                "choice_logprobs": {c: _finite(scores[c], "choice score") for c in ("A", "B")},
                "token_ids": normalized_tokens,
                "direct_logit_method": _text(probability.get("direct_logit_method"), "direct method"),
                "prompt_sha256": _hash_text(probability.get("prompt_sha256"), "prompt hash"),
            }
        if reference_metadata is None:
            reference_metadata = metadata_by_id
        elif metadata_by_id != reference_metadata:
            raise ValueError("TabFact four-condition sample membership mismatch")
        all_rows.extend(derived[key] for key in sorted(derived))

    assert shared_identity is not None
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tabfact-text-free-", dir=output.parent))
    try:
        records_path = temporary / "analysis_records.jsonl"
        records_path.write_text(
            "".join(_canonical_json(row) + "\n" for row in all_rows), encoding="utf-8"
        )
        manifest: dict[str, Any] = {
            "protocol": PROTOCOL, "dataset_text_included": False,
            "conditions": {name: expected for name in CONDITIONS},
            "paired_samples": expected, "records": len(all_rows),
            "split_sha256": shared_identity[0],
            "evidence_sha256": shared_identity[1],
            "prompt_contract_sha256": shared_identity[2],
            "records_jsonl_sha256": _file_sha256(records_path),
            "source_sha256": source_hashes,
        }
        manifest["manifest_sha256"] = _record_hash(manifest, "manifest_sha256")
        manifest_path = temporary / "analysis_records_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        sidecar = temporary / "analysis_records_manifest.json.sha256"
        sidecar.write_text(f"{_file_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8")
        _install_idempotently(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "records": output / "analysis_records.jsonl",
        "manifest": output / "analysis_records_manifest.json",
        "detached_sha256": output / "analysis_records_manifest.json.sha256",
    }
