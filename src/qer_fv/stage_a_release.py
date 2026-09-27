"""Text-free, source-bound Stage A four-model GGUF release."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .major_revision_records import (
    _canonical_json, _contains_dataset_text_key, _derive_quartets, _file_sha256,
    _install_idempotently, _read_object, _record_hash, _text,
)


PROTOCOL = "vitaminc-stage-a-text-free-v1-20260927"
CONDITIONS = {
    "gemma_f16": ("gemma4_e4b", "F16"),
    "gemma_q4": ("gemma4_e4b", "Q4_K_M"),
    "ministral_f16": ("ministral3_8b", "F16"),
    "ministral_q4": ("ministral3_8b", "Q4_K_M"),
    "olmo_f16": ("olmo3_7b", "F16"),
    "olmo_q4": ("olmo3_7b", "Q4_K_M"),
    "qwen_f16": ("qwen35_9b", "F16"),
    "qwen_q4": ("qwen35_9b", "Q4_K_M"),
}


def write_stage_a_text_free(
    exports: Mapping[str, str | Path],
    subset_ids_path: str | Path,
    output_directory: str | Path,
    *,
    formal: bool = True,
) -> dict[str, Path]:
    """Export the frozen common 1,200 quartets without original claim/evidence text."""

    if set(exports) != set(CONDITIONS):
        raise ValueError("Stage A release requires exactly eight GGUF conditions")
    ids_path = Path(subset_ids_path)
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    expected = 1200 if formal else 1
    if len(ids) != expected or len(set(ids)) != expected or any(not item for item in ids):
        raise ValueError("Stage A subset identity count mismatch")
    selected = set(ids)
    all_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {"subset_ids": _file_sha256(ids_path)}
    split_hashes: dict[str, str] = {}
    prompt_hash: str | None = None
    reference_metadata: dict[str, tuple[str, str]] | None = None

    for condition, (model, precision) in CONDITIONS.items():
        root = Path(exports[condition])
        if any(part.casefold() == "recovery" for part in root.parts):
            raise ValueError("recovery sources are forbidden")
        manifest_path, records_path = root / "manifest.json", root / "records.jsonl"
        manifest = _read_object(manifest_path)
        records_hash = _file_sha256(records_path)
        if manifest.get("records_jsonl_sha256") != records_hash:
            raise ValueError(f"{condition} source hash mismatch")
        source_hashes[f"{condition}_manifest"] = _file_sha256(manifest_path)
        source_hashes[f"{condition}_records"] = records_hash
        identity = manifest.get("run_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{condition} run identity missing")
        split = "confirmatory" if model == "qwen35_9b" else "dose_subset"
        if (identity.get("model_key") != model
            or identity.get("quantization") != precision
            or identity.get("split") != split):
            raise ValueError(f"{condition} model, precision, or split mismatch")
        split_hash = _text(identity.get("split_sha256"), "split SHA-256")
        if split in split_hashes and split_hashes[split] != split_hash:
            raise ValueError(f"{condition} split hash mismatch")
        split_hashes[split] = split_hash
        current_prompt = _text(identity.get("prompt_contract_sha256"), "prompt SHA-256")
        if prompt_hash is not None and prompt_hash != current_prompt:
            raise ValueError("Stage A prompt contract mismatch")
        prompt_hash = current_prompt
        if (manifest.get("hard_failures") != 0
            or manifest.get("probability_failures") != 0):
            raise ValueError(f"{condition} source contains errors")
        registered = 0
        full_count = 0
        selected_rows = []
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                registered += 1
                if row.get("control") == "full":
                    full_count += 1
                    if (isinstance(row.get("metadata"), Mapping)
                        and row["metadata"].get("case_id") in selected):
                        selected_rows.append(row)
        if registered != manifest.get("units_registered"):
            raise ValueError(f"{condition} registered owner count mismatch")
        expected_full = (9932 if formal else 8) if model == "qwen35_9b" else expected * 4
        if full_count != expected_full or manifest.get("full_registered") != expected_full:
            raise ValueError(f"{condition} full owner count mismatch")
        if len(selected_rows) != expected * 4:
            raise ValueError(f"{condition} selected owner count mismatch")
        owner_keys = [_text(row.get("owner_key"), "owner key") for row in selected_rows]
        if len(set(owner_keys)) != len(owner_keys):
            raise ValueError(f"{condition} duplicate owner identity")
        derived = _derive_quartets(condition, model, precision, selected_rows)
        if len(derived) != expected or {row["case_id"] for row in derived} != selected:
            raise ValueError(f"{condition} quartet membership mismatch")
        metadata = {row["case_id"]: (row["page"], row["negative_label"]) for row in derived}
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError("Stage A cross-model quartet metadata mismatch")
        if _contains_dataset_text_key(derived):
            raise ValueError("Stage A derived records contain original dataset text")
        all_rows.extend(sorted(derived, key=lambda row: row["case_id"]))

    assert reference_metadata is not None and prompt_hash is not None
    pages = len({page for page, _ in reference_metadata.values()})
    if formal and pages != 1078:
        raise ValueError("Stage A common page count mismatch")
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".stage-a-text-free-", dir=output.parent))
    try:
        records_path = temporary / "analysis_records.jsonl"
        records_path.write_text(
            "".join(_canonical_json(row) + "\n" for row in all_rows), encoding="utf-8",
        )
        manifest: dict[str, Any] = {
            "protocol": PROTOCOL, "dataset_text_included": False,
            "conditions": {name: expected for name in CONDITIONS},
            "paired_quartets": expected, "paired_pages": pages,
            "records": len(all_rows), "split_sha256": split_hashes,
            "prompt_contract_sha256": prompt_hash,
            "records_jsonl_sha256": _file_sha256(records_path),
            "source_sha256": source_hashes,
            "quartet_interaction_formula":
                "(m0-m1-m2+m3)/2; m=logp(SUPPORTS)-logp(negative_label)",
        }
        manifest["manifest_sha256"] = _record_hash(manifest, "manifest_sha256")
        manifest_path = temporary / "analysis_records_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
