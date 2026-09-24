"""Text-free paired release of the frozen VitaminC GGUF confirmatory comparison."""

from __future__ import annotations

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
    _text,
)


CONDITIONS = {"f16": "F16", "q4": "Q4_K_M"}
PROTOCOL = "vitaminc-confirmatory-text-free-v1-20260924"


def write_confirmatory_text_free(
    exports: Mapping[str, str | Path],
    output_directory: str | Path,
    *,
    formal: bool = True,
) -> dict[str, Path]:
    """Verify both source exports and write only paired, analysis-level fields."""

    if set(exports) != set(CONDITIONS):
        raise ValueError("confirmatory release requires exactly F16 and Q4 exports")
    expected_quartets = 2483 if formal else 1
    expected_full = expected_quartets * 4
    all_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    shared_identity: tuple[str, str, str] | None = None
    paired_metadata: dict[str, tuple[str, str]] | None = None

    for condition, precision in CONDITIONS.items():
        root = Path(exports[condition])
        if any(part.casefold() == "recovery" for part in root.parts):
            raise ValueError("recovery sources are forbidden")
        manifest_path = root / "manifest.json"
        records_path = root / "records.jsonl"
        manifest = _read_object(manifest_path)
        source_hash = _file_sha256(records_path)
        if manifest.get("records_jsonl_sha256") != source_hash:
            raise ValueError(f"{condition} source records hash mismatch")
        source_hashes[f"{condition}_records"] = source_hash
        source_hashes[f"{condition}_manifest"] = _file_sha256(manifest_path)
        identity = manifest.get("run_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{condition} run identity missing")
        if identity.get("model_key") != "qwen35_9b" or identity.get("quantization") != precision:
            raise ValueError(f"{condition} model or precision identity mismatch")
        current_identity = (
            str(identity.get("split", "")),
            str(identity.get("split_sha256", "")),
            str(identity.get("prompt_contract_sha256", "")),
        )
        if current_identity[0] != "confirmatory" or not all(current_identity):
            raise ValueError(f"{condition} source split identity incomplete")
        if shared_identity is None:
            shared_identity = current_identity
        elif current_identity != shared_identity:
            raise ValueError("F16/Q4 split or prompt identity mismatch")
        if manifest.get("hard_failures") != 0 or manifest.get("probability_failures") != 0:
            raise ValueError(f"{condition} source contains errors")
        raw_rows = _read_jsonl(records_path)
        if len(raw_rows) != manifest.get("units_registered"):
            raise ValueError(f"{condition} registered unit count mismatch")
        full_rows = [row for row in raw_rows if row.get("control") == "full"]
        if len(full_rows) != expected_full or manifest.get("full_registered") != expected_full:
            raise ValueError(f"{condition} full-evidence count mismatch")
        owner_keys = [_text(row.get("owner_key"), "owner key") for row in full_rows]
        if len(owner_keys) != len(set(owner_keys)):
            raise ValueError(f"{condition} duplicate owner identities")
        derived = _derive_quartets(condition, "qwen35_9b", precision, full_rows)
        if len(derived) != expected_quartets:
            raise ValueError(f"{condition} quartet count mismatch")
        metadata = {row["case_id"]: (row["page"], row["negative_label"]) for row in derived}
        if paired_metadata is None:
            paired_metadata = metadata
        elif metadata != paired_metadata:
            raise ValueError("F16/Q4 quartet membership mismatch")
        if _contains_dataset_text_key(derived):
            raise ValueError("derived export contains dataset text")
        all_rows.extend(derived)

    assert shared_identity is not None and paired_metadata is not None
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".confirmatory-text-free-", dir=output.parent))
    try:
        records_path = temporary / "analysis_records.jsonl"
        records_path.write_text(
            "".join(_canonical_json(row) + "\n" for row in all_rows),
            encoding="utf-8",
        )
        manifest: dict[str, Any] = {
            "protocol": PROTOCOL,
            "dataset_text_included": False,
            "model_key": "qwen35_9b",
            "conditions": {name: expected_quartets for name in CONDITIONS},
            "paired_quartets": expected_quartets,
            "paired_pages": len({page for page, _ in paired_metadata.values()}),
            "records": len(all_rows),
            "split": shared_identity[0],
            "split_sha256": shared_identity[1],
            "prompt_contract_sha256": shared_identity[2],
            "records_jsonl_sha256": _file_sha256(records_path),
            "source_sha256": source_hashes,
            "quartet_interaction_formula": (
                "(m0-m1-m2+m3)/2; m=logp(SUPPORTS)-logp(negative_label)"
            ),
        }
        manifest["manifest_sha256"] = _record_hash(manifest, "manifest_sha256")
        output_manifest = temporary / "analysis_records_manifest.json"
        output_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        sidecar = temporary / "analysis_records_manifest.json.sha256"
        sidecar.write_text(
            f"{_file_sha256(output_manifest)}  {output_manifest.name}\n",
            encoding="utf-8",
        )
        _install_idempotently(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "records": output / "analysis_records.jsonl",
        "manifest": output / "analysis_records_manifest.json",
        "detached_sha256": output / "analysis_records_manifest.json.sha256",
    }
