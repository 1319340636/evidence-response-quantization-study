"""Text-free paired CUB/DRUID diagnostic records from frozen condition exports."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .cub_results import CubResultsError, load_cub_condition_export, pair_cub_conditions
from .major_revision_records import (
    _canonical_json, _file_sha256, _install_idempotently, _record_hash,
)


COMPARISONS = {
    "ministral3_3b_q4": ("ministral3_3b", "Q4_K_M"),
    "ministral3_8b_q4": ("ministral3_8b", "Q4_K_M"),
    "qwen35_4b_q4": ("qwen35_4b", "Q4_K_M"),
    "qwen35_9b_q4": ("qwen35_9b", "Q4_K_M"),
    "qwen35_9b_q5": ("qwen35_9b", "Q5_K_M"),
    "qwen35_9b_q8": ("qwen35_9b", "Q8_0"),
}
PROTOCOL = "cub-druid-paired-text-free-v1-20260924"


def write_cub_text_free(
    pairs: Mapping[str, tuple[str | Path, str | Path]],
    output_directory: str | Path,
    *,
    formal: bool = True,
) -> dict[str, Path]:
    """Validate hashed raw sources but release only paired diagnostic scalars."""

    if formal and set(pairs) != set(COMPARISONS):
        raise ValueError("formal CUB release requires six frozen comparisons")
    if not pairs or any(name not in COMPARISONS for name in pairs):
        raise ValueError("unsupported CUB comparison")
    all_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for comparison in sorted(pairs):
        f16_root, quant_root = (Path(path) for path in pairs[comparison])
        if any(part.casefold() == "recovery" for root in (f16_root, quant_root) for part in root.parts):
            raise ValueError("recovery source forbidden")
        try:
            f16 = load_cub_condition_export(f16_root)
            quantized = load_cub_condition_export(quant_root)
            observations = pair_cub_conditions(f16, quantized)
        except CubResultsError as error:
            raise ValueError(f"{comparison} source hash or pair validation failed: {error}") from error
        model, precision = COMPARISONS[comparison]
        if (f16.model_key, quantized.model_key) != (model, model):
            raise ValueError(f"{comparison} model identity mismatch")
        if f16.quantization != "F16" or quantized.quantization != precision:
            raise ValueError(f"{comparison} quantization identity mismatch")
        if formal and len(observations) != 4302:
            raise ValueError(f"{comparison} sample count mismatch")
        counts[comparison] = len(observations)
        for role, root in (("f16", f16_root), ("quantized", quant_root)):
            for name in ("manifest.json", "contexts.jsonl", "queries.jsonl"):
                source_hashes[f"{comparison}_{role}_{name}"] = _file_sha256(root / name)
        for observation in observations:
            record = asdict(observation)
            if record["analysis_context_type"] not in {"gold", "conflicting", "irrelevant"}:
                raise ValueError(f"{comparison} unexpected context type")
            all_rows.append({
                "comparison": comparison,
                "model_key": model,
                "quantization": precision,
                **record,
            })

    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cub-text-free-", dir=output.parent))
    try:
        records_path = temporary / "analysis_records.jsonl"
        records_path.write_text(
            "".join(_canonical_json(row) + "\n" for row in all_rows), encoding="utf-8"
        )
        manifest: dict[str, Any] = {
            "protocol": PROTOCOL,
            "dataset_text_included": False,
            "comparisons": counts,
            "records": len(all_rows),
            "records_jsonl_sha256": _file_sha256(records_path),
            "source_sha256": source_hashes,
            "reconstructable_estimands": ["BCU", "CCU", "query-cluster effects"],
        }
        manifest["manifest_sha256"] = _record_hash(manifest, "manifest_sha256")
        manifest_path = temporary / "analysis_records_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        sidecar = temporary / "analysis_records_manifest.json.sha256"
        sidecar.write_text(
            f"{_file_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
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
