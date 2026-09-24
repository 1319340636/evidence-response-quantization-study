import hashlib
import json

import pytest

from qer_fv.cub_release import write_cub_text_free


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source(tmp_path, name, precision):
    root = tmp_path / name
    root.mkdir()
    context = {
        "sample_id": "sample-1", "query_identity_sha256": "e" * 64,
        "hard_status": "ok", "probability_status": "ok",
        "hard_payload": {
            "analysis_context_type": "gold", "bcu": precision == "F16",
            "context_identity_sha256": "1" * 64, "target_new": "False",
        },
        "probability_payload": {
            "analysis_context_type": "gold", "ccu": 0.5 if precision == "F16" else 0.25,
            "ccu_target_label": "False", "context_identity_sha256": "1" * 64,
            "target_new": "False",
        },
    }
    contexts = root / "contexts.jsonl"
    queries = root / "queries.jsonl"
    contexts.write_text(json.dumps(context) + "\n", encoding="utf-8")
    queries.write_text(json.dumps({"hard_payload": {"claim": "PRIVATE CLAIM"}}) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "cub-store-v4-20260715",
        "contexts_registered": 1,
        "contexts_jsonl_sha256": hashlib.sha256(contexts.read_bytes()).hexdigest(),
        "queries_jsonl_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
        "run_identity": {
            "protocol_version": "cub-paired-run-v4-20260715",
            "model_key": "qwen35_4b", "quantization": precision,
            "audit_certificate_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "prompt_contract_sha256": "c" * 64,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_cub_release_exports_paired_numeric_fields_without_claims(tmp_path):
    f16 = _source(tmp_path, "f16", "F16")
    q4 = _source(tmp_path, "q4", "Q4_K_M")
    paths = write_cub_text_free({"qwen35_4b_q4": (f16, q4)}, tmp_path / "release", formal=False)
    content = paths["records"].read_text(encoding="utf-8")
    row = json.loads(content)
    assert row["comparison"] == "qwen35_4b_q4"
    assert row["bcu_f16"] is True and row["bcu_quantized"] is False
    assert row["ccu_f16"] == 0.5 and row["ccu_quantized"] == 0.25
    assert "PRIVATE CLAIM" not in content and "claim" not in row
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["comparisons"] == {"qwen35_4b_q4": 1}


def test_cub_release_rejects_source_hash_drift(tmp_path):
    f16 = _source(tmp_path, "f16", "F16")
    q4 = _source(tmp_path, "q4", "Q4_K_M")
    (q4 / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        write_cub_text_free({"qwen35_4b_q4": (f16, q4)}, tmp_path / "release", formal=False)
