import hashlib
import json

import pytest

from qer_fv.tabfact_release import write_tabfact_text_free


CONDITIONS = {
    "qwen_f16": ("qwen35_9b", "F16"),
    "qwen_q4": ("qwen35_9b", "Q4_K_M"),
    "gemma_f16": ("gemma4_e4b", "F16"),
    "gemma_q4": ("gemma4_e4b", "Q4_K_M"),
}


def _source(tmp_path, name, model, precision):
    root = tmp_path / name
    root.mkdir()
    row = {
        "owner_key": "full:sample-1", "hard_status": "ok", "probability_status": "ok",
        "metadata": {
            "sample_id": "sample-1", "table_id": "table-1", "claim_index": 0,
            "gold_label": "Entailed", "table_rows": "PRIVATE TABLE",
            "claim": "PRIVATE CLAIM", "table_truncated": False,
        },
        "hard_payload": {"scored_label": "Entailed", "raw_response": "PRIVATE PROMPT"},
        "probability_payload": {
            "choice_logprobs": {"A": -1.0, "B": -2.0},
            "token_ids": {"A": 32, "B": 33},
            "direct_logit_method": "direct-v1", "prompt_sha256": "a" * 64,
            "raw_response": "PRIVATE PROMPT",
        },
    }
    records = root / "records.jsonl"
    records.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "run_identity": {
            "model_key": model, "quantization": precision,
            "expected_units": 1, "split_sha256": "b" * 64,
            "evidence_sha256": "c" * 64, "prompt_contract_sha256": "d" * 64,
        },
        "owners": 1, "hard_completed": 1, "probability_completed": 1,
        "hard_failures": 0, "probability_failures": 0,
        "records_jsonl_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _exports(tmp_path):
    return {name: _source(tmp_path, name, model, precision)
            for name, (model, precision) in CONDITIONS.items()}


def test_tabfact_release_strips_source_text_and_pairs_conditions(tmp_path):
    paths = write_tabfact_text_free(_exports(tmp_path), tmp_path / "release", formal=False)
    content = paths["records"].read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines()]
    assert len(rows) == 4
    assert all(row["sample_id"] == "sample-1" for row in rows)
    assert all(set(row["choice_logprobs"]) == {"A", "B"} for row in rows)
    assert all(value not in content for value in ("PRIVATE TABLE", "PRIVATE CLAIM", "PRIVATE PROMPT"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["records"] == 4
    assert manifest["conditions"] == {name: 1 for name in CONDITIONS}


def test_tabfact_release_fails_closed_on_hash_drift(tmp_path):
    exports = _exports(tmp_path)
    records = exports["qwen_f16"] / "records.jsonl"
    records.write_text(records.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        write_tabfact_text_free(exports, tmp_path / "release", formal=False)
