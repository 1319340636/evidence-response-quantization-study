import hashlib
import json

import pytest

from qer_fv.confirmatory_release import write_confirmatory_text_free


def _source(tmp_path, name, precision, *, corrupt=False):
    root = tmp_path / name
    root.mkdir()
    rows = []
    for index in range(4):
        rows.append({
            "owner_key": f"full:case-1:cell:{index}",
            "control": "full",
            "metadata": {
                "case_id": "case-1", "page": "page-1", "cell_index": index,
                "claim_index": index // 2, "evidence_index": index % 2,
                "negative_label": "REFUTES", "gold_label": "SUPPORTS",
                "claim": "PRIVATE CLAIM", "evidence": "PRIVATE EVIDENCE",
            },
            "hard_status": "ok", "hard_payload": {"scored_label": "SUPPORTS"},
            "probability_status": "ok",
            "probability_payload": {
                "choice_logprobs": {"A": -1.0 - index, "B": -2.0, "C": -3.0},
                "token_ids": {"A": 32, "B": 33, "C": 34},
                "direct_logit_method": "direct-v1", "prompt_sha256": f"{index+1:064x}",
            },
        })
    records = root / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "run_identity": {
            "model_key": "qwen35_9b", "quantization": precision,
            "split": "confirmatory", "split_sha256": "b" * 64,
            "prompt_contract_sha256": "c" * 64,
        },
        "units_registered": 4, "full_registered": 4,
        "hard_completed": 4, "probability_completed": 4,
        "hard_failures": 0, "probability_failures": 0,
        "records_jsonl_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
    }
    if corrupt:
        manifest["records_jsonl_sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_confirmatory_release_is_paired_text_free_and_source_bound(tmp_path):
    exports = {
        "f16": _source(tmp_path, "f16", "F16"),
        "q4": _source(tmp_path, "q4", "Q4_K_M"),
    }
    paths = write_confirmatory_text_free(exports, tmp_path / "release", formal=False)
    content = paths["records"].read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines()]
    assert len(rows) == 2
    assert {row["precision"] for row in rows} == {"F16", "Q4_K_M"}
    assert "PRIVATE CLAIM" not in content and "PRIVATE EVIDENCE" not in content
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["conditions"] == {"f16": 1, "q4": 1}
    assert manifest["records_jsonl_sha256"] == hashlib.sha256(paths["records"].read_bytes()).hexdigest()


def test_confirmatory_release_rejects_source_hash_mismatch(tmp_path):
    exports = {
        "f16": _source(tmp_path, "f16", "F16", corrupt=True),
        "q4": _source(tmp_path, "q4", "Q4_K_M"),
    }
    with pytest.raises(ValueError, match="hash"):
        write_confirmatory_text_free(exports, tmp_path / "release", formal=False)
    assert not (tmp_path / "release").exists()
