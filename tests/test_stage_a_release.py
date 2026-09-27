import hashlib
import json

import pytest

from qer_fv.stage_a_release import CONDITIONS, write_stage_a_text_free


def _source(tmp_path, condition, model, precision, case_ids, *, corrupt=False):
    root = tmp_path / condition
    root.mkdir()
    rows = []
    for case_id in case_ids:
        for index in range(4):
            rows.append({
                "owner_key": f"full:{case_id}:cell:{index}", "control": "full",
                "metadata": {
                    "case_id": case_id, "page": "page-1", "cell_index": index,
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
            "model_key": model, "quantization": precision,
            "split": "confirmatory" if model == "qwen35_9b" else "dose_subset",
            "split_sha256": ("a" if model == "qwen35_9b" else "b") * 64,
            "prompt_contract_sha256": "c" * 64,
        },
        "units_registered": len(rows), "full_registered": len(rows),
        "hard_completed": len(rows), "probability_completed": len(rows),
        "hard_failures": 0, "probability_failures": 0,
        "records_jsonl_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
    }
    if corrupt:
        manifest["records_jsonl_sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _exports(tmp_path, *, corrupt=None):
    return {
        condition: _source(tmp_path, condition, model, precision,
                        ("case-1", "case-extra") if model == "qwen35_9b" else ("case-1",),
                        corrupt=condition == corrupt)
        for condition, (model, precision) in CONDITIONS.items()
    }


def test_stage_a_release_filters_confirmatory_and_excludes_text(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("case-1\n", encoding="utf-8")
    paths = write_stage_a_text_free(_exports(tmp_path), ids, tmp_path / "release", formal=False)
    content = paths["records"].read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines()]
    assert len(rows) == 8
    assert {row["case_id"] for row in rows} == {"case-1"}
    assert "PRIVATE CLAIM" not in content and "PRIVATE EVIDENCE" not in content
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["paired_quartets"] == 1
    assert manifest["paired_pages"] == 1
    assert manifest["dataset_text_included"] is False


def test_stage_a_release_rejects_corrupt_source(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("case-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        write_stage_a_text_free(_exports(tmp_path, corrupt="olmo_q4"), ids,
                                tmp_path / "release", formal=False)
    assert not (tmp_path / "release").exists()
