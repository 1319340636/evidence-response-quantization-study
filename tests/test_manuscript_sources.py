import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _verifier():
    spec = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts/verify_release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    source = tmp_path / "reports/example/results.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({
        "analysis": {
            "effect": 0.25, "interval": [0.1, 0.4],
            "hard_labels": {"units": 100, "agreement": 0.91},
        },
        "rows": [{"row_id": "abc", "metrics": {"effect": 0.5}}],
    }), encoding="utf-8")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    table = tmp_path / "manuscript_source/tables/example.json"
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps({"cells": [{
        "source_path": "reports/example/results.json",
        "source_sha256": sha,
        "source_key": "rows[row_id=abc].metrics.effect",
        "value": 0.5,
    }]}), encoding="utf-8")
    figure = tmp_path / "manuscript_source/figures/example.json"
    figure.parent.mkdir(parents=True)
    figure.write_text(json.dumps({"marks": [{
        "source_path": "reports/example/results.json",
        "source_sha256": sha,
        "source_key": "/analysis/effect",
        "value": 0.25,
        "lower_source_key": "/analysis/interval/0",
        "lower": 0.1,
        "upper_source_key": "/analysis/interval/1",
        "upper": 0.4,
    }]}), encoding="utf-8")
    return source, table


def test_manuscript_source_links_check_hash_and_source_values(tmp_path):
    source, table = _fixture(tmp_path)
    verifier = _verifier()
    assert verifier.verify_manuscript_source_links(tmp_path) == {
        "source_references": 2,
        "distinct_sources": 1,
        "value_links": 4,
    }
    assert verifier.resolve_json_source_key(
        json.loads(source.read_text(encoding="utf-8")),
        "/analysis/hard_labels/units,/agreement",
    ) == 9
    payload = json.loads(table.read_text(encoding="utf-8"))
    payload["cells"][0]["value"] = 0.6
    table.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source value mismatch"):
        verifier.verify_manuscript_source_links(tmp_path)
    payload["cells"][0]["value"] = 0.5
    table.write_text(json.dumps(payload), encoding="utf-8")
    source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        verifier.verify_manuscript_source_links(tmp_path)


def test_static_gptq_route_source_key_is_resolved_without_executing_code():
    verifier = _verifier()
    path = ROOT / "src/qer_fv/gptq_route.py"
    assert verifier.resolve_code_source_key(
        path, "GPTQRouteConfig.olmo3_7b.group_size"
    ) == 128


def test_mapping_figure_and_table_trace_to_frozen_report(tmp_path):
    verifier = _verifier()
    for relative in (
        "results/mapping/scale_sensitivity.json",
        "manuscript_source/figures/fig_balanced_mapping.csv",
        "manuscript_source/tables/tab_balanced_mapping.tex",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    assert verifier.verify_mapping_display_sources(tmp_path) == {
        "figure_rows": 36,
        "table_cells": 18,
    }
    csv_path = tmp_path / "manuscript_source/figures/fig_balanced_mapping.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace(
            "0.486358033146514", "0.586358033146514", 1
        ), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mapping figure value mismatch"):
        verifier.verify_mapping_display_sources(tmp_path)


def test_stage_a_four_model_point_estimates_recompute_from_safe_records():
    verifier = _verifier()
    assert verifier.verify_stage_a(ROOT) == {"conditions": 8, "quartets": 1200, "pages": 1078}
