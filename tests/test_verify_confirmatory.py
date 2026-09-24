import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _verifier():
    spec = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts/verify_release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmatory_verifier_recalculates_frozen_estimate(tmp_path):
    shutil.copytree(ROOT / "data/confirmatory", tmp_path / "data/confirmatory")
    shutil.copytree(ROOT / "results/confirmatory", tmp_path / "results/confirmatory")
    verifier = _verifier()
    verifier.verify_confirmatory(tmp_path)
    report = tmp_path / "results/confirmatory/primary_results.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["primary"]["estimate"] += 0.01
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmatory.*estimate"):
        verifier.verify_confirmatory(tmp_path)


def test_release_file_inventory_ignores_test_cache(tmp_path):
    verifier = _verifier()
    verifier.ROOT = tmp_path
    verifier.MANIFEST = tmp_path / "SHA256SUMS.txt"
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "nodeids").write_text("generated", encoding="utf-8")
    (tmp_path / "README.md").write_text("public", encoding="utf-8")
    assert set(verifier.release_files()) == {"README.md"}
