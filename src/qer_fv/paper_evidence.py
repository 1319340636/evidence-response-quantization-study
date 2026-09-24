"""Load publication-facing evidence from the frozen result inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from qer_fv.publication_readiness import verify_report_manifest


PAPER_EVIDENCE_PROTOCOL = "paper-evidence-v1-20260728"
PUBLICATION_REPORT = Path("reports/publication_readiness_v1")


def load_verified_publication_package(
    project_root: str | Path,
) -> dict[str, Any]:
    """Load the hashed publication inventory after all validation gates."""
    root = Path(project_root).resolve()
    report = root / PUBLICATION_REPORT
    manifest = verify_report_manifest(report)
    validation = _load_json_object(report / "validation.json")
    inventory_path = report / "inventory.json"
    inventory = _load_json_object(inventory_path)
    if validation.get("status") != "valid":
        raise ValueError("publication validation is not valid")
    checks = validation.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("publication validation check failed")
    return {
        "root": root,
        "report": report,
        "manifest": manifest,
        "validation": validation,
        "inventory": inventory,
        "inventory_sha256": _file_sha256(inventory_path),
    }


def select_unique_row(
    rows: Iterable[Mapping[str, Any]],
    **criteria: object,
) -> dict[str, Any]:
    """Select exactly one inventory row matching all supplied criteria."""
    selected = [
        dict(row)
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected exactly one row for {criteria}; found {len(selected)}"
        )
    return selected[0]


def build_paper_evidence(project_root: str | Path) -> dict[str, Any]:
    """Adapt the verified inventory into explicit manuscript evidence groups."""
    package = load_verified_publication_package(project_root)
    inventory = package["inventory"]
    rows = inventory.get("rows")
    models = inventory.get("models")
    if not isinstance(rows, list) or not isinstance(models, list):
        raise ValueError("publication inventory rows or models are invalid")

    evidence = {
        "evidence_protocol": PAPER_EVIDENCE_PROTOCOL,
        "source_inventory_sha256": package["inventory_sha256"],
        "source_publication_manifest_sha256": package["manifest"].get(
            "manifest_sha256"
        ),
        "canonical_stage_a": inventory.get("canonical_stage_a"),
        "superseded_reports": list(inventory.get("superseded_reports", [])),
        "stage_a_olmo_boundary": dict(
            inventory.get("stage_a_olmo_boundary", {})
        ),
        "models": [dict(model) for model in models],
        "main": select_unique_row(
            rows, source_role="vitaminc_confirmatory_anchor"
        ),
        "stage_a": _rows_for_role(rows, "vitaminc_stage_a"),
        "hf_gptq": sorted(
            _rows_for_roles(rows, {"qwen_hf_gptq", "gemma_hf_gptq"}),
            key=lambda row: row["model_key"],
        ),
        "route_interaction": select_unique_row(
            rows, source_role="route_model_interaction"
        ),
        "tabfact": _rows_for_role(rows, "tabfact_external_validation"),
        "cub": _rows_for_role(rows, "cub_diagnostic"),
        "sensitivity": _rows_for_roles(
            rows, {"mapping_sensitivity", "strengthening_audit"}
        ),
        "awq": _rows_for_role(rows, "hf_awq_d5"),
        "qwen_awq_formal": _load_qwen_awq_formal(package["root"]),
    }
    validate_paper_evidence(evidence)
    return evidence


def validate_paper_evidence(evidence: Mapping[str, Any]) -> None:
    """Fail closed when a publication-facing scientific boundary drifts."""
    if evidence.get("canonical_stage_a") != (
        "reports/vitaminc_stage_a_v4"
    ):
        raise ValueError("Stage A v4 is not canonical")
    if evidence.get("superseded_reports") != [
        "reports/vitaminc_stage_a_v1",
        "reports/vitaminc_stage_a_v2",
        "reports/vitaminc_stage_a_v3",
    ]:
        raise ValueError("superseded Stage A registry mismatch")

    models = evidence.get("models")
    if not isinstance(models, list) or [
        model.get("key") for model in models
    ] != [
        "qwen35_4b",
        "qwen35_9b",
        "ministral3_3b",
        "ministral3_8b",
        "gemma4_e4b",
        "olmo3_7b",
    ]:
        raise ValueError("six-model registry mismatch")

    stage_a = evidence.get("stage_a")
    if not isinstance(stage_a, list) or {
        row.get("model_key") for row in stage_a
    } != {"qwen35_9b", "ministral3_8b", "gemma4_e4b", "olmo3_7b"}:
        raise ValueError("Stage A common-support model registry mismatch")
    olmo = select_unique_row(stage_a, model_key="olmo3_7b")
    if (
        "negative_point_estimate_not_significant_after_holm"
        not in olmo.get("boundaries", [])
        or olmo.get("metrics", {}).get(
            "balanced_accuracy_holm_pvalue", 0.0
        )
        < 0.05
    ):
        raise ValueError("OLMo Holm boundary mismatch")

    awq = evidence.get("awq")
    if not isinstance(awq, list) or {
        row.get("model_key") for row in awq
    } != {"qwen35_9b", "gemma4_e4b"}:
        raise ValueError("AWQ model registry mismatch")
    if any(
        row.get("status") != "not_formally_completed"
        or "not_a_formal_result" not in row.get("boundaries", [])
        for row in awq
    ):
        raise ValueError("AWQ completion boundary mismatch")

    qwen_awq = evidence.get("qwen_awq_formal")
    if (
        not isinstance(qwen_awq, Mapping)
        or qwen_awq.get("source_path")
        != "reports/qwen_hf_triplet_formal_v1/results.json"
        or qwen_awq.get("analysis", {}).get("population")
        != {"quartets": 1200, "pages": 1078}
        or qwen_awq.get("analysis", {}).get("contrast_order")
        != ["awq_minus_fp16", "awq_minus_gptq"]
    ):
        raise ValueError("Qwen AWQ formal evidence mismatch")

    route = evidence.get("route_interaction")
    if not isinstance(route, dict) or (
        "joint_deployment_route_effect"
        not in route.get("boundaries", [])
    ):
        raise ValueError("joint deployment-route boundary mismatch")


def _rows_for_role(
    rows: Iterable[Mapping[str, Any]], role: str
) -> list[dict[str, Any]]:
    return _rows_for_roles(rows, {role})


def _rows_for_roles(
    rows: Iterable[Mapping[str, Any]], roles: set[str]
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows if row.get("source_role") in roles),
        key=lambda row: (
            str(row.get("model_key", "")),
            str(row.get("route", "")),
            str(row.get("scope", "")),
        ),
    )


def _load_qwen_awq_formal(root: Path) -> dict[str, Any]:
    report = root / "reports" / "qwen_hf_triplet_formal_v1"
    manifest = _load_json_object(report / "report_manifest.json")
    supplied_manifest_hash = manifest.get("manifest_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    if supplied_manifest_hash != _record_hash(unsigned_manifest):
        raise ValueError("Qwen AWQ report manifest self-hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Qwen AWQ report file manifest is missing")
    for name, digest in files.items():
        path = report / str(name)
        if not path.is_file() or _file_sha256(path) != digest:
            raise ValueError(f"Qwen AWQ report file hash mismatch: {name}")
    results_path = report / "results.json"
    results = _load_json_object(results_path)
    supplied_results_hash = results.get("results_sha256")
    unsigned_results = dict(results)
    unsigned_results.pop("results_sha256", None)
    if supplied_results_hash != _record_hash(unsigned_results):
        raise ValueError("Qwen AWQ results self-hash mismatch")
    if results.get("report_protocol") != "qwen-hf-triplet-report-v1-20260802":
        raise ValueError("Qwen AWQ report protocol mismatch")
    analysis = results.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Qwen AWQ analysis is missing")
    return {
        "source_path": "reports/qwen_hf_triplet_formal_v1/results.json",
        "source_sha256": _file_sha256(results_path),
        "report_manifest_sha256": supplied_manifest_hash,
        "analysis": analysis,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
