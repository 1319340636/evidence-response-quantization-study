"""Deterministic row-level governance ledger for publication results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paper_evidence import load_verified_publication_package
from .publication_readiness import load_publication_freeze
from .qwen_triplet_freeze import load_qwen_triplet_freeze


RESULT_GOVERNANCE_PROTOCOL = "result-governance-v1-20260728"
RESULT_GOVERNANCE_VALIDATION_PROTOCOL = (
    "result-governance-validation-v1-20260728"
)
RESULT_GOVERNANCE_MANIFEST_PROTOCOL = (
    "result-governance-manifest-v1-20260728"
)
_MOJIBAKE_MARKERS = ("\ufffd", "鈥", "锟")


def build_result_governance(project_root: str | Path) -> dict[str, Any]:
    """Bind every publication row to a canonical source path and hash."""
    root = Path(project_root).resolve()
    package = load_verified_publication_package(root)
    inventory = package["inventory"]
    validation = package["validation"]
    freeze = load_publication_freeze(root / "configs/publication_freeze_v1.json")
    qwen_config = root / "configs/qwen_hf_triplet_only_v1.json"
    qwen_sidecar = root / "configs/qwen_hf_triplet_only_v1.sha256"
    load_qwen_triplet_freeze(qwen_config, sha256_path=qwen_sidecar)
    qwen_sha256 = _sha256_file(qwen_config)
    qwen_rows, qwen_report_manifest_sha256 = _load_qwen_triplet_rows(root)
    ministral_rows, ministral_results_sha256 = _load_ministral_triplet_rows(root)
    interaction_rows, interaction_results_sha256 = (
        _load_qwen_ministral_interaction_rows(root)
    )

    role_paths = {
        item["role"]: item["path"] for item in freeze["canonical_reports"]
    }
    audited_sources = validation["sources"]
    rows = []
    for source_row in inventory["rows"]:
        row = json.loads(json.dumps(source_row, ensure_ascii=False))
        if row["source_role"] == "hf_awq_d5":
            if row["model_key"] == "qwen35_9b":
                continue
            source_path = "configs/qwen_hf_triplet_only_v1.json"
            source_sha256 = qwen_sha256
            governance_status = "retired_without_result"
        else:
            source_path = role_paths[row["source_role"]]
            source_sha256 = _audited_source_sha256(
                source_path, audited_sources[source_path]
            )
            governance_status = "canonical_result"
        row.update(
            {
                "source_path": source_path,
                "source_sha256": source_sha256,
                "governance_status": governance_status,
            }
        )
        row["row_id"] = _sha256_json(
            {
                "dataset": row["dataset"],
                "model_key": row["model_key"],
                "route": row["route"],
                "comparison": row["comparison"],
                "source_role": row["source_role"],
            }
        )
        rows.append(row)
    rows.extend(qwen_rows)
    rows.extend(ministral_rows)
    rows.extend(interaction_rows)

    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("governance row identifiers are not unique")
    return {
        "protocol": RESULT_GOVERNANCE_PROTOCOL,
        "canonical_stage_a": inventory["canonical_stage_a"],
        "superseded_stage_a": inventory["superseded_reports"],
        "source_publication_inventory_sha256": package[
            "inventory_sha256"
        ],
        "source_publication_manifest_sha256": package["manifest"].get(
            "manifest_sha256"
        ),
        "qwen_triplet_freeze_sha256": qwen_sha256,
        "qwen_triplet_report_manifest_sha256": qwen_report_manifest_sha256,
        "ministral_triplet_results_sha256": ministral_results_sha256,
        "qwen_ministral_interaction_results_sha256": interaction_results_sha256,
        "rows": rows,
    }


def verify_governance_markdown_text(text: str) -> None:
    """Reject common Unicode replacement and mojibake markers."""
    if any(marker in text for marker in _MOJIBAKE_MARKERS):
        raise ValueError("governance Markdown contains mojibake")
    text.encode("utf-8", errors="strict")


def render_governance_markdown(ledger: Mapping[str, Any]) -> str:
    lines = [
        "# Result Governance Master Table",
        "",
        "## Frozen boundaries",
        "",
        "- Stage A v4 is the sole canonical cross-family analysis.",
        "- Stage A v1-v3 are retained for audit only and never enter rows.",
        "- OLMo balanced-accuracy change is a negative point estimate; ",
        "  p=.0503; not significant after Holm correction.",
        "- Qwen3.5-9B and Ministral-3-8B have completed HF AWQ extensions.",
        "- Gemma AWQ is retired, not formally completed, and not a formal result.",
        "",
        "## Model-data-route-sample-result-hash ledger",
        "",
        "| Dataset | Model | Route | Sample | Result | Status | Source | SHA-256 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in ledger["rows"]:
        sample = _compact_mapping(row["population"])
        result = _compact_mapping(row["metrics"]) or "-"
        status = f"{row['status']}; {row['governance_status']}"
        lines.append(
            f"| {row['dataset']} | {row['model_key']} | {row['route']} | "
            f"{sample or '-'} | {result} | {status} | "
            f"{row['source_path']} | `{row['source_sha256']}` |"
        )
    lines.append("")
    text = "\n".join(lines)
    verify_governance_markdown_text(text)
    return text


def run_result_governance(
    *, project_root: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    ledger = build_result_governance(project_root)
    validation = _build_validation(ledger)
    files = {
        "master_table.json": _json_text(ledger),
        "master_table.md": render_governance_markdown(ledger),
        "validation.json": _json_text(validation),
    }
    return _write_report(files, output_directory, ledger)


def verify_result_governance_manifest(
    output_directory: str | Path,
) -> dict[str, Any]:
    root = Path(output_directory)
    manifest_path = root / "report_manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("protocol") != RESULT_GOVERNANCE_MANIFEST_PROTOCOL:
        raise ValueError("result governance manifest protocol mismatch")
    expected_self = manifest.get("manifest_sha256")
    without_self = dict(manifest)
    without_self.pop("manifest_sha256", None)
    if expected_self != _sha256_json(without_self):
        raise ValueError("result governance manifest self-hash mismatch")
    for name, expected in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"result governance file hash mismatch: {name}")
    return manifest


def _build_validation(ledger: Mapping[str, Any]) -> dict[str, Any]:
    olmo = next(
        row for row in ledger["rows"]
        if row["source_role"] == "vitaminc_stage_a"
        and row["model_key"] == "olmo3_7b"
    )
    awq = {
        row["model_key"]: row for row in ledger["rows"]
        if row["route"] == "HF_AWQ"
    }
    checks = {
        "canonical_stage_a_v4": ledger["canonical_stage_a"]
        == "reports/vitaminc_stage_a_v4",
        "all_rows_source_hashed": all(
            len(row["source_sha256"]) == 64 for row in ledger["rows"]
        ),
        "row_ids_unique": len({row["row_id"] for row in ledger["rows"]})
        == len(ledger["rows"]),
        "olmo_holm_boundary": (
            olmo["metrics"]["balanced_accuracy_holm_pvalue"] >= 0.05
            and "negative_point_estimate_not_significant_after_holm"
            in olmo["boundaries"]
        ),
        "qwen_awq_formal_completed": all(
            row["governance_status"] == "formal_completed_qwen_only"
            for row in ledger["rows"]
            if row["route"] == "HF_AWQ" and row["model_key"] == "qwen35_9b"
        ),
        "gemma_awq_retired": awq["gemma4_e4b"]["governance_status"]
        == "retired_without_result",
    }
    if not all(checks.values()):
        raise ValueError("result governance validation failed")
    return {
        "protocol": RESULT_GOVERNANCE_VALIDATION_PROTOCOL,
        "status": "valid",
        "checks": checks,
        "row_count": len(ledger["rows"]),
    }


def _write_report(
    files: Mapping[str, str], output_directory: str | Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.exists() and any(
        path.is_file() or path.is_symlink() for path in output.rglob("*")
    ):
        raise ValueError("result governance output must be immutable-empty")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, text in sorted(files.items()):
        path = output / name
        _atomic_write_text(path, text)
        hashes[name] = _sha256_file(path)
    manifest = {
        "protocol": RESULT_GOVERNANCE_MANIFEST_PROTOCOL,
        "files": hashes,
        "source_publication_inventory_sha256": ledger[
            "source_publication_inventory_sha256"
        ],
        "source_publication_manifest_sha256": ledger[
            "source_publication_manifest_sha256"
        ],
        "qwen_triplet_freeze_sha256": ledger["qwen_triplet_freeze_sha256"],
        "qwen_triplet_report_manifest_sha256": ledger[
            "qwen_triplet_report_manifest_sha256"
        ],
        "ministral_triplet_results_sha256": ledger[
            "ministral_triplet_results_sha256"
        ],
        "qwen_ministral_interaction_results_sha256": ledger[
            "qwen_ministral_interaction_results_sha256"
        ],
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _atomic_write_text(output / "report_manifest.json", _json_text(manifest))
    return verify_result_governance_manifest(output)


def _audited_source_sha256(path: str, audit: Mapping[str, Any]) -> str:
    if "manifest_sha256" in audit:
        return str(audit["manifest_sha256"])
    files = audit.get("files")
    if isinstance(files, Mapping) and files:
        return _sha256_json({"path": path, "files": dict(files)})
    raise ValueError(f"canonical source has no audit hash: {path}")


def _load_qwen_triplet_rows(root: Path) -> tuple[list[dict[str, Any]], str]:
    report_root = root / "reports" / "qwen_hf_triplet_formal_v1"
    manifest_path = report_root / "report_manifest.json"
    results_path = report_root / "results.json"
    manifest = _load_json_object(manifest_path)
    manifest_self = manifest.get("manifest_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    if manifest_self != _sha256_json(unsigned_manifest):
        raise ValueError("Qwen triplet report manifest self-hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Qwen triplet report file manifest is missing")
    for name, digest in files.items():
        path = report_root / str(name)
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"Qwen triplet report file hash mismatch: {name}")

    results = _load_json_object(results_path)
    supplied_results_hash = results.get("results_sha256")
    unsigned_results = dict(results)
    unsigned_results.pop("results_sha256", None)
    if supplied_results_hash != _sha256_json(unsigned_results):
        raise ValueError("Qwen triplet results self-hash mismatch")
    if results.get("report_protocol") != "qwen-hf-triplet-report-v1-20260802":
        raise ValueError("Qwen triplet report protocol mismatch")
    analysis = results.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("Qwen triplet analysis is missing")
    if analysis.get("population") != {"quartets": 1200, "pages": 1078}:
        raise ValueError("Qwen triplet formal population drifted")
    contrasts = analysis["contrasts"]
    fp16_contrast = contrasts["awq_minus_fp16"]
    gptq_contrast = contrasts["awq_minus_gptq"]
    fp16 = analysis["hard_labels"]["awq_vs_fp16"]
    gptq = analysis["hard_labels"]["awq_vs_gptq"]
    source_path = "reports/qwen_hf_triplet_formal_v1/results.json"
    source_sha256 = _sha256_file(results_path)
    common = {
        "dataset": "VitaminC",
        "model_key": "qwen35_9b",
        "route": "HF_AWQ",
        "population": {"quartets": 1200, "pages": 1078, "full_inputs": 4800},
        "scope": "confirmatory_within_hf_method_associated",
        "status": "formal_completed",
        "source_role": "qwen_hf_triplet_formal",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "governance_status": "formal_completed_qwen_only",
        "boundaries": [
            "single_model_qwen_only",
            "method_associated_not_pure_algorithm_effect",
            "packing_loader_kernel_differences_remain",
        ],
    }
    rows = [
        {
            **common,
            "metric_family": "interaction",
            "comparison": "AWQ_INT4 minus FP16 and GPTQ_INT4",
            "metrics": {
                "awq_minus_fp16_estimate": fp16_contrast["page_bootstrap"]["estimate"],
                "awq_minus_fp16_ci": [
                    fp16_contrast["page_bootstrap"]["lower"],
                    fp16_contrast["page_bootstrap"]["upper"],
                ],
                "awq_minus_fp16_holm_pvalue": fp16_contrast["holm_adjusted_pvalue"],
                "awq_minus_gptq_estimate": gptq_contrast["page_bootstrap"]["estimate"],
                "awq_minus_gptq_ci": [
                    gptq_contrast["page_bootstrap"]["lower"],
                    gptq_contrast["page_bootstrap"]["upper"],
                ],
                "awq_minus_gptq_holm_pvalue": gptq_contrast["holm_adjusted_pvalue"],
            },
        },
        {
            **common,
            "metric_family": "accuracy",
            "comparison": "FP16 vs GPTQ_INT4 vs AWQ_INT4",
            "metrics": {
                "fp16": fp16["accuracy"]["reference"],
                "gptq": gptq["accuracy"]["reference"],
                "awq": fp16["accuracy"]["awq"],
            },
        },
        {
            **common,
            "metric_family": "balanced_accuracy",
            "comparison": "FP16 vs GPTQ_INT4 vs AWQ_INT4",
            "metrics": {
                "fp16": fp16["balanced_accuracy"]["reference"],
                "gptq": gptq["balanced_accuracy"]["reference"],
                "awq": fp16["balanced_accuracy"]["awq"],
            },
        },
        {
            **common,
            "metric_family": "mcc",
            "comparison": "FP16 vs GPTQ_INT4 vs AWQ_INT4",
            "metrics": {
                "fp16": fp16["mcc"]["reference"],
                "gptq": gptq["mcc"]["reference"],
                "awq": fp16["mcc"]["awq"],
            },
        },
    ]
    for row in rows:
        row["row_id"] = _sha256_json(
            {
                "dataset": row["dataset"],
                "model_key": row["model_key"],
                "route": row["route"],
                "comparison": row["comparison"],
                "source_role": row["source_role"],
                "metric_family": row["metric_family"],
            }
        )
    return rows, str(manifest_self)


def _load_ministral_triplet_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], str]:
    results_path = (
        root / "reports" / "ministral_hf_triplet_formal_v1" / "results.json"
    )
    source_sha256 = _verified_detached_sha256(results_path)
    results = _load_json_object(results_path)
    if (
        results.get("analysis_protocol")
        != "ministral-hf-triplet-analysis-v1-20260809"
    ):
        raise ValueError("Ministral triplet analysis protocol mismatch")
    if results.get("population") != {"quartets": 1200, "pages": 1078}:
        raise ValueError("Ministral triplet formal population drifted")
    if (
        results.get("causal_boundary")
        != "quantizer_route_associated_not_pure_algorithm_effect"
    ):
        raise ValueError("Ministral triplet causal boundary drifted")

    contrasts = results.get("contrasts")
    hard_labels = results.get("hard_labels")
    if not isinstance(contrasts, Mapping) or not isinstance(hard_labels, Mapping):
        raise ValueError("Ministral triplet analysis blocks are missing")
    awq_fp16 = contrasts["awq_minus_fp16"]
    awq_gptq = contrasts["awq_minus_gptq"]
    hard_fp16 = hard_labels["awq_vs_fp16"]
    hard_gptq = hard_labels["awq_vs_gptq"]
    source_path = "reports/ministral_hf_triplet_formal_v1/results.json"
    common = {
        "dataset": "VitaminC",
        "model_key": "ministral3_8b",
        "route": "HF_AWQ",
        "population": {"quartets": 1200, "pages": 1078, "full_inputs": 4800},
        "scope": "prospectively_frozen_within_hf_method_associated",
        "status": "formal_completed",
        "source_role": "ministral_hf_triplet_formal",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "governance_status": "formal_completed_second_hf_family",
        "boundaries": [
            "method_associated_not_pure_algorithm_effect",
            "packing_loader_kernel_differences_remain",
            "raw_probability_scale_not_for_cross_model_ranking",
        ],
    }
    rows = [
        {
            **common,
            "metric_family": "interaction",
            "comparison": "AWQ_INT4 minus FP16 and GPTQ_INT4",
            "metrics": {
                "awq_minus_fp16_estimate": awq_fp16["page_bootstrap"][
                    "estimate"
                ],
                "awq_minus_fp16_ci": [
                    awq_fp16["page_bootstrap"]["lower"],
                    awq_fp16["page_bootstrap"]["upper"],
                ],
                "awq_minus_fp16_holm_pvalue": awq_fp16[
                    "holm_adjusted_pvalue"
                ],
                "awq_minus_gptq_estimate": awq_gptq["page_bootstrap"][
                    "estimate"
                ],
                "awq_minus_gptq_ci": [
                    awq_gptq["page_bootstrap"]["lower"],
                    awq_gptq["page_bootstrap"]["upper"],
                ],
                "awq_minus_gptq_holm_pvalue": awq_gptq[
                    "holm_adjusted_pvalue"
                ],
            },
        },
        {
            **common,
            "metric_family": "accuracy",
            "comparison": "FP16 vs GPTQ_INT4 vs AWQ_INT4",
            "metrics": {
                "fp16": hard_fp16["accuracy"]["reference"],
                "gptq": hard_gptq["accuracy"]["reference"],
                "awq": hard_fp16["accuracy"]["awq"],
            },
        },
        {
            **common,
            "metric_family": "balanced_accuracy",
            "comparison": "FP16 vs GPTQ_INT4 vs AWQ_INT4",
            "metrics": {
                "fp16": hard_fp16["balanced_accuracy"]["reference"],
                "gptq": hard_gptq["balanced_accuracy"]["reference"],
                "awq": hard_fp16["balanced_accuracy"]["awq"],
            },
        },
        {
            **common,
            "metric_family": "mcc",
            "comparison": "FP16 vs GPTQ_INT4 vs AWQ_INT4",
            "metrics": {
                "fp16": hard_fp16["mcc"]["reference"],
                "gptq": hard_gptq["mcc"]["reference"],
                "awq": hard_fp16["mcc"]["awq"],
            },
        },
    ]
    _assign_governance_row_ids(rows)
    return rows, source_sha256


def _load_qwen_ministral_interaction_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], str]:
    results_path = (
        root
        / "reports"
        / "qwen_ministral_method_interaction_v1"
        / "analysis.json"
    )
    source_sha256 = _verified_detached_sha256(results_path)
    report = _load_json_object(results_path)
    analysis = report.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("Qwen--Ministral interaction analysis is missing")
    if report.get("analysis_sha256") != _sha256_json({"analysis": analysis}):
        raise ValueError("Qwen--Ministral interaction self-hash mismatch")
    if (
        analysis.get("analysis_protocol")
        != "qwen-ministral-hf-method-interaction-v1-20260809"
    ):
        raise ValueError("Qwen--Ministral interaction protocol mismatch")
    if analysis.get("population") != {"quartets": 1200, "pages": 1078}:
        raise ValueError("Qwen--Ministral interaction population drifted")
    if (
        analysis.get("estimand")
        != "ministral_route_effect_minus_qwen_route_effect"
        or analysis.get("causal_boundary")
        != "route_by_model_heterogeneity_not_pure_quantizer_causality"
    ):
        raise ValueError("Qwen--Ministral interaction boundary drifted")
    interactions = analysis.get("interactions")
    if not isinstance(interactions, Mapping):
        raise ValueError("Qwen--Ministral interaction estimates are missing")

    source_path = "reports/qwen_ministral_method_interaction_v1/analysis.json"
    rows = []
    for contrast in ("awq_minus_fp16", "awq_minus_gptq"):
        result = interactions[contrast]
        bootstrap = result["page_bootstrap"]
        rows.append(
            {
                "dataset": "VitaminC",
                "model_key": "qwen35_9b__ministral3_8b",
                "route": "HF_FP16_GPTQ_AWQ_INTERACTION",
                "population": {
                    "quartets": 1200,
                    "pages": 1078,
                    "full_inputs_per_condition": 4800,
                },
                "scope": "prospectively_frozen_route_by_model_interaction",
                "status": "formal_completed",
                "source_role": "qwen_ministral_hf_method_interaction",
                "source_path": source_path,
                "source_sha256": source_sha256,
                "governance_status": "formal_completed_central_interaction",
                "metric_family": f"{contrast}_interaction",
                "comparison": (
                    f"Ministral {contrast} minus Qwen {contrast}"
                ),
                "metrics": {
                    "estimate": bootstrap["estimate"],
                    "ci95": [bootstrap["lower"], bootstrap["upper"]],
                    "raw_pvalue": result["page_wald"]["pvalue"],
                    "holm_adjusted_pvalue": result["holm_adjusted_pvalue"],
                },
                "boundaries": [
                    "route_by_model_heterogeneity_not_pure_quantizer_causality",
                    "not_architecture_causality",
                    "packing_loader_kernel_differences_remain",
                ],
            }
        )
    _assign_governance_row_ids(rows)
    return rows, source_sha256


def _assign_governance_row_ids(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["row_id"] = _sha256_json(
            {
                "dataset": row["dataset"],
                "model_key": row["model_key"],
                "route": row["route"],
                "comparison": row["comparison"],
                "source_role": row["source_role"],
                "metric_family": row["metric_family"],
            }
        )


def _verified_detached_sha256(path: Path) -> str:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"detached SHA-256 is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"detached SHA-256 is malformed: {sidecar}")
    try:
        int(fields[0], 16)
    except ValueError as error:
        raise ValueError(f"detached SHA-256 is malformed: {sidecar}") from error
    digest = _sha256_file(path)
    if digest != fields[0].lower():
        raise ValueError(f"detached SHA-256 mismatch: {path}")
    return digest


def _compact_mapping(value: Mapping[str, Any]) -> str:
    return "; ".join(
        f"{key}={_compact_value(item)}" for key, item in sorted(value.items())
    )


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(_compact_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen result-governance master table."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    output = (root / args.output_dir).resolve()
    if root != output and root not in output.parents:
        parser.error("--output-dir must resolve inside --project-root")
    manifest = run_result_governance(
        project_root=root, output_directory=output
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0
