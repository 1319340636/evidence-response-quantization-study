"""Build a deterministic audit of publication-facing result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


PUBLICATION_FREEZE_PROTOCOL = "publication-freeze-v1-20260726"
PUBLICATION_INVENTORY_PROTOCOL = "publication-inventory-v1-20260726"


def load_publication_freeze(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a publication freeze registry."""
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("publication freeze must be a JSON object")
    if value.get("protocol") != PUBLICATION_FREEZE_PROTOCOL:
        raise ValueError("publication freeze protocol mismatch")

    reports = value.get("canonical_reports")
    if not isinstance(reports, list):
        raise ValueError("canonical_reports must be a list")
    roles: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("canonical report must be an object")
        role = report.get("role")
        report_path = report.get("path")
        if not isinstance(role, str) or not role:
            raise ValueError("canonical report role is invalid")
        if role in roles:
            raise ValueError(f"duplicate canonical role: {role}")
        if not isinstance(report_path, str) or not report_path:
            raise ValueError(f"canonical report path is invalid: {role}")
        roles.add(role)

    models = value.get("expected_models", [])
    if models:
        if not isinstance(models, list) or not all(
            isinstance(model, str) and model for model in models
        ):
            raise ValueError("expected_models must be non-empty strings")
        if len(models) != len(set(models)):
            raise ValueError("duplicate expected model")

    superseded = value.get("superseded_reports", [])
    if not isinstance(superseded, list):
        raise ValueError("superseded_reports must be a list")
    superseded_paths: set[str] = set()
    for item in superseded:
        if not isinstance(item, dict):
            raise ValueError("superseded report must be an object")
        item_path = item.get("path")
        replacement = item.get("replacement_role")
        if not isinstance(item_path, str) or not item_path:
            raise ValueError("superseded report path is invalid")
        if item_path in superseded_paths:
            raise ValueError(f"duplicate superseded report: {item_path}")
        if replacement not in roles:
            raise ValueError("superseded replacement role is not canonical")
        superseded_paths.add(item_path)

    if "scientific_guards" in value:
        guards = value["scientific_guards"]
        if not isinstance(guards, dict):
            raise ValueError("scientific_guards must be an object")
        stage_path = guards.get("stage_a_path")
        if stage_path != "reports/vitaminc_stage_a_v4":
            raise ValueError("canonical Stage A path must be v4")

    validate_awq_completion(value, canonical_roles=roles)
    return value


def verify_publication_markdown(path: str | Path) -> None:
    """Reject invalid UTF-8 and replacement characters in Markdown."""
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="strict")
    if "\ufffd" in text:
        raise ValueError(
            f"publication Markdown contains replacement character: {source}"
        )


def verify_report_manifest(report_directory: str | Path) -> dict[str, Any]:
    """Verify a hashed report manifest and every listed file."""
    root = Path(report_directory)
    manifest_path = root / "report_manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("manifest_protocol") != "hashed-report-manifest-v1":
        raise ValueError(f"report manifest protocol mismatch: {root}")
    expected_self = manifest.get("manifest_sha256")
    actual_self = _record_hash(manifest, excluded="manifest_sha256")
    if expected_self != actual_self:
        raise ValueError(f"manifest self-hash mismatch: {root}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"report manifest files are invalid: {root}")
    for name, expected in sorted(files.items()):
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError(f"report manifest file entry is invalid: {root}")
        source = root / name
        if not source.is_file():
            raise ValueError(f"report manifest file is missing: {source}")
        actual = _file_sha256(source)
        if actual != expected:
            raise ValueError(f"file hash mismatch: {source}")
        if source.suffix.casefold() == ".md":
            verify_publication_markdown(source)
    return manifest


def validate_stage_a_results(
    results: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    report_path: str,
) -> dict[str, Any]:
    """Enforce the frozen Stage A version, families, and OLMo boundary."""
    guards = freeze.get("scientific_guards")
    if not isinstance(guards, Mapping):
        raise ValueError("scientific guards are missing")
    if report_path != guards.get("stage_a_path"):
        raise ValueError("canonical Stage A path must be v4")
    expected = guards.get("stage_a_families")
    observed = results.get("family_order")
    if not isinstance(expected, list) or observed != expected:
        raise ValueError("Stage A family order mismatch")
    families = results.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("Stage A families are missing")
    try:
        balanced = families["olmo3_7b"]["balanced_accuracy"]
        difference = float(balanced["difference"])
        adjusted = float(balanced["holm_adjusted_pvalue"])
        minimum = float(guards["olmo_balanced_accuracy_holm_minimum"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OLMo Balanced Accuracy guard is missing") from error
    if difference >= 0.0 or adjusted < minimum:
        raise ValueError("OLMo Holm boundary mismatch")
    return {
        "difference": difference,
        "holm_adjusted_pvalue": adjusted,
        "significant_after_holm": adjusted < 0.05,
        "boundary": "negative_point_estimate_not_significant_after_holm",
    }


def validate_awq_completion(
    freeze: Mapping[str, Any], *, canonical_roles: set[str]
) -> None:
    """Prevent AWQ from being counted complete without a formal report role."""
    policy = freeze.get("awq_policy")
    if policy is None:
        return
    if not isinstance(policy, Mapping):
        raise ValueError("awq_policy must be an object")
    models = policy.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("awq_policy models must be an object")
    allowed = {"not_formally_completed", "formal_complete"}
    for model, item in models.items():
        if not isinstance(model, str) or not isinstance(item, Mapping):
            raise ValueError("AWQ model policy is invalid")
        status = item.get("status")
        if status not in allowed:
            raise ValueError(f"AWQ status is invalid: {model}")
        required_role = f"{model}_hf_awq"
        if (
            status == "formal_complete"
            and policy.get("formal_report_required") is True
            and required_role not in canonical_roles
        ):
            raise ValueError(f"AWQ formal report is missing: {model}")


def audit_publication_sources(
    project_root: str | Path, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify every source declared by the publication freeze."""
    root = Path(project_root).resolve()
    verified: dict[str, Any] = {}
    pinned = freeze.get("pinned_sources", {})
    if not isinstance(pinned, Mapping):
        raise ValueError("pinned_sources must be an object")
    for relative, expected in sorted(pinned.items()):
        source = root / relative
        actual = _file_sha256(source)
        if actual != expected:
            raise ValueError(f"pinned source hash mismatch: {relative}")
        verified[relative] = actual

    for report in freeze["canonical_reports"]:
        relative_root = report["path"]
        report_root = root / relative_root
        if not report_root.is_dir():
            raise ValueError(f"canonical report is missing: {relative_root}")
        verification = report.get("verification")
        if verification == "hashed_manifest":
            manifest = verify_report_manifest(report_root)
            validation_path = report_root / "validation.json"
            if validation_path.is_file():
                validation = _load_json_object(validation_path)
                if validation.get("status") != "valid":
                    raise ValueError(
                        f"canonical validation is not valid: {relative_root}"
                    )
            verified[relative_root] = {
                "verification": verification,
                "manifest_sha256": manifest["manifest_sha256"],
            }
        elif verification == "legacy_pinned":
            files = report.get("files")
            if not isinstance(files, Mapping) or not files:
                raise ValueError(
                    f"legacy pinned files are missing: {relative_root}"
                )
            file_hashes: dict[str, str] = {}
            for name, expected in sorted(files.items()):
                source = report_root / name
                if not source.is_file():
                    raise ValueError(f"legacy report file is missing: {source}")
                actual = _file_sha256(source)
                if actual != expected:
                    raise ValueError(f"legacy file hash mismatch: {source}")
                if source.suffix.casefold() == ".md":
                    verify_publication_markdown(source)
                file_hashes[name] = actual
            verified[relative_root] = {
                "verification": verification,
                "files": file_hashes,
            }
        else:
            raise ValueError(
                f"unknown publication verification mode: {verification}"
            )
    return verified


def build_publication_inventory(
    project_root: str | Path, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the frozen six-model result inventory from explicit schemas."""
    root = Path(project_root).resolve()
    audit_publication_sources(root, freeze)
    roles = {item["role"]: item for item in freeze["canonical_reports"]}
    models_value = _load_json_object(root / "configs/models_v1.json")
    model_by_key = {
        item["key"]: item for item in models_value.get("models", [])
    }
    expected_models = freeze["expected_models"]
    if set(model_by_key) != set(expected_models):
        raise ValueError("registered model set does not match publication freeze")
    models = [
        {
            "key": key,
            "family": model_by_key[key]["family"],
            "capacity": model_by_key[key]["capacity"],
            "repo_id": model_by_key[key]["repo_id"],
            "revision": model_by_key[key]["revision"],
        }
        for key in expected_models
    ]

    rows: list[dict[str, Any]] = []
    rows.extend(_adapt_vitaminc_main(root, roles["vitaminc_confirmatory_anchor"]))
    stage_rows, stage_boundary = _adapt_stage_a(
        root, roles["vitaminc_stage_a"], freeze
    )
    rows.extend(stage_rows)
    rows.extend(_adapt_hf(root, roles["qwen_hf_gptq"]))
    rows.extend(_adapt_hf(root, roles["gemma_hf_gptq"]))
    rows.extend(_adapt_route_interaction(root, roles["route_model_interaction"]))
    rows.extend(_adapt_tabfact(root, roles["tabfact_external_validation"]))
    rows.extend(_adapt_cub(root, roles["cub_diagnostic"]))
    rows.extend(_adapt_mapping(root, roles["mapping_sensitivity"]))
    rows.extend(_adapt_strengthening(root, roles["strengthening_audit"]))
    for model, item in sorted(freeze["awq_policy"]["models"].items()):
        rows.append(
            _inventory_row(
                source_role="hf_awq_d5",
                dataset="VitaminC",
                model_key=model,
                route="HF_AWQ",
                comparison="AWQ_INT4 vs FP16/GPTQ_INT4",
                scope="prospectively_frozen_incomplete",
                status=item["status"],
                population={"quartets": 1200, "pages": 1078},
                metrics={},
                boundaries=["not_a_formal_result"],
            )
        )

    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["model_key"],
            row["route"],
            row["source_role"],
        )
    )
    return {
        "inventory_protocol": PUBLICATION_INVENTORY_PROTOCOL,
        "freeze_protocol": freeze["protocol"],
        "models": models,
        "canonical_stage_a": freeze["scientific_guards"]["stage_a_path"],
        "superseded_reports": [
            item["path"] for item in freeze["superseded_reports"]
        ],
        "stage_a_olmo_boundary": stage_boundary,
        "rows": rows,
    }


def _adapt_vitaminc_main(
    root: Path, report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    report_root = root / report["path"]
    primary = _load_json_object(report_root / "primary_results.json")
    hard = _load_json_object(report_root / "hard_metrics.json")
    interval = primary["primary"]["ci95_percentile"]
    balanced = hard["balanced_accuracy"]
    return [
        _inventory_row(
            source_role=report["role"],
            dataset="VitaminC",
            model_key=primary["identity"]["model_key"],
            route="GGUF_Q4_K_M",
            comparison="Q4_K_M minus F16",
            scope=report["scope"],
            status="formal_complete",
            population={
                "quartets": primary["data_quality"]["paired_quartets"],
                "pages": primary["data_quality"]["paired_pages"],
                "full_evidence_rows": hard["rows"],
            },
            metrics={
                "delta_i": primary["primary"]["estimate"],
                "delta_i_ci95": interval,
                "balanced_accuracy_f16": balanced["f16"],
                "balanced_accuracy_quantized": balanced["q4"],
                "balanced_accuracy_difference": balanced["q4_minus_f16"],
            },
            boundaries=["confirmatory_h1_not_supported"],
        )
    ]


def _adapt_stage_a(
    root: Path, report: Mapping[str, Any], freeze: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = _load_json_object(root / report["path"] / "results.json")
    olmo_boundary = validate_stage_a_results(
        results, freeze, report_path=report["path"]
    )
    scale_boundary = freeze["scientific_guards"][
        "cross_model_scale_boundary"
    ]
    rows = []
    for model in results["family_order"]:
        item = results["families"][model]
        interval = item["page_effect"]
        balanced = item["balanced_accuracy"]
        boundaries = [scale_boundary]
        if model == "olmo3_7b":
            boundaries.insert(0, olmo_boundary["boundary"])
        rows.append(
            _inventory_row(
                source_role=report["role"],
                dataset="VitaminC",
                model_key=model,
                route="GGUF_Q4_K_M",
                comparison="Q4_K_M minus F16",
                scope=report["scope"],
                status="exploratory_complete",
                population={
                    "quartets": results["common_case_count"],
                    "pages": results["common_page_count"],
                },
                metrics={
                    "delta_i": interval["estimate"],
                    "delta_i_ci95": [interval["lower"], interval["upper"]],
                    "balanced_accuracy_difference": balanced["difference"],
                    "balanced_accuracy_holm_pvalue": balanced[
                        "holm_adjusted_pvalue"
                    ],
                    "negative_page_fraction": item["direction"][
                        "negative_fraction"
                    ],
                },
                boundaries=boundaries,
            )
        )
    return rows, olmo_boundary


def _adapt_hf(root: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = _load_json_object(root / report["path"] / "results.json")
    item = results["confirmatory_hf"]
    identity = item["identity"]
    interaction = item["interaction"]["page_bootstrap"]
    hard = item["hard_labels"]
    balanced = hard["balanced_accuracy"]["page_bootstrap"]
    return [
        _inventory_row(
            source_role=report["role"],
            dataset="VitaminC",
            model_key=identity["model_key"],
            route="HF_GPTQ_INT4",
            comparison="GPTQ_INT4 minus FP16",
            scope=report["scope"],
            status="formal_complete",
            population=item["population"],
            metrics={
                "delta_i": interaction["estimate"],
                "delta_i_ci95": [interaction["lower"], interaction["upper"]],
                "accuracy_difference": hard["accuracy"]["difference"],
                "balanced_accuracy_difference": balanced["difference"],
                "balanced_accuracy_ci95": [
                    balanced["lower"],
                    balanced["upper"],
                ],
            },
            boundaries=["same_hf_backend_paired_quantization_route"],
        )
    ]


def _adapt_route_interaction(
    root: Path, report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results = _load_json_object(root / report["path"] / "results.json")
    item = results["analysis"]
    interval = item["interaction"]["page_bootstrap"]
    return [
        _inventory_row(
            source_role=report["role"],
            dataset="VitaminC",
            model_key="qwen35_9b_x_gemma4_e4b",
            route="GGUF_vs_HF_interaction",
            comparison="Gemma route minus Qwen route",
            scope=report["scope"],
            status="formal_complete",
            population=item["population"],
            metrics={
                "four_way_interaction": interval["estimate"],
                "ci95": [interval["lower"], interval["upper"]],
            },
            boundaries=[
                "joint_deployment_route_effect",
                "not_a_pure_quantizer_backend_bitwidth_or_architecture_effect",
            ],
        )
    ]


def _adapt_tabfact(
    root: Path, report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results = _load_json_object(root / report["path"] / "results.json")
    item = results["fresh_tabfact_formal"]
    rows = []
    for model, effect in sorted(item["within_family_effects"].items()):
        rows.append(
            _inventory_row(
                source_role=report["role"],
                dataset="Fresh TabFact",
                model_key=model,
                route="GGUF_Q4_K_M",
                comparison="Q4_K_M minus F16",
                scope=report["scope"],
                status="formal_complete",
                population=item["population"],
                metrics={
                    "accuracy_difference": effect["accuracy"]["difference"],
                    "accuracy_ci95": [
                        effect["accuracy"]["cluster_bootstrap_95_ci"]["lower"],
                        effect["accuracy"]["cluster_bootstrap_95_ci"]["upper"],
                    ],
                    "balanced_accuracy_difference": effect[
                        "balanced_accuracy"
                    ]["difference"],
                },
                boundaries=["external_structured_evidence_validation"],
            )
        )
    return rows


def _adapt_cub(root: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    report_root = root / report["path"]
    rows = []
    for path in sorted(report_root.glob("*.json")):
        if path.name == "confirmatory_holm.json":
            continue
        item = _load_json_object(path)
        scope = item["inference_scope"]
        status = (
            "formal_complete"
            if scope == "confirmatory_external_validation"
            else "exploratory_complete"
        )
        rows.append(
            _inventory_row(
                source_role=report["role"],
                dataset="CUB/DRUID",
                model_key=item["model_key"],
                route=f"GGUF_{item['quantization']}",
                comparison=(
                    f"{item['quantization']} minus "
                    f"{item['reference_quantization']}"
                ),
                scope=scope,
                status=status,
                population={
                    "gold_samples": item["gold"]["ccu"]["samples"],
                    "conflicting_samples": item["conflicting"]["ccu"][
                        "samples"
                    ],
                    "irrelevant_samples": item["irrelevant"]["samples"],
                },
                metrics={
                    "gold_bcu_effect": item["gold"]["bcu"]["estimate"],
                    "gold_ccu_effect": item["gold"]["ccu"]["estimate"],
                    "conflicting_bcu_effect": item["conflicting"]["bcu"][
                        "estimate"
                    ],
                    "conflicting_ccu_effect": item["conflicting"]["ccu"][
                        "estimate"
                    ],
                },
                boundaries=["secondary_behavioral_diagnostic"],
            )
        )
    return rows


def _adapt_mapping(
    root: Path, report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results = _load_json_object(root / report["path"] / "results.json")
    mapping = {
        "qwen_gguf": ("qwen35_9b", "GGUF_Q4_K_M"),
        "qwen_hf": ("qwen35_9b", "HF_GPTQ_INT4"),
        "gemma_gguf": ("gemma4_e4b", "GGUF_Q4_K_M"),
    }
    rows = []
    for name, (model, route) in mapping.items():
        item = results["routes"][name]
        original = item["original"]["page_bootstrap"]
        reversed_value = item["reversed"]["page_bootstrap"]
        rows.append(
            _inventory_row(
                source_role=report["role"],
                dataset="VitaminC discovery",
                model_key=model,
                route=route,
                comparison="reversed label mapping sensitivity",
                scope=report["scope"],
                status="formal_complete",
                population=results["population"],
                metrics={
                    "original_delta_i": original["estimate"],
                    "reversed_delta_i": reversed_value["estimate"],
                    "direction_stable": item["direction_stable"],
                },
                boundaries=["cannot_select_or_rewrite_primary_mapping"],
            )
        )
    return rows


def _adapt_strengthening(
    root: Path, report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results = _load_json_object(root / report["path"] / "results.json")
    edit = results["edit_strength_robustness"]
    backend = results["backend_comparability"]
    return [
        _inventory_row(
            source_role=report["role"],
            dataset="VitaminC",
            model_key="qwen35_9b",
            route="GGUF_Q4_K_M",
            comparison="edit-strength robustness",
            scope=report["scope"],
            status="formal_complete",
            population=edit["population"],
            metrics={
                "delta_i": edit["overall"]["page_bootstrap"]["estimate"],
                "leave_one_page_out_crosses_zero": edit[
                    "high_volume_page_sensitivity"
                ]["leave_one_page_out"]["crosses_zero"],
            },
            boundaries=["operational_edit_difficulty_only"],
        ),
        _inventory_row(
            source_role=report["role"],
            dataset="VitaminC",
            model_key="qwen35_9b",
            route="HF_FP16_vs_GGUF_F16",
            comparison="high-precision backend comparability",
            scope=report["scope"],
            status="formal_complete",
            population=backend["identity"],
            metrics={
                "hard_label_agreement": backend["hard_labels"]["agreement"],
                "per_page_interaction_correlation": backend["interaction"][
                    "per_page_pearson_correlation"
                ],
                "interaction_offset": backend["interaction"]["hf_minus_gguf"][
                    "estimate"
                ],
            },
            boundaries=["descriptive_comparability_not_equivalence"],
        ),
    ]


def _inventory_row(
    *,
    source_role: str,
    dataset: str,
    model_key: str,
    route: str,
    comparison: str,
    scope: str,
    status: str,
    population: Mapping[str, Any],
    metrics: Mapping[str, Any],
    boundaries: list[str],
) -> dict[str, Any]:
    return {
        "source_role": source_role,
        "dataset": dataset,
        "model_key": model_key,
        "route": route,
        "comparison": comparison,
        "scope": scope,
        "status": status,
        "population": dict(population),
        "metrics": dict(metrics),
        "boundaries": list(boundaries),
    }


def write_publication_readiness_report(
    inventory: Mapping[str, Any],
    validation: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write a deterministic immutable publication-readiness package."""
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("publication output directory is not immutable-empty")
    output.mkdir(parents=True, exist_ok=True)

    inventory_value = dict(inventory)
    validation_value = dict(validation)
    if validation_value.get("status") != "valid":
        raise ValueError("publication validation status must be valid")
    _atomic_write_text(output / "inventory.json", _json_text(inventory_value))
    _atomic_write_text(output / "inventory.md", _inventory_markdown(inventory_value))
    _atomic_write_text(output / "validation.json", _json_text(validation_value))

    manifest = {
        "manifest_protocol": "hashed-report-manifest-v1",
        "files": {
            name: _file_sha256(output / name)
            for name in ("inventory.json", "inventory.md", "validation.json")
        },
    }
    manifest["manifest_sha256"] = _record_hash(
        manifest, excluded="manifest_sha256"
    )
    _atomic_write_text(output / "report_manifest.json", _json_text(manifest))
    return manifest


def run_publication_readiness(
    *,
    project_root: str | Path,
    freeze_config: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Audit frozen sources and write the publication-readiness package."""
    root = Path(project_root).resolve()
    config_path = Path(freeze_config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    try:
        config_identity = config_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "freeze_config must be located inside project_root"
        ) from exc
    freeze = load_publication_freeze(config_path)
    sources = audit_publication_sources(root, freeze)
    inventory = build_publication_inventory(root, freeze)
    validation = {
        "validation_protocol": freeze["output_protocols"]["validation"],
        "status": "valid",
        "freeze_config": {
            "path": config_identity,
            "sha256": _file_sha256(config_path),
        },
        "sources": sources,
        "checks": {
            "canonical_stage_a_v4": True,
            "superseded_stage_a_excluded": True,
            "six_model_registry_exact": True,
            "awq_not_counted_as_complete": True,
            "olmo_holm_boundary_preserved": True,
            "publication_markdown_utf8_valid": True,
        },
        "scientific_boundaries": dict(freeze["scientific_guards"]),
    }
    return write_publication_readiness_report(
        inventory, validation, output_directory
    )


def _inventory_markdown(inventory: Mapping[str, Any]) -> str:
    boundary = inventory["stage_a_olmo_boundary"]
    lines = [
        "# Publication Readiness Inventory",
        "",
        "## Frozen status",
        "",
        "- Canonical cross-family analysis: Stage A v4.",
        (
            "- Superseded Stage A reports retained for audit only: "
            + ", ".join(inventory["superseded_reports"])
            + "."
        ),
        (
            "- OLMo Balanced Accuracy: negative point estimate; "
            f"Holm-adjusted p={boundary['holm_adjusted_pvalue']:.4f}; "
            "not significant after Holm correction."
        ),
        "- AWQ is not a completed result unless a formal hashed report and gate pass.",
        "",
        "## Registered models",
        "",
        "| Model key | Family | Capacity | Repository | Revision |",
        "|---|---|---:|---|---|",
    ]
    for model in inventory["models"]:
        lines.append(
            f"| {model['key']} | {model['family']} | {model['capacity']} | "
            f"{model['repo_id']} | `{model['revision']}` |"
        )
    lines.extend(
        [
            "",
            "## Model-data-route results",
            "",
            "| Dataset | Model | Route | Comparison | Scope | Status | Key result | Boundaries |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in inventory["rows"]:
        metrics = "; ".join(
            f"{name}={_compact_value(value)}"
            for name, value in sorted(row["metrics"].items())
        )
        boundaries = "; ".join(row["boundaries"])
        lines.append(
            f"| {row['dataset']} | {row['model_key']} | {row['route']} | "
            f"{row['comparison']} | {row['scope']} | {row['status']} | "
            f"{metrics or '-'} | {boundaries or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Publication boundaries",
            "",
            "- The original confirmatory H1 was not supported and remains reported as such.",
            "- Cross-model raw interaction magnitudes are scale- and calibration-sensitive.",
            "- GGUF-versus-HF comparisons are joint deployment-route effects, not pure quantizer or bit-width effects.",
            "- `not_formally_completed` rows are engineering status records, not scientific outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(_compact_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(
            handle, "w", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(
    value: Mapping[str, Any], *, excluded: str | None = None
) -> str:
    payload = {
        key: item for key, item in value.items() if key != excluded
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
