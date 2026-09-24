"""Fail-closed formal report for the frozen Qwen HF precision triplet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .hf_awq_triplet_gate import (
    HFAWQTripletGateError,
    _load_export,
    validate_hf_awq_triplet,
)
from .hf_runtime_audit import HFRuntimeAuditError, load_hf_runtime_audit
from .hf_vitaminc_analysis import _load_condition
from .qwen_hf_triplet_analysis import ANALYSIS_PROTOCOL, analyze_qwen_hf_triplet
from .qwen_hf_triplet_figures import render_qwen_hf_triplet_figures
from .vitaminc_results import VitaminCConditionResults


REPORT_PROTOCOL = "qwen-hf-triplet-report-v1-20260802"
VALIDATION_PROTOCOL = "qwen-hf-triplet-report-validation-v1-20260802"
CAUSAL_LIMITATION = (
    "The contrasts are method-associated within the Qwen3.5-9B HF route. "
    "They hold the source checkpoint, data, prompt, scoring rule, and broad "
    "backend fixed, but do not isolate a pure quantization-algorithm effect "
    "because packing, loaders, and quantized kernels may still differ."
)
_REQUIRED_SOURCE_ROLES = {
    "fp16_manifest",
    "fp16_records",
    "gptq_manifest",
    "gptq_records",
    "awq_manifest",
    "awq_records",
    "fp16_runtime_audit",
    "gptq_runtime_audit",
    "awq_runtime_audit",
    "awq_artifact_audit",
    "qwen_triplet_gate",
    "qwen_triplet_freeze",
    "qwen_triplet_freeze_sha256",
}
_REPORT_FILES = (
    "results.json",
    "analysis-report.md",
    "stats-appendix.md",
    "metrics.csv",
    "validation.json",
    "figure-catalog.md",
    "figures/figure-01-method-contrasts.png",
    "figures/figure-01-method-contrasts.pdf",
    "figures/figure-02-hard-label-metrics.png",
    "figures/figure-02-hard-label-metrics.pdf",
    "report_manifest.json",
)


class QwenHFTripletReportError(RuntimeError):
    """Raised when immutable inputs or deterministic report outputs drift."""


def build_qwen_hf_triplet_report(
    *,
    project_root: Path,
    output_dir: Path,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Path]:
    """Revalidate the formal exports and create their strict report bundle."""
    root = Path(project_root)
    paths = _formal_paths(root)
    _reject_recovery_paths(paths.values())
    try:
        regenerated = validate_hf_awq_triplet(
            fp16_export=paths["fp16_export"],
            gptq_export=paths["gptq_export"],
            awq_export=paths["awq_export"],
            fp16_audit=paths["fp16_runtime_audit"],
            gptq_audit=paths["gptq_runtime_audit"],
            awq_audit=paths["awq_runtime_audit"],
            awq_artifact_audit=paths["awq_artifact_audit"],
            model_key="qwen35_9b",
            mode="qwen_formal4800",
            qwen_freeze_config=paths["qwen_triplet_freeze"],
            qwen_freeze_sha256_path=paths["qwen_triplet_freeze_sha256"],
        )
        saved_gate = _read_json_object(paths["qwen_triplet_gate"], "saved Gate")
        if saved_gate != regenerated:
            raise QwenHFTripletReportError(
                "saved Qwen triplet Gate does not match regenerated Gate"
            )
        conditions: dict[str, VitaminCConditionResults] = {}
        for route, precision, protocol in (
            ("fp16", "FP16", "hf-vitaminc-formal-v1-20260718"),
            ("gptq", "GPTQ_INT4", "hf-vitaminc-formal-v1-20260718"),
            ("awq", "AWQ_INT4", "hf-awq-d5-formal-v1-20260721"),
        ):
            certificate = load_hf_runtime_audit(paths[f"{route}_runtime_audit"])
            manifest, rows = _load_export(paths[f"{route}_export"])
            conditions[route] = _load_condition(
                manifest,
                rows,
                certificate_sha256=certificate.certificate_sha256,
                expected_precision=precision,
                mode="formal",
                expected_protocol=protocol,
                expected_model_key="qwen35_9b",
            )
    except (HFAWQTripletGateError, HFRuntimeAuditError, OSError, TypeError, ValueError) as error:
        if isinstance(error, QwenHFTripletReportError):
            raise
        raise QwenHFTripletReportError(str(error)) from error

    analysis = analyze_qwen_hf_triplet(
        conditions, gate=regenerated, formal=True, draws=draws, seed=seed
    )
    source_files = {
        role: paths[role]
        for role in _REQUIRED_SOURCE_ROLES
    }
    return write_qwen_hf_triplet_report(
        analysis, output_dir, source_files=source_files
    )


def write_qwen_hf_triplet_report(
    result: Mapping[str, Any],
    output_directory: str | Path,
    *,
    source_files: Mapping[str, str | Path],
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Write one atomic, deterministic and idempotent report transaction."""
    _validate_analysis(result)
    if set(source_files) != _REQUIRED_SOURCE_ROLES:
        raise QwenHFTripletReportError("formal report source roles drifted")
    _reject_recovery_paths(source_files.values())
    source_hashes: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    for role, supplied in sorted(source_files.items()):
        path = Path(supplied)
        if not path.is_file():
            raise QwenHFTripletReportError(f"source is missing: {role}")
        source_hashes[role] = _file_sha256(path)
        source_paths[role] = str(path)
    if expected_source_hashes is not None and {
        str(key): str(value) for key, value in expected_source_hashes.items()
    } != source_hashes:
        raise QwenHFTripletReportError("source hash drifted")

    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        results = {
            "report_protocol": REPORT_PROTOCOL,
            "analysis": dict(result),
        }
        results["results_sha256"] = _record_hash(results, excluded="results_sha256")
        _write_text(temporary / "results.json", _json_text(results))
        _write_text(temporary / "analysis-report.md", _analysis_markdown(result))
        _write_text(temporary / "stats-appendix.md", _stats_markdown(result))
        _write_text(temporary / "metrics.csv", _metrics_csv(result))
        render_qwen_hf_triplet_figures(result, temporary)
        validation = {
            "validation_protocol": VALIDATION_PROTOCOL,
            "status": "valid",
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "bootstrap_contract": dict(result["bootstrap_contract"]),
            "source_paths": source_paths,
            "source_sha256": source_hashes,
            "results_record_sha256": results["results_sha256"],
            "recovery_access": False,
            "causal_limitation_present": True,
        }
        _write_text(temporary / "validation.json", _json_text(validation))
        payload_files = sorted(set(_REPORT_FILES) - {"report_manifest.json"})
        manifest = {
            "manifest_protocol": "hashed-report-manifest-v1",
            "report_protocol": REPORT_PROTOCOL,
            "files": {
                name: _file_sha256(temporary / name) for name in payload_files
            },
            "source_sha256": source_hashes,
        }
        manifest["manifest_sha256"] = _record_hash(
            manifest, excluded="manifest_sha256"
        )
        _write_text(temporary / "report_manifest.json", _json_text(manifest))
        _install_idempotently(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: output / name for name in _REPORT_FILES}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--project-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--draws", type=int, default=10_000)
    value.add_argument("--seed", type=int, default=20260711)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = build_qwen_hf_triplet_report(
        project_root=args.project_root,
        output_dir=args.output,
        draws=args.draws,
        seed=args.seed,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, sort_keys=True))
    return 0


def _formal_paths(root: Path) -> dict[str, Path]:
    baseline = root / "runs" / "qwen_hf_formal_v1"
    triplet = root / "runs" / "qwen_hf_triplet_formal_v1"
    fp16 = baseline / "qwen35_9b_FP16"
    gptq = baseline / "qwen35_9b_GPTQ_INT4"
    awq = triplet / "qwen35_9b_AWQ_INT4"
    return {
        "fp16_export": fp16 / "export",
        "gptq_export": gptq / "export",
        "awq_export": awq / "export",
        "fp16_manifest": fp16 / "export" / "manifest.json",
        "fp16_records": fp16 / "export" / "records.jsonl",
        "gptq_manifest": gptq / "export" / "manifest.json",
        "gptq_records": gptq / "export" / "records.jsonl",
        "awq_manifest": awq / "export" / "manifest.json",
        "awq_records": awq / "export" / "records.jsonl",
        "fp16_runtime_audit": fp16 / "audit.json",
        "gptq_runtime_audit": gptq / "audit.json",
        "awq_runtime_audit": awq / "audit.json",
        "awq_artifact_audit": root
        / "runs"
        / "hf_awq_artifacts_v1"
        / "qwen35_9b"
        / "artifact_audit.json",
        "qwen_triplet_gate": triplet / "qwen_triplet_formal_gate_v1.json",
        "qwen_triplet_freeze": root / "configs" / "qwen_hf_triplet_only_v1.json",
        "qwen_triplet_freeze_sha256": root
        / "configs"
        / "qwen_hf_triplet_only_v1.sha256",
    }


def _validate_analysis(result: Mapping[str, Any]) -> None:
    if result.get("analysis_protocol") != ANALYSIS_PROTOCOL:
        raise QwenHFTripletReportError("analysis protocol drifted")
    if result.get("population") != {"quartets": 1200, "pages": 1078}:
        raise QwenHFTripletReportError("formal analysis population drifted")
    expected_contract = {
        "cluster": "page",
        "draws": 10_000,
        "seed": 20260711,
        "confidence_level": 0.95,
        "alternative": "two_sided",
        "shared_page_indices": True,
    }
    if result.get("bootstrap_contract") != expected_contract:
        raise QwenHFTripletReportError("formal bootstrap contract drifted")
    if result.get("contrast_order") != ["awq_minus_fp16", "awq_minus_gptq"]:
        raise QwenHFTripletReportError("formal contrast order drifted")
    _require_finite(result)


def _require_finite(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise QwenHFTripletReportError("report values must be finite")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)
        return
    raise QwenHFTripletReportError("report contains an unsupported value")


def _analysis_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen HF FP16/GPTQ/AWQ formal analysis",
        "",
        "## Frozen paired contrasts",
        "",
        "| Contrast | Page-balanced estimate | 95% bootstrap CI | Holm-adjusted p |",
        "|---|---:|---:|---:|",
    ]
    for name in result["contrast_order"]:
        row = result["contrasts"][name]
        interval = row["page_bootstrap"]
        lines.append(
            f"| {name} | {interval['estimate']:.6f} | "
            f"[{interval['lower']:.6f}, {interval['upper']:.6f}] | "
            f"{_format_p(row['holm_adjusted_pvalue'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            CAUSAL_LIMITATION,
            "",
        ]
    )
    return "\n".join(lines)


def _stats_markdown(result: Mapping[str, Any]) -> str:
    contract = result["bootstrap_contract"]
    lines = [
        "# Statistical appendix",
        "",
        f"Population: {result['population']['quartets']} quartets on "
        f"{result['population']['pages']} page clusters.",
        "",
        f"Bootstrap: {contract['draws']} shared page-cluster draws; "
        f"seed {contract['seed']}; two-sided 95% percentile intervals.",
        "",
        "The two frozen Wald tests form one Holm family. Confidence intervals "
        "are unadjusted descriptive intervals; quartet-weighted estimates are "
        "secondary sensitivity analyses.",
        "",
    ]
    return "\n".join(lines)


def _metrics_csv(result: Mapping[str, Any]) -> str:
    columns = (
        "row_id",
        "row_type",
        "estimate",
        "ci_lower",
        "ci_upper",
        "raw_pvalue",
        "holm_adjusted_pvalue",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "agreement",
    )
    rows: list[dict[str, Any]] = []
    for name in result["contrast_order"]:
        contrast = result["contrasts"][name]
        interval = contrast["page_bootstrap"]
        rows.append(
            {
                "row_id": name,
                "row_type": "interaction_contrast",
                "estimate": interval["estimate"],
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
                "raw_pvalue": contrast["page_wald"]["pvalue"],
                "holm_adjusted_pvalue": contrast["holm_adjusted_pvalue"],
            }
        )
    fp16 = result["hard_labels"]["awq_vs_fp16"]
    gptq = result["hard_labels"]["awq_vs_gptq"]
    for route, pair, role in (
        ("fp16", fp16, "reference"),
        ("gptq", gptq, "reference"),
        ("awq", fp16, "awq"),
    ):
        rows.append(
            {
                "row_id": route,
                "row_type": "hard_label_route",
                "accuracy": pair["accuracy"][role],
                "balanced_accuracy": pair["balanced_accuracy"][role],
                "mcc": pair["mcc"][role],
            }
        )
    for name, pair in (("awq_vs_fp16", fp16), ("awq_vs_gptq", gptq)):
        rows.append(
            {
                "row_id": name,
                "row_type": "hard_label_pair",
                "estimate": pair["accuracy"]["difference"],
                "agreement": pair["agreement"],
            }
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in columns})
    return buffer.getvalue()


def _format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.4f}"


def _reject_recovery_paths(paths: Any) -> None:
    for supplied in paths:
        if any("recovery" in part.casefold() for part in Path(supplied).parts):
            raise QwenHFTripletReportError("recovery sources are forbidden")


def _install_idempotently(temporary: Path, output: Path) -> None:
    if output.exists():
        current = {
            path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
        proposed = {
            path.relative_to(temporary).as_posix(): path.read_bytes()
            for path in temporary.rglob("*")
            if path.is_file()
        }
        if current != proposed:
            raise QwenHFTripletReportError(
                "existing report is non-identical and cannot be overwritten"
            )
        shutil.rmtree(temporary)
        return
    os.replace(temporary, output)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QwenHFTripletReportError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise QwenHFTripletReportError(f"{label} must be a JSON object")
    return value


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(record: Mapping[str, Any], *, excluded: str) -> str:
    payload = json.dumps(
        {key: value for key, value in record.items() if key != excluded},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
