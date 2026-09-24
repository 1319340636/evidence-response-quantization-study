"""Atomic deterministic report writer for the frozen D5 analysis."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .hf_awq_method_analysis import HF_AWQ_ANALYSIS_PROTOCOL


HF_AWQ_REPORT_PROTOCOL = "hf-awq-method-report-v1-20260721"
CAUSAL_LIMITATION = (
    "This extension holds the data, prompt, scoring rule, high-precision "
    "checkpoint, and broad HF route fixed, but AWQ and GPTQ still use "
    "different compression procedures, packed representations, loaders, "
    "and quantized kernels. The contrast is method-associated and does not "
    "identify a pure quantizer causal effect."
)


class HFAWQMethodReportError(RuntimeError):
    """Raised when a D5 report cannot be created without drift."""


def write_hf_awq_method_report(
    result: Mapping[str, Any],
    output_directory: str | Path,
    *,
    source_files: Mapping[str, str | Path],
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Create all report files as one fail-closed directory transaction."""
    output = Path(output_directory)
    if output.exists():
        raise HFAWQMethodReportError(f"report output already exists: {output}")
    if result.get("analysis_protocol") != HF_AWQ_ANALYSIS_PROTOCOL:
        raise HFAWQMethodReportError("analysis protocol drifted")
    population = result.get("population")
    if not isinstance(population, Mapping) or population.get("quartets") != 1200 or population.get("pages") != 1078:
        raise HFAWQMethodReportError("formal analysis population drifted")
    if not source_files:
        raise HFAWQMethodReportError("report sources are required")
    source_hashes: dict[str, str] = {}
    for role, raw_path in sorted(source_files.items()):
        if not isinstance(role, str) or not role:
            raise HFAWQMethodReportError("source roles must be nonempty")
        path = Path(raw_path)
        if any("recovery" in part.casefold() for part in path.parts):
            raise HFAWQMethodReportError("recovery sources are forbidden")
        if not path.is_file():
            raise HFAWQMethodReportError(f"source is missing: {role}")
        source_hashes[role] = _file_sha256(path)
    if expected_source_hashes is not None:
        normalized_expected = {
            str(role): str(digest)
            for role, digest in expected_source_hashes.items()
        }
        if normalized_expected != source_hashes:
            raise HFAWQMethodReportError("source hash drifted")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        results_path = temporary / "results.json"
        results_path.write_text(
            json.dumps(
                dict(result),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        analysis_path = temporary / "analysis.md"
        analysis_path.write_text(
            _render_markdown(result), encoding="utf-8", newline="\n"
        )
        validation = {
            "protocol_version": HF_AWQ_REPORT_PROTOCOL,
            "analysis_protocol": result["analysis_protocol"],
            "source_sha256": source_hashes,
            "results_sha256": _file_sha256(results_path),
            "causal_limitation_present": True,
        }
        validation_path = temporary / "validation.json"
        validation_path.write_text(
            json.dumps(
                validation,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        files = {
            name: _file_sha256(temporary / name)
            for name in ("results.json", "analysis.md", "validation.json")
        }
        manifest: dict[str, Any] = {
            "protocol_version": HF_AWQ_REPORT_PROTOCOL,
            "files": files,
            "source_sha256": source_hashes,
        }
        manifest["manifest_sha256"] = _record_hash(
            manifest, excluded="manifest_sha256"
        )
        manifest_path = temporary / "report_manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        name: output / name
        for name in (
            "results.json",
            "analysis.md",
            "validation.json",
            "report_manifest.json",
        )
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    primary = result.get("primary")
    if not isinstance(primary, Mapping):
        raise HFAWQMethodReportError("primary result is missing")
    interval = primary.get("page_bootstrap")
    wald = primary.get("page_wald")
    if not isinstance(interval, Mapping) or not isinstance(wald, Mapping):
        raise HFAWQMethodReportError("primary inference is incomplete")
    return (
        "# HF AWQ Method-Robustness Extension\n\n"
        "## Confirmatory primary endpoint\n\n"
        f"Estimand: `{primary.get('estimand')}`.\n\n"
        f"Page-balanced estimate: {float(interval['estimate']):.8g}; "
        f"two-sided 95% percentile CI "
        f"[{float(interval['lower']):.8g}, {float(interval['upper']):.8g}]; "
        f"page-level Wald p={float(wald['pvalue']):.8g}.\n\n"
        "The four frozen secondary comparisons use Holm-adjusted p-values; "
        "their unadjusted confidence intervals are descriptive. The "
        "quartet-weighted analysis is an unequal-page-weighted secondary "
        "sensitivity analysis.\n\n"
        "## Interpretation boundary\n\n"
        f"{CAUSAL_LIMITATION}\n"
    )


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
