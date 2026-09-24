"""Boundary-aware reporting for frozen paired CUB analyses."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cub_results import load_cub_condition_export, pair_cub_conditions
from .cub_statistics import (
    PairedCubObservation,
    cluster_effects,
    ccu_missing_effect_bounds,
    paired_cluster_bootstrap_interval,
    paired_cluster_sign_flip_pvalue,
)
from .multiplicity import holm_adjust


def build_cub_pair_report(
    rows: Sequence[PairedCubObservation],
    *,
    model_key: str,
    quantization: str,
    protocol_version: str | None = None,
) -> dict[str, Any]:
    if quantization == "Q4_K_M":
        scope = "confirmatory_external_validation"
    elif quantization in {"Q8_0", "Q5_K_M"}:
        scope = "exploratory_dose_extension"
    else:
        raise ValueError("unsupported CUB quantized comparison")
    report: dict[str, Any] = {
        "model_key": model_key,
        "quantization": quantization,
        "reference_quantization": "F16",
        "inference_scope": scope,
        "run_protocol_version": protocol_version,
    }
    for context_type in ("gold", "conflicting"):
        selected = [row for row in rows if row.analysis_context_type == context_type]
        bcu_effects = cluster_effects(rows, metric="bcu", context_type=context_type)
        bcu = _metric_record(bcu_effects, len(selected), scope == "confirmatory_external_validation")
        complete = [
            row for row in selected
            if row.ccu_f16 is not None and row.ccu_quantized is not None
        ]
        ccu_bounds = ccu_missing_effect_bounds(rows, context_type=context_type)
        if complete:
            ccu_effects = cluster_effects(complete, metric="ccu", context_type=context_type)
            ccu = _metric_record(ccu_effects, len(complete), scope == "confirmatory_external_validation")
        else:
            ccu = {"estimate": None, "clusters": 0, "samples": 0}
        ccu.update(
            {
                "observed_pairs": ccu_bounds.observed_pairs,
                "total_pairs": ccu_bounds.total_pairs,
                "missing_effect_bounds": {
                    "lower": ccu_bounds.lower,
                    "upper": ccu_bounds.upper,
                },
            }
        )
        report[context_type] = {
            "bcu": bcu,
            "ccu": ccu,
            "hard_coverage": _coverage_record(selected, "hard"),
            "probability_coverage": _coverage_record(selected, "probability"),
        }
    report["irrelevant"] = _irrelevant_record(rows)
    return report


def apply_confirmatory_holm(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if len(reports) != 4 or len({report.get("model_key") for report in reports}) != 4:
        raise ValueError("Holm families require exactly four checkpoints")
    families: dict[str, list[dict[str, Any]]] = {}
    for metric in ("bcu", "ccu"):
        raw: dict[str, float] = {}
        metadata: dict[str, tuple[str, str]] = {}
        for report in reports:
            if report.get("inference_scope") != "confirmatory_external_validation":
                raise ValueError("Holm family contains a non-confirmatory report")
            model = str(report["model_key"])
            for context in ("gold", "conflicting"):
                key = f"{model}|{context}"
                raw[key] = float(report[context][metric]["p_value"])
                metadata[key] = (model, context)
        adjusted = holm_adjust(raw)
        families[metric] = [
            {
                "model_key": metadata[key][0],
                "context_type": metadata[key][1],
                "raw_p_value": raw[key],
                "adjusted_p_value": adjusted[key],
            }
            for key in sorted(raw)
        ]
    return families


def _metric_record(
    effects: Mapping[str, float], samples: int, confirmatory: bool
) -> dict[str, Any]:
    interval = paired_cluster_bootstrap_interval(effects)
    record = {
        "estimate": interval.estimate,
        "ci_95": {"lower": interval.lower, "upper": interval.upper},
        "bootstrap_draws": interval.draws,
        "bootstrap_seed": interval.seed,
        "clusters": len(effects),
        "samples": samples,
    }
    if confirmatory:
        record["p_value"] = paired_cluster_sign_flip_pvalue(effects)
        record["p_value_method"] = "paired_query_cluster_sign_flip"
    return record


def _irrelevant_record(rows: Sequence[PairedCubObservation]) -> dict[str, Any]:
    selected = [row for row in rows if row.analysis_context_type == "irrelevant"]
    bcu_effects = [float(row.bcu_quantized) - float(row.bcu_f16) for row in selected]
    ccu_effects = [
        row.ccu_quantized - row.ccu_f16
        for row in selected
        if row.ccu_f16 is not None and row.ccu_quantized is not None
    ]
    return {
        "inference_scope": "descriptive_only",
        "samples": len(selected),
        "bcu_effect": statistics.fmean(bcu_effects) if bcu_effects else None,
        "ccu_effect": statistics.fmean(ccu_effects) if ccu_effects else None,
        "ccu_observed_pairs": len(ccu_effects),
        "hard_coverage": _coverage_record(selected, "hard"),
        "probability_coverage": _coverage_record(selected, "probability"),
    }


def _coverage_record(
    rows: Sequence[PairedCubObservation], stage: str
) -> dict[str, int]:
    if stage == "hard":
        f16 = [row.hard_ok_f16 for row in rows]
        quantized = [row.hard_ok_quantized for row in rows]
    elif stage == "probability":
        f16 = [row.probability_ok_f16 for row in rows]
        quantized = [row.probability_ok_quantized for row in rows]
    else:
        raise ValueError("coverage stage must be hard or probability")
    return {
        "f16_ok": sum(f16),
        "quantized_ok": sum(quantized),
        "paired_ok": sum(left and right for left, right in zip(f16, quantized, strict=True)),
        "total_pairs": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f16", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    f16 = load_cub_condition_export(args.f16)
    quantized = load_cub_condition_export(args.quantized)
    rows = pair_cub_conditions(f16, quantized)
    report = build_cub_pair_report(
        rows,
        model_key=f16.model_key,
        quantization=quantized.quantization,
        protocol_version=f16.protocol_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
