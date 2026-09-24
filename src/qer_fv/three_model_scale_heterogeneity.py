"""Scale-invariant route-by-model heterogeneity across three HF families."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from .major_revision_scale import (
    CONTRASTS,
    ENDPOINTS,
    _direction,
    _page_means,
    _subtract,
    _validate_values,
    classify_scale_robust_gate,
)
from .multiplicity import holm_adjust
from .vitaminc_statistics import (
    paired_page_bootstrap_intervals,
    paired_page_wald_test,
)


ANALYSIS_PROTOCOL = "three-model-hf-scale-heterogeneity-v1-20260902"
MODEL_ORDER = ("qwen", "ministral", "olmo")
PAIR_DEFINITIONS = {
    "ministral_minus_qwen": ("ministral", "qwen"),
    "olmo_minus_qwen": ("olmo", "qwen"),
    "olmo_minus_ministral": ("olmo", "ministral"),
}


def analyze_three_model_scale_heterogeneity(
    *,
    model_values: Mapping[str, Mapping[str, Mapping[str, float]]],
    pages: Mapping[str, str],
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Estimate all pairwise family differences on scale-invariant endpoints."""

    if set(model_values) != set(MODEL_ORDER):
        raise ValueError("scale heterogeneity requires exactly three models")
    if type(draws) is not int or draws < 1 or type(seed) is not int:
        raise ValueError("bootstrap policy is invalid")

    case_sets = {model: _validate_values(model_values[model]) for model in MODEL_ORDER}
    cases = case_sets[MODEL_ORDER[0]]
    if any(case_sets[model] != cases for model in MODEL_ORDER[1:]) or set(pages) != cases:
        raise ValueError("cross-model case and page identities must match")

    endpoint_cases: dict[str, dict[str, dict[str, float]]] = {}
    within_model: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        values = model_values[model]
        endpoint_cases[model] = {}
        within_model[model] = {}
        for contrast, (route_a, route_b) in CONTRASTS.items():
            delta = _subtract(values[route_a], values[route_b])
            reference = [float(values[route_b][case]) for case in sorted(cases)]
            reference_sd = statistics.stdev(reference)
            if reference_sd == 0.0:
                raise ValueError(f"{model} {contrast} reference route scale must be nonzero")
            standardized = {
                case: delta[case] / reference_sd for case in sorted(cases)
            }
            directional = {
                case: _direction(delta[case]) for case in sorted(cases)
            }
            endpoint_cases[model][contrast] = {
                "reference_standardized_route_effect": standardized,
                "directional_common_language_effect": directional,
            }
            within_model[model][contrast] = {
                "raw_route_effect_estimate": statistics.fmean(
                    _page_means(delta, pages).values()
                ),
                "reference_sd": reference_sd,
                "reference_standardized_route_effect": statistics.fmean(
                    _page_means(standardized, pages).values()
                ),
                "directional_common_language_effect": statistics.fmean(
                    _page_means(directional, pages).values()
                ),
            }

    metric_pages: dict[str, dict[str, float]] = {}
    for pair, (left_model, right_model) in PAIR_DEFINITIONS.items():
        for contrast in CONTRASTS:
            for endpoint in ENDPOINTS:
                key = f"{pair}.{contrast}.{endpoint}"
                difference = _subtract(
                    endpoint_cases[left_model][contrast][endpoint],
                    endpoint_cases[right_model][contrast][endpoint],
                )
                metric_pages[key] = _page_means(difference, pages)

    intervals = paired_page_bootstrap_intervals(metric_pages, draws=draws, seed=seed)
    tests = {key: paired_page_wald_test(values) for key, values in metric_pages.items()}
    adjusted = holm_adjust({key: test.pvalue for key, test in tests.items()})

    interactions: dict[str, Any] = {}
    for pair in PAIR_DEFINITIONS:
        interactions[pair] = {}
        for contrast in CONTRASTS:
            endpoints: dict[str, Any] = {}
            for endpoint in ENDPOINTS:
                key = f"{pair}.{contrast}.{endpoint}"
                endpoints[endpoint] = {
                    **asdict(intervals[key]),
                    "raw_pvalue": tests[key].pvalue,
                    "holm_adjusted_pvalue": adjusted[key],
                    "confidence_interval_status": "unadjusted_descriptive",
                }
            interactions[pair][contrast] = {
                "endpoints": endpoints,
                "claim_gate": classify_scale_robust_gate(
                    standardized_estimate=endpoints[
                        "reference_standardized_route_effect"
                    ]["estimate"],
                    standardized_adjusted_pvalue=endpoints[
                        "reference_standardized_route_effect"
                    ]["holm_adjusted_pvalue"],
                    directional_estimate=endpoints[
                        "directional_common_language_effect"
                    ]["estimate"],
                    directional_adjusted_pvalue=endpoints[
                        "directional_common_language_effect"
                    ]["holm_adjusted_pvalue"],
                ),
            }

    return {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "estimand": "pairwise_difference_of_within_model_route_effects",
        "population": {
            "quartets": len(cases),
            "pages": len(set(pages.values())),
            "models": len(MODEL_ORDER),
        },
        "bootstrap_contract": {
            "cluster": "page",
            "draws": draws,
            "seed": seed,
            "confidence_level": 0.95,
            "shared_page_indices": True,
            "holm_tests": len(metric_pages),
        },
        "model_order": list(MODEL_ORDER),
        "pair_order": list(PAIR_DEFINITIONS),
        "contrast_order": list(CONTRASTS),
        "endpoint_order": list(ENDPOINTS),
        "within_model_effects": within_model,
        "interactions": interactions,
        "holm_family": [
            {
                "name": key,
                "raw_pvalue": tests[key].pvalue,
                "holm_adjusted_pvalue": adjusted[key],
            }
            for key in metric_pages
        ],
        "causal_boundary": (
            "route_by_model_heterogeneity_not_pure_quantizer_kernel_"
            "packing_or_architecture_causality"
        ),
    }


def write_three_model_scale_result(
    result: Mapping[str, Any],
    output_path: str | Path,
    *,
    source_files: Mapping[str, str | Path],
    formal: bool = True,
) -> dict[str, Path]:
    """Write an atomic, source-bound analysis record and detached digest."""

    if result.get("analysis_protocol") != ANALYSIS_PROTOCOL:
        raise ValueError("three-model analysis protocol drifted")
    if formal:
        if result.get("population") != {
            "quartets": 1200,
            "pages": 1078,
            "models": 3,
        }:
            raise ValueError("formal three-model population drifted")
        bootstrap = result.get("bootstrap_contract", {})
        if (
            bootstrap.get("draws") != 10_000
            or bootstrap.get("seed") != 20260711
            or bootstrap.get("holm_tests") != 12
        ):
            raise ValueError("formal three-model inference policy drifted")
    if not source_files:
        raise ValueError("three-model analysis requires source files")

    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for role, supplied in sorted(source_files.items()):
        path = Path(supplied)
        if any(part.casefold() == "recovery" for part in path.parts):
            raise ValueError("recovery sources are forbidden")
        if not path.is_file():
            raise ValueError(f"three-model source is missing: {role}")
        source_paths[str(role)] = str(path)
        source_hashes[str(role)] = hashlib.sha256(path.read_bytes()).hexdigest()

    record: dict[str, Any] = {
        "analysis": dict(result),
        "source_paths": source_paths,
        "source_sha256": source_hashes,
    }
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    record["analysis_record_sha256"] = hashlib.sha256(canonical).hexdigest()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    detached = output.with_name(output.name + ".sha256")
    detached_temporary = detached.with_name(detached.name + ".tmp")
    detached_temporary.write_text(
        hashlib.sha256(output.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    detached_temporary.replace(detached)
    return {"result": output, "detached_sha256": detached}
