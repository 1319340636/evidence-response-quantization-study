"""Post-review sensitivity checks for the three-model scale analysis.

This module does not replace the frozen primary analysis.  It propagates the
sampling uncertainty of the reference-route standard deviation by recomputing
that denominator inside every shared page-cluster bootstrap draw and compares
page-balanced with quartet-weighted aggregation.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping

from .major_revision_scale import (
    CONTRASTS,
    ENDPOINTS,
    _direction,
    _subtract,
    _validate_values,
    classify_scale_robust_gate,
)
from .multiplicity import holm_adjust
from .three_model_scale_heterogeneity import MODEL_ORDER, PAIR_DEFINITIONS


SENSITIVITY_PROTOCOL = "three-model-scale-sensitivity-v1-20260903"
WEIGHTINGS = ("page_balanced", "quartet_weighted")


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _normal_pvalue(estimate: float, standard_error: float) -> float:
    if standard_error == 0.0:
        return 0.0 if estimate != 0.0 else 1.0
    return math.erfc(abs(estimate / standard_error) / math.sqrt(2.0))


def _case_groups(pages: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for case, page in pages.items():
        grouped[page].append(case)
    return {
        page: tuple(sorted(grouped[page]))
        for page in sorted(grouped)
    }


def _sample_sd(*, total: float, total_squares: float, count: int) -> float:
    if count < 2:
        return 0.0
    variance = max(0.0, (total_squares - total * total / count) / (count - 1))
    return math.sqrt(variance)


def analyze_three_model_scale_sensitivity(
    *,
    model_values: Mapping[str, Mapping[str, Mapping[str, float]]],
    pages: Mapping[str, str],
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Re-estimate scale uncertainty under shared page-cluster resampling."""

    if set(model_values) != set(MODEL_ORDER):
        raise ValueError("scale sensitivity requires exactly three models")
    if type(draws) is not int or draws < 1 or type(seed) is not int:
        raise ValueError("bootstrap policy is invalid")
    case_sets = {model: _validate_values(model_values[model]) for model in MODEL_ORDER}
    cases = case_sets[MODEL_ORDER[0]]
    if any(case_sets[model] != cases for model in MODEL_ORDER[1:]) or set(pages) != cases:
        raise ValueError("cross-model case and page identities must match")

    grouped = _case_groups(pages)
    page_order = tuple(grouped)
    work: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODEL_ORDER:
        work[model] = {}
        for contrast, (route_a, route_b) in CONTRASTS.items():
            delta = _subtract(model_values[model][route_a], model_values[model][route_b])
            reference = {
                case: float(model_values[model][route_b][case]) for case in cases
            }
            directional = {case: _direction(delta[case]) for case in cases}
            page_stats: dict[str, dict[str, float | int]] = {}
            for page, page_cases in grouped.items():
                ref_values = [reference[case] for case in page_cases]
                delta_values = [delta[case] for case in page_cases]
                direction_values = [directional[case] for case in page_cases]
                page_stats[page] = {
                    "count": len(page_cases),
                    "reference_sum": math.fsum(ref_values),
                    "reference_sumsq": math.fsum(value * value for value in ref_values),
                    "delta_sum": math.fsum(delta_values),
                    "delta_mean": statistics.fmean(delta_values),
                    "direction_sum": math.fsum(direction_values),
                    "direction_mean": statistics.fmean(direction_values),
                }
            observed_reference_sd = statistics.stdev(reference.values())
            if observed_reference_sd == 0.0:
                raise ValueError(f"{model} {contrast} reference route scale must be nonzero")
            work[model][contrast] = {
                "page_stats": page_stats,
                "observed_reference_sd": observed_reference_sd,
            }

    def estimates(sampled_pages: tuple[str, ...], weighting: str) -> dict[str, float] | None:
        within: dict[str, dict[str, dict[str, float]]] = {}
        for model in MODEL_ORDER:
            within[model] = {}
            for contrast in CONTRASTS:
                stats_by_page = work[model][contrast]["page_stats"]
                count = sum(int(stats_by_page[page]["count"]) for page in sampled_pages)
                reference_sum = math.fsum(
                    float(stats_by_page[page]["reference_sum"]) for page in sampled_pages
                )
                reference_sumsq = math.fsum(
                    float(stats_by_page[page]["reference_sumsq"]) for page in sampled_pages
                )
                reference_sd = _sample_sd(
                    total=reference_sum,
                    total_squares=reference_sumsq,
                    count=count,
                )
                if reference_sd == 0.0:
                    return None
                if weighting == "page_balanced":
                    delta_mean = statistics.fmean(
                        float(stats_by_page[page]["delta_mean"]) for page in sampled_pages
                    )
                    direction_mean = statistics.fmean(
                        float(stats_by_page[page]["direction_mean"]) for page in sampled_pages
                    )
                else:
                    delta_mean = math.fsum(
                        float(stats_by_page[page]["delta_sum"]) for page in sampled_pages
                    ) / count
                    direction_mean = math.fsum(
                        float(stats_by_page[page]["direction_sum"]) for page in sampled_pages
                    ) / count
                within[model][contrast] = {
                    "reference_standardized_route_effect": delta_mean / reference_sd,
                    "directional_common_language_effect": direction_mean,
                }
        output: dict[str, float] = {}
        for pair, (left_model, right_model) in PAIR_DEFINITIONS.items():
            for contrast in CONTRASTS:
                for endpoint in ENDPOINTS:
                    key = f"{pair}.{contrast}.{endpoint}"
                    output[key] = (
                        within[left_model][contrast][endpoint]
                        - within[right_model][contrast][endpoint]
                    )
        return output

    observed: dict[str, dict[str, float]] = {}
    for weighting in WEIGHTINGS:
        current = estimates(page_order, weighting)
        assert current is not None
        observed[weighting] = current

    replicates: dict[str, dict[str, list[float]]] = {
        weighting: {key: [] for key in observed[weighting]}
        for weighting in WEIGHTINGS
    }
    generator = random.Random(seed)
    rejected_degenerate_draws = 0
    while len(next(iter(replicates[WEIGHTINGS[0]].values()))) < draws:
        sampled = tuple(
            page_order[generator.randrange(len(page_order))]
            for _ in range(len(page_order))
        )
        draw_values = {weighting: estimates(sampled, weighting) for weighting in WEIGHTINGS}
        if any(value is None for value in draw_values.values()):
            rejected_degenerate_draws += 1
            if rejected_degenerate_draws > max(1000, draws * 10):
                raise ValueError("too many degenerate reference-scale bootstrap draws")
            continue
        for weighting in WEIGHTINGS:
            assert draw_values[weighting] is not None
            for key, value in draw_values[weighting].items():
                replicates[weighting][key].append(value)

    families: dict[str, Any] = {}
    for weighting in WEIGHTINGS:
        endpoint_results: dict[str, dict[str, float | int | str]] = {}
        raw_pvalues: dict[str, float] = {}
        for key, values in replicates[weighting].items():
            ordered = sorted(values)
            standard_error = statistics.stdev(values) if len(values) > 1 else 0.0
            raw_pvalue = _normal_pvalue(observed[weighting][key], standard_error)
            raw_pvalues[key] = raw_pvalue
            endpoint_results[key] = {
                "estimate": observed[weighting][key],
                "lower": _quantile(ordered, 0.025),
                "upper": _quantile(ordered, 0.975),
                "bootstrap_standard_error": standard_error,
                "raw_pvalue": raw_pvalue,
                "draws": draws,
                "clusters": len(page_order),
                "confidence_interval_status": "unadjusted_sensitivity",
            }
        adjusted = holm_adjust(raw_pvalues)
        interactions: dict[str, Any] = {}
        for pair in PAIR_DEFINITIONS:
            interactions[pair] = {}
            for contrast in CONTRASTS:
                endpoints: dict[str, Any] = {}
                for endpoint in ENDPOINTS:
                    key = f"{pair}.{contrast}.{endpoint}"
                    endpoints[endpoint] = {
                        **endpoint_results[key],
                        "holm_adjusted_pvalue": adjusted[key],
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
        families[weighting] = {
            "weighting": weighting,
            "holm_tests": len(endpoint_results),
            "pvalue_method": "normal_approximation_from_bootstrap_standard_error",
            "interactions": interactions,
        }

    return {
        "analysis_protocol": SENSITIVITY_PROTOCOL,
        "analysis_role": "post_review_sensitivity_not_frozen_primary",
        "population": {
            "quartets": len(cases),
            "pages": len(page_order),
            "models": len(MODEL_ORDER),
        },
        "bootstrap_contract": {
            "cluster": "page",
            "draws": draws,
            "seed": seed,
            "shared_page_indices": True,
            "reference_sd_reestimated_each_draw": True,
            "rejected_degenerate_draws": rejected_degenerate_draws,
        },
        "model_order": list(MODEL_ORDER),
        "pair_order": list(PAIR_DEFINITIONS),
        "contrast_order": list(CONTRASTS),
        "endpoint_order": list(ENDPOINTS),
        "weighting_sensitivities": families,
        "causal_boundary": (
            "sensitivity_of_route_by_model_heterogeneity_not_component_causality"
        ),
    }


def write_three_model_scale_sensitivity(
    result: Mapping[str, Any],
    output_path: str | Path,
    *,
    source_files: Mapping[str, str | Path],
    formal: bool = True,
) -> dict[str, Path]:
    """Write a source-bound sensitivity result and detached digest."""

    if result.get("analysis_protocol") != SENSITIVITY_PROTOCOL:
        raise ValueError("three-model sensitivity protocol drifted")
    if formal:
        if result.get("population") != {"quartets": 1200, "pages": 1078, "models": 3}:
            raise ValueError("formal sensitivity population drifted")
        bootstrap = result.get("bootstrap_contract", {})
        if (
            bootstrap.get("draws") != 10_000
            or bootstrap.get("seed") != 20260711
            or bootstrap.get("reference_sd_reestimated_each_draw") is not True
        ):
            raise ValueError("formal sensitivity bootstrap policy drifted")
    if not source_files:
        raise ValueError("three-model sensitivity requires source files")

    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for role, supplied in sorted(source_files.items()):
        path = Path(supplied)
        if any(part.casefold() == "recovery" for part in path.parts):
            raise ValueError("recovery sources are forbidden")
        if not path.is_file():
            raise ValueError(f"three-model sensitivity source is missing: {role}")
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
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    detached_temporary = detached.with_name(detached.name + ".tmp")
    detached_temporary.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    detached_temporary.replace(detached)
    return {"result": output, "detached_sha256": detached}
