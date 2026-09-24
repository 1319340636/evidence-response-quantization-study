"""Scale-robust analyses frozen for the CI manuscript major revision."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from .multiplicity import holm_adjust
from .vitaminc_statistics import (
    paired_page_bootstrap_intervals,
    paired_page_wald_test,
)
from .vitaminc_results import VitaminCConditionResults


PROTOCOL = "ci-major-revision-scale-v1-20260829"
CONTRASTS = {
    "awq_minus_fp16": ("awq", "fp16"),
    "awq_minus_gptq": ("awq", "gptq"),
}
ENDPOINTS = (
    "reference_standardized_route_effect",
    "directional_common_language_effect",
)
SCALE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)


def _validate_values(values: Mapping[str, Mapping[str, float]]) -> set[str]:
    if set(values) != {"fp16", "gptq", "awq"}:
        raise ValueError("scale analysis requires fp16, gptq, and awq")
    case_sets = [set(values[route]) for route in ("fp16", "gptq", "awq")]
    if not case_sets[0] or any(cases != case_sets[0] for cases in case_sets[1:]):
        raise ValueError("route case identities must match")
    return case_sets[0]


def _page_means(values: Mapping[str, float], pages: Mapping[str, str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in values.items():
        grouped[pages[case_id]].append(float(value))
    return {
        page: statistics.fmean(grouped[page]) for page in sorted(grouped)
    }


def _subtract(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    return {case_id: float(left[case_id]) - float(right[case_id]) for case_id in left}


def _direction(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return 0.0
    return 0.5


def _mad_scale(values: list[float]) -> float:
    median = statistics.median(values)
    return 1.4826 * statistics.median(abs(value - median) for value in values)


def classify_scale_robust_gate(
    *,
    standardized_estimate: float,
    standardized_adjusted_pvalue: float,
    directional_estimate: float,
    directional_adjusted_pvalue: float,
) -> str:
    """Apply the frozen MR-G1 claim gate."""

    same_direction = (
        standardized_estimate != 0.0
        and directional_estimate != 0.0
        and (standardized_estimate > 0.0) == (directional_estimate > 0.0)
    )
    significant = sum(
        pvalue < 0.05
        for pvalue in (
            standardized_adjusted_pvalue,
            directional_adjusted_pvalue,
        )
    )
    if same_direction and significant == 2:
        return "bounded_scale_robust"
    if same_direction and significant == 1:
        return "qualitative_scale_robust"
    return "fixed_protocol_descriptive"


def analyze_scale_robust_cases(
    *,
    qwen_values: Mapping[str, Mapping[str, float]],
    ministral_values: Mapping[str, Mapping[str, float]],
    pages: Mapping[str, str],
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Analyze aligned per-quartet interactions on scale-invariant endpoints."""

    q_cases = _validate_values(qwen_values)
    m_cases = _validate_values(ministral_values)
    if q_cases != m_cases or set(pages) != q_cases:
        raise ValueError("cross-model case identities and pages must match")

    metric_pages: dict[str, dict[str, float]] = {}
    contrast_work: dict[str, dict[str, Any]] = {}
    scale_grid: dict[str, dict[str, float]] = {}
    for contrast, (route_a, route_b) in CONTRASTS.items():
        q_delta = _subtract(qwen_values[route_a], qwen_values[route_b])
        m_delta = _subtract(ministral_values[route_a], ministral_values[route_b])
        q_reference = [float(qwen_values[route_b][case]) for case in sorted(q_cases)]
        m_reference = [float(ministral_values[route_b][case]) for case in sorted(m_cases)]
        q_sd = statistics.stdev(q_reference)
        m_sd = statistics.stdev(m_reference)
        if q_sd == 0.0 or m_sd == 0.0:
            raise ValueError("reference route scale must be nonzero")
        standardized = {
            case: m_delta[case] / m_sd - q_delta[case] / q_sd
            for case in sorted(q_cases)
        }
        directional = {
            case: _direction(m_delta[case]) - _direction(q_delta[case])
            for case in sorted(q_cases)
        }
        standardized_key = f"{contrast}.reference_standardized_route_effect"
        directional_key = f"{contrast}.directional_common_language_effect"
        metric_pages[standardized_key] = _page_means(standardized, pages)
        metric_pages[directional_key] = _page_means(directional, pages)

        q_mad = _mad_scale(q_reference)
        m_mad = _mad_scale(m_reference)
        mad_interaction = None
        if q_mad > 0.0 and m_mad > 0.0:
            mad_interaction = statistics.fmean(
                m_delta[case] / m_mad - q_delta[case] / q_mad
                for case in sorted(q_cases)
            )
        contrast_work[contrast] = {
            "qwen_reference_sd": q_sd,
            "ministral_reference_sd": m_sd,
            "qwen_reference_mad_scale": q_mad,
            "ministral_reference_mad_scale": m_mad,
            "mad_standardized_quartet_weighted_interaction": mad_interaction,
        }
        scale_grid[contrast] = {}
        for q_scale in SCALE_GRID:
            for m_scale in SCALE_GRID:
                raw = {
                    case: m_scale * m_delta[case] - q_scale * q_delta[case]
                    for case in sorted(q_cases)
                }
                raw_pages = _page_means(raw, pages)
                key = f"qwen={q_scale},ministral={m_scale}"
                scale_grid[contrast][key] = statistics.fmean(raw_pages.values())

    intervals = paired_page_bootstrap_intervals(
        metric_pages, draws=draws, seed=seed
    )
    wald = {name: paired_page_wald_test(values) for name, values in metric_pages.items()}
    adjusted = holm_adjust({name: test.pvalue for name, test in wald.items()})

    contrasts: dict[str, Any] = {}
    for contrast in CONTRASTS:
        endpoints: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            name = f"{contrast}.{endpoint}"
            interval = asdict(intervals[name])
            endpoints[endpoint] = {
                **interval,
                "raw_pvalue": wald[name].pvalue,
                "holm_adjusted_pvalue": adjusted[name],
            }
        contrasts[contrast] = {
            **contrast_work[contrast],
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
        "analysis_protocol": PROTOCOL,
        "population": {
            "quartets": len(q_cases),
            "pages": len(set(pages.values())),
        },
        "bootstrap_contract": {
            "cluster": "page",
            "draws": draws,
            "seed": seed,
            "shared_page_indices": True,
            "holm_tests": 4,
        },
        "contrasts": contrasts,
        "scale_grid": scale_grid,
        "causal_boundary": (
            "exact_executable_route_association_not_component_causality"
        ),
    }


def analyze_scale_robust_conditions(
    conditions: Mapping[str, VitaminCConditionResults],
    *,
    qwen_model_key: str,
    ministral_model_key: str,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Revalidate six frozen conditions and run the scale-robust analysis."""

    expected = {
        "qwen_fp16",
        "qwen_gptq",
        "qwen_awq",
        "ministral_fp16",
        "ministral_gptq",
        "ministral_awq",
    }
    if set(conditions) != expected:
        raise ValueError("scale analysis requires exactly six frozen conditions")
    route_precisions = {
        "fp16": "FP16",
        "gptq": "GPTQ_INT4",
        "awq": "AWQ_INT4",
    }
    model_keys = {"qwen": qwen_model_key, "ministral": ministral_model_key}
    indexed: dict[str, dict[str, Any]] = {}
    reference_identity: tuple[str, str] | None = None
    reference_metadata: dict[str, tuple[str, str]] | None = None
    for family in ("qwen", "ministral"):
        for route in ("fp16", "gptq", "awq"):
            key = f"{family}_{route}"
            condition = conditions[key]
            if condition.model_key != model_keys[family]:
                raise ValueError(f"{key} model identity mismatch")
            if condition.quantization != route_precisions[route]:
                raise ValueError(f"{key} quantization identity mismatch")
            if condition.hard_coverage != 1.0 or condition.probability_coverage != 1.0:
                raise ValueError(f"{key} requires complete hard and probability coverage")
            identity = (condition.split, condition.split_sha256)
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise ValueError("six-condition split identity mismatch")
            current: dict[str, Any] = {}
            metadata: dict[str, tuple[str, str]] = {}
            for quartet in condition.quartets:
                if quartet.case_id in current:
                    raise ValueError(f"{key} contains duplicate quartet identities")
                if quartet.interaction is None:
                    raise ValueError(f"{key} has incomplete selected-token interaction coverage")
                current[quartet.case_id] = float(quartet.interaction)
                metadata[quartet.case_id] = (quartet.page, quartet.negative_label)
            if not current:
                raise ValueError(f"{key} has no quartets")
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise ValueError("six-condition quartet metadata identities mismatch")
            indexed[key] = current

    assert reference_metadata is not None
    qwen_values = {
        route: indexed[f"qwen_{route}"] for route in ("fp16", "gptq", "awq")
    }
    ministral_values = {
        route: indexed[f"ministral_{route}"]
        for route in ("fp16", "gptq", "awq")
    }
    pages = {case: page for case, (page, _) in reference_metadata.items()}
    result = analyze_scale_robust_cases(
        qwen_values=qwen_values,
        ministral_values=ministral_values,
        pages=pages,
        draws=draws,
        seed=seed,
    )
    result["input_contract"] = {
        "six_condition_alignment": True,
        "complete_hard_coverage": True,
        "complete_probability_coverage": True,
        "split": reference_identity[0],
        "split_sha256": reference_identity[1],
        "qwen_model_key": qwen_model_key,
        "ministral_model_key": ministral_model_key,
    }
    return result


def write_scale_robust_result(
    result: Mapping[str, Any],
    output_path: str | Path,
    *,
    source_files: Mapping[str, str | Path],
    source_path_root: str | Path | None = None,
    formal: bool = True,
) -> dict[str, Path]:
    """Atomically write a source-bound result and detached SHA-256."""

    if result.get("analysis_protocol") != PROTOCOL:
        raise ValueError("scale-robust analysis protocol drifted")
    if formal:
        if result.get("population") != {"quartets": 1200, "pages": 1078}:
            raise ValueError("formal scale analysis population drifted")
        expected_bootstrap = {
            "cluster": "page",
            "draws": 10_000,
            "seed": 20260711,
            "shared_page_indices": True,
            "holm_tests": 4,
        }
        if result.get("bootstrap_contract") != expected_bootstrap:
            raise ValueError("formal scale analysis bootstrap contract drifted")
    if not source_files:
        raise ValueError("scale analysis requires source files")
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    portable_root = Path(source_path_root).resolve() if source_path_root else None
    for role, supplied in sorted(source_files.items()):
        path = Path(supplied)
        if any(part.casefold() == "recovery" for part in path.parts):
            raise ValueError("recovery sources are forbidden")
        if not path.is_file():
            raise ValueError(f"scale analysis source is missing: {role}")
        if portable_root is None:
            serialized_path = str(path)
        else:
            try:
                serialized_path = path.resolve().relative_to(portable_root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"scale analysis source is outside portable root: {role}"
                ) from error
        source_paths[str(role)] = serialized_path
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
    text = json.dumps(
        record,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    detached = output.with_name(output.name + ".sha256")
    output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    detached_temporary = detached.with_name(f".{detached.name}.tmp")
    detached_temporary.write_text(
        f"{output_hash}  {output.name}\n", encoding="utf-8"
    )
    detached_temporary.replace(detached)
    return {"result": output, "detached_sha256": detached}
