"""Frozen Qwen--Ministral heterogeneity in HF quantization-route effects."""

from __future__ import annotations

from dataclasses import asdict
import statistics
from typing import Any, Mapping

from .hf_awq_method_analysis import _page_means, _subtract_cases
from .ministral_hf_triplet_analysis import (
    MinistralHFTripletAnalysisError,
    _aligned_triplet,
    _validate_condition_identities as _validate_ministral_conditions,
    _validate_gate as _validate_ministral_gate,
    _validate_policy,
)
from .multiplicity import holm_adjust
from .qwen_hf_triplet_analysis import (
    QwenHFTripletAnalysisError,
    _validate_condition_identities as _validate_qwen_conditions,
    _validate_gate as _validate_qwen_gate,
)
from .vitaminc_results import VitaminCConditionResults
from .vitaminc_statistics import paired_page_bootstrap_intervals, paired_page_wald_test


INTERACTION_PROTOCOL = "qwen-ministral-hf-method-interaction-v1-20260809"
INTERACTION_ORDER = ("awq_minus_fp16", "awq_minus_gptq")


class QwenMinistralMethodInteractionError(RuntimeError):
    """Raised when the frozen cross-model paired comparison drifts."""


def analyze_qwen_ministral_method_interaction(
    *,
    qwen_conditions: Mapping[str, VitaminCConditionResults],
    ministral_conditions: Mapping[str, VitaminCConditionResults],
    qwen_gate: Mapping[str, Any],
    ministral_gate: Mapping[str, Any],
    formal: bool = True,
    draws: int = 10_000,
    seed: int = 20260711,
    qwen_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        _validate_policy(formal=formal, draws=draws, seed=seed)
        _validate_qwen_conditions(qwen_conditions, formal=formal)
        _validate_ministral_conditions(ministral_conditions, formal=formal)
        _validate_qwen_gate(
            qwen_gate,
            formal=formal,
            split_sha256=qwen_conditions["fp16"].split_sha256,
        )
        _validate_ministral_gate(
            ministral_gate,
            formal=formal,
            split_sha256=ministral_conditions["fp16"].split_sha256,
        )
        if (
            qwen_conditions["fp16"].split_sha256
            != ministral_conditions["fp16"].split_sha256
            or qwen_conditions["fp16"].prompt_contract_sha256
            != ministral_conditions["fp16"].prompt_contract_sha256
        ):
            raise QwenMinistralMethodInteractionError(
                "cross-model source identity drifted"
            )
        q_index, q_cases, q_pages, q_values = _generic_aligned_qwen(
            qwen_conditions, formal=formal
        )
        m_index, m_cases, m_pages, m_values = _aligned_triplet(
            ministral_conditions, formal=formal
        )
        if q_cases != m_cases:
            raise QwenMinistralMethodInteractionError(
                "cross-model case identities are not matched"
            )
        for case_id in q_cases:
            q_ref = q_index["fp16"][case_id]
            m_ref = m_index["fp16"][case_id]
            if (
                q_pages[case_id] != m_pages[case_id]
                or q_ref.negative_label != m_ref.negative_label
                or q_ref.hard_gold_labels != m_ref.hard_gold_labels
            ):
                raise QwenMinistralMethodInteractionError(
                    f"cross-model quartet metadata drifted for {case_id}"
                )
        q_effects = {
            "awq_minus_fp16": _subtract_cases(q_values["awq"], q_values["fp16"]),
            "awq_minus_gptq": _subtract_cases(q_values["awq"], q_values["gptq"]),
        }
        if formal:
            if not isinstance(qwen_reference, Mapping):
                raise QwenMinistralMethodInteractionError(
                    "formal interaction requires immutable Qwen reference"
                )
            for name in INTERACTION_ORDER:
                expected = qwen_reference.get(f"{name}_estimate")
                observed = statistics.fmean(
                    _page_means(q_effects[name], q_pages).values()
                )
                if not isinstance(expected, (int, float)) or abs(observed - float(expected)) > 1e-12:
                    raise QwenMinistralMethodInteractionError(
                        f"immutable Qwen reference drifted at {name}"
                    )
        m_effects = {
            "awq_minus_fp16": _subtract_cases(m_values["awq"], m_values["fp16"]),
            "awq_minus_gptq": _subtract_cases(m_values["awq"], m_values["gptq"]),
        }
        quartet_interactions = {
            name: _subtract_cases(m_effects[name], q_effects[name])
            for name in INTERACTION_ORDER
        }
        page_interactions = {
            name: _page_means(values, q_pages)
            for name, values in quartet_interactions.items()
        }
        intervals = paired_page_bootstrap_intervals(
            page_interactions, draws=draws, seed=seed
        )
        wald = {
            name: paired_page_wald_test(page_interactions[name])
            for name in INTERACTION_ORDER
        }
        adjusted = holm_adjust({name: wald[name].pvalue for name in INTERACTION_ORDER})
    except (
        MinistralHFTripletAnalysisError,
        QwenHFTripletAnalysisError,
        ValueError,
        TypeError,
    ) as error:
        if isinstance(error, QwenMinistralMethodInteractionError):
            raise
        raise QwenMinistralMethodInteractionError(str(error)) from error

    return {
        "analysis_protocol": INTERACTION_PROTOCOL,
        "estimand": "ministral_route_effect_minus_qwen_route_effect",
        "causal_boundary": "route_by_model_heterogeneity_not_pure_quantizer_causality",
        "population": {"quartets": len(q_cases), "pages": len(set(q_pages.values()))},
        "bootstrap_contract": {
            "cluster": "page",
            "draws": draws,
            "seed": seed,
            "confidence_level": 0.95,
            "alternative": "two_sided",
            "shared_page_indices": True,
        },
        "interaction_order": list(INTERACTION_ORDER),
        "interactions": {
            name: {
                "page_bootstrap": asdict(intervals[name]),
                "page_wald": asdict(wald[name]),
                "holm_adjusted_pvalue": adjusted[name],
            }
            for name in INTERACTION_ORDER
        },
        "holm_family": [
            {
                "name": name,
                "raw_pvalue": wald[name].pvalue,
                "holm_adjusted_pvalue": adjusted[name],
            }
            for name in INTERACTION_ORDER
        ],
        "source_gates": {
            "qwen": qwen_gate["gate_sha256"],
            "ministral": ministral_gate["gate_sha256"],
        },
        "qwen_reference": dict(qwen_reference) if qwen_reference is not None else None,
    }


def _generic_aligned_qwen(conditions, *, formal):
    """Reuse the shared alignment core after Qwen's identity validator."""
    converted = {
        key: VitaminCConditionResults(
            protocol_version=(
                "ministral-hf-triplet-formal-v1-20260809"
                if formal
                else "ministral-hf-triplet-smoke-v1-20260809"
            ),
            split=value.split,
            split_sha256=value.split_sha256,
            prompt_contract_sha256=value.prompt_contract_sha256,
            model_key="ministral3_8b",
            quantization=value.quantization,
            hard_coverage=value.hard_coverage,
            probability_coverage=value.probability_coverage,
            quartets=value.quartets,
        )
        for key, value in conditions.items()
    }
    return _aligned_triplet(converted, formal=formal)
