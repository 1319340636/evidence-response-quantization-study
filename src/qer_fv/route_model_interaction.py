"""Matched Qwen × Gemma deployment-route interaction analysis."""

from __future__ import annotations

import statistics
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .vitaminc_results import (
    PairedVitaminCQuartet,
    family_effect_heterogeneity,
)
from .vitaminc_statistics import (
    page_effects,
    page_direction_summary,
    paired_direction_mcnemar_test,
    paired_page_bootstrap_intervals,
    paired_page_wald_test,
)


ANALYSIS_PROTOCOL = "qwen-gemma-route-model-interaction-v1-20260721"
CAUSAL_BOUNDARY = (
    "The interaction compares two joint deployment routes across two model "
    "families. It combines quantization algorithm, runtime backend, weight "
    "packing, numerical kernels, and model-specific response scale; it is "
    "not a pure quantizer, bit-width, backend, or architecture effect."
)


class RouteModelInteractionError(RuntimeError):
    """Raised when four deployment routes cannot be matched safely."""


def analyze_route_model_interaction(
    *,
    qwen_hf_pairs: Sequence[PairedVitaminCQuartet],
    qwen_gguf_pairs: Sequence[PairedVitaminCQuartet],
    gemma_hf_pairs: Sequence[PairedVitaminCQuartet],
    gemma_gguf_pairs: Sequence[PairedVitaminCQuartet],
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Compute the frozen Gemma-minus-Qwen route interaction."""
    indexed = _validate_four_way(
        {
            "qwen_hf": qwen_hf_pairs,
            "qwen_gguf": qwen_gguf_pairs,
            "gemma_hf": gemma_hf_pairs,
            "gemma_gguf": gemma_gguf_pairs,
        }
    )
    qwen_hf = page_effects(qwen_hf_pairs)
    qwen_gguf = page_effects(qwen_gguf_pairs)
    gemma_hf = page_effects(gemma_hf_pairs)
    gemma_gguf = page_effects(gemma_gguf_pairs)
    try:
        qwen_route = family_effect_heterogeneity(qwen_hf, qwen_gguf)
        gemma_route = family_effect_heterogeneity(gemma_hf, gemma_gguf)
        interaction = family_effect_heterogeneity(gemma_route, qwen_route)
        intervals = paired_page_bootstrap_intervals(
            {
                "qwen_hf": qwen_hf,
                "qwen_gguf": qwen_gguf,
                "qwen_route": qwen_route,
                "gemma_hf": gemma_hf,
                "gemma_gguf": gemma_gguf,
                "gemma_route": gemma_route,
                "interaction": interaction,
            },
            draws=draws,
            seed=seed,
        )
    except ValueError as error:
        raise RouteModelInteractionError(str(error)) from error
    return {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "inference_scope": "prospectively_frozen_d4_followup_secondary",
        "causal_boundary": CAUSAL_BOUNDARY,
        "parameters": {
            "bootstrap_draws": draws,
            "seed": seed,
            "cluster": "page",
        },
        "population": {
            "quartets": len(indexed["qwen_hf"]),
            "pages": len(interaction),
        },
        "components": {
            name: asdict(intervals[name])
            for name in (
                "qwen_hf",
                "qwen_gguf",
                "qwen_route",
                "gemma_hf",
                "gemma_gguf",
                "gemma_route",
            )
        },
        "interaction": {
            "order": "gemma_route_minus_qwen_route",
            "page_bootstrap": asdict(intervals["interaction"]),
            "page_wald": asdict(paired_page_wald_test(interaction)),
            "quartet_weighted_sensitivity": _quartet_sensitivity(indexed),
        },
        "direction_diagnostic": {
            "scope": "secondary_scale_reduced_unadjusted",
            "qwen_route": asdict(page_direction_summary(qwen_route)),
            "gemma_route": asdict(page_direction_summary(gemma_route)),
            "paired_mcnemar": asdict(
                paired_direction_mcnemar_test(qwen_route, gemma_route)
            ),
        },
    }


def _validate_four_way(
    routes: Mapping[str, Sequence[PairedVitaminCQuartet]],
) -> dict[str, dict[str, PairedVitaminCQuartet]]:
    indexed: dict[str, dict[str, PairedVitaminCQuartet]] = {}
    for name, pairs in routes.items():
        by_case = {pair.case_id: pair for pair in pairs}
        if not by_case or len(by_case) != len(pairs):
            raise RouteModelInteractionError(
                f"{name} case identities must be unique and nonempty"
            )
        indexed[name] = by_case
    reference = set(indexed["qwen_hf"])
    if any(set(by_case) != reference for by_case in indexed.values()):
        raise RouteModelInteractionError(
            "four-way case identities must match exactly"
        )
    for case_id in sorted(reference):
        rows = [indexed[name][case_id] for name in sorted(indexed)]
        metadata = {
            (row.page, row.negative_label, row.hard_gold_labels)
            for row in rows
        }
        if len(metadata) != 1:
            raise RouteModelInteractionError(
                f"four-way metadata mismatch for {case_id}"
            )
    return indexed


def _quartet_sensitivity(
    indexed: Mapping[str, Mapping[str, PairedVitaminCQuartet]],
) -> float:
    values: list[float] = []
    for case_id in sorted(indexed["qwen_hf"]):
        effects = {
            name: indexed[name][case_id].interaction_effect
            for name in indexed
        }
        if any(value is None for value in effects.values()):
            raise RouteModelInteractionError(
                "quartet sensitivity requires complete effects"
            )
        values.append(
            (float(effects["gemma_hf"]) - float(effects["gemma_gguf"]))
            - (float(effects["qwen_hf"]) - float(effects["qwen_gguf"]))
        )
    return statistics.fmean(values)
