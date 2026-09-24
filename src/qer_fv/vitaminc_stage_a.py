"""Zero-new-inference Stage A analyses for the VitaminC cross-model study."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import asdict
from typing import Mapping, Sequence

from .multiplicity import holm_adjust
from .vitaminc_hard_metrics import (
    HardPredictionPair,
    paired_page_permutation_balanced_accuracy,
)
from .vitaminc_results import (
    PairedVitaminCQuartet,
    family_effect_heterogeneity,
)
from .vitaminc_statistics import (
    matched_direction_omnibus,
    model_internal_standardized_effect,
    omnibus_page_wald_test,
    page_direction_summary,
    page_effects,
    paired_hard_label_summary,
    paired_direction_mcnemar_test,
    paired_page_bootstrap_intervals,
    paired_page_wald_test,
)


def analyze_vitaminc_stage_a(
    paired_families: Mapping[str, Sequence[PairedVitaminCQuartet]],
    *,
    draws: int = 10_000,
    ba_draws: int | None = None,
    seed: int = 20260711,
) -> dict[str, object]:
    """Analyze the common-case four-family matrix without additional inference."""
    if ba_draws is None:
        ba_draws = draws
    if type(ba_draws) is not int or ba_draws < 1:
        raise ValueError("Stage A BA draws must be a positive integer")
    if len(paired_families) < 2:
        raise ValueError("Stage A requires at least two model families")
    family_names = tuple(sorted(str(name) for name in paired_families))
    if len(family_names) != len(paired_families):
        raise ValueError("Stage A family names must be unique strings")
    indexed: dict[str, dict[str, PairedVitaminCQuartet]] = {}
    for family in family_names:
        rows = paired_families[family]
        if not rows:
            raise ValueError("Stage A family inputs must be nonempty")
        family_index = {row.case_id: row for row in rows}
        if len(family_index) != len(rows):
            raise ValueError("Stage A family inputs contain duplicate cases")
        indexed[family] = family_index
    common_cases = set.intersection(*(set(indexed[name]) for name in family_names))
    if not common_cases:
        raise ValueError("Stage A families have no common cases")
    ordered_cases = tuple(sorted(common_cases))
    _validate_common_metadata(indexed, family_names, ordered_cases)
    common_pairs = {
        family: tuple(indexed[family][case_id] for case_id in ordered_cases)
        for family in family_names
    }
    effects = {
        family: page_effects(common_pairs[family]) for family in family_names
    }
    page_sets = [set(effects[family]) for family in family_names]
    if any(pages != page_sets[0] for pages in page_sets[1:]):
        raise ValueError("Stage A common cases do not yield identical pages")

    family_intervals = paired_page_bootstrap_intervals(
        effects, draws=draws, seed=seed
    )
    balanced_accuracy_tests = {
        family: paired_page_permutation_balanced_accuracy(
            _hard_rows(common_pairs[family]), draws=ba_draws, seed=seed
        )
        for family in family_names
    }
    balanced_accuracy_adjusted = holm_adjust(
        {
            family: test.pvalue
            for family, test in balanced_accuracy_tests.items()
        }
    )
    families: dict[str, object] = {}
    for family in family_names:
        interval = family_intervals[family]
        direction = page_direction_summary(effects[family])
        hard = paired_hard_label_summary(common_pairs[family])
        mass_f16 = _complete_mean(
            pair.label_probability_mass_f16 for pair in common_pairs[family]
        )
        mass_quantized = _complete_mean(
            pair.label_probability_mass_quantized for pair in common_pairs[family]
        )
        families[family] = {
            "quartets": len(common_pairs[family]),
            "pages": len(effects[family]),
            "page_effect": asdict(interval),
            "model_internal_standardized_effect": model_internal_standardized_effect(
                effects[family]
            ),
            "direction": asdict(direction),
            "hard_accuracy_f16": statistics.fmean(
                pair.hard_accuracy_f16 for pair in common_pairs[family]
            ),
            "hard_accuracy_quantized": statistics.fmean(
                pair.hard_accuracy_quantized for pair in common_pairs[family]
            ),
            "hard_accuracy_effect": statistics.fmean(
                pair.hard_accuracy_effect for pair in common_pairs[family]
            ),
            "hard_labels": asdict(hard),
            "balanced_accuracy": {
                **asdict(balanced_accuracy_tests[family]),
                "holm_adjusted_pvalue": balanced_accuracy_adjusted[family],
            },
            "mean_label_probability_mass_f16": mass_f16,
            "mean_label_probability_mass_quantized": mass_quantized,
            "mean_label_probability_mass_effect": mass_quantized - mass_f16,
        }

    pairwise_effects = {
        _comparison_name(left, right): family_effect_heterogeneity(
            effects[left], effects[right]
        )
        for left, right in itertools.combinations(family_names, 2)
    }
    pairwise_intervals = paired_page_bootstrap_intervals(
        pairwise_effects, draws=draws, seed=seed
    )
    pairwise_tests = {
        name: paired_page_wald_test(values)
        for name, values in pairwise_effects.items()
    }
    adjusted = holm_adjust(
        {name: result.pvalue for name, result in pairwise_tests.items()}
    )
    pairwise: dict[str, object] = {}
    for name in sorted(pairwise_effects):
        test = pairwise_tests[name]
        pairwise[name] = {
            "bootstrap_interval": asdict(pairwise_intervals[name]),
            "wald_test": asdict(test),
            "holm_adjusted_pvalue": adjusted[name],
        }

    direction_tests = {
        _comparison_name(left, right): paired_direction_mcnemar_test(
            effects[left], effects[right]
        )
        for left, right in itertools.combinations(family_names, 2)
    }
    direction_adjusted = holm_adjust(
        {name: result.pvalue for name, result in direction_tests.items()}
    )
    direction_pairwise = {
        name: {
            "mcnemar_test": asdict(direction_tests[name]),
            "holm_adjusted_pvalue": direction_adjusted[name],
        }
        for name in sorted(direction_tests)
    }

    return {
        "analysis_version": "vitaminc-stage-a-v4",
        "bootstrap_draws": draws,
        "balanced_accuracy_permutation_draws": ba_draws,
        "seed": seed,
        "family_order": family_names,
        "common_case_count": len(ordered_cases),
        "common_page_count": len(page_sets[0]),
        "families": families,
        "omnibus": asdict(omnibus_page_wald_test(effects)),
        "pairwise": pairwise,
        "direction_omnibus": asdict(matched_direction_omnibus(effects)),
        "direction_pairwise": direction_pairwise,
        "label_set_renormalization": {
            "interaction_invariant": True,
            "reason": (
                "For any cell, log softmax(A)-log softmax(negative) equals "
                "raw logp(A)-raw logp(negative); the common normalizer cancels."
            ),
        },
    }


def _comparison_name(left: str, right: str) -> str:
    return f"{left}-minus-{right}"


def _validate_common_metadata(
    indexed: Mapping[str, Mapping[str, PairedVitaminCQuartet]],
    families: Sequence[str],
    case_ids: Sequence[str],
) -> None:
    reference = families[0]
    for case_id in case_ids:
        expected = indexed[reference][case_id]
        for family in families[1:]:
            candidate = indexed[family][case_id]
            if (
                candidate.page != expected.page
                or candidate.negative_label != expected.negative_label
            ):
                raise ValueError("Stage A common-case metadata mismatch")


def _complete_mean(values: Sequence[float | None] | object) -> float:
    collected = tuple(values)
    if not collected or any(value is None for value in collected):
        raise ValueError("Stage A requires complete label-probability mass")
    return statistics.fmean(float(value) for value in collected)


def _hard_rows(
    pairs: Sequence[PairedVitaminCQuartet],
) -> tuple[HardPredictionPair, ...]:
    rows: list[HardPredictionPair] = []
    for pair in pairs:
        if not (
            pair.hard_predictions_f16
            and len(pair.hard_predictions_f16)
            == len(pair.hard_predictions_quantized)
            == len(pair.hard_gold_labels)
        ):
            raise ValueError("Stage A hard predictions are incomplete")
        for index, (f16, quantized, gold) in enumerate(
            zip(
                pair.hard_predictions_f16,
                pair.hard_predictions_quantized,
                pair.hard_gold_labels,
                strict=True,
            )
        ):
            if f16 is None or quantized is None:
                raise ValueError("Stage A hard predictions require full coverage")
            rows.append(
                HardPredictionPair(
                    sample_id=f"{pair.case_id}:{index}",
                    page=pair.page,
                    gold_label=gold,
                    prediction_f16=f16,
                    prediction_quantized=quantized,
                )
            )
    return tuple(rows)
