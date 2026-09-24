"""Deterministic edit-strength features for frozen VitaminC quartets."""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .multiplicity import holm_adjust
from .vitaminc import NEGATIVE_LABELS, VitaminCQuartet
from .vitaminc_results import PairedVitaminCQuartet
from .vitaminc_statistics import (
    page_effects,
    paired_page_bootstrap_interval,
    paired_page_wald_test,
)


EDIT_STRENGTH_PROTOCOL = "vitaminc-edit-strength-v1-20260719"

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)",
    flags=re.UNICODE,
)
_DATE_PATTERN = re.compile(
    r"(?ix)"
    r"(?<!\w)(?:"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?"
    r"\s+\d{4}"
    r")"
    r"(?!\w)"
)
_NEGATION_PATTERN = re.compile(r"\b\w+n['’]t\b", flags=re.IGNORECASE | re.UNICODE)
_NEGATION_TOKENS = frozenset(
    {
        "no",
        "not",
        "never",
        "neither",
        "nor",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "without",
        "否",
        "不",
        "没",
        "没有",
        "并非",
        "从未",
    }
)


@dataclass(frozen=True)
class VitaminCEditFeatures:
    """Outcome-independent edit features for one strict 2x2 quartet."""

    protocol_version: str
    case_id: str
    page: str
    negative_label: str
    claim_distance_absolute: int
    claim_distance_normalized: float
    evidence_distance_absolute: int
    evidence_distance_normalized: float
    claim_left_tokens: int
    claim_right_tokens: int
    evidence_left_tokens: int
    evidence_right_tokens: int
    claim_numeric_change: bool
    evidence_numeric_change: bool
    claim_date_change: bool
    evidence_date_change: bool
    claim_negation_change: bool
    evidence_negation_change: bool
    entity_status: str


@dataclass(frozen=True)
class _EditStrengthRow:
    case_id: str
    page: str
    negative_label: str
    effect: float
    claim_quartile: int
    evidence_quartile: int
    numeric_change: bool
    date_change: bool
    negation_change: bool


class EditStrengthAnalysisError(ValueError):
    """Raised when edit features cannot be joined to frozen paired outcomes."""


def tokenize_edit_text(text: str) -> tuple[str, ...]:
    """Tokenize with a frozen Unicode word-or-punctuation regex."""
    clean = _required_text(text, "edit text")
    return tuple(token.casefold() for token in _TOKEN_PATTERN.findall(clean))


def token_levenshtein_distance(
    left: Sequence[str], right: Sequence[str]
) -> int:
    """Return exact insertion/deletion/substitution distance between token lists."""
    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
        raise ValueError("token Levenshtein inputs must be token sequences")
    left_tokens = tuple(_required_text(token, "left token") for token in left)
    right_tokens = tuple(_required_text(token, "right token") for token in right)
    if len(left_tokens) > len(right_tokens):
        left_tokens, right_tokens = right_tokens, left_tokens
    previous = list(range(len(left_tokens) + 1))
    for right_index, right_token in enumerate(right_tokens, start=1):
        current = [right_index]
        for left_index, left_token in enumerate(left_tokens, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1]
                    + int(left_token.casefold() != right_token.casefold()),
                )
            )
        previous = current
    return previous[-1]


def extract_edit_features(quartet: VitaminCQuartet) -> VitaminCEditFeatures:
    """Extract frozen, prediction-independent edit features from one quartet."""
    if not isinstance(quartet, VitaminCQuartet):
        raise ValueError("edit features require a VitaminCQuartet")
    if len(quartet.claims) != 2:
        raise ValueError("VitaminC edit features require exactly two claims")
    if len(quartet.evidences) != 2:
        raise ValueError("VitaminC edit features require exactly two evidences")
    case_id = _required_text(quartet.case_id, "case_id")
    page = _required_text(quartet.page, "page")
    if quartet.negative_label not in NEGATIVE_LABELS:
        raise ValueError("VitaminC edit features require a supported negative label")
    claim_left = _required_text(quartet.claims[0], "claim")
    claim_right = _required_text(quartet.claims[1], "claim")
    evidence_left = _required_text(quartet.evidences[0], "evidence")
    evidence_right = _required_text(quartet.evidences[1], "evidence")
    claim_left_tokens = tokenize_edit_text(claim_left)
    claim_right_tokens = tokenize_edit_text(claim_right)
    evidence_left_tokens = tokenize_edit_text(evidence_left)
    evidence_right_tokens = tokenize_edit_text(evidence_right)
    claim_distance = token_levenshtein_distance(
        claim_left_tokens, claim_right_tokens
    )
    evidence_distance = token_levenshtein_distance(
        evidence_left_tokens, evidence_right_tokens
    )
    return VitaminCEditFeatures(
        protocol_version=EDIT_STRENGTH_PROTOCOL,
        case_id=case_id,
        page=page,
        negative_label=quartet.negative_label,
        claim_distance_absolute=claim_distance,
        claim_distance_normalized=_normalized_distance(
            claim_distance, claim_left_tokens, claim_right_tokens
        ),
        evidence_distance_absolute=evidence_distance,
        evidence_distance_normalized=_normalized_distance(
            evidence_distance, evidence_left_tokens, evidence_right_tokens
        ),
        claim_left_tokens=len(claim_left_tokens),
        claim_right_tokens=len(claim_right_tokens),
        evidence_left_tokens=len(evidence_left_tokens),
        evidence_right_tokens=len(evidence_right_tokens),
        claim_numeric_change=_signature(_NUMBER_PATTERN, claim_left)
        != _signature(_NUMBER_PATTERN, claim_right),
        evidence_numeric_change=_signature(_NUMBER_PATTERN, evidence_left)
        != _signature(_NUMBER_PATTERN, evidence_right),
        claim_date_change=_signature(_DATE_PATTERN, claim_left)
        != _signature(_DATE_PATTERN, claim_right),
        evidence_date_change=_signature(_DATE_PATTERN, evidence_left)
        != _signature(_DATE_PATTERN, evidence_right),
        claim_negation_change=_negation_signature(claim_left)
        != _negation_signature(claim_right),
        evidence_negation_change=_negation_signature(evidence_left)
        != _negation_signature(evidence_right),
        entity_status="unavailable",
    )


def freeze_quartile_cutpoints(values: Iterable[float]) -> tuple[float, float, float]:
    """Freeze linearly interpolated 25/50/75% cut points before outcome joins."""
    clean = sorted(_finite(value, "quartile value") for value in values)
    if not clean:
        raise ValueError("quartile values must be nonempty")
    return (
        _linear_quantile(clean, 0.25),
        _linear_quantile(clean, 0.50),
        _linear_quantile(clean, 0.75),
    )


def assign_frozen_quartile(
    value: float, cutpoints: Sequence[float]
) -> int:
    """Assign ties to the lower frozen bin; never adapt cut points by outcome."""
    number = _finite(value, "quartile value")
    if len(cutpoints) != 3:
        raise ValueError("quartile assignment requires three cutpoints")
    first, second, third = (
        _finite(item, "quartile cutpoint") for item in cutpoints
    )
    if not first <= second <= third:
        raise ValueError("quartile cutpoints must be nondecreasing")
    if number <= first:
        return 1
    if number <= second:
        return 2
    if number <= third:
        return 3
    return 4


def analyze_edit_strength(
    pairs: Sequence[PairedVitaminCQuartet],
    features: Sequence[VitaminCEditFeatures],
    *,
    draws: int = 10_000,
    seed: int = 20260711,
    page_weight_cap: int = 5,
) -> dict[str, Any]:
    """Run post-confirmatory edit robustness with shared page resamples."""
    if type(draws) is not int or draws < 1:
        raise EditStrengthAnalysisError(
            "edit-strength bootstrap draws must be a positive integer"
        )
    if type(seed) is not int:
        raise EditStrengthAnalysisError(
            "edit-strength bootstrap seed must be an integer"
        )
    if type(page_weight_cap) is not int or page_weight_cap < 1:
        raise EditStrengthAnalysisError(
            "edit-strength page weight cap must be a positive integer"
        )
    pair_by = _unique_pairs(pairs)
    feature_by = _unique_features(features)
    if set(pair_by) != set(feature_by):
        raise EditStrengthAnalysisError(
            "edit-strength pair and feature case identities mismatch"
        )

    claim_cutpoints = freeze_quartile_cutpoints(
        feature.claim_distance_normalized for feature in feature_by.values()
    )
    evidence_cutpoints = freeze_quartile_cutpoints(
        feature.evidence_distance_normalized for feature in feature_by.values()
    )
    rows: list[_EditStrengthRow] = []
    for case_id in sorted(pair_by):
        pair = pair_by[case_id]
        feature = feature_by[case_id]
        if (
            pair.page != feature.page
            or pair.negative_label != feature.negative_label
        ):
            raise EditStrengthAnalysisError(
                "edit-strength pair and feature metadata mismatch"
            )
        if pair.interaction_effect is None:
            raise EditStrengthAnalysisError(
                "edit-strength analysis requires complete interaction effects"
            )
        rows.append(
            _EditStrengthRow(
                case_id=case_id,
                page=pair.page,
                negative_label=pair.negative_label,
                effect=_finite(pair.interaction_effect, "interaction effect"),
                claim_quartile=assign_frozen_quartile(
                    feature.claim_distance_normalized, claim_cutpoints
                ),
                evidence_quartile=assign_frozen_quartile(
                    feature.evidence_distance_normalized, evidence_cutpoints
                ),
                numeric_change=(
                    feature.claim_numeric_change
                    or feature.evidence_numeric_change
                ),
                date_change=(
                    feature.claim_date_change or feature.evidence_date_change
                ),
                negation_change=(
                    feature.claim_negation_change
                    or feature.evidence_negation_change
                ),
            )
        )

    strata_rows = _build_strata(rows)
    shared_intervals = _shared_page_bootstrap(
        strata_rows, draws=draws, seed=seed
    )
    strata_output: dict[str, dict[str, dict[str, Any]]] = {}
    raw_pvalues: dict[str, float] = {}
    for dimension in sorted(strata_rows):
        strata_output[dimension] = {}
        for label in sorted(strata_rows[dimension]):
            key = f"{dimension}/{label}"
            selected = strata_rows[dimension][label]
            page_values = _page_means(selected)
            summary: dict[str, Any] = {
                "quartets": len(selected),
                "pages": len(page_values),
                "estimate": statistics.fmean(page_values.values()),
                "page_bootstrap": shared_intervals[key],
            }
            if len(page_values) >= 2:
                raw_pvalue = paired_page_wald_test(page_values).pvalue
                summary["raw_pvalue"] = raw_pvalue
                if dimension != "joint_quartile":
                    raw_pvalues[key] = raw_pvalue
            else:
                summary["raw_pvalue"] = None
            strata_output[dimension][label] = summary

    overall_page_effects = page_effects(tuple(pairs))
    label_counts = Counter(pair.negative_label for pair in pairs)
    leave_one_out = _leave_one_page_out(overall_page_effects)
    cap_effect = _page_weight_cap_effect(
        rows, cap=page_weight_cap
    )
    adjusted = holm_adjust(raw_pvalues) if raw_pvalues else {}
    for key, adjusted_value in adjusted.items():
        dimension, label = key.split("/", maxsplit=1)
        strata_output[dimension][label][
            "holm_adjusted_pvalue"
        ] = adjusted_value

    overall_estimate = statistics.fmean(overall_page_effects.values())
    return {
        "analysis_protocol": EDIT_STRENGTH_PROTOCOL,
        "inference_scope": "post_confirmatory_edit_robustness",
        "interpretation_boundary": (
            "Token edit distance is an operational difficulty measure, not "
            "semantic edit magnitude. This analysis cannot rewrite the "
            "original confirmatory hypothesis."
        ),
        "population": {
            "quartets": len(pairs),
            "pages": len(overall_page_effects),
            "negative_labels": dict(sorted(label_counts.items())),
        },
        "cutpoints": {
            "claim_distance_normalized": claim_cutpoints,
            "evidence_distance_normalized": evidence_cutpoints,
            "frozen_before_outcome_join": True,
            "tie_policy": "lower_bin",
        },
        "bootstrap_contract": {
            "draws": draws,
            "seed": seed,
            "cluster": "page",
            "shared_global_page_indices": True,
        },
        "overall": {
            "page_bootstrap": asdict(
                paired_page_bootstrap_interval(
                    overall_page_effects, draws=draws, seed=seed
                )
            )
        },
        "strata": strata_output,
        "robustness_family": {
            "scope": (
                "negative-label, marginal edit-quartile, and edit-type strata"
            ),
            "raw_pvalues": dict(sorted(raw_pvalues.items())),
            "holm_adjusted_pvalues": dict(sorted(adjusted.items())),
        },
        "high_volume_page_sensitivity": {
            "page_weight_cap": page_weight_cap,
            "capped_page_weighted_estimate": cap_effect,
            "uncapped_quartet_weighted_estimate": statistics.fmean(
                row.effect for row in rows
            ),
            "page_balanced_estimate": overall_estimate,
            "leave_one_page_out": leave_one_out,
        },
        "claim_narrowing_flags": {
            "leave_one_page_out_crosses_zero": leave_one_out["crosses_zero"],
            "negative_label_direction_instability": _dimension_has_both_signs(
                strata_output["negative_label"]
            ),
            "claim_quartile_direction_instability": _dimension_has_both_signs(
                strata_output["claim_quartile"]
            ),
            "evidence_quartile_direction_instability": _dimension_has_both_signs(
                strata_output["evidence_quartile"]
            ),
            "entity_stratum_omitted": True,
        },
    }


def _unique_pairs(
    pairs: Sequence[PairedVitaminCQuartet],
) -> dict[str, PairedVitaminCQuartet]:
    output = {pair.case_id: pair for pair in pairs}
    if not output or len(output) != len(pairs):
        raise EditStrengthAnalysisError(
            "edit-strength pairs require unique nonempty case identities"
        )
    return output


def _unique_features(
    features: Sequence[VitaminCEditFeatures],
) -> dict[str, VitaminCEditFeatures]:
    output = {feature.case_id: feature for feature in features}
    if not output or len(output) != len(features):
        raise EditStrengthAnalysisError(
            "edit-strength features require unique nonempty case identities"
        )
    for feature in output.values():
        if feature.protocol_version != EDIT_STRENGTH_PROTOCOL:
            raise EditStrengthAnalysisError(
                "edit-strength feature protocol mismatch"
            )
        if feature.entity_status != "unavailable":
            raise EditStrengthAnalysisError(
                "entity features require a separate versioned protocol"
            )
    return output


def _build_strata(
    rows: Sequence[_EditStrengthRow],
) -> dict[str, dict[str, tuple[_EditStrengthRow, ...]]]:
    output: dict[str, dict[str, list[_EditStrengthRow]]] = {
        "negative_label": defaultdict(list),
        "claim_quartile": defaultdict(list),
        "evidence_quartile": defaultdict(list),
        "joint_quartile": defaultdict(list),
        "numeric_change": defaultdict(list),
        "date_change": defaultdict(list),
        "negation_change": defaultdict(list),
    }
    for row in rows:
        output["negative_label"][row.negative_label].append(row)
        output["claim_quartile"][f"Q{row.claim_quartile}"].append(row)
        output["evidence_quartile"][f"Q{row.evidence_quartile}"].append(row)
        output["joint_quartile"][
            f"claim_Q{row.claim_quartile}__evidence_Q{row.evidence_quartile}"
        ].append(row)
        output["numeric_change"][
            "changed" if row.numeric_change else "unchanged"
        ].append(row)
        output["date_change"][
            "changed" if row.date_change else "unchanged"
        ].append(row)
        output["negation_change"][
            "changed" if row.negation_change else "unchanged"
        ].append(row)
    return {
        dimension: {
            label: tuple(selected)
            for label, selected in sorted(groups.items())
        }
        for dimension, groups in sorted(output.items())
    }


def _page_means(
    rows: Sequence[_EditStrengthRow],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.page].append(row.effect)
    return {
        page: statistics.fmean(values)
        for page, values in sorted(grouped.items())
    }


def _shared_page_bootstrap(
    strata: Mapping[str, Mapping[str, Sequence[_EditStrengthRow]]],
    *,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    keyed = {
        f"{dimension}/{label}": _page_means(selected)
        for dimension, groups in strata.items()
        for label, selected in groups.items()
    }
    pages = sorted(
        {
            row.page
            for groups in strata.values()
            for selected in groups.values()
            for row in selected
        }
    )
    if not pages:
        raise EditStrengthAnalysisError(
            "edit-strength bootstrap requires nonempty pages"
        )
    generator = random.Random(seed)
    sampled_indices = [
        tuple(generator.randrange(len(pages)) for _ in pages)
        for _ in range(draws)
    ]
    output: dict[str, dict[str, float | int]] = {}
    for key in sorted(keyed):
        page_values = keyed[key]
        replicates: list[float] = []
        for indices in sampled_indices:
            selected = [
                page_values[pages[index]]
                for index in indices
                if pages[index] in page_values
            ]
            if selected:
                replicates.append(statistics.fmean(selected))
        replicates.sort()
        if not replicates:
            raise EditStrengthAnalysisError(
                "edit-strength stratum has no valid bootstrap draws"
            )
        output[key] = {
            "estimate": statistics.fmean(page_values.values()),
            "lower": _linear_quantile(replicates, 0.025),
            "upper": _linear_quantile(replicates, 0.975),
            "draws": draws,
            "valid_draws": len(replicates),
            "seed": seed,
            "clusters": len(page_values),
        }
    return output


def _leave_one_page_out(
    effects: Mapping[str, float],
) -> dict[str, float | str | bool]:
    if len(effects) < 2:
        raise EditStrengthAnalysisError(
            "leave-one-page-out requires at least two pages"
        )
    observed = statistics.fmean(effects.values())
    estimates = {
        page: statistics.fmean(
            value for other, value in effects.items() if other != page
        )
        for page in sorted(effects)
    }
    maximum_shift_page = max(
        estimates,
        key=lambda page: (abs(estimates[page] - observed), page),
    )
    minimum = min(estimates.values())
    maximum = max(estimates.values())
    return {
        "minimum": minimum,
        "maximum": maximum,
        "crosses_zero": minimum <= 0.0 <= maximum,
        "maximum_absolute_shift": abs(
            estimates[maximum_shift_page] - observed
        ),
        "maximum_shift_page": maximum_shift_page,
    }


def _page_weight_cap_effect(
    rows: Sequence[_EditStrengthRow], *, cap: int
) -> float:
    page_values = _page_means(rows)
    counts = Counter(row.page for row in rows)
    weights = {page: min(counts[page], cap) for page in page_values}
    denominator = sum(weights.values())
    return sum(
        page_values[page] * weights[page] for page in sorted(page_values)
    ) / denominator


def _dimension_has_both_signs(
    summaries: Mapping[str, Mapping[str, Any]],
) -> bool:
    signs = {
        -1 if summary["estimate"] < 0.0 else 1
        for summary in summaries.values()
        if summary["estimate"] != 0.0
    }
    return signs == {-1, 1}


def _normalized_distance(
    distance: int, left: Sequence[str], right: Sequence[str]
) -> float:
    denominator = max(len(left), len(right), 1)
    return distance / denominator


def _signature(pattern: re.Pattern[str], text: str) -> Counter[str]:
    return Counter(match.group(0).casefold() for match in pattern.finditer(text))


def _negation_signature(text: str) -> Counter[str]:
    tokens = tokenize_edit_text(text)
    output = Counter(token for token in tokens if token in _NEGATION_TOKENS)
    output.update(match.group(0).casefold() for match in _NEGATION_PATTERN.finditer(text))
    return output


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
