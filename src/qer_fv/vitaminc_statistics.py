"""Frozen page-balanced paired inference for VitaminC formal comparisons."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .vitaminc_results import PairedVitaminCQuartet


@dataclass(frozen=True)
class VitaminCBootstrapInterval:
    estimate: float
    lower: float
    upper: float
    draws: int
    seed: int
    clusters: int


@dataclass(frozen=True)
class VitaminCWaldTest:
    estimate: float
    standard_error: float
    statistic: float
    pvalue: float
    clusters: int


@dataclass(frozen=True)
class VitaminCOmnibusWaldTest:
    models: tuple[str, ...]
    reference: str
    estimates: dict[str, float]
    contrasts: dict[str, float]
    statistic: float
    degrees_of_freedom: int
    pvalue: float
    clusters: int


@dataclass(frozen=True)
class VitaminCDirectionSummary:
    median: float
    negative: int
    positive: int
    zero: int
    negative_fraction: float
    sign_test_pvalue: float
    clusters: int


@dataclass(frozen=True)
class VitaminCDirectionOmnibus:
    models: tuple[str, ...]
    negative_counts: dict[str, int]
    statistic: float
    degrees_of_freedom: int
    pvalue: float
    clusters: int


@dataclass(frozen=True)
class VitaminCDirectionMcNemar:
    left_negative_right_nonnegative: int
    left_nonnegative_right_negative: int
    concordant: int
    statistic: float
    pvalue: float
    clusters: int


@dataclass(frozen=True)
class VitaminCHardLabelSummary:
    units: int
    agreement: float
    f16_correct_quantized_wrong: int
    f16_wrong_quantized_correct: int
    both_correct: int
    both_wrong: int
    transitions: dict[str, int]
    mcnemar_exact_pvalue: float


def page_effects(
    pairs: Sequence[PairedVitaminCQuartet],
) -> dict[str, float]:
    """Average quartet effects within page before any between-page averaging."""
    if not pairs:
        raise ValueError("VitaminC paired quartets must be nonempty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        if pair.interaction_effect is None:
            raise ValueError("page effects require complete interaction pairs")
        if not isinstance(pair.page, str) or not pair.page:
            raise ValueError("VitaminC page identity must be nonempty")
        grouped[pair.page].append(_finite(pair.interaction_effect))
    return {
        page: statistics.fmean(values) for page, values in sorted(grouped.items())
    }


def page_balanced_effect(effects: Mapping[str, float]) -> float:
    """Return the equally page-weighted mean effect."""
    values = _effect_values(effects)
    return statistics.fmean(values)


def quartet_weighted_effect(pairs: Sequence[PairedVitaminCQuartet]) -> float:
    """Return the preregistered quartet-weighted sensitivity estimate."""
    if not pairs:
        raise ValueError("VitaminC paired quartets must be nonempty")
    values: list[float] = []
    for pair in pairs:
        if pair.interaction_effect is None:
            raise ValueError("quartet-weighted effect requires complete interactions")
        values.append(_finite(pair.interaction_effect))
    return statistics.fmean(values)


def paired_page_wald_test(effects: Mapping[str, float]) -> VitaminCWaldTest:
    """Test a paired page-level mean against zero with a cluster-level Wald test."""
    values = _effect_values(effects)
    if len(values) < 2:
        raise ValueError("VitaminC Wald test requires at least two pages")
    estimate = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    if standard_error == 0.0:
        statistic = math.inf if estimate != 0.0 else 0.0
        pvalue = 0.0 if estimate != 0.0 else 1.0
    else:
        z_score = estimate / standard_error
        statistic = z_score * z_score
        pvalue = _chi_square_survival(statistic, 1)
    return VitaminCWaldTest(
        estimate=estimate,
        standard_error=standard_error,
        statistic=statistic,
        pvalue=pvalue,
        clusters=len(values),
    )


def omnibus_page_wald_test(
    model_effects: Mapping[str, Mapping[str, float]],
) -> VitaminCOmnibusWaldTest:
    """Test equality of matched page-level mean effects across all models."""
    if len(model_effects) < 2:
        raise ValueError("VitaminC omnibus test requires at least two models")
    models = tuple(sorted(str(model) for model in model_effects))
    if len(models) != len(model_effects):
        raise ValueError("VitaminC model identities must be unique strings")
    page_sets = [set(model_effects[model]) for model in models]
    if not page_sets[0] or any(pages != page_sets[0] for pages in page_sets[1:]):
        raise ValueError("omnibus model effects must share identical nonempty pages")
    pages = sorted(page_sets[0])
    if len(pages) <= len(models) - 1:
        raise ValueError("VitaminC omnibus test has too few page clusters")
    reference = models[0]
    estimates = {
        model: statistics.fmean(
            _finite(model_effects[model][page]) for page in pages
        )
        for model in models
    }
    contrast_names = models[1:]
    contrast_rows = [
        [
            _finite(model_effects[model][page])
            - _finite(model_effects[reference][page])
            for model in contrast_names
        ]
        for page in pages
    ]
    contrast_means = [
        statistics.fmean(row[index] for row in contrast_rows)
        for index in range(len(contrast_names))
    ]
    covariance = _sample_covariance_matrix(contrast_rows)
    covariance_of_mean = [
        [value / len(pages) for value in row] for row in covariance
    ]
    solution = _solve_linear_system(covariance_of_mean, contrast_means)
    statistic = sum(
        contrast_means[index] * solution[index]
        for index in range(len(contrast_means))
    )
    statistic = max(0.0, statistic)
    degrees_of_freedom = len(contrast_means)
    return VitaminCOmnibusWaldTest(
        models=models,
        reference=reference,
        estimates=estimates,
        contrasts={
            f"{model}-minus-{reference}": contrast_means[index]
            for index, model in enumerate(contrast_names)
        },
        statistic=statistic,
        degrees_of_freedom=degrees_of_freedom,
        pvalue=_chi_square_survival(statistic, degrees_of_freedom),
        clusters=len(pages),
    )


def page_direction_summary(
    effects: Mapping[str, float],
) -> VitaminCDirectionSummary:
    """Summarize page-level direction and an exact two-sided sign test."""
    values = _effect_values(effects)
    negative = sum(value < 0.0 for value in values)
    positive = sum(value > 0.0 for value in values)
    zero = len(values) - negative - positive
    nonzero = negative + positive
    pvalue = 1.0
    if nonzero:
        pvalue = _two_sided_sign_pvalue(negative, positive)
    return VitaminCDirectionSummary(
        median=statistics.median(values),
        negative=negative,
        positive=positive,
        zero=zero,
        negative_fraction=negative / len(values),
        sign_test_pvalue=pvalue,
        clusters=len(values),
    )


def matched_direction_omnibus(
    model_effects: Mapping[str, Mapping[str, float]],
) -> VitaminCDirectionOmnibus:
    """Cochran's Q test for matched negative-effect indicators across models."""
    if len(model_effects) < 2:
        raise ValueError("direction omnibus requires at least two models")
    models = tuple(sorted(str(model) for model in model_effects))
    page_sets = [set(model_effects[model]) for model in models]
    if not page_sets[0] or any(pages != page_sets[0] for pages in page_sets[1:]):
        raise ValueError("direction model effects must share identical nonempty pages")
    pages = sorted(page_sets[0])
    indicators = [
        [
            int(_finite(model_effects[model][page]) < 0.0)
            for model in models
        ]
        for page in pages
    ]
    column_totals = [
        sum(row[index] for row in indicators) for index in range(len(models))
    ]
    row_totals = [sum(row) for row in indicators]
    total = sum(column_totals)
    denominator = len(models) * total - sum(value * value for value in row_totals)
    if denominator == 0:
        raise ValueError("direction omnibus is undefined without discordance")
    numerator = (len(models) - 1) * (
        len(models) * sum(value * value for value in column_totals)
        - total * total
    )
    statistic = numerator / denominator
    degrees_of_freedom = len(models) - 1
    return VitaminCDirectionOmnibus(
        models=models,
        negative_counts=dict(zip(models, column_totals, strict=True)),
        statistic=statistic,
        degrees_of_freedom=degrees_of_freedom,
        pvalue=_chi_square_survival(statistic, degrees_of_freedom),
        clusters=len(pages),
    )


def paired_direction_mcnemar_test(
    left_effects: Mapping[str, float],
    right_effects: Mapping[str, float],
) -> VitaminCDirectionMcNemar:
    """Exact paired test for different negative-effect rates between two models."""
    if not left_effects or set(left_effects) != set(right_effects):
        raise ValueError("direction pair must share identical nonempty pages")
    left_negative_right_nonnegative = 0
    left_nonnegative_right_negative = 0
    concordant = 0
    for page in sorted(left_effects):
        left_negative = _finite(left_effects[page]) < 0.0
        right_negative = _finite(right_effects[page]) < 0.0
        if left_negative and not right_negative:
            left_negative_right_nonnegative += 1
        elif right_negative and not left_negative:
            left_nonnegative_right_negative += 1
        else:
            concordant += 1
    discordant = (
        left_negative_right_nonnegative + left_nonnegative_right_negative
    )
    statistic = 0.0
    if discordant:
        statistic = (
            abs(left_negative_right_nonnegative - left_nonnegative_right_negative)
            - 1.0
        ) ** 2 / discordant
    return VitaminCDirectionMcNemar(
        left_negative_right_nonnegative=left_negative_right_nonnegative,
        left_nonnegative_right_negative=left_nonnegative_right_negative,
        concordant=concordant,
        statistic=statistic,
        pvalue=_two_sided_sign_pvalue(
            left_negative_right_nonnegative,
            left_nonnegative_right_negative,
        ),
        clusters=len(left_effects),
    )


def model_internal_standardized_effect(effects: Mapping[str, float]) -> float:
    """Return the page-level mean divided by its within-model sample deviation."""
    values = _effect_values(effects)
    if len(values) < 2:
        raise ValueError("standardized effect requires at least two pages")
    deviation = statistics.stdev(values)
    if deviation == 0.0:
        raise ValueError("standardized effect is undefined at zero deviation")
    return statistics.fmean(values) / deviation


def paired_hard_label_summary(
    pairs: Sequence[PairedVitaminCQuartet],
) -> VitaminCHardLabelSummary:
    """Summarize paired cell-level hard-label transitions without re-inference."""
    if not pairs:
        raise ValueError("VitaminC hard-label pairs must be nonempty")
    transitions: dict[str, int] = defaultdict(int)
    agreement = 0
    f16_correct_quantized_wrong = 0
    f16_wrong_quantized_correct = 0
    both_correct = 0
    both_wrong = 0
    units = 0
    for pair in pairs:
        left = pair.hard_predictions_f16
        right = pair.hard_predictions_quantized
        gold = pair.hard_gold_labels
        if not left or len(left) != len(right) or len(left) != len(gold):
            raise ValueError("VitaminC hard-label pairs are incomplete")
        for f16_label, quantized_label, gold_label in zip(left, right, gold):
            if f16_label is None or quantized_label is None:
                raise ValueError("VitaminC hard-label pairs require complete coverage")
            transitions[f"{f16_label}->{quantized_label}"] += 1
            agreement += f16_label == quantized_label
            f16_correct = f16_label == gold_label
            quantized_correct = quantized_label == gold_label
            if f16_correct and quantized_correct:
                both_correct += 1
            elif f16_correct:
                f16_correct_quantized_wrong += 1
            elif quantized_correct:
                f16_wrong_quantized_correct += 1
            else:
                both_wrong += 1
            units += 1
    return VitaminCHardLabelSummary(
        units=units,
        agreement=agreement / units,
        f16_correct_quantized_wrong=f16_correct_quantized_wrong,
        f16_wrong_quantized_correct=f16_wrong_quantized_correct,
        both_correct=both_correct,
        both_wrong=both_wrong,
        transitions=dict(sorted(transitions.items())),
        mcnemar_exact_pvalue=_two_sided_sign_pvalue(
            f16_correct_quantized_wrong,
            f16_wrong_quantized_correct,
        ),
    )


def paired_page_bootstrap_interval(
    effects: Mapping[str, float],
    *,
    draws: int = 10_000,
    seed: int = 20260711,
) -> VitaminCBootstrapInterval:
    """Bootstrap pages with replacement under the frozen seed and draw count."""
    return paired_page_bootstrap_intervals(
        {"effect": effects}, draws=draws, seed=seed
    )["effect"]


def paired_page_bootstrap_intervals(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, VitaminCBootstrapInterval]:
    """Use identical sampled page indices for every supplied paired metric."""
    if not metrics:
        raise ValueError("VitaminC bootstrap metrics must be nonempty")
    if type(draws) is not int or draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    names = sorted(metrics)
    page_sets = [set(metrics[name]) for name in names]
    if not page_sets[0] or any(pages != page_sets[0] for pages in page_sets[1:]):
        raise ValueError("bootstrap metrics must share identical nonempty pages")
    pages = sorted(page_sets[0])
    values = {
        name: tuple(_finite(metrics[name][page]) for page in pages) for name in names
    }
    generator = random.Random(seed)
    replicates: dict[str, list[float]] = {name: [] for name in names}
    n_pages = len(pages)
    for _ in range(draws):
        indices = [generator.randrange(n_pages) for _ in range(n_pages)]
        for name in names:
            metric_values = values[name]
            replicates[name].append(
                statistics.fmean(metric_values[index] for index in indices)
            )
    output: dict[str, VitaminCBootstrapInterval] = {}
    for name in names:
        sorted_replicates = sorted(replicates[name])
        output[name] = VitaminCBootstrapInterval(
            estimate=statistics.fmean(values[name]),
            lower=_quantile(sorted_replicates, 0.025),
            upper=_quantile(sorted_replicates, 0.975),
            draws=draws,
            seed=seed,
            clusters=n_pages,
        )
    return output


def _effect_values(effects: Mapping[str, float]) -> tuple[float, ...]:
    if not effects:
        raise ValueError("VitaminC page effects must be nonempty")
    if any(not isinstance(page, str) or not page for page in effects):
        raise ValueError("VitaminC page identities must be nonempty")
    return tuple(_finite(value) for _, value in sorted(effects.items()))


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("VitaminC effects must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("VitaminC effects must be finite")
    return number


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _sample_covariance_matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(rows) < 2 or not rows or not rows[0]:
        raise ValueError("covariance requires at least two nonempty rows")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("covariance rows must have equal width")
    means = [
        statistics.fmean(row[index] for row in rows) for index in range(width)
    ]
    denominator = len(rows) - 1
    return [
        [
            sum(
                (row[left] - means[left]) * (row[right] - means[right])
                for row in rows
            )
            / denominator
            for right in range(width)
        ]
        for left in range(width)
    ]


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    size = len(vector)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("Wald covariance matrix shape mismatch")
    augmented = [
        [_finite(value) for value in matrix[row]] + [_finite(vector[row])]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise ValueError("Wald covariance matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


def _chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    if statistic < 0.0 or not math.isfinite(statistic):
        return 0.0 if statistic == math.inf else 1.0
    if type(degrees_of_freedom) is not int or degrees_of_freedom < 1:
        raise ValueError("chi-square degrees of freedom must be positive")
    x = statistic / 2.0
    if degrees_of_freedom % 2:
        shape = 0.5
        survival = math.erfc(math.sqrt(x))
    else:
        shape = 1.0
        survival = math.exp(-x)
    target_shape = degrees_of_freedom / 2.0
    while shape < target_shape:
        survival += math.exp(shape * math.log(x) - x - math.lgamma(shape + 1.0)) if x else 0.0
        shape += 1.0
    return min(1.0, max(0.0, survival))


def _two_sided_sign_pvalue(negative: int, positive: int) -> float:
    total = negative + positive
    if total == 0:
        return 1.0
    smaller = min(negative, positive)
    log_terms = [
        math.lgamma(total + 1.0)
        - math.lgamma(k + 1.0)
        - math.lgamma(total - k + 1.0)
        - total * math.log(2.0)
        for k in range(smaller + 1)
    ]
    maximum = max(log_terms)
    log_lower_tail = maximum + math.log(
        sum(math.exp(value - maximum) for value in log_terms)
    )
    return min(1.0, 2.0 * math.exp(log_lower_tail))
