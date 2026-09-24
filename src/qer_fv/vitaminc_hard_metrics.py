"""Frozen three-class hard-label metrics for paired VitaminC conditions."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence


VITAMINC_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


@dataclass(frozen=True)
class HardPredictionPair:
    sample_id: str
    page: str
    gold_label: str
    prediction_f16: str
    prediction_quantized: str


@dataclass(frozen=True)
class BalancedAccuracyBootstrap:
    f16: float
    quantized: float
    difference: float
    lower: float
    upper: float
    draws: int
    seed: int
    clusters: int


@dataclass(frozen=True)
class BalancedAccuracyPermutation:
    f16: float
    quantized: float
    difference: float
    statistic: float
    pvalue: float
    draws: int
    seed: int
    clusters: int
    exact: bool


def confusion_matrix(
    rows: Sequence[HardPredictionPair],
    *,
    prediction: Literal["f16", "quantized"],
) -> tuple[tuple[int, ...], ...]:
    """Return a fixed gold-row, predicted-column A/B/C confusion matrix."""
    if not rows:
        raise ValueError("hard prediction rows must be nonempty")
    label_index = {label: index for index, label in enumerate(VITAMINC_LABELS)}
    matrix = [[0 for _ in VITAMINC_LABELS] for _ in VITAMINC_LABELS]
    attribute = (
        "prediction_f16" if prediction == "f16" else "prediction_quantized"
    )
    for row in rows:
        try:
            gold_index = label_index[row.gold_label]
            predicted_index = label_index[getattr(row, attribute)]
        except (KeyError, AttributeError) as error:
            raise ValueError("hard prediction contains an unsupported label") from error
        matrix[gold_index][predicted_index] += 1
    return tuple(tuple(values) for values in matrix)


def balanced_accuracy(matrix: Sequence[Sequence[int]]) -> float:
    """Return macro recall across the three frozen VitaminC gold classes."""
    values = _validated_matrix(matrix)
    recalls: list[float] = []
    for index, row in enumerate(values):
        total = sum(row)
        if total == 0:
            raise ValueError("balanced accuracy requires every gold class")
        recalls.append(row[index] / total)
    return sum(recalls) / len(recalls)


def multiclass_mcc(matrix: Sequence[Sequence[int]]) -> float:
    """Return the standard multiclass Matthews correlation coefficient."""
    values = _validated_matrix(matrix)
    total = sum(sum(row) for row in values)
    if total == 0:
        raise ValueError("MCC requires at least one observation")
    correct = sum(values[index][index] for index in range(len(values)))
    true_totals = [sum(row) for row in values]
    predicted_totals = [
        sum(values[row][column] for row in range(len(values)))
        for column in range(len(values))
    ]
    numerator = correct * total - sum(
        predicted * true
        for predicted, true in zip(predicted_totals, true_totals, strict=True)
    )
    first = total * total - sum(value * value for value in predicted_totals)
    second = total * total - sum(value * value for value in true_totals)
    denominator = math.sqrt(first * second)
    return 0.0 if denominator == 0.0 else numerator / denominator


def paired_page_bootstrap_balanced_accuracy(
    rows: Sequence[HardPredictionPair],
    *,
    draws: int = 10_000,
    seed: int = 20260711,
) -> BalancedAccuracyBootstrap:
    """Bootstrap whole pages and evaluate the paired BA difference."""
    if type(draws) is not int or draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")
    f16_observed = balanced_accuracy(confusion_matrix(rows, prediction="f16"))
    quantized_observed = balanced_accuracy(
        confusion_matrix(rows, prediction="quantized")
    )
    grouped: dict[str, list[HardPredictionPair]] = defaultdict(list)
    for row in rows:
        if not isinstance(row.page, str) or not row.page:
            raise ValueError("hard prediction page must be nonempty")
        grouped[row.page].append(row)
    pages = sorted(grouped)
    page_matrices = [
        (
            confusion_matrix(grouped[page], prediction="f16"),
            confusion_matrix(grouped[page], prediction="quantized"),
        )
        for page in pages
    ]
    generator = random.Random(seed)
    differences: list[float] = []
    for _ in range(draws):
        f16_matrix = [[0] * 3 for _ in range(3)]
        quantized_matrix = [[0] * 3 for _ in range(3)]
        for _ in pages:
            first, second = page_matrices[generator.randrange(len(pages))]
            _add_matrix(f16_matrix, first)
            _add_matrix(quantized_matrix, second)
        differences.append(
            balanced_accuracy(quantized_matrix) - balanced_accuracy(f16_matrix)
        )
    differences.sort()
    return BalancedAccuracyBootstrap(
        f16=f16_observed,
        quantized=quantized_observed,
        difference=quantized_observed - f16_observed,
        lower=_quantile(differences, 0.025),
        upper=_quantile(differences, 0.975),
        draws=draws,
        seed=seed,
        clusters=len(pages),
    )


def paired_page_permutation_balanced_accuracy(
    rows: Sequence[HardPredictionPair],
    *,
    draws: int = 10_000,
    seed: int = 20260711,
) -> BalancedAccuracyPermutation:
    """Swap paired page conditions under the sharp no-condition-effect null."""
    if type(draws) is not int or draws < 1:
        raise ValueError("permutation draws must be a positive integer")
    if type(seed) is not int:
        raise ValueError("permutation seed must be an integer")
    f16_observed = balanced_accuracy(confusion_matrix(rows, prediction="f16"))
    quantized_observed = balanced_accuracy(
        confusion_matrix(rows, prediction="quantized")
    )
    observed = quantized_observed - f16_observed
    grouped: dict[str, list[HardPredictionPair]] = defaultdict(list)
    for row in rows:
        if not isinstance(row.page, str) or not row.page:
            raise ValueError("hard prediction page must be nonempty")
        grouped[row.page].append(row)
    pages = sorted(grouped)
    gold_totals = {
        label: sum(row.gold_label == label for row in rows)
        for label in VITAMINC_LABELS
    }
    if any(total == 0 for total in gold_totals.values()):
        raise ValueError("balanced accuracy requires every gold class")
    page_contributions = [
        sum(
            (
                sum(
                    row.gold_label == label
                    and row.prediction_quantized == label
                    for row in grouped[page]
                )
                - sum(
                    row.gold_label == label and row.prediction_f16 == label
                    for row in grouped[page]
                )
            )
            / gold_totals[label]
            for label in VITAMINC_LABELS
        )
        / len(VITAMINC_LABELS)
        for page in pages
    ]
    if not math.isclose(
        sum(page_contributions), observed, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("page contributions do not reconstruct BA difference")
    threshold = abs(observed) - 1e-15
    if len(pages) <= 20:
        total_draws = 1 << len(pages)
        extreme = sum(
            abs(_permuted_ba_difference(page_contributions, mask=mask)) >= threshold
            for mask in range(total_draws)
        )
        pvalue = extreme / total_draws
        exact = True
    else:
        generator = random.Random(seed)
        extreme = sum(
            abs(_permuted_ba_difference(page_contributions, generator=generator))
            >= threshold
            for _ in range(draws)
        )
        total_draws = draws
        pvalue = (extreme + 1.0) / (draws + 1.0)
        exact = False
    return BalancedAccuracyPermutation(
        f16=f16_observed,
        quantized=quantized_observed,
        difference=observed,
        statistic=abs(observed),
        pvalue=pvalue,
        draws=total_draws,
        seed=seed,
        clusters=len(pages),
        exact=exact,
    )


def _validated_matrix(
    matrix: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if len(matrix) != len(VITAMINC_LABELS):
        raise ValueError("confusion matrix must be 3x3")
    output: list[tuple[int, ...]] = []
    for row in matrix:
        if len(row) != len(VITAMINC_LABELS):
            raise ValueError("confusion matrix must be 3x3")
        if any(type(value) is not int or value < 0 for value in row):
            raise ValueError("confusion counts must be nonnegative integers")
        output.append(tuple(row))
    return tuple(output)


def _add_matrix(target: list[list[int]], source: Sequence[Sequence[int]]) -> None:
    for row in range(3):
        for column in range(3):
            target[row][column] += source[row][column]


def _permuted_ba_difference(
    page_contributions: Sequence[float],
    *,
    mask: int | None = None,
    generator: random.Random | None = None,
) -> float:
    if (mask is None) == (generator is None):
        raise ValueError("permutation requires exactly one swap source")
    difference = 0.0
    for index, contribution in enumerate(page_contributions):
        swap = bool(mask & (1 << index)) if mask is not None else bool(generator.getrandbits(1))
        difference += -contribution if swap else contribution
    return difference


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
