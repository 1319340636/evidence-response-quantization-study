"""Frozen query-cluster statistics for paired CUB conditions."""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PairedCubObservation:
    sample_id: str
    query_identity_sha256: str
    analysis_context_type: str
    bcu_f16: bool
    bcu_quantized: bool
    ccu_f16: float | None
    ccu_quantized: float | None
    hard_ok_f16: bool = True
    hard_ok_quantized: bool = True
    probability_ok_f16: bool = True
    probability_ok_quantized: bool = True


@dataclass(frozen=True)
class EffectBounds:
    lower: float
    upper: float
    observed_pairs: int
    total_pairs: int


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    draws: int
    seed: int


def cluster_effects(
    rows: Sequence[PairedCubObservation], *, metric: str, context_type: str
) -> dict[str, float]:
    selected = [row for row in rows if row.analysis_context_type == context_type]
    if not selected:
        raise ValueError("no rows for requested CUB context type")
    values: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        if metric == "bcu":
            if type(row.bcu_f16) is not bool or type(row.bcu_quantized) is not bool:
                raise ValueError("BCU values must be boolean")
            effect = float(row.bcu_quantized) - float(row.bcu_f16)
        elif metric == "ccu":
            if row.ccu_f16 is None or row.ccu_quantized is None:
                raise ValueError("CCU cluster effects require complete CCU pairs")
            f16 = _ccu(row.ccu_f16)
            quantized = _ccu(row.ccu_quantized)
            effect = quantized - f16
        else:
            raise ValueError("metric must be bcu or ccu")
        values[row.query_identity_sha256].append(effect)
    return {
        query: statistics.fmean(query_values)
        for query, query_values in sorted(values.items())
    }


def ccu_missing_effect_bounds(
    rows: Sequence[PairedCubObservation], *, context_type: str
) -> EffectBounds:
    selected = [row for row in rows if row.analysis_context_type == context_type]
    if not selected:
        raise ValueError("no rows for requested CUB context type")
    cluster_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    observed = 0
    for row in selected:
        f16 = _score_interval(row.ccu_f16)
        quantized = _score_interval(row.ccu_quantized)
        if row.ccu_f16 is not None and row.ccu_quantized is not None:
            observed += 1
        cluster_intervals[row.query_identity_sha256].append(
            (quantized[0] - f16[1], quantized[1] - f16[0])
        )
    lowers = [
        statistics.fmean(interval[0] for interval in intervals)
        for intervals in cluster_intervals.values()
    ]
    uppers = [
        statistics.fmean(interval[1] for interval in intervals)
        for intervals in cluster_intervals.values()
    ]
    return EffectBounds(
        lower=statistics.fmean(lowers),
        upper=statistics.fmean(uppers),
        observed_pairs=observed,
        total_pairs=len(selected),
    )


def paired_cluster_bootstrap_interval(
    effects: Mapping[str, float], *, draws: int = 10_000, seed: int = 20260711
) -> BootstrapInterval:
    values = _effect_values(effects)
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    generator = random.Random(seed)
    replicates = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(draws)
    )
    return BootstrapInterval(
        estimate=statistics.fmean(values),
        lower=_quantile(replicates, 0.025),
        upper=_quantile(replicates, 0.975),
        draws=draws,
        seed=seed,
    )


def paired_cluster_sign_flip_pvalue(
    effects: Mapping[str, float], *, draws: int = 100_000, seed: int = 20260711
) -> float:
    values = _effect_values(effects)
    observed = abs(statistics.fmean(values))
    tolerance = 1e-15
    if len(values) <= 20:
        total = 2 ** len(values)
        extreme = sum(
            abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
            >= observed - tolerance
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        )
        return extreme / total
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
        raise ValueError("sign-flip draws must be a positive integer")
    generator = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        permuted = statistics.fmean(
            value if generator.getrandbits(1) else -value for value in values
        )
        extreme += abs(permuted) >= observed - tolerance
    return (extreme + 1) / (draws + 1)


def _effect_values(effects: Mapping[str, float]) -> tuple[float, ...]:
    if not effects:
        raise ValueError("cluster effects must be nonempty")
    values = tuple(float(value) for _, value in sorted(effects.items()))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cluster effects must be finite")
    return values


def _score_interval(value: float | None) -> tuple[float, float]:
    if value is None:
        return (-1.0, 1.0)
    score = _ccu(value)
    return (score, score)


def _ccu(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("CCU value must be a finite score in [-1, 1]")
    score = float(value)
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("CCU value must be a finite score in [-1, 1]")
    return score


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
