"""Descriptive comparability audit for matched HF and GGUF FP16 endpoints."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

from .vitaminc_hard_metrics import (
    VITAMINC_LABELS,
    HardPredictionPair,
    balanced_accuracy,
    confusion_matrix,
    multiclass_mcc,
)
from .vitaminc_results import VitaminCConditionResults, VitaminCQuartetResult
from .vitaminc_statistics import paired_page_bootstrap_interval


BACKEND_COMPARABILITY_PROTOCOL = "vitaminc-backend-comparability-v1-20260719"


class BackendComparabilityError(ValueError):
    """Raised when FP16 endpoints cannot enter a matched descriptive audit."""


def derive_condition_subset(
    condition: VitaminCConditionResults,
    *,
    case_ids: Sequence[str],
    split: str,
    split_sha256: str,
) -> VitaminCConditionResults:
    """Derive an exact frozen subset while preserving endpoint provenance."""
    ordered_ids = tuple(case_ids)
    if (
        not ordered_ids
        or any(not isinstance(case_id, str) or not case_id for case_id in ordered_ids)
    ):
        raise BackendComparabilityError(
            "condition subset requires nonempty case identities"
        )
    if len(set(ordered_ids)) != len(ordered_ids):
        raise BackendComparabilityError(
            "condition subset case identities contain duplicates"
        )
    if not isinstance(split, str) or not split:
        raise BackendComparabilityError(
            "condition subset split identity is invalid"
        )
    if (
        not isinstance(split_sha256, str)
        or len(split_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in split_sha256)
    ):
        raise BackendComparabilityError(
            "condition subset split hash is invalid"
        )
    source = _unique_quartets(condition)
    missing = [case_id for case_id in ordered_ids if case_id not in source]
    if missing:
        raise BackendComparabilityError(
            "condition subset case identities are missing from source"
        )
    return replace(
        condition,
        split=split,
        split_sha256=split_sha256.lower(),
        quartets=tuple(source[case_id] for case_id in ordered_ids),
    )


def compare_high_precision_backends(
    *,
    hf_fp16: VitaminCConditionResults,
    gguf_f16: VitaminCConditionResults,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Compare matched high-precision endpoints without claiming equivalence."""
    _validate_condition_identity(hf_fp16, gguf_f16)
    hf_by = _unique_quartets(hf_fp16)
    gguf_by = _unique_quartets(gguf_f16)
    if set(hf_by) != set(gguf_by):
        raise BackendComparabilityError(
            "backend comparability requires identical quartet identities"
        )

    hard_rows: list[HardPredictionPair] = []
    transitions: Counter[str] = Counter()
    agreement = 0
    hf_page_interactions: dict[str, list[float]] = defaultdict(list)
    gguf_page_interactions: dict[str, list[float]] = defaultdict(list)
    hf_masses: list[float] = []
    gguf_masses: list[float] = []
    for case_id in sorted(hf_by):
        hf = hf_by[case_id]
        gguf = gguf_by[case_id]
        _validate_quartet_match(hf, gguf)
        if hf.interaction is None or gguf.interaction is None:
            raise BackendComparabilityError(
                "backend comparability requires complete probability coverage"
            )
        if (
            hf.mean_label_probability_mass is None
            or gguf.mean_label_probability_mass is None
        ):
            raise BackendComparabilityError(
                "backend comparability requires complete label probability mass"
            )
        hf_page_interactions[hf.page].append(
            _finite(hf.interaction, "HF interaction")
        )
        gguf_page_interactions[gguf.page].append(
            _finite(gguf.interaction, "GGUF interaction")
        )
        hf_masses.append(
            _unit_interval(hf.mean_label_probability_mass, "HF label mass")
        )
        gguf_masses.append(
            _unit_interval(gguf.mean_label_probability_mass, "GGUF label mass")
        )
        for cell_index, (hf_label, gguf_label, gold_label) in enumerate(
            zip(
                hf.hard_predictions,
                gguf.hard_predictions,
                hf.hard_gold_labels,
                strict=True,
            )
        ):
            if hf_label is None or gguf_label is None:
                raise BackendComparabilityError(
                    "backend comparability requires complete hard-label coverage"
                )
            transitions[f"{hf_label}->{gguf_label}"] += 1
            agreement += hf_label == gguf_label
            hard_rows.append(
                HardPredictionPair(
                    sample_id=f"{case_id}:cell:{cell_index}",
                    page=hf.page,
                    gold_label=gold_label,
                    prediction_f16=hf_label,
                    prediction_quantized=gguf_label,
                )
            )

    hf_page = {
        page: statistics.fmean(values)
        for page, values in sorted(hf_page_interactions.items())
    }
    gguf_page = {
        page: statistics.fmean(values)
        for page, values in sorted(gguf_page_interactions.items())
    }
    if set(hf_page) != set(gguf_page):
        raise BackendComparabilityError(
            "backend comparability page identities are not matched"
        )
    hf_minus_gguf = {
        page: hf_page[page] - gguf_page[page] for page in sorted(hf_page)
    }
    hf_matrix = confusion_matrix(hard_rows, prediction="f16")
    gguf_matrix = confusion_matrix(hard_rows, prediction="quantized")
    return {
        "analysis_protocol": BACKEND_COMPARABILITY_PROTOCOL,
        "inference_scope": "descriptive_backend_comparability",
        "interpretation_boundary": (
            "This audit describes matched HF FP16 and GGUF F16 behavior. It "
            "does not establish statistical equivalence because no prospective "
            "equivalence margin was frozen."
        ),
        "route_claim_boundary": (
            "Any later HF GPTQ versus GGUF Q4 contrast remains a joint "
            "deployment-route effect, not a pure quantizer or bit-width effect."
        ),
        "parameters": {
            "bootstrap_draws": draws,
            "seed": seed,
            "cluster": "page",
        },
        "identity": {
            "model_key": hf_fp16.model_key,
            "split": hf_fp16.split,
            "split_sha256": hf_fp16.split_sha256,
            "prompt_contract_sha256": hf_fp16.prompt_contract_sha256,
            "hf_protocol_version": hf_fp16.protocol_version,
            "gguf_protocol_version": gguf_f16.protocol_version,
            "hf_precision": hf_fp16.quantization,
            "gguf_precision": gguf_f16.quantization,
            "quartets": len(hf_by),
            "pages": len(hf_page),
            "hard_units": len(hard_rows),
        },
        "hard_labels": {
            "agreement": agreement / len(hard_rows),
            "transitions": dict(sorted(transitions.items())),
            "accuracy": {
                "hf_fp16": _accuracy(hf_matrix),
                "gguf_f16": _accuracy(gguf_matrix),
            },
            "balanced_accuracy": {
                "hf_fp16": balanced_accuracy(hf_matrix),
                "gguf_f16": balanced_accuracy(gguf_matrix),
            },
            "mcc": {
                "hf_fp16": multiclass_mcc(hf_matrix),
                "gguf_f16": multiclass_mcc(gguf_matrix),
            },
            "recall": {
                "hf_fp16": _recalls(hf_matrix),
                "gguf_f16": _recalls(gguf_matrix),
            },
            "confusion_gold_rows_predicted_columns": {
                "labels": VITAMINC_LABELS,
                "hf_fp16": hf_matrix,
                "gguf_f16": gguf_matrix,
            },
        },
        "interaction": {
            "estimand": "page-balanced mean(I_HF_FP16-I_GGUF_F16)",
            "hf_minus_gguf": asdict(
                paired_page_bootstrap_interval(
                    hf_minus_gguf, draws=draws, seed=seed
                )
            ),
            "per_page_pearson_correlation": _pearson(hf_page, gguf_page),
            "hf_fp16_scale": _distribution_summary(tuple(hf_page.values())),
            "gguf_f16_scale": _distribution_summary(tuple(gguf_page.values())),
        },
        "label_probability_mass": {
            "hf_fp16": statistics.fmean(hf_masses),
            "gguf_f16": statistics.fmean(gguf_masses),
            "hf_minus_gguf": statistics.fmean(
                left - right
                for left, right in zip(hf_masses, gguf_masses, strict=True)
            ),
        },
    }


def _validate_condition_identity(
    hf: VitaminCConditionResults, gguf: VitaminCConditionResults
) -> None:
    if (
        hf.split != gguf.split
        or hf.split_sha256 != gguf.split_sha256
    ):
        raise BackendComparabilityError(
            "backend comparability split identity mismatch"
        )
    if hf.prompt_contract_sha256 != gguf.prompt_contract_sha256:
        raise BackendComparabilityError(
            "backend comparability prompt identity mismatch"
        )
    if hf.model_key != gguf.model_key:
        raise BackendComparabilityError(
            "backend comparability model identity mismatch"
        )
    if hf.quantization not in {"FP16", "F16"}:
        raise BackendComparabilityError("HF endpoint is not high precision")
    if gguf.quantization not in {"FP16", "F16"}:
        raise BackendComparabilityError("GGUF endpoint is not high precision")
    if (
        hf.hard_coverage != 1.0
        or hf.probability_coverage != 1.0
        or gguf.hard_coverage != 1.0
        or gguf.probability_coverage != 1.0
    ):
        raise BackendComparabilityError(
            "backend comparability requires complete coverage"
        )


def _unique_quartets(
    condition: VitaminCConditionResults,
) -> dict[str, VitaminCQuartetResult]:
    output = {item.case_id: item for item in condition.quartets}
    if not output or len(output) != len(condition.quartets):
        raise BackendComparabilityError(
            "backend comparability requires unique nonempty quartets"
        )
    return output


def _validate_quartet_match(
    hf: VitaminCQuartetResult, gguf: VitaminCQuartetResult
) -> None:
    if (
        hf.page != gguf.page
        or hf.negative_label != gguf.negative_label
        or hf.hard_gold_labels != gguf.hard_gold_labels
    ):
        raise BackendComparabilityError(
            "backend comparability quartet metadata mismatch"
        )
    if (
        len(hf.hard_predictions) != 4
        or len(gguf.hard_predictions) != 4
        or len(hf.hard_gold_labels) != 4
    ):
        raise BackendComparabilityError(
            "backend comparability quartet hard labels are incomplete"
        )


def _accuracy(matrix: Sequence[Sequence[int]]) -> float:
    total = sum(sum(row) for row in matrix)
    return sum(matrix[index][index] for index in range(len(matrix))) / total


def _recalls(matrix: Sequence[Sequence[int]]) -> dict[str, float]:
    return {
        label: matrix[index][index] / sum(matrix[index])
        for index, label in enumerate(VITAMINC_LABELS)
    }


def _pearson(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float | None:
    if set(left) != set(right):
        raise BackendComparabilityError(
            "backend correlation requires identical page identities"
        )
    if len(left) < 2:
        return None
    left_values = tuple(left[page] for page in sorted(left))
    right_values = tuple(right[page] for page in sorted(right))
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (first - left_mean) * (second - right_mean)
        for first, second in zip(left_values, right_values, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_values))
    right_scale = math.sqrt(
        sum((value - right_mean) ** 2 for value in right_values)
    )
    denominator = left_scale * right_scale
    return None if denominator == 0.0 else numerator / denominator


def _distribution_summary(values: Sequence[float]) -> dict[str, float | None]:
    clean = tuple(_finite(value, "interaction") for value in values)
    if not clean:
        raise BackendComparabilityError(
            "backend interaction distribution must be nonempty"
        )
    return {
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "standard_deviation": statistics.stdev(clean) if len(clean) > 1 else None,
        "minimum": min(clean),
        "maximum": max(clean),
    }


def _unit_interval(value: object, label: str) -> float:
    number = _finite(value, label)
    if number < 0.0 or number > 1.0 + 1e-9:
        raise BackendComparabilityError(f"{label} must be in [0, 1]")
    return number


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendComparabilityError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise BackendComparabilityError(f"{label} must be finite")
    return number
