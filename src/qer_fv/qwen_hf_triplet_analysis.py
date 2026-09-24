"""Frozen Qwen3.5-9B within-HF FP16/GPTQ/AWQ analysis."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from typing import Any, Mapping

from .hf_awq_method_analysis import (
    HFAWQMethodAnalysisError,
    _hard_label_summary,
    _index_quartets,
    _page_means,
    _subtract_cases,
)
from .hf_awq_triplet_gate import QWEN_HF_TRIPLET_GATE_PROTOCOL
from .multiplicity import holm_adjust
from .vitaminc_results import VitaminCConditionResults
from .vitaminc_statistics import (
    paired_page_bootstrap_intervals,
    paired_page_wald_test,
)


ANALYSIS_PROTOCOL = "qwen-hf-triplet-analysis-v1-20260802"
CONTRAST_ORDER = ("awq_minus_fp16", "awq_minus_gptq")
_CONDITION_KEYS = ("fp16", "gptq", "awq")
_PRECISIONS = {
    "fp16": "FP16",
    "gptq": "GPTQ_INT4",
    "awq": "AWQ_INT4",
}
_PROTOCOLS = {
    False: {
        "fp16": "hf-vitaminc-smoke-v2-20260718",
        "gptq": "hf-vitaminc-smoke-v2-20260718",
        "awq": "hf-awq-d5-smoke-v1-20260721",
    },
    True: {
        "fp16": "hf-vitaminc-formal-v1-20260718",
        "gptq": "hf-vitaminc-formal-v1-20260718",
        "awq": "hf-awq-d5-formal-v1-20260721",
    },
}


class QwenHFTripletAnalysisError(RuntimeError):
    """Raised when a frozen Qwen triplet input or analysis policy drifts."""


def analyze_qwen_hf_triplet(
    conditions: Mapping[str, VitaminCConditionResults],
    *,
    gate: Mapping[str, Any],
    formal: bool = True,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Analyze the two frozen Qwen AWQ contrasts on matched page clusters."""
    _validate_policy(formal=formal, draws=draws, seed=seed)
    if set(conditions) != set(_CONDITION_KEYS):
        raise QwenHFTripletAnalysisError("three frozen conditions are required")
    normalized: dict[str, VitaminCConditionResults] = {}
    for key in _CONDITION_KEYS:
        condition = conditions[key]
        if not isinstance(condition, VitaminCConditionResults):
            raise QwenHFTripletAnalysisError(
                f"{key} must be a complete VitaminC condition object"
            )
        normalized[key] = condition
    _validate_gate(gate, formal=formal, split_sha256=normalized["fp16"].split_sha256)
    _validate_condition_identities(normalized, formal=formal)

    try:
        indexed = {
            key: _index_quartets(condition.quartets, key)
            for key, condition in normalized.items()
        }
        case_sets = [set(indexed[key]) for key in _CONDITION_KEYS]
        if not case_sets[0] or any(value != case_sets[0] for value in case_sets[1:]):
            raise QwenHFTripletAnalysisError(
                "condition case identities are not matched"
            )
        case_ids = sorted(case_sets[0])
        expected_quartets = 1200 if formal else 2
        if len(case_ids) != expected_quartets:
            raise QwenHFTripletAnalysisError(
                f"analysis requires exactly {expected_quartets} quartets"
            )
        for case_id in case_ids:
            reference = indexed["fp16"][case_id]
            for key in ("gptq", "awq"):
                candidate = indexed[key][case_id]
                if (
                    candidate.page != reference.page
                    or candidate.negative_label != reference.negative_label
                    or candidate.hard_gold_labels != reference.hard_gold_labels
                ):
                    raise QwenHFTripletAnalysisError(
                        f"quartet metadata alignment drifted for {case_id}"
                    )

        quartet_values = {
            key: {
                case_id: float(indexed[key][case_id].interaction)  # validated above
                for case_id in case_ids
            }
            for key in _CONDITION_KEYS
        }
        quartet_contrasts = {
            "awq_minus_fp16": _subtract_cases(
                quartet_values["awq"], quartet_values["fp16"]
            ),
            "awq_minus_gptq": _subtract_cases(
                quartet_values["awq"], quartet_values["gptq"]
            ),
        }
        pages = {case_id: indexed["fp16"][case_id].page for case_id in case_ids}
        page_contrasts = {
            name: _page_means(values, pages)
            for name, values in quartet_contrasts.items()
        }
        page_set = set(page_contrasts[CONTRAST_ORDER[0]])
        if any(set(page_contrasts[name]) != page_set for name in CONTRAST_ORDER):
            raise QwenHFTripletAnalysisError("contrast page identities drifted")
        if formal and len(page_set) != 1078:
            raise QwenHFTripletAnalysisError(
                "formal analysis requires exactly 1078 pages"
            )

        intervals = paired_page_bootstrap_intervals(
            page_contrasts, draws=draws, seed=seed
        )
        wald = {
            name: paired_page_wald_test(page_contrasts[name])
            for name in CONTRAST_ORDER
        }
        adjusted = holm_adjust({name: wald[name].pvalue for name in CONTRAST_ORDER})
        contrasts = {
            name: {
                "estimand": name,
                "page_bootstrap": asdict(intervals[name]),
                "page_wald": asdict(wald[name]),
                "holm_adjusted_pvalue": adjusted[name],
                "confidence_interval_status": "unadjusted_descriptive",
            }
            for name in CONTRAST_ORDER
        }
        hard_labels = {
            "awq_vs_fp16": _hard_label_summary(indexed["fp16"], indexed["awq"]),
            "awq_vs_gptq": _hard_label_summary(indexed["gptq"], indexed["awq"]),
        }
    except (HFAWQMethodAnalysisError, ValueError, TypeError) as error:
        if isinstance(error, QwenHFTripletAnalysisError):
            raise
        raise QwenHFTripletAnalysisError(str(error)) from error

    return {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "inference_scope": "qwen35_9b_within_hf_method_associated_contrasts",
        "causal_boundary": "quantizer_route_associated_not_pure_algorithm_effect",
        "population": {"quartets": len(case_ids), "pages": len(page_set)},
        "bootstrap_contract": {
            "cluster": "page",
            "draws": draws,
            "seed": seed,
            "confidence_level": 0.95,
            "alternative": "two_sided",
            "shared_page_indices": True,
        },
        "gate": {
            "mode": gate["mode"],
            "model_key": gate["model_key"],
            "gate_sha256": gate["gate_sha256"],
            "qwen_triplet_freeze_sha256": gate["qwen_triplet_freeze_sha256"],
            "aligned_full_owners": gate["aligned_full_owners"],
        },
        "contrast_order": list(CONTRAST_ORDER),
        "contrasts": contrasts,
        "holm_family": [
            {
                "name": name,
                "raw_pvalue": wald[name].pvalue,
                "holm_adjusted_pvalue": adjusted[name],
            }
            for name in CONTRAST_ORDER
        ],
        "quartet_weighted_sensitivity": {
            "status": "secondary",
            "weighting": "unequal_page_quartet_weighted",
            "estimates": {
                name: statistics.fmean(quartet_contrasts[name].values())
                for name in CONTRAST_ORDER
            },
        },
        "hard_labels": hard_labels,
    }


def _validate_policy(*, formal: bool, draws: int, seed: int) -> None:
    if type(formal) is not bool:
        raise QwenHFTripletAnalysisError("formal mode must be boolean")
    if type(draws) is not int or draws < 1:
        raise QwenHFTripletAnalysisError("bootstrap draws must be positive")
    if type(seed) is not int:
        raise QwenHFTripletAnalysisError("bootstrap seed must be an integer")
    if formal and (draws != 10_000 or seed != 20260711):
        raise QwenHFTripletAnalysisError("formal bootstrap policy drifted")


def _validate_gate(
    gate: Mapping[str, Any], *, formal: bool, split_sha256: str
) -> None:
    if not isinstance(gate, Mapping):
        raise QwenHFTripletAnalysisError("triplet gate must be a mapping")
    if _contains_recovery(gate):
        raise QwenHFTripletAnalysisError("recovery-derived inputs are forbidden")
    expected = {
        "protocol_version": QWEN_HF_TRIPLET_GATE_PROTOCOL,
        "mode": "qwen_formal4800" if formal else "qwen_smoke16",
        "model_key": "qwen35_9b",
        "gate_passed": True,
        "aligned_full_owners": 4800 if formal else 16,
        "controls": {"full": 4800 if formal else 16},
        "split_sha256": split_sha256,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise QwenHFTripletAnalysisError("Qwen triplet gate drifted")
    freeze_hash = gate.get("qwen_triplet_freeze_sha256")
    if (
        not isinstance(freeze_hash, str)
        or len(freeze_hash) != 64
        or any(character not in "0123456789abcdef" for character in freeze_hash)
    ):
        raise QwenHFTripletAnalysisError("Qwen triplet freeze hash drifted")
    supplied = gate.get("gate_sha256")
    if supplied != _record_hash(gate, excluded="gate_sha256"):
        raise QwenHFTripletAnalysisError("Qwen triplet gate hash drifted")


def _validate_condition_identities(
    conditions: Mapping[str, VitaminCConditionResults], *, formal: bool
) -> None:
    expected_split = "dose_subset" if formal else "pilot"
    if len({value.split_sha256 for value in conditions.values()}) != 1:
        raise QwenHFTripletAnalysisError("condition source identity drifted")
    if len({value.prompt_contract_sha256 for value in conditions.values()}) != 1:
        raise QwenHFTripletAnalysisError("condition prompt identity drifted")
    for key, condition in conditions.items():
        if (
            condition.model_key != "qwen35_9b"
            or condition.quantization != _PRECISIONS[key]
            or condition.protocol_version != _PROTOCOLS[formal][key]
            or condition.split != expected_split
            or condition.hard_coverage != 1.0
            or condition.probability_coverage != 1.0
        ):
            raise QwenHFTripletAnalysisError(f"condition identity drifted for {key}")


def _contains_recovery(value: Any) -> bool:
    if isinstance(value, str):
        return "recovery" in value.casefold()
    if isinstance(value, Mapping):
        return any(_contains_recovery(key) or _contains_recovery(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_recovery(item) for item in value)
    return False


def _record_hash(record: Mapping[str, Any], *, excluded: str) -> str:
    payload = json.dumps(
        {key: value for key, value in record.items() if key != excluded},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
