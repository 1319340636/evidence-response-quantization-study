"""Frozen within-model analysis for the OLMo 3 HF precision triplet."""

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
from .multiplicity import holm_adjust
from .olmo3_hf_triplet_gate import OLMO3_HF_TRIPLET_GATE_PROTOCOL
from .olmo3_triplet_freeze import DOSE_SUBSET_SHA256
from .vitaminc_results import VitaminCConditionResults
from .vitaminc_statistics import (
    paired_page_bootstrap_intervals,
    paired_page_wald_test,
)


ANALYSIS_PROTOCOL = "olmo3-hf-triplet-analysis-v1-20260901"
PRIMARY_CONTRASTS = ("awq_minus_fp16", "awq_minus_gptq")
SECONDARY_CONTRAST = "gptq_minus_fp16"
_ROUTES = ("fp16", "gptq", "awq")
_PRECISIONS = {"fp16": "FP16", "gptq": "GPTQ_INT4", "awq": "AWQ_INT4"}


class Olmo3HFTripletAnalysisError(RuntimeError):
    """Raised when a frozen OLMo3 analysis input or policy drifts."""


def analyze_olmo3_hf_triplet(
    conditions: Mapping[str, VitaminCConditionResults],
    *,
    gate: Mapping[str, Any],
    formal: bool = True,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    if (
        formal
        and "fp16" in conditions
        and conditions["fp16"].split_sha256 != DOSE_SUBSET_SHA256
    ):
        raise Olmo3HFTripletAnalysisError(
            "formal population does not match frozen split"
        )
    _validate_policy(formal=formal, draws=draws, seed=seed)
    _validate_condition_identities(conditions, formal=formal)
    indexed, case_ids, pages, values = _aligned_triplet(
        conditions, formal=formal
    )
    _validate_gate(
        gate, formal=formal, split_sha256=conditions["fp16"].split_sha256
    )

    quartet_contrasts = {
        "awq_minus_fp16": _subtract_cases(values["awq"], values["fp16"]),
        "awq_minus_gptq": _subtract_cases(values["awq"], values["gptq"]),
        "gptq_minus_fp16": _subtract_cases(values["gptq"], values["fp16"]),
    }
    page_contrasts = {
        name: _page_means(values_by_case, pages)
        for name, values_by_case in quartet_contrasts.items()
    }
    intervals = paired_page_bootstrap_intervals(
        page_contrasts, draws=draws, seed=seed
    )
    wald = {
        name: paired_page_wald_test(page_contrasts[name])
        for name in quartet_contrasts
    }
    adjusted = holm_adjust(
        {name: wald[name].pvalue for name in PRIMARY_CONTRASTS}
    )
    contrasts: dict[str, Any] = {}
    for name in (*PRIMARY_CONTRASTS, SECONDARY_CONTRAST):
        contrasts[name] = {
            "estimand": name,
            "page_bootstrap": asdict(intervals[name]),
            "page_wald": asdict(wald[name]),
            "confidence_interval_status": "unadjusted_descriptive",
            "multiplicity_status": (
                "holm_primary_family"
                if name in PRIMARY_CONTRASTS
                else "secondary_not_in_holm_family"
            ),
        }
        if name in adjusted:
            contrasts[name]["holm_adjusted_pvalue"] = adjusted[name]

    return {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "inference_scope": "olmo3_7b_within_hf_method_associated_contrasts",
        "causal_boundary": (
            "model_route_associated_not_pure_quantizer_kernel_packing_or_"
            "architecture_effect"
        ),
        "population": {
            "quartets": len(case_ids),
            "pages": len(set(pages.values())),
        },
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
            "olmo3_triplet_freeze_sha256": gate[
                "olmo3_triplet_freeze_sha256"
            ],
            "aligned_full_owners": gate["aligned_full_owners"],
        },
        "primary_contrast_order": list(PRIMARY_CONTRASTS),
        "secondary_contrast": SECONDARY_CONTRAST,
        "contrasts": contrasts,
        "holm_family": [
            {
                "name": name,
                "raw_pvalue": wald[name].pvalue,
                "holm_adjusted_pvalue": adjusted[name],
            }
            for name in PRIMARY_CONTRASTS
        ],
        "quartet_weighted_sensitivity": {
            "status": "secondary",
            "estimates": {
                name: statistics.fmean(values_by_case.values())
                for name, values_by_case in quartet_contrasts.items()
            },
        },
        "hard_labels": {
            "awq_vs_fp16": _hard_label_summary(
                indexed["fp16"], indexed["awq"]
            ),
            "awq_vs_gptq": _hard_label_summary(
                indexed["gptq"], indexed["awq"]
            ),
            "gptq_vs_fp16": _relabel_candidate(
                _hard_label_summary(indexed["fp16"], indexed["gptq"]),
                "gptq",
            ),
        },
    }


def _relabel_candidate(value: object, route: str) -> object:
    """Replace the AWQ-specific candidate key used by the shared helper."""
    if isinstance(value, dict):
        return {
            (route if key == "awq" else key): _relabel_candidate(child, route)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_relabel_candidate(child, route) for child in value]
    return value


def _aligned_triplet(
    conditions: Mapping[str, VitaminCConditionResults], *, formal: bool
) -> tuple[dict[str, dict], list[str], dict[str, str], dict[str, dict[str, float]]]:
    try:
        indexed = {
            key: _index_quartets(conditions[key].quartets, key)
            for key in _ROUTES
        }
        case_sets = [set(indexed[key]) for key in _ROUTES]
        if not case_sets[0] or any(
            value != case_sets[0] for value in case_sets[1:]
        ):
            raise Olmo3HFTripletAnalysisError(
                "condition case identities are not matched"
            )
        case_ids = sorted(case_sets[0])
        expected = 1200 if formal else 2
        if len(case_ids) != expected:
            raise Olmo3HFTripletAnalysisError(
                f"analysis requires exactly {expected} quartets"
            )
        pages: dict[str, str] = {}
        for case_id in case_ids:
            reference = indexed["fp16"][case_id]
            pages[case_id] = reference.page
            for key in ("gptq", "awq"):
                candidate = indexed[key][case_id]
                if (
                    candidate.page != reference.page
                    or candidate.negative_label != reference.negative_label
                    or candidate.hard_gold_labels
                    != reference.hard_gold_labels
                ):
                    raise Olmo3HFTripletAnalysisError(
                        f"quartet metadata alignment drifted for {case_id}"
                    )
        if formal and len(set(pages.values())) != 1078:
            raise Olmo3HFTripletAnalysisError(
                "formal analysis requires exactly 1078 pages"
            )
        values = {
            key: {
                case_id: float(indexed[key][case_id].interaction)
                for case_id in case_ids
            }
            for key in _ROUTES
        }
    except (HFAWQMethodAnalysisError, ValueError, TypeError) as error:
        if isinstance(error, Olmo3HFTripletAnalysisError):
            raise
        raise Olmo3HFTripletAnalysisError(str(error)) from error
    return indexed, case_ids, pages, values


def _validate_condition_identities(
    conditions: Mapping[str, VitaminCConditionResults], *, formal: bool
) -> None:
    if set(conditions) != set(_ROUTES):
        raise Olmo3HFTripletAnalysisError(
            "three frozen conditions are required"
        )
    if any(
        not isinstance(conditions[key], VitaminCConditionResults)
        for key in _ROUTES
    ):
        raise Olmo3HFTripletAnalysisError(
            "each route must be a complete condition object"
        )
    expected_split = "dose_subset" if formal else "pilot"
    expected_protocol = (
        "olmo3-hf-triplet-formal-v1-20260901"
        if formal
        else "olmo3-hf-triplet-smoke-v1-20260901"
    )
    if len({condition.split_sha256 for condition in conditions.values()}) != 1:
        raise Olmo3HFTripletAnalysisError("condition source identity drifted")
    if len(
        {condition.prompt_contract_sha256 for condition in conditions.values()}
    ) != 1:
        raise Olmo3HFTripletAnalysisError("condition prompt identity drifted")
    for key, condition in conditions.items():
        if (
            condition.model_key != "olmo3_7b"
            or condition.quantization != _PRECISIONS[key]
            or condition.protocol_version != expected_protocol
            or condition.split != expected_split
            or condition.hard_coverage != 1.0
            or condition.probability_coverage != 1.0
        ):
            raise Olmo3HFTripletAnalysisError(
                f"condition identity drifted for {key}"
            )


def _validate_gate(
    gate: Mapping[str, Any], *, formal: bool, split_sha256: str
) -> None:
    expected = {
        "protocol_version": OLMO3_HF_TRIPLET_GATE_PROTOCOL,
        "mode": "olmo3_formal4800" if formal else "olmo3_smoke16",
        "model_key": "olmo3_7b",
        "gate_passed": True,
        "aligned_full_owners": 4800 if formal else 16,
        "controls": {"full": 4800 if formal else 16},
        "split_sha256": split_sha256,
    }
    if not isinstance(gate, Mapping) or any(
        gate.get(key) != value for key, value in expected.items()
    ):
        raise Olmo3HFTripletAnalysisError("OLMo3 triplet gate drifted")
    if _contains_recovery(gate):
        raise Olmo3HFTripletAnalysisError(
            "recovery-derived gate inputs are forbidden"
        )
    freeze_hash = gate.get("olmo3_triplet_freeze_sha256")
    if not isinstance(freeze_hash, str) or len(freeze_hash) != 64:
        raise Olmo3HFTripletAnalysisError("OLMo3 freeze hash drifted")
    if gate.get("gate_sha256") != _record_hash(
        gate, excluded="gate_sha256"
    ):
        raise Olmo3HFTripletAnalysisError("OLMo3 gate hash drifted")


def _validate_policy(*, formal: bool, draws: int, seed: int) -> None:
    if (
        type(formal) is not bool
        or type(draws) is not int
        or draws < 1
        or type(seed) is not int
    ):
        raise Olmo3HFTripletAnalysisError("bootstrap policy is invalid")
    if formal and (draws != 10_000 or seed != 20260711):
        raise Olmo3HFTripletAnalysisError("formal bootstrap policy drifted")


def _contains_recovery(value: object) -> bool:
    if isinstance(value, str):
        return "recovery" in value.casefold()
    if isinstance(value, Mapping):
        return any(
            _contains_recovery(key) or _contains_recovery(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_recovery(child) for child in value)
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
