"""Frozen D5 within-HF AWQ/GPTQ method-robustness analysis."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hf_awq_triplet_gate import (
    HF_AWQ_TRIPLET_GATE_PROTOCOL,
    HFAWQTripletGateError,
    _load_export,
    validate_hf_awq_triplet,
)
from .hf_runtime_audit import HFRuntimeAuditError, load_hf_runtime_audit
from .hf_vitaminc_analysis import _load_condition
from .vitaminc_hard_metrics import (
    HardPredictionPair,
    balanced_accuracy,
    confusion_matrix,
    multiclass_mcc,
)
from .vitaminc_results import VitaminCConditionResults, VitaminCQuartetResult
from .vitaminc_statistics import (
    paired_page_bootstrap_intervals,
    paired_page_wald_test,
)


HF_AWQ_ANALYSIS_PROTOCOL = "hf-awq-method-robustness-v1-20260721"
SECONDARY_ORDER = (
    "qwen_awq_minus_fp16",
    "qwen_awq_minus_gptq",
    "gemma_awq_minus_fp16",
    "gemma_awq_minus_gptq",
)
_CONDITION_KEYS = (
    "qwen_fp16",
    "qwen_gptq",
    "qwen_awq",
    "gemma_fp16",
    "gemma_gptq",
    "gemma_awq",
)
_PRECISIONS = {
    "qwen_fp16": ("qwen35_9b", "FP16"),
    "qwen_gptq": ("qwen35_9b", "GPTQ_INT4"),
    "qwen_awq": ("qwen35_9b", "AWQ_INT4"),
    "gemma_fp16": ("gemma4_e4b", "FP16"),
    "gemma_gptq": ("gemma4_e4b", "GPTQ_INT4"),
    "gemma_awq": ("gemma4_e4b", "AWQ_INT4"),
}
_PROTOCOLS = {
    False: {
        "qwen_fp16": "hf-vitaminc-smoke-v2-20260718",
        "qwen_gptq": "hf-vitaminc-smoke-v2-20260718",
        "qwen_awq": "hf-awq-d5-smoke-v1-20260721",
        "gemma_fp16": "gemma-hf-vitaminc-smoke-v1-20260720",
        "gemma_gptq": "gemma-hf-vitaminc-smoke-v1-20260720",
        "gemma_awq": "hf-awq-d5-smoke-v1-20260721",
    },
    True: {
        "qwen_fp16": "hf-vitaminc-formal-v1-20260718",
        "qwen_gptq": "hf-vitaminc-formal-v1-20260718",
        "qwen_awq": "hf-awq-d5-formal-v1-20260721",
        "gemma_fp16": "gemma-hf-vitaminc-formal-v1-20260720",
        "gemma_gptq": "gemma-hf-vitaminc-formal-v1-20260720",
        "gemma_awq": "hf-awq-d5-formal-v1-20260721",
    },
}


class HFAWQMethodAnalysisError(RuntimeError):
    """Raised when D5 inputs or the frozen inference policy drift."""


@dataclass(frozen=True)
class LoadedHFAWQMethodInputs:
    conditions: Mapping[str, VitaminCConditionResults]
    gates: Mapping[str, Mapping[str, Any]]
    source_files: Mapping[str, Path]


def load_hf_awq_method_inputs(
    *,
    exports: Mapping[str, str | Path],
    audits: Mapping[str, str | Path],
    artifact_audits: Mapping[str, str | Path],
    gate_paths: Mapping[str, str | Path],
    formal: bool = True,
) -> LoadedHFAWQMethodInputs:
    """Revalidate all six immutable exports before reading outcomes."""
    if set(exports) != set(_CONDITION_KEYS) or set(audits) != set(_CONDITION_KEYS):
        raise HFAWQMethodAnalysisError("six frozen export/audit inputs are required")
    if set(artifact_audits) != {"qwen35_9b", "gemma4_e4b"} or set(gate_paths) != {
        "qwen35_9b",
        "gemma4_e4b",
    }:
        raise HFAWQMethodAnalysisError("two artifact audits and Gates are required")
    all_paths = [
        *(Path(value) for value in exports.values()),
        *(Path(value) for value in audits.values()),
        *(Path(value) for value in artifact_audits.values()),
        *(Path(value) for value in gate_paths.values()),
    ]
    for path in all_paths:
        if any("recovery" in part.casefold() for part in path.parts):
            raise HFAWQMethodAnalysisError("recovery inputs are forbidden")
    mode = "formal" if formal else "smoke"
    regenerated: dict[str, Mapping[str, Any]] = {}
    model_conditions = {
        "qwen35_9b": ("qwen_fp16", "qwen_gptq", "qwen_awq"),
        "gemma4_e4b": ("gemma_fp16", "gemma_gptq", "gemma_awq"),
    }
    try:
        for model_key, (fp_key, gptq_key, awq_key) in model_conditions.items():
            current = validate_hf_awq_triplet(
                fp16_export=exports[fp_key],
                gptq_export=exports[gptq_key],
                awq_export=exports[awq_key],
                fp16_audit=audits[fp_key],
                gptq_audit=audits[gptq_key],
                awq_audit=audits[awq_key],
                awq_artifact_audit=artifact_audits[model_key],
                model_key=model_key,
                mode=mode,
            )
            saved = _load_json_object(Path(gate_paths[model_key]), "triplet Gate")
            if saved != current:
                raise HFAWQMethodAnalysisError(
                    f"saved {model_key} triplet Gate does not match sources"
                )
            regenerated[model_key] = current
        loaded_conditions: dict[str, VitaminCConditionResults] = {}
        for key in _CONDITION_KEYS:
            certificate = load_hf_runtime_audit(audits[key])
            manifest, rows = _load_export(Path(exports[key]))
            model_key, precision = _PRECISIONS[key]
            loaded_conditions[key] = _load_condition(
                manifest,
                rows,
                certificate_sha256=certificate.certificate_sha256,
                expected_precision=precision,
                mode=mode,
                expected_protocol=_PROTOCOLS[formal][key],
                expected_model_key=model_key,
            )
    except (HFAWQTripletGateError, HFRuntimeAuditError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, HFAWQMethodAnalysisError):
            raise
        raise HFAWQMethodAnalysisError(str(error)) from error
    source_files: dict[str, Path] = {}
    for key in _CONDITION_KEYS:
        root = Path(exports[key])
        source_files[f"{key}_manifest"] = root / "manifest.json"
        source_files[f"{key}_records"] = root / "records.jsonl"
        source_files[f"{key}_runtime_audit"] = Path(audits[key])
    for model_key in ("qwen35_9b", "gemma4_e4b"):
        source_files[f"{model_key}_artifact_audit"] = Path(
            artifact_audits[model_key]
        )
        source_files[f"{model_key}_triplet_gate"] = Path(gate_paths[model_key])
    return LoadedHFAWQMethodInputs(
        conditions=loaded_conditions,
        gates=regenerated,
        source_files=source_files,
    )


def analyze_hf_awq_method(
    conditions: Mapping[str, VitaminCConditionResults],
    *,
    gates: Mapping[str, Mapping[str, Any]],
    formal: bool = True,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Estimate the five frozen D5 contrasts on matched pages."""
    if type(formal) is not bool:
        raise HFAWQMethodAnalysisError("formal mode must be boolean")
    if type(draws) is not int or draws < 1:
        raise HFAWQMethodAnalysisError("bootstrap draws must be positive")
    if type(seed) is not int:
        raise HFAWQMethodAnalysisError("bootstrap seed must be an integer")
    if formal and (draws != 10_000 or seed != 20260711):
        raise HFAWQMethodAnalysisError("formal bootstrap policy drifted")
    if set(conditions) != set(_CONDITION_KEYS):
        raise HFAWQMethodAnalysisError("six frozen conditions are required")
    normalized = {key: conditions[key] for key in _CONDITION_KEYS}
    _validate_gates(gates, normalized, formal=formal)
    _validate_condition_identities(normalized, formal=formal)

    by_key = {
        key: _index_quartets(value.quartets, key)
        for key, value in normalized.items()
    }
    case_sets = [set(items) for items in by_key.values()]
    if not case_sets[0] or any(items != case_sets[0] for items in case_sets[1:]):
        raise HFAWQMethodAnalysisError("condition case identities are not matched")
    case_ids = sorted(case_sets[0])
    expected_quartets = 1200 if formal else 2
    if len(case_ids) != expected_quartets:
        raise HFAWQMethodAnalysisError(
            f"D5 requires exactly {expected_quartets} quartets"
        )
    for case_id in case_ids:
        reference = by_key["qwen_fp16"][case_id]
        for key in _CONDITION_KEYS[1:]:
            candidate = by_key[key][case_id]
            if (
                candidate.page != reference.page
                or candidate.negative_label != reference.negative_label
                or candidate.hard_gold_labels != reference.hard_gold_labels
            ):
                raise HFAWQMethodAnalysisError(
                    f"quartet metadata alignment drifted for {case_id}"
                )

    quartet_values = {
        key: {
            case_id: _finite(by_key[key][case_id].interaction)
            for case_id in case_ids
        }
        for key in _CONDITION_KEYS
    }
    quartet_contrasts = {
        "qwen_awq_minus_fp16": _subtract_cases(
            quartet_values["qwen_awq"], quartet_values["qwen_fp16"]
        ),
        "qwen_awq_minus_gptq": _subtract_cases(
            quartet_values["qwen_awq"], quartet_values["qwen_gptq"]
        ),
        "gemma_awq_minus_fp16": _subtract_cases(
            quartet_values["gemma_awq"], quartet_values["gemma_fp16"]
        ),
        "gemma_awq_minus_gptq": _subtract_cases(
            quartet_values["gemma_awq"], quartet_values["gemma_gptq"]
        ),
    }
    quartet_contrasts["primary"] = _subtract_cases(
        quartet_contrasts["gemma_awq_minus_gptq"],
        quartet_contrasts["qwen_awq_minus_gptq"],
    )
    pages = {
        case_id: by_key["qwen_fp16"][case_id].page for case_id in case_ids
    }
    page_contrasts = {
        name: _page_means(values, pages)
        for name, values in quartet_contrasts.items()
    }
    page_set = set(page_contrasts["primary"])
    if formal and len(page_set) != 1078:
        raise HFAWQMethodAnalysisError("D5 formal mode requires exactly 1078 pages")

    intervals = paired_page_bootstrap_intervals(
        page_contrasts, draws=draws, seed=seed
    )
    secondary_wald = {
        name: paired_page_wald_test(page_contrasts[name]) for name in SECONDARY_ORDER
    }
    holm = _holm_adjust(
        [(name, secondary_wald[name].pvalue) for name in SECONDARY_ORDER]
    )
    contrast_results: dict[str, Any] = {}
    for name in SECONDARY_ORDER:
        contrast_results[name] = {
            "estimand": name,
            "page_bootstrap": asdict(intervals[name]),
            "page_wald": asdict(secondary_wald[name]),
            "holm_adjusted_pvalue": holm[name],
            "confidence_interval_status": "unadjusted_descriptive",
        }
    primary_wald = paired_page_wald_test(page_contrasts["primary"])
    quartet_sensitivity = {
        name: statistics.fmean(values.values())
        for name, values in quartet_contrasts.items()
    }
    return {
        "analysis_protocol": HF_AWQ_ANALYSIS_PROTOCOL,
        "inference_scope": "within_hf_method_associated_robustness",
        "population": {"quartets": len(case_ids), "pages": len(page_set)},
        "bootstrap_contract": {
            "cluster": "page",
            "draws": draws,
            "seed": seed,
            "confidence_level": 0.95,
            "alternative": "two_sided",
            "shared_page_indices": True,
        },
        "gates": {
            model_key: {
                "gate_sha256": gates[model_key]["gate_sha256"],
                "aligned_full_owners": gates[model_key]["aligned_full_owners"],
            }
            for model_key in ("qwen35_9b", "gemma4_e4b")
        },
        "primary": {
            "estimand": "(Gemma_AWQ-Gemma_GPTQ)-(Qwen_AWQ-Qwen_GPTQ)",
            "page_bootstrap": asdict(intervals["primary"]),
            "page_wald": asdict(primary_wald),
            "multiplicity_adjustment": "none_single_primary",
        },
        "secondary_order": list(SECONDARY_ORDER),
        "contrasts": contrast_results,
        "holm_family": [
            {
                "name": name,
                "raw_pvalue": secondary_wald[name].pvalue,
                "holm_adjusted_pvalue": holm[name],
            }
            for name in SECONDARY_ORDER
        ],
        "quartet_weighted_sensitivity": {
            "status": "secondary",
            "weighting": "unequal_page_quartet_weighted",
            "estimates": quartet_sensitivity,
        },
        "hard_labels": {
            "qwen_awq_vs_fp16": _hard_label_summary(
                by_key["qwen_fp16"], by_key["qwen_awq"]
            ),
            "qwen_awq_vs_gptq": _hard_label_summary(
                by_key["qwen_gptq"], by_key["qwen_awq"]
            ),
            "gemma_awq_vs_fp16": _hard_label_summary(
                by_key["gemma_fp16"], by_key["gemma_awq"]
            ),
            "gemma_awq_vs_gptq": _hard_label_summary(
                by_key["gemma_gptq"], by_key["gemma_awq"]
            ),
        },
    }


def _validate_gates(
    gates: Mapping[str, Mapping[str, Any]],
    conditions: Mapping[str, VitaminCConditionResults],
    *,
    formal: bool,
) -> None:
    if set(gates) != {"qwen35_9b", "gemma4_e4b"}:
        raise HFAWQMethodAnalysisError("both formal triplet gates are required")
    expected_mode = "formal" if formal else "smoke"
    expected_owners = 4800 if formal else 8
    split_sha256 = conditions["qwen_fp16"].split_sha256
    for model_key in ("qwen35_9b", "gemma4_e4b"):
        gate = gates[model_key]
        if not isinstance(gate, Mapping):
            raise HFAWQMethodAnalysisError("triplet gate must be a mapping")
        expected = {
            "protocol_version": HF_AWQ_TRIPLET_GATE_PROTOCOL,
            "mode": expected_mode,
            "model_key": model_key,
            "gate_passed": True,
            "aligned_full_owners": expected_owners,
            "controls": {"full": expected_owners},
            "split_sha256": split_sha256,
        }
        if any(gate.get(field) != value for field, value in expected.items()):
            raise HFAWQMethodAnalysisError(f"{model_key} triplet gate drifted")
        supplied = gate.get("gate_sha256")
        if supplied != _record_hash(gate, excluded="gate_sha256"):
            raise HFAWQMethodAnalysisError(f"{model_key} triplet gate hash drifted")


def _validate_condition_identities(
    conditions: Mapping[str, VitaminCConditionResults], *, formal: bool
) -> None:
    expected_split = "dose_subset" if formal else "pilot"
    split_hashes = {value.split_sha256 for value in conditions.values()}
    prompt_hashes = {value.prompt_contract_sha256 for value in conditions.values()}
    if len(split_hashes) != 1 or len(prompt_hashes) != 1:
        raise HFAWQMethodAnalysisError("condition source identity drifted")
    for key, condition in conditions.items():
        model_key, precision = _PRECISIONS[key]
        if (
            condition.model_key != model_key
            or condition.quantization != precision
            or condition.protocol_version != _PROTOCOLS[formal][key]
            or condition.split != expected_split
            or condition.hard_coverage != 1.0
            or condition.probability_coverage != 1.0
        ):
            raise HFAWQMethodAnalysisError(f"condition identity drifted for {key}")


def _index_quartets(
    quartets: Sequence[VitaminCQuartetResult], key: str
) -> dict[str, VitaminCQuartetResult]:
    indexed: dict[str, VitaminCQuartetResult] = {}
    for quartet in quartets:
        if not isinstance(quartet.case_id, str) or not quartet.case_id or quartet.case_id in indexed:
            raise HFAWQMethodAnalysisError(f"quartet identities drifted for {key}")
        _finite(quartet.interaction)
        indexed[quartet.case_id] = quartet
    return indexed


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HFAWQMethodAnalysisError("interaction values must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise HFAWQMethodAnalysisError("interaction values must be finite")
    return number


def _subtract_cases(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    if set(left) != set(right):
        raise HFAWQMethodAnalysisError("contrast case identities drifted")
    return {case_id: left[case_id] - right[case_id] for case_id in sorted(left)}


def _page_means(
    values: Mapping[str, float], pages: Mapping[str, str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, value in values.items():
        page = pages.get(case_id)
        if not isinstance(page, str) or not page:
            raise HFAWQMethodAnalysisError("page identity drifted")
        grouped[page].append(_finite(value))
    return {
        page: statistics.fmean(page_values)
        for page, page_values in sorted(grouped.items())
    }


def _holm_adjust(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    if len(pairs) != 4 or tuple(name for name, _ in pairs) != SECONDARY_ORDER:
        raise HFAWQMethodAnalysisError("Holm family order drifted")
    ordered = sorted(pairs, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalue))
        adjusted[name] = running
    return adjusted


def _hard_label_summary(
    reference: Mapping[str, VitaminCQuartetResult],
    awq: Mapping[str, VitaminCQuartetResult],
) -> dict[str, Any]:
    transitions: Counter[str] = Counter()
    reference_correct = 0
    awq_correct = 0
    agreement = 0
    units = 0
    hard_rows: list[HardPredictionPair] = []
    for case_id in sorted(reference):
        left = reference[case_id]
        right = awq[case_id]
        if (
            not left.hard_predictions
            or len(left.hard_predictions)
            != len(right.hard_predictions)
            or len(left.hard_predictions) != len(left.hard_gold_labels)
            or left.hard_gold_labels != right.hard_gold_labels
        ):
            raise HFAWQMethodAnalysisError("hard-label inputs are incomplete")
        for left_label, awq_label, gold in zip(
            left.hard_predictions,
            right.hard_predictions,
            left.hard_gold_labels,
            strict=True,
        ):
            if left_label is None or awq_label is None:
                raise HFAWQMethodAnalysisError("hard-label coverage is incomplete")
            transitions[f"{left_label}->{awq_label}"] += 1
            reference_correct += left_label == gold
            awq_correct += awq_label == gold
            agreement += left_label == awq_label
            hard_rows.append(
                HardPredictionPair(
                    sample_id=f"{case_id}:{units}",
                    page=left.page,
                    gold_label=gold,
                    prediction_f16=left_label,
                    prediction_quantized=awq_label,
                )
            )
            units += 1
    reference_matrix = confusion_matrix(hard_rows, prediction="f16")
    awq_matrix = confusion_matrix(hard_rows, prediction="quantized")
    return {
        "units": units,
        "accuracy": {
            "reference": reference_correct / units,
            "awq": awq_correct / units,
            "difference": (awq_correct - reference_correct) / units,
        },
        "balanced_accuracy": {
            "reference": balanced_accuracy(reference_matrix),
            "awq": balanced_accuracy(awq_matrix),
        },
        "mcc": {
            "reference": multiclass_mcc(reference_matrix),
            "awq": multiclass_mcc(awq_matrix),
        },
        "agreement": agreement / units,
        "transitions": dict(sorted(transitions.items())),
    }


def _record_hash(record: Mapping[str, Any], *, excluded: str) -> str:
    payload = json.dumps(
        {key: value for key, value in record.items() if key != excluded},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFAWQMethodAnalysisError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise HFAWQMethodAnalysisError(f"{label} must be a JSON object")
    return value
