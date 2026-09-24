"""Audited result loading for paired Hugging Face VitaminC runs."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .hf_pair_gate import HFPairGateError, _load_export, validate_hf_pair
from .hf_model_identity import HFModelIdentityError, hf_model_identity
from .hf_runtime import HF_DIRECT_LOGIT_METHOD
from .hf_runtime_audit import HFRuntimeAuditError, load_hf_runtime_audit
from .hf_vitaminc_runner import (
    HF_VITAMINC_FORMAL_PROTOCOL,
    HF_VITAMINC_SMOKE_PROTOCOL,
)
from .vitaminc_results import (
    PairedVitaminCQuartet,
    VitaminCConditionResults,
    _summarize_quartet,
    pair_vitaminc_conditions,
    family_effect_heterogeneity,
)
from .vitaminc_hard_metrics import (
    VITAMINC_LABELS,
    HardPredictionPair,
    balanced_accuracy,
    confusion_matrix,
    multiclass_mcc,
    paired_page_bootstrap_balanced_accuracy,
    paired_page_permutation_balanced_accuracy,
)
from .vitaminc_statistics import (
    model_internal_standardized_effect,
    page_direction_summary,
    page_effects,
    paired_direction_mcnemar_test,
    paired_hard_label_summary,
    paired_page_bootstrap_interval,
    paired_page_wald_test,
    quartet_weighted_effect,
)
from .vitaminc_store_v4 import VITAMINC_STORE_V4_SCHEMA_VERSION


_MODE_IDENTITIES = {
    "smoke": (HF_VITAMINC_SMOKE_PROTOCOL, "pilot"),
    "formal": (HF_VITAMINC_FORMAL_PROTOCOL, "dose_subset"),
}


class HFVitaminCAnalysisError(RuntimeError):
    """Raised when an HF export cannot enter audited result analysis."""


@dataclass(frozen=True)
class LoadedHFVitaminCPair:
    gate: Mapping[str, Any]
    fp16: VitaminCConditionResults
    gptq: VitaminCConditionResults
    pairs: tuple[PairedVitaminCQuartet, ...]
    fp16_rows: tuple[Mapping[str, Any], ...]
    gptq_rows: tuple[Mapping[str, Any], ...]


HF_VITAMINC_ANALYSIS_PROTOCOL = "qwen-hf-formal-analysis-v1-20260718"
_ANALYSIS_PROTOCOLS = {
    "qwen35_9b": HF_VITAMINC_ANALYSIS_PROTOCOL,
    "gemma4_e4b": "gemma-hf-d4-analysis-v1-20260721",
}


def hf_analysis_protocol(model_key: str) -> str:
    """Return the frozen outcome-analysis protocol for one HF identity."""
    try:
        return _ANALYSIS_PROTOCOLS[model_key]
    except (KeyError, TypeError) as error:
        raise HFVitaminCAnalysisError(
            f"unsupported HF analysis identity: {model_key!r}"
        ) from error


def load_hf_vitaminc_pair(
    *,
    fp16_export: str | Path,
    gptq_export: str | Path,
    fp16_audit: str | Path,
    gptq_audit: str | Path,
    mode: str = "formal",
    model_key: str = "qwen35_9b",
) -> LoadedHFVitaminCPair:
    """Revalidate and load one immutable FP16/GPTQ VitaminC pair."""
    if mode not in _MODE_IDENTITIES:
        raise HFVitaminCAnalysisError("HF analysis mode must be smoke or formal")
    try:
        model_identity = hf_model_identity(model_key)
        expected_protocol = (
            model_identity.smoke_protocol
            if mode == "smoke"
            else model_identity.formal_protocol
        )
        gate = validate_hf_pair(
            fp16_export=fp16_export,
            gptq_export=gptq_export,
            fp16_audit=fp16_audit,
            gptq_audit=gptq_audit,
            mode=mode,
            model_key=model_key,
        )
        fp_certificate = load_hf_runtime_audit(fp16_audit)
        gptq_certificate = load_hf_runtime_audit(gptq_audit)
        fp_manifest, fp_rows = _load_export(Path(fp16_export))
        gptq_manifest, gptq_rows = _load_export(Path(gptq_export))
        fp16 = _load_condition(
            fp_manifest,
            fp_rows,
            certificate_sha256=fp_certificate.certificate_sha256,
            expected_precision="FP16",
            mode=mode,
            expected_protocol=expected_protocol,
            expected_model_key=model_key,
        )
        gptq = _load_condition(
            gptq_manifest,
            gptq_rows,
            certificate_sha256=gptq_certificate.certificate_sha256,
            expected_precision="GPTQ_INT4",
            mode=mode,
            expected_protocol=expected_protocol,
            expected_model_key=model_key,
        )
        pairs = pair_vitaminc_conditions(fp16, gptq)
    except (
        HFPairGateError,
        HFModelIdentityError,
        HFRuntimeAuditError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise HFVitaminCAnalysisError(str(error)) from error
    return LoadedHFVitaminCPair(
        gate=gate,
        fp16=fp16,
        gptq=gptq,
        pairs=pairs,
        fp16_rows=tuple(fp_rows),
        gptq_rows=tuple(gptq_rows),
    )


def analyze_hf_vitaminc_pair(
    loaded: LoadedHFVitaminCPair,
    *,
    draws: int = 10_000,
    ba_draws: int = 100_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Calculate the frozen paired HF outcomes after structural unblinding."""
    pairs = loaded.pairs
    effects = page_effects(pairs)
    hard_rows = _hard_rows(pairs)
    fp_matrix = confusion_matrix(hard_rows, prediction="f16")
    gptq_matrix = confusion_matrix(hard_rows, prediction="quantized")
    ba_interval = paired_page_bootstrap_balanced_accuracy(
        hard_rows, draws=draws, seed=seed
    )
    ba_test = paired_page_permutation_balanced_accuracy(
        hard_rows, draws=ba_draws, seed=seed
    )
    return {
        "analysis_protocol": hf_analysis_protocol(loaded.fp16.model_key),
        "inference_scope": "confirmatory_second_route",
        "parameters": {
            "bootstrap_draws": draws,
            "balanced_accuracy_permutation_draws": ba_draws,
            "seed": seed,
            "cluster": "page",
        },
        "population": {
            "quartets": len(pairs),
            "pages": len(effects),
            "full_units": len(hard_rows),
            "units_per_condition": len(loaded.fp16_rows),
        },
        "identity": {
            "model_key": loaded.fp16.model_key,
            "fp16_quantization": loaded.fp16.quantization,
            "gptq_quantization": loaded.gptq.quantization,
            "split": loaded.fp16.split,
            "split_sha256": loaded.fp16.split_sha256,
            "prompt_contract_sha256": (
                loaded.fp16.prompt_contract_sha256
            ),
            "hard_coverage_fp16": loaded.fp16.hard_coverage,
            "hard_coverage_gptq_int4": loaded.gptq.hard_coverage,
            "probability_coverage_fp16": loaded.fp16.probability_coverage,
            "probability_coverage_gptq_int4": (
                loaded.gptq.probability_coverage
            ),
        },
        "pair_gate": dict(loaded.gate),
        "interaction": {
            "estimand": "page-balanced mean(I_GPTQ_INT4-I_FP16)",
            "page_bootstrap": asdict(
                paired_page_bootstrap_interval(
                    effects, draws=draws, seed=seed
                )
            ),
            "page_wald": asdict(paired_page_wald_test(effects)),
            "page_direction": asdict(page_direction_summary(effects)),
            "model_internal_standardized_mean": (
                model_internal_standardized_effect(effects)
            ),
            "quartet_weighted_sensitivity": quartet_weighted_effect(pairs),
        },
        "hard_labels": {
            "accuracy": {
                "fp16": _accuracy(fp_matrix),
                "gptq_int4": _accuracy(gptq_matrix),
                "difference": _accuracy(gptq_matrix) - _accuracy(fp_matrix),
            },
            "balanced_accuracy": {
                "page_bootstrap": asdict(ba_interval),
                "page_permutation": asdict(ba_test),
            },
            "mcc": {
                "fp16": multiclass_mcc(fp_matrix),
                "gptq_int4": multiclass_mcc(gptq_matrix),
            },
            "recall": {
                "fp16": _recalls(fp_matrix),
                "gptq_int4": _recalls(gptq_matrix),
            },
            "confusion_gold_rows_predicted_columns": {
                "labels": VITAMINC_LABELS,
                "fp16": fp_matrix,
                "gptq_int4": gptq_matrix,
            },
            "paired_summary": asdict(paired_hard_label_summary(pairs)),
        },
        "label_probability_mass": {
            "fp16": _complete_mean(
                pair.label_probability_mass_f16 for pair in pairs
            ),
            "gptq_int4": _complete_mean(
                pair.label_probability_mass_quantized for pair in pairs
            ),
            "difference": _complete_mean(
                pair.label_probability_mass_quantized
                - pair.label_probability_mass_f16
                for pair in pairs
            ),
        },
        "controls": {
            control: _control_summary(
                loaded.fp16_rows, loaded.gptq_rows, control
            )
            for control in ("no_evidence", "knowledge_only")
        },
    }


def analyze_route_heterogeneity(
    hf_pairs: tuple[PairedVitaminCQuartet, ...],
    gguf_pairs: tuple[PairedVitaminCQuartet, ...],
    *,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Compare matched within-route effects without a pure-quantizer claim."""
    hf_by = {pair.case_id: pair for pair in hf_pairs}
    gguf_by = {pair.case_id: pair for pair in gguf_pairs}
    if (
        not hf_by
        or len(hf_by) != len(hf_pairs)
        or len(gguf_by) != len(gguf_pairs)
        or set(hf_by) != set(gguf_by)
    ):
        raise HFVitaminCAnalysisError(
            "route comparison requires identical unique case identities"
        )
    for case_id in sorted(hf_by):
        hf = hf_by[case_id]
        gguf = gguf_by[case_id]
        if (
            hf.page != gguf.page
            or hf.negative_label != gguf.negative_label
            or hf.hard_gold_labels != gguf.hard_gold_labels
        ):
            raise HFVitaminCAnalysisError(
                "route comparison metadata is not matched"
            )
    hf_effects = page_effects(hf_pairs)
    gguf_effects = page_effects(gguf_pairs)
    contrast = family_effect_heterogeneity(hf_effects, gguf_effects)
    fp16_backend = _page_endpoint_difference(
        hf_by, gguf_by, "interaction_f16"
    )
    quantized_backend = _page_endpoint_difference(
        hf_by, gguf_by, "interaction_quantized"
    )
    return {
        "inference_scope": "exploratory_route_heterogeneity",
        "causal_boundary": (
            "The contrast combines quantization method, runtime backend, "
            "weight packing, and numerical kernels; it is a deployment-route "
            "effect, not a pure bit-width or quantizer effect."
        ),
        "parameters": {
            "bootstrap_draws": draws,
            "seed": seed,
            "cluster": "page",
        },
        "hf_gptq_effect": asdict(
            paired_page_bootstrap_interval(
                hf_effects, draws=draws, seed=seed
            )
        ),
        "gguf_q4_effect": asdict(
            paired_page_bootstrap_interval(
                gguf_effects, draws=draws, seed=seed
            )
        ),
        "effect_difference_hf_minus_gguf": {
            "page_bootstrap": asdict(
                paired_page_bootstrap_interval(
                    contrast, draws=draws, seed=seed
                )
            ),
            "page_wald": asdict(paired_page_wald_test(contrast)),
            "direction_mcnemar": asdict(
                paired_direction_mcnemar_test(hf_effects, gguf_effects)
            ),
        },
        "directions": {
            "hf_gptq": asdict(page_direction_summary(hf_effects)),
            "gguf_q4": asdict(page_direction_summary(gguf_effects)),
        },
        "fp16_backend_interaction_hf_minus_gguf": {
            "page_bootstrap": asdict(
                paired_page_bootstrap_interval(
                    fp16_backend, draws=draws, seed=seed
                )
            ),
            "page_wald": asdict(paired_page_wald_test(fp16_backend)),
        },
        "quantized_interaction_gptq_minus_gguf_q4": {
            "page_bootstrap": asdict(
                paired_page_bootstrap_interval(
                    quantized_backend, draws=draws, seed=seed
                )
            ),
            "page_wald": asdict(
                paired_page_wald_test(quantized_backend)
            ),
        },
        "fp16_backend_hard_labels": _endpoint_hard_summary(
            hf_by, gguf_by, "hard_predictions_f16"
        ),
        "quantized_backend_hard_labels": _endpoint_hard_summary(
            hf_by, gguf_by, "hard_predictions_quantized"
        ),
    }


def _load_condition(
    manifest: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    certificate_sha256: str,
    expected_precision: str,
    mode: str,
    expected_protocol: str,
    expected_model_key: str,
) -> VitaminCConditionResults:
    _, expected_split = _MODE_IDENTITIES[mode]
    if manifest.get("schema_version") != VITAMINC_STORE_V4_SCHEMA_VERSION:
        raise ValueError("HF VitaminC export schema version mismatch")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("HF VitaminC run identity is missing")
    expected_identity = {
        "protocol_version": expected_protocol,
        "split": expected_split,
        "audit_certificate_sha256": certificate_sha256,
        "model_key": expected_model_key,
        "quantization": expected_precision,
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"HF VitaminC identity mismatch at {field}")
    for row in rows:
        payload = row.get("probability_payload")
        if not isinstance(payload, Mapping):
            raise ValueError("HF VitaminC probability payload is missing")
        if payload.get("direct_logit_method") != HF_DIRECT_LOGIT_METHOD:
            raise ValueError("HF VitaminC direct-logit method mismatch")

    full_records = [
        row for row in rows if row.get("control") == "full"
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in full_records:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("HF VitaminC record metadata is missing")
        case_id = metadata.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("HF VitaminC case identity is invalid")
        grouped[case_id].append(row)
    quartets = tuple(
        _summarize_quartet(
            case_id,
            grouped[case_id],
            expected_direct_logit_method=HF_DIRECT_LOGIT_METHOD,
        )
        for case_id in sorted(grouped)
    )
    return VitaminCConditionResults(
        protocol_version=str(identity["protocol_version"]),
        split=str(identity["split"]),
        split_sha256=str(identity["split_sha256"]),
        prompt_contract_sha256=str(identity["prompt_contract_sha256"]),
        model_key=str(identity["model_key"]),
        quantization=str(identity["quantization"]),
        hard_coverage=1.0,
        probability_coverage=1.0,
        quartets=quartets,
    )


def _hard_rows(
    pairs: tuple[PairedVitaminCQuartet, ...],
) -> tuple[HardPredictionPair, ...]:
    rows: list[HardPredictionPair] = []
    for pair in pairs:
        if not (
            pair.hard_predictions_f16
            and len(pair.hard_predictions_f16)
            == len(pair.hard_predictions_quantized)
            == len(pair.hard_gold_labels)
        ):
            raise HFVitaminCAnalysisError(
                "HF hard-label predictions are incomplete"
            )
        for index, (fp16, gptq, gold) in enumerate(
            zip(
                pair.hard_predictions_f16,
                pair.hard_predictions_quantized,
                pair.hard_gold_labels,
                strict=True,
            )
        ):
            if fp16 is None or gptq is None:
                raise HFVitaminCAnalysisError(
                    "HF hard-label coverage is incomplete"
                )
            rows.append(
                HardPredictionPair(
                    sample_id=f"{pair.case_id}:{index}",
                    page=pair.page,
                    gold_label=gold,
                    prediction_f16=fp16,
                    prediction_quantized=gptq,
                )
            )
    return tuple(rows)


def _accuracy(matrix: tuple[tuple[int, ...], ...]) -> float:
    total = sum(sum(row) for row in matrix)
    return sum(matrix[index][index] for index in range(len(matrix))) / total


def _recalls(
    matrix: tuple[tuple[int, ...], ...],
) -> dict[str, float]:
    return {
        label: matrix[index][index] / sum(matrix[index])
        for index, label in enumerate(VITAMINC_LABELS)
    }


def _complete_mean(values: Any) -> float:
    collected = tuple(values)
    if not collected or any(value is None for value in collected):
        raise HFVitaminCAnalysisError(
            "HF probability analysis requires complete values"
        )
    return statistics.fmean(float(value) for value in collected)


def _control_summary(
    fp16_rows: tuple[Mapping[str, Any], ...],
    gptq_rows: tuple[Mapping[str, Any], ...],
    control: str,
) -> dict[str, Any]:
    def labels(
        rows: tuple[Mapping[str, Any], ...],
    ) -> dict[str, str]:
        output: dict[str, str] = {}
        for row in rows:
            if row.get("control") != control:
                continue
            owner = row.get("owner_key")
            payload = row.get("hard_payload")
            scored = (
                payload.get("scored_label")
                if isinstance(payload, Mapping)
                else None
            )
            if (
                not isinstance(owner, str)
                or not owner
                or owner in output
                or not isinstance(scored, str)
                or not scored
            ):
                raise HFVitaminCAnalysisError(
                    f"{control} hard-label payload is invalid"
                )
            output[owner] = scored
        return output

    fp16 = labels(fp16_rows)
    gptq = labels(gptq_rows)
    if not fp16 or set(fp16) != set(gptq):
        raise HFVitaminCAnalysisError(
            f"{control} owner identities are not paired"
        )
    units = len(fp16)
    return {
        "units": units,
        "fp16_counts": dict(sorted(Counter(fp16.values()).items())),
        "gptq_int4_counts": dict(
            sorted(Counter(gptq.values()).items())
        ),
        "fp16_rates": {
            label: count / units
            for label, count in sorted(Counter(fp16.values()).items())
        },
        "gptq_int4_rates": {
            label: count / units
            for label, count in sorted(Counter(gptq.values()).items())
        },
        "agreement": sum(
            fp16[owner] == gptq[owner] for owner in fp16
        )
        / units,
    }


def _page_endpoint_difference(
    hf_by: Mapping[str, PairedVitaminCQuartet],
    gguf_by: Mapping[str, PairedVitaminCQuartet],
    attribute: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id in sorted(hf_by):
        hf_value = getattr(hf_by[case_id], attribute)
        gguf_value = getattr(gguf_by[case_id], attribute)
        if hf_value is None or gguf_value is None:
            raise HFVitaminCAnalysisError(
                "route endpoint interaction is incomplete"
            )
        grouped[hf_by[case_id].page].append(hf_value - gguf_value)
    return {
        page: statistics.fmean(values)
        for page, values in sorted(grouped.items())
    }


def _endpoint_hard_summary(
    hf_by: Mapping[str, PairedVitaminCQuartet],
    gguf_by: Mapping[str, PairedVitaminCQuartet],
    attribute: str,
) -> dict[str, Any]:
    agreement = 0
    hf_correct_gguf_wrong = 0
    hf_wrong_gguf_correct = 0
    both_correct = 0
    both_wrong = 0
    units = 0
    for case_id in sorted(hf_by):
        hf_pair = hf_by[case_id]
        gguf_pair = gguf_by[case_id]
        hf_predictions = getattr(hf_pair, attribute)
        gguf_predictions = getattr(gguf_pair, attribute)
        for hf_label, gguf_label, gold in zip(
            hf_predictions,
            gguf_predictions,
            hf_pair.hard_gold_labels,
            strict=True,
        ):
            if hf_label is None or gguf_label is None:
                raise HFVitaminCAnalysisError(
                    "route endpoint hard labels are incomplete"
                )
            agreement += hf_label == gguf_label
            hf_correct = hf_label == gold
            gguf_correct = gguf_label == gold
            if hf_correct and gguf_correct:
                both_correct += 1
            elif hf_correct:
                hf_correct_gguf_wrong += 1
            elif gguf_correct:
                hf_wrong_gguf_correct += 1
            else:
                both_wrong += 1
            units += 1
    return {
        "units": units,
        "agreement": agreement / units,
        "hf_accuracy": (
            both_correct + hf_correct_gguf_wrong
        )
        / units,
        "gguf_accuracy": (
            both_correct + hf_wrong_gguf_correct
        )
        / units,
        "hf_correct_gguf_wrong": hf_correct_gguf_wrong,
        "hf_wrong_gguf_correct": hf_wrong_gguf_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }
