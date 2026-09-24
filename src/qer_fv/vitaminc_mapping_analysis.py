"""Mapping-aware analysis for frozen VitaminC D3 discovery runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hf_runtime import HF_DIRECT_LOGIT_METHOD
from .llama_client import SELECTED_TOKEN_LOGPROB_METHOD
from .multiplicity import holm_adjust
from .prompts import load_prompt_contract, render_prompt
from .vitaminc_mapping_protocol import (
    MappingCondition,
    load_mapping_protocol,
    load_mapping_units,
    require_d3_path,
)
from .vitaminc_mapping_runner import build_mapping_run_protocol
from .vitaminc_results import (
    PairedVitaminCQuartet,
    VitaminCConditionResults,
    VitaminCQuartetResult,
    pair_vitaminc_conditions,
)
from .vitaminc_statistics import (
    page_effects,
    paired_page_bootstrap_interval,
    paired_page_wald_test,
    quartet_weighted_effect,
)
from .vitaminc_store_v4 import VITAMINC_STORE_V4_SCHEMA_VERSION


D3_ANALYSIS_PROTOCOL = (
    "vitaminc-mapping-sensitivity-analysis-v1-20260719"
)
_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


@dataclass(frozen=True)
class LoadedMappingCondition:
    condition: str
    mapping_variant: str
    backend: str
    results: VitaminCConditionResults
    rows: tuple[Mapping[str, Any], ...]


def mapping_interaction_margin(
    choice_logprobs: Mapping[str, float],
    *,
    negative_label: str,
    choice_to_label: Mapping[str, str],
) -> float:
    """Return SUPPORTS-minus-negative margin under the rendered mapping."""
    if (
        set(choice_logprobs) != {"A", "B", "C"}
        or set(choice_to_label) != {"A", "B", "C"}
        or set(choice_to_label.values()) != set(_LABELS)
    ):
        raise ValueError("VitaminC mapping is incomplete or non-bijective")
    if negative_label not in {"REFUTES", "NOT ENOUGH INFO"}:
        raise ValueError("VitaminC negative label is invalid")
    label_to_choice = {
        label: choice for choice, label in choice_to_label.items()
    }
    values = {
        choice: _finite(value, "choice log-probability")
        for choice, value in choice_logprobs.items()
    }
    return (
        values[label_to_choice["SUPPORTS"]]
        - values[label_to_choice[negative_label]]
    )


def analyze_mapping_route(
    *,
    original_pairs: Sequence[PairedVitaminCQuartet],
    reversed_pairs: Sequence[PairedVitaminCQuartet],
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Compare one within-route precision effect across two prompt mappings."""
    original_by = _pairs_by_case(original_pairs)
    reversed_by = _pairs_by_case(reversed_pairs)
    if set(original_by) != set(reversed_by):
        raise ValueError("D3 mapping route case identities mismatch")
    for case_id in sorted(original_by):
        left = original_by[case_id]
        right = reversed_by[case_id]
        if (
            left.page != right.page
            or left.negative_label != right.negative_label
            or left.hard_gold_labels != right.hard_gold_labels
        ):
            raise ValueError("D3 mapping route metadata mismatch")
    original_effects = page_effects(tuple(original_pairs))
    reversed_effects = page_effects(tuple(reversed_pairs))
    if set(original_effects) != set(reversed_effects):
        raise ValueError("D3 mapping route page identities mismatch")
    difference = {
        page: reversed_effects[page] - original_effects[page]
        for page in sorted(original_effects)
    }
    original_estimate = statistics.fmean(original_effects.values())
    reversed_estimate = statistics.fmean(reversed_effects.values())
    return {
        "original": _effect_summary(
            tuple(original_pairs), original_effects, draws=draws, seed=seed
        ),
        "reversed": _effect_summary(
            tuple(reversed_pairs), reversed_effects, draws=draws, seed=seed
        ),
        "reversed_minus_original": {
            "page_bootstrap": asdict(
                paired_page_bootstrap_interval(
                    difference, draws=draws, seed=seed
                )
            ),
            "page_wald": asdict(paired_page_wald_test(difference)),
        },
        "direction_stable": _sign(original_estimate)
        == _sign(reversed_estimate),
        "hard_label_transitions": {
            "high_precision": _mapping_transitions(
                original_by, reversed_by, "hard_predictions_f16"
            ),
            "quantized": _mapping_transitions(
                original_by,
                reversed_by,
                "hard_predictions_quantized",
            ),
        },
        "label_probability_mass": {
            "original": _mass_summary(tuple(original_pairs)),
            "reversed": _mass_summary(tuple(reversed_pairs)),
        },
    }


def load_mapping_condition(
    *,
    project_root: str | Path,
    export_directory: str | Path,
    condition: str,
    mapping_variant: str,
    population: str = "formal",
) -> LoadedMappingCondition:
    """Load one D3 export and reconstruct interactions under its mapping."""
    root = Path(project_root).resolve()
    frozen = load_mapping_protocol(root)
    if condition not in frozen.condition_identities:
        raise ValueError("D3 condition is not frozen")
    if mapping_variant not in frozen.mappings:
        raise ValueError("D3 mapping variant is not frozen")
    units, population_identity = load_mapping_units(
        root, population=population
    )
    unit_by_owner = {unit.owner_key: unit for unit in units}
    condition_identity = frozen.condition_identities[condition]
    export = require_d3_path(export_directory)
    manifest_path = require_d3_path(export / "manifest.json")
    records_path = require_d3_path(export / "records.jsonl")
    manifest = _read_object(manifest_path)
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != _record_hash(unsigned):
        raise ValueError("D3 formal manifest self-hash mismatch")
    if (
        manifest.get("schema_version") != VITAMINC_STORE_V4_SCHEMA_VERSION
        or manifest.get("records_jsonl_sha256")
        != _file_sha256(records_path)
    ):
        raise ValueError("D3 formal export hash or schema mismatch")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("D3 formal run identity is missing")
    expected_identity = {
        "protocol_version": build_mapping_run_protocol(
            backend=condition_identity.backend,
            mapping_variant=mapping_variant,
            mapping_config_sha256=frozen.config_sha256,
        ),
        "split": "discovery",
        "split_sha256": population_identity.case_ids_sha256,
        "prompt_contract_sha256": frozen.prompt_contract_sha256,
        "model_key": condition_identity.model_key,
        "quantization": condition_identity.quantization,
        "expected_full": population_identity.unit_count,
        "expected_no_evidence": 0,
        "expected_knowledge_only": 0,
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"D3 formal identity mismatch at {field}")
    for field, expected in {
        "units_registered": population_identity.unit_count,
        "full_registered": population_identity.unit_count,
        "no_evidence_registered": 0,
        "knowledge_only_registered": 0,
        "hard_completed": population_identity.unit_count,
        "hard_failures": 0,
        "probability_completed": population_identity.unit_count,
        "probability_failures": 0,
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"D3 formal count mismatch at {field}")
    rows = _read_jsonl(records_path)
    if (
        len(rows) != population_identity.unit_count
        or [row.get("owner_key") for row in rows]
        != sorted(unit_by_owner)
    ):
        raise ValueError("D3 formal owner identities mismatch")
    contract = load_prompt_contract(
        root / "configs/prompt_contract_v2.json"
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        owner = row["owner_key"]
        unit = unit_by_owner[owner]
        metadata = row.get("metadata")
        if (
            row.get("control") != "full"
            or row.get("hard_status") != "ok"
            or row.get("probability_status") != "ok"
            or not isinstance(metadata, Mapping)
            or metadata.get("case_id") != unit.case_id
            or metadata.get("mapping_variant") != mapping_variant
        ):
            raise ValueError("D3 formal record structure mismatch")
        expected_prompt = render_prompt(
            contract,
            unit.prompt_input,
            control="full",
            mapping_variant=mapping_variant,
        ).prompt_sha256
        for payload_name in ("hard_payload", "probability_payload"):
            payload = row.get(payload_name)
            if (
                not isinstance(payload, Mapping)
                or payload.get("prompt_sha256") != expected_prompt
            ):
                raise ValueError("D3 formal prompt hash mismatch")
        grouped[unit.case_id].append(row)
    choice_to_label = frozen.mappings[mapping_variant]
    expected_method = (
        SELECTED_TOKEN_LOGPROB_METHOD
        if condition_identity.backend == "gguf"
        else HF_DIRECT_LOGIT_METHOD
    )
    quartets = tuple(
        _summarize_mapping_quartet(
            case_id,
            grouped[case_id],
            choice_to_label=choice_to_label,
            expected_direct_logit_method=expected_method,
        )
        for case_id in sorted(grouped)
    )
    results = VitaminCConditionResults(
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
    return LoadedMappingCondition(
        condition=condition,
        mapping_variant=mapping_variant,
        backend=condition_identity.backend,
        results=results,
        rows=tuple(rows),
    )


def analyze_mapping_run(
    *,
    project_root: str | Path,
    run_root: str | Path,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Load all 12 formal cells and analyze the three frozen paired routes."""
    root = Path(project_root).resolve()
    run = require_d3_path(run_root)
    frozen = load_mapping_protocol(root)
    loaded: dict[tuple[str, str], LoadedMappingCondition] = {}
    for condition in frozen.conditions:
        for mapping in ("original", "reversed"):
            loaded[(condition, mapping)] = load_mapping_condition(
                project_root=root,
                export_directory=run / f"{condition}__{mapping}" / "export",
                condition=condition,
                mapping_variant=mapping,
                population="formal",
            )
    routes: dict[str, dict[str, Any]] = {}
    raw_pvalues: dict[str, float] = {}
    for route, (high_name, quantized_name) in frozen.routes.items():
        pairs_by_mapping = {
            mapping: pair_vitaminc_conditions(
                loaded[(high_name, mapping)].results,
                loaded[(quantized_name, mapping)].results,
            )
            for mapping in ("original", "reversed")
        }
        route_result = analyze_mapping_route(
            original_pairs=pairs_by_mapping["original"],
            reversed_pairs=pairs_by_mapping["reversed"],
            draws=draws,
            seed=seed,
        )
        routes[route] = route_result
        raw_pvalues[route] = route_result["reversed_minus_original"][
            "page_wald"
        ]["pvalue"]
    adjusted = holm_adjust(raw_pvalues)
    for route in routes:
        routes[route]["mapping_difference_raw_pvalue"] = raw_pvalues[route]
        routes[route]["mapping_difference_holm_pvalue"] = adjusted[route]
    units, population = load_mapping_units(root, population="formal")
    return {
        "analysis_protocol": D3_ANALYSIS_PROTOCOL,
        "inference_scope": "post_hoc_prompt_robustness",
        "interpretation_boundary": (
            "Mapping sensitivity cannot select the primary mapping or "
            "rewrite the original confirmatory hypothesis."
        ),
        "parameters": {
            "draws": draws,
            "seed": seed,
            "cluster": "page",
            "mapping_difference_family": "Holm across three routes",
        },
        "population": {
            "quartets": population.case_count,
            "pages": len({unit.page for unit in units}),
            "full_units_per_cell": population.unit_count,
            "mapping_route_cells": 12,
        },
        "routes": routes,
    }


def _summarize_mapping_quartet(
    case_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    choice_to_label: Mapping[str, str],
    expected_direct_logit_method: str,
) -> VitaminCQuartetResult:
    if len(records) != 4:
        raise ValueError("D3 quartet must contain four cells")
    by_cell = {row["metadata"]["cell_index"]: row for row in records}
    if set(by_cell) != {0, 1, 2, 3}:
        raise ValueError("D3 quartet cells are incomplete")
    metadata_rows = [by_cell[index]["metadata"] for index in range(4)]
    pages = {row.get("page") for row in metadata_rows}
    negatives = {row.get("negative_label") for row in metadata_rows}
    if len(pages) != 1 or len(negatives) != 1:
        raise ValueError("D3 quartet metadata is inconsistent")
    page = next(iter(pages))
    negative = next(iter(negatives))
    if not isinstance(page, str) or not isinstance(negative, str):
        raise ValueError("D3 quartet metadata is invalid")
    hard_predictions: list[str] = []
    hard_gold: list[str] = []
    margins: list[float] = []
    masses: list[float] = []
    hard_bits: list[str] = []
    for index in range(4):
        row = by_cell[index]
        hard = row["hard_payload"]
        probability = row["probability_payload"]
        gold = row["metadata"].get("gold_label")
        predicted = hard.get("scored_label")
        if gold not in _LABELS or predicted not in _LABELS:
            raise ValueError("D3 hard label is invalid")
        if probability.get("direct_logit_method") != expected_direct_logit_method:
            raise ValueError("D3 direct-logit method mismatch")
        scores = probability.get("choice_logprobs")
        if not isinstance(scores, Mapping):
            raise ValueError("D3 choice log-probabilities are missing")
        margins.append(
            mapping_interaction_margin(
                scores,
                negative_label=negative,
                choice_to_label=choice_to_label,
            )
        )
        mass = _finite(
            probability.get("label_probability_mass"),
            "label probability mass",
        )
        if not 0.0 <= mass <= 1.0 + 1e-9:
            raise ValueError("D3 label probability mass is outside [0,1]")
        masses.append(mass)
        hard_predictions.append(predicted)
        hard_gold.append(gold)
        hard_bits.append("1" if predicted == gold else "0")
    interaction = (
        margins[0] - margins[1] - margins[2] + margins[3]
    ) / 2.0
    return VitaminCQuartetResult(
        case_id=case_id,
        page=page,
        negative_label=negative,
        interaction=interaction,
        hard_accuracy=hard_bits.count("1") / 4.0,
        hard_pattern="".join(hard_bits),
        probability_coverage=1.0,
        mean_label_probability_mass=statistics.fmean(masses),
        hard_predictions=tuple(hard_predictions),
        hard_gold_labels=tuple(hard_gold),
    )


def _effect_summary(
    pairs: tuple[PairedVitaminCQuartet, ...],
    effects: Mapping[str, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "page_bootstrap": asdict(
            paired_page_bootstrap_interval(
                effects, draws=draws, seed=seed
            )
        ),
        "page_wald": asdict(paired_page_wald_test(effects)),
        "quartet_weighted": quartet_weighted_effect(pairs),
    }


def _pairs_by_case(
    pairs: Sequence[PairedVitaminCQuartet],
) -> dict[str, PairedVitaminCQuartet]:
    values = {pair.case_id: pair for pair in pairs}
    if not values or len(values) != len(pairs):
        raise ValueError("D3 mapping route requires unique nonempty cases")
    return values


def _mapping_transitions(
    original: Mapping[str, PairedVitaminCQuartet],
    reversed_values: Mapping[str, PairedVitaminCQuartet],
    attribute: str,
) -> dict[str, int]:
    transitions: Counter[str] = Counter()
    for case_id in sorted(original):
        left = getattr(original[case_id], attribute)
        right = getattr(reversed_values[case_id], attribute)
        if len(left) != 4 or len(right) != 4 or any(
            value is None for value in (*left, *right)
        ):
            raise ValueError("D3 hard-label mapping coverage is incomplete")
        for before, after in zip(left, right, strict=True):
            transitions[f"{before}->{after}"] += 1
    return dict(sorted(transitions.items()))


def _mass_summary(
    pairs: tuple[PairedVitaminCQuartet, ...],
) -> dict[str, float]:
    high = [pair.label_probability_mass_f16 for pair in pairs]
    quantized = [
        pair.label_probability_mass_quantized for pair in pairs
    ]
    if any(value is None for value in (*high, *quantized)):
        raise ValueError("D3 label probability mass is incomplete")
    return {
        "high_precision": statistics.fmean(float(value) for value in high),
        "quantized": statistics.fmean(
            float(value) for value in quantized
        ),
    }


def _sign(value: float) -> int:
    return -1 if value < 0.0 else 1 if value > 0.0 else 0


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"D3 {label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"D3 {label} must be finite")
    return number


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D3 expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("D3 records must be JSON objects")
        values.append(value)
    return values


def _record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

