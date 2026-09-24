"""Protocol-aware VitaminC v4 loading and frozen interaction metrics."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .llama_client import SELECTED_TOKEN_LOGPROB_METHOD
from .vitaminc_runner_v4 import VITAMINC_RUN_V4_PROTOCOL_VERSION
from .vitaminc_store_v4 import VITAMINC_STORE_V4_SCHEMA_VERSION


_LABEL_TO_CHOICE = {
    "SUPPORTS": "A",
    "REFUTES": "B",
    "NOT ENOUGH INFO": "C",
}


@dataclass(frozen=True)
class VitaminCQuartetResult:
    case_id: str
    page: str
    negative_label: str
    interaction: float | None
    hard_accuracy: float
    hard_pattern: str
    probability_coverage: float
    mean_label_probability_mass: float | None
    hard_predictions: tuple[str | None, ...] = ()
    hard_gold_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class VitaminCConditionResults:
    protocol_version: str
    split: str
    split_sha256: str
    prompt_contract_sha256: str
    model_key: str
    quantization: str
    hard_coverage: float
    probability_coverage: float
    quartets: tuple[VitaminCQuartetResult, ...]


@dataclass(frozen=True)
class PairedVitaminCQuartet:
    case_id: str
    page: str
    negative_label: str
    interaction_f16: float | None
    interaction_quantized: float | None
    interaction_effect: float | None
    hard_accuracy_f16: float
    hard_accuracy_quantized: float
    hard_accuracy_effect: float
    hard_predictions_f16: tuple[str | None, ...] = ()
    hard_predictions_quantized: tuple[str | None, ...] = ()
    hard_gold_labels: tuple[str, ...] = ()
    label_probability_mass_f16: float | None = None
    label_probability_mass_quantized: float | None = None


def interaction_margin(
    choice_logprobs: Mapping[str, float], negative_label: str
) -> float:
    """Return log p(SUPPORTS) minus log p(the quartet's negative label)."""
    if set(choice_logprobs) != {"A", "B", "C"}:
        raise ValueError("VitaminC choice log-probabilities must contain A/B/C")
    try:
        negative_choice = _LABEL_TO_CHOICE[negative_label]
    except KeyError as error:
        raise ValueError("unsupported VitaminC negative label") from error
    if negative_choice == "A":
        raise ValueError("VitaminC negative label cannot be SUPPORTS")
    values = {choice: _finite(value, "choice log-probability") for choice, value in choice_logprobs.items()}
    return values["A"] - values[negative_choice]


def label_set_normalized_logprobs(
    choice_logprobs: Mapping[str, float],
) -> dict[str, float]:
    """Condition A/B/C log-probabilities on the frozen label set."""
    if set(choice_logprobs) != {"A", "B", "C"}:
        raise ValueError("VitaminC choice log-probabilities must contain A/B/C")
    values = {
        choice: _finite(value, "choice log-probability")
        for choice, value in choice_logprobs.items()
    }
    maximum = max(values.values())
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values.values())
    )
    return {
        choice: value - log_normalizer for choice, value in sorted(values.items())
    }


def quartet_interaction(
    choice_logprobs_by_cell: Mapping[int, Mapping[str, float]],
    negative_label: str,
) -> float:
    """Compute the preregistered oriented 2x2 interaction for one quartet."""
    if set(choice_logprobs_by_cell) != {0, 1, 2, 3}:
        raise ValueError("VitaminC interaction requires cells 0, 1, 2, and 3")
    margins = [
        interaction_margin(choice_logprobs_by_cell[index], negative_label)
        for index in range(4)
    ]
    return (margins[0] - margins[1] - margins[2] + margins[3]) / 2.0


def load_vitaminc_condition_export(
    export_directory: str | Path,
) -> VitaminCConditionResults:
    root = Path(export_directory)
    manifest = _read_object(root / "manifest.json")
    expected_manifest_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if expected_manifest_hash != _record_hash(unsigned):
        raise ValueError("VitaminC export manifest hash mismatch")
    if manifest.get("schema_version") != VITAMINC_STORE_V4_SCHEMA_VERSION:
        raise ValueError("VitaminC export schema version mismatch")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("VitaminC export run identity is missing")
    if identity.get("protocol_version") != VITAMINC_RUN_V4_PROTOCOL_VERSION:
        raise ValueError("VitaminC export protocol version mismatch")
    records_path = root / "records.jsonl"
    if manifest.get("records_jsonl_sha256") != _file_sha256(records_path):
        raise ValueError("VitaminC records hash mismatch")
    records = _read_jsonl(records_path)
    if len(records) != manifest.get("units_registered"):
        raise ValueError("VitaminC exported unit count mismatch")
    owner_keys = [record.get("owner_key") for record in records]
    if any(not isinstance(key, str) or not key for key in owner_keys):
        raise ValueError("VitaminC export contains an invalid owner key")
    if len(set(owner_keys)) != len(owner_keys) or owner_keys != sorted(owner_keys):
        raise ValueError("VitaminC export owner identities are not canonical")

    full_records = [record for record in records if record.get("control") == "full"]
    expected_full = identity.get("expected_full")
    if not isinstance(expected_full, int) or len(full_records) != expected_full:
        raise ValueError("VitaminC full-evidence count mismatch")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in full_records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("VitaminC record metadata is missing")
        case_id = metadata.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("VitaminC record case identity is invalid")
        grouped[case_id].append(record)

    quartets = tuple(
        _summarize_quartet(case_id, grouped[case_id]) for case_id in sorted(grouped)
    )
    hard_ok = sum(
        record.get("hard_status") == "ok" for record in full_records
    )
    probability_ok = sum(
        record.get("probability_status") == "ok" for record in full_records
    )
    denominator = len(full_records)
    return VitaminCConditionResults(
        protocol_version=str(identity["protocol_version"]),
        split=str(identity["split"]),
        split_sha256=str(identity["split_sha256"]),
        prompt_contract_sha256=str(identity["prompt_contract_sha256"]),
        model_key=str(identity["model_key"]),
        quantization=str(identity["quantization"]),
        hard_coverage=hard_ok / denominator,
        probability_coverage=probability_ok / denominator,
        quartets=quartets,
    )


def pair_vitaminc_conditions(
    f16: VitaminCConditionResults,
    quantized: VitaminCConditionResults,
) -> tuple[PairedVitaminCQuartet, ...]:
    if (
        f16.protocol_version != quantized.protocol_version
        or f16.split != quantized.split
        or f16.split_sha256 != quantized.split_sha256
    ):
        raise ValueError("VitaminC paired split identity mismatch")
    if f16.prompt_contract_sha256 != quantized.prompt_contract_sha256:
        raise ValueError("VitaminC paired prompt identity mismatch")
    if f16.model_key != quantized.model_key:
        raise ValueError("VitaminC paired model identity mismatch")
    if f16.quantization == quantized.quantization:
        raise ValueError("VitaminC paired conditions must use different precisions")
    left = {item.case_id: item for item in f16.quartets}
    right = {item.case_id: item for item in quantized.quartets}
    if set(left) != set(right):
        raise ValueError("VitaminC paired quartet identities mismatch")
    output: list[PairedVitaminCQuartet] = []
    for case_id in sorted(left):
        first = left[case_id]
        second = right[case_id]
        if (
            first.page != second.page
            or first.negative_label != second.negative_label
        ):
            raise ValueError("VitaminC paired quartet metadata mismatch")
        effect = None
        if first.interaction is not None and second.interaction is not None:
            effect = second.interaction - first.interaction
        output.append(
            PairedVitaminCQuartet(
                case_id=case_id,
                page=first.page,
                negative_label=first.negative_label,
                interaction_f16=first.interaction,
                interaction_quantized=second.interaction,
                interaction_effect=effect,
                hard_accuracy_f16=first.hard_accuracy,
                hard_accuracy_quantized=second.hard_accuracy,
                hard_accuracy_effect=second.hard_accuracy - first.hard_accuracy,
                hard_predictions_f16=first.hard_predictions,
                hard_predictions_quantized=second.hard_predictions,
                hard_gold_labels=first.hard_gold_labels,
                label_probability_mass_f16=first.mean_label_probability_mass,
                label_probability_mass_quantized=second.mean_label_probability_mass,
            )
        )
    return tuple(output)


def family_effect_heterogeneity(
    left_family_page_effects: Mapping[str, float],
    right_family_page_effects: Mapping[str, float],
) -> dict[str, float]:
    """Subtract matched within-family Q4-minus-F16 page effects."""
    if set(left_family_page_effects) != set(right_family_page_effects):
        raise ValueError("VitaminC family-effect page identities mismatch")
    return {
        page: _finite(left_family_page_effects[page], "family page effect")
        - _finite(right_family_page_effects[page], "family page effect")
        for page in sorted(left_family_page_effects)
    }


def _summarize_quartet(
    case_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_direct_logit_method: str = SELECTED_TOKEN_LOGPROB_METHOD,
) -> VitaminCQuartetResult:
    if len(records) != 4:
        raise ValueError("VitaminC quartet must contain exactly four full cells")
    by_cell: dict[int, Mapping[str, Any]] = {}
    for record in records:
        metadata = record["metadata"]
        cell_index = metadata.get("cell_index")
        if type(cell_index) is not int or cell_index not in range(4):
            raise ValueError("VitaminC cell index is invalid")
        if cell_index in by_cell:
            raise ValueError("VitaminC quartet contains a duplicate cell")
        by_cell[cell_index] = record
    if set(by_cell) != {0, 1, 2, 3}:
        raise ValueError("VitaminC quartet cells are incomplete")
    metadata_rows = [by_cell[index]["metadata"] for index in range(4)]
    pages = {row.get("page") for row in metadata_rows}
    negative_labels = {row.get("negative_label") for row in metadata_rows}
    if len(pages) != 1 or len(negative_labels) != 1:
        raise ValueError("VitaminC quartet metadata is inconsistent")
    page = next(iter(pages))
    negative_label = next(iter(negative_labels))
    if not isinstance(page, str) or not isinstance(negative_label, str):
        raise ValueError("VitaminC quartet metadata is invalid")

    hard_bits: list[str] = []
    hard_predictions: list[str | None] = []
    hard_gold_labels: list[str] = []
    probability_scores: dict[int, Mapping[str, float]] = {}
    masses: list[float] = []
    probability_ok = 0
    for index in range(4):
        record = by_cell[index]
        metadata = record["metadata"]
        hard_payload = record.get("hard_payload")
        gold_label = metadata.get("gold_label")
        if gold_label not in _LABEL_TO_CHOICE:
            raise ValueError("VitaminC gold label is invalid")
        hard_gold_labels.append(str(gold_label))
        scored_label = (
            hard_payload.get("scored_label")
            if record.get("hard_status") == "ok"
            and isinstance(hard_payload, Mapping)
            else None
        )
        if scored_label is not None and scored_label not in _LABEL_TO_CHOICE:
            raise ValueError("VitaminC scored hard label is invalid")
        hard_predictions.append(scored_label)
        hard_correct = (
            record.get("hard_status") == "ok"
            and isinstance(hard_payload, Mapping)
            and scored_label == gold_label
        )
        hard_bits.append("1" if hard_correct else "0")
        if record.get("probability_status") != "ok":
            continue
        payload = record.get("probability_payload")
        if not isinstance(payload, Mapping):
            raise ValueError("VitaminC probability payload is missing")
        if payload.get("direct_logit_method") != expected_direct_logit_method:
            raise ValueError("VitaminC direct-logit method mismatch")
        scores = payload.get("choice_logprobs")
        if not isinstance(scores, Mapping):
            raise ValueError("VitaminC choice log-probabilities are missing")
        probability_scores[index] = {
            str(choice): _nonpositive_finite(value)
            for choice, value in scores.items()
        }
        mass = _finite(payload.get("label_probability_mass"), "label mass")
        if mass < 0.0 or mass > 1.0 + 1e-9:
            raise ValueError("VitaminC label probability mass is outside [0, 1]")
        masses.append(mass)
        probability_ok += 1
    interaction = None
    if probability_ok == 4:
        interaction = quartet_interaction(probability_scores, negative_label)
    return VitaminCQuartetResult(
        case_id=case_id,
        page=page,
        negative_label=negative_label,
        interaction=interaction,
        hard_accuracy=hard_bits.count("1") / 4.0,
        hard_pattern="".join(hard_bits),
        probability_coverage=probability_ok / 4.0,
        mean_label_probability_mass=(statistics.fmean(masses) if masses else None),
        hard_predictions=tuple(hard_predictions),
        hard_gold_labels=tuple(hard_gold_labels),
    )


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"VitaminC {label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"VitaminC {label} must be finite")
    return number


def _nonpositive_finite(value: object) -> float:
    number = _finite(value, "choice log-probability")
    if number > 0.0:
        raise ValueError("VitaminC choice log-probability must be nonpositive")
    return number


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("VitaminC JSONL rows must be objects")
        output.append(value)
    return output


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
