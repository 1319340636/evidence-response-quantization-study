"""Fail-closed analysis of the four-condition Fresh TabFact formal run."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .llama_client import SELECTED_TOKEN_LOGPROB_METHOD
from .scoring import HARD_LABEL_METHOD
from .tabfact_runner import (
    TABFACT_FORMAL_EVIDENCE_SHA256,
    TABFACT_FORMAL_MANIFEST_SHA256,
    TABFACT_FORMAL_PROTOCOL_VERSION,
    TABFACT_FORMAL_SPLIT_SHA256,
)
from .tabfact_store import TABFACT_STORE_SCHEMA_VERSION
from .vitaminc import id_manifest_sha256


TABFACT_FORMAL_ANALYSIS_PROTOCOL = "tabfact-formal-analysis-v1-20260719"
TABFACT_FORMAL_GATE_PROTOCOL = "tabfact-fresh-formal-gate-v1-20260718"
TABFACT_FORMAL_PROMPT_SHA256 = (
    "62be3ad998b851887ce433bb80a8269f020dbea63c048d418f0e1277590352e1"
)
TABFACT_LABELS = ("Entailed", "Refuted")
TABFACT_FORMAL_CONDITIONS = (
    "qwen35_9b_F16",
    "qwen35_9b_Q4_K_M",
    "gemma4_e4b_F16",
    "gemma4_e4b_Q4_K_M",
)
_CONDITION_IDENTITIES = {
    "qwen35_9b_F16": ("qwen35_9b", "F16"),
    "qwen35_9b_Q4_K_M": ("qwen35_9b", "Q4_K_M"),
    "gemma4_e4b_F16": ("gemma4_e4b", "F16"),
    "gemma4_e4b_Q4_K_M": ("gemma4_e4b", "Q4_K_M"),
}
_FAMILY_PAIRS = {
    "qwen35_9b": ("qwen35_9b_F16", "qwen35_9b_Q4_K_M"),
    "gemma4_e4b": ("gemma4_e4b_F16", "gemma4_e4b_Q4_K_M"),
}


class TabFactFormalAnalysisError(RuntimeError):
    """Raised when formal TabFact artifacts cannot enter analysis."""


@dataclass(frozen=True)
class TabFactPrediction:
    owner_key: str
    sample_id: str
    table_id: str
    gold_label: str
    hard_label: str
    probability_label: str


@dataclass(frozen=True)
class TabFactCondition:
    name: str
    model_key: str
    quantization: str
    run_identity: Mapping[str, Any]
    export_manifest_sha256: str
    records_sha256: str
    records: tuple[TabFactPrediction, ...]


@dataclass(frozen=True)
class LoadedTabFactFormal:
    run_root: str
    gate: Mapping[str, Any]
    gate_sha256: str
    conditions: Mapping[str, TabFactCondition]


def load_tabfact_formal_run(run_root: str | Path) -> LoadedTabFactFormal:
    """Revalidate the gate and all four immutable condition exports."""
    root = Path(run_root)
    try:
        gate_path = root / "formal_gate.json"
        gate_sha256 = _file_sha256(gate_path)
        _validate_gate_sidecar(root / "formal_gate.sha256", gate_sha256)
        gate = _load_json_object(gate_path)
        if gate.get("protocol_version") != TABFACT_FORMAL_GATE_PROTOCOL:
            raise ValueError("formal gate protocol mismatch")
        gate_conditions = gate.get("conditions")
        if not isinstance(gate_conditions, list):
            raise ValueError("formal gate conditions are missing")
        names = tuple(
            item.get("name") if isinstance(item, Mapping) else None
            for item in gate_conditions
        )
        if names != TABFACT_FORMAL_CONDITIONS:
            raise ValueError("formal gate condition order mismatch")

        conditions: dict[str, TabFactCondition] = {}
        for gate_condition in gate_conditions:
            assert isinstance(gate_condition, Mapping)
            name = str(gate_condition["name"])
            conditions[name] = _load_condition(root, name, gate_condition)
        _validate_aligned_population(conditions)
    except (
        AssertionError,
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise TabFactFormalAnalysisError(str(error)) from error
    return LoadedTabFactFormal(
        run_root=str(root),
        gate=gate,
        gate_sha256=gate_sha256,
        conditions=conditions,
    )


def analyze_tabfact_formal(
    loaded: LoadedTabFactFormal,
    *,
    draws: int = 10_000,
    seed: int = 20260711,
) -> dict[str, Any]:
    """Calculate frozen paired binary outcomes with table-cluster intervals."""
    if type(draws) is not int or draws < 1:
        raise TabFactFormalAnalysisError("bootstrap draws must be positive")
    if type(seed) is not int:
        raise TabFactFormalAnalysisError("bootstrap seed must be an integer")
    try:
        conditions = {
            name: loaded.conditions[name] for name in TABFACT_FORMAL_CONDITIONS
        }
        aligned = _aligned_records(conditions)
        condition_metrics = {
            name: _condition_metrics(condition.records)
            for name, condition in conditions.items()
        }
        bootstrap = _joint_cluster_bootstrap(
            aligned, draws=draws, seed=seed
        )
        family_effects = {
            family: _family_effect(
                conditions[f16],
                conditions[q4],
                condition_metrics[f16],
                condition_metrics[q4],
                bootstrap=bootstrap[family],
            )
            for family, (f16, q4) in _FAMILY_PAIRS.items()
        }
        baseline = conditions[TABFACT_FORMAL_CONDITIONS[0]]
        tables = {record.table_id for record in baseline.records}
        gold = Counter(record.gold_label for record in baseline.records)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise TabFactFormalAnalysisError(str(error)) from error
    return {
        "analysis_protocol": TABFACT_FORMAL_ANALYSIS_PROTOCOL,
        "inference_scope": "external_structured_evidence_validation",
        "parameters": {
            "bootstrap_draws": draws,
            "seed": seed,
            "cluster": "table",
            "primary_comparison": "within_family_q4_k_m_minus_f16",
        },
        "population": {
            "claims": len(baseline.records),
            "tables": len(tables),
            "gold_counts": {
                label: gold[label] for label in TABFACT_LABELS
            },
            "conditions": len(conditions),
        },
        "integrity": {
            "formal_gate_sha256": loaded.gate_sha256,
            "owner_alignment": "exact",
            "hard_probability_label_disagreements": {
                name: sum(
                    record.hard_label != record.probability_label
                    for record in condition.records
                )
                for name, condition in conditions.items()
            },
            "condition_sources": {
                name: {
                    "export_manifest_sha256": (
                        condition.export_manifest_sha256
                    ),
                    "records_sha256": condition.records_sha256,
                    "audit_certificate_sha256": condition.run_identity.get(
                        "audit_certificate_sha256"
                    ),
                }
                for name, condition in conditions.items()
            },
        },
        "condition_metrics": condition_metrics,
        "within_family_effects": family_effects,
        "family_effect_heterogeneity": {
            metric: {
                "contrast": "qwen_effect_minus_gemma_effect",
                "estimate": (
                    family_effects["qwen35_9b"][metric]["difference"]
                    - family_effects["gemma4_e4b"][metric]["difference"]
                ),
                "cluster_bootstrap_95_ci": bootstrap["heterogeneity"][metric],
            }
            for metric in ("accuracy", "balanced_accuracy")
        }
        | {
            "interpretation_boundary": (
                "This is heterogeneity of matched within-family effects, not "
                "a comparison of raw model quality or an internal mechanism."
            )
        },
    }


def _load_condition(
    root: Path, name: str, gate_condition: Mapping[str, Any]
) -> TabFactCondition:
    run = root / name
    export = run / "export"
    audit = _load_json_object(run / "audit.json")
    manifest_path = export / "manifest.json"
    report_path = export / "structural_report.json"
    if _file_sha256(manifest_path) != gate_condition.get(
        "export_manifest_file_sha256"
    ):
        raise ValueError(f"{name} export manifest file hash mismatch")
    if _file_sha256(report_path) != gate_condition.get(
        "structural_report_file_sha256"
    ):
        raise ValueError(f"{name} structural report file hash mismatch")
    certificate_sha256 = audit.get("certificate_sha256")
    if (
        not isinstance(certificate_sha256, str)
        or certificate_sha256
        != gate_condition.get("audit_certificate_sha256")
    ):
        raise ValueError(f"{name} audit certificate hash mismatch")
    if (
        gate_condition.get("owners") != 2_000
        or gate_condition.get("hard_failures") != 0
        or gate_condition.get("probability_failures") != 0
    ):
        raise ValueError(f"{name} formal gate is incomplete")

    report = _load_json_object(report_path)
    expected_report = {
        "schema_version": "tabfact-structural-formal-v1-20260718",
        "gate_passed": True,
        "owners": 2_000,
        "hard_completed": 2_000,
        "hard_failures": 0,
        "probability_completed": 2_000,
        "probability_failures": 0,
    }
    for field, expected in expected_report.items():
        if report.get(field) != expected:
            raise ValueError(f"{name} structural report mismatch at {field}")

    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != TABFACT_STORE_SCHEMA_VERSION:
        raise ValueError(f"{name} export schema mismatch")
    expected_manifest_hash = _record_hash(
        manifest, excluded="manifest_sha256"
    )
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError(f"{name} export manifest self-hash mismatch")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{name} run identity is missing")
    model_key, quantization = _CONDITION_IDENTITIES[name]
    expected_identity = {
        "protocol_version": TABFACT_FORMAL_PROTOCOL_VERSION,
        "split_sha256": TABFACT_FORMAL_SPLIT_SHA256,
        "manifest_sha256": TABFACT_FORMAL_MANIFEST_SHA256,
        "evidence_sha256": TABFACT_FORMAL_EVIDENCE_SHA256,
        "audit_certificate_sha256": certificate_sha256,
        "prompt_contract_sha256": TABFACT_FORMAL_PROMPT_SHA256,
        "model_key": model_key,
        "quantization": quantization,
        "expected_units": 2_000,
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"{name} run identity mismatch at {field}")
    for field, expected in (
        ("owners", 2_000),
        ("hard_completed", 2_000),
        ("hard_failures", 0),
        ("probability_completed", 2_000),
        ("probability_failures", 0),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"{name} export count mismatch at {field}")

    records_path = export / "records.jsonl"
    records_sha256 = _file_sha256(records_path)
    if manifest.get("records_jsonl_sha256") != records_sha256:
        raise ValueError(f"{name} records hash mismatch")
    records = _load_records(records_path, name)
    return TabFactCondition(
        name=name,
        model_key=model_key,
        quantization=quantization,
        run_identity=dict(identity),
        export_manifest_sha256=str(manifest["manifest_sha256"]),
        records_sha256=records_sha256,
        records=records,
    )


def _load_records(path: Path, name: str) -> tuple[TabFactPrediction, ...]:
    output: list[TabFactPrediction] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise ValueError(f"{name} records contain a blank line")
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} record {line_number} is not an object")
        owner_key = row.get("owner_key")
        metadata = row.get("metadata")
        hard = row.get("hard_payload")
        probability = row.get("probability_payload")
        if (
            not isinstance(owner_key, str)
            or not owner_key
            or owner_key in seen
            or not isinstance(metadata, Mapping)
            or not isinstance(hard, Mapping)
            or not isinstance(probability, Mapping)
        ):
            raise ValueError(f"{name} record identity or payload is invalid")
        seen.add(owner_key)
        if row.get("metadata_sha256") != _record_hash(metadata):
            raise ValueError(f"{name} record metadata hash mismatch")
        if row.get("hard_payload_sha256") != _record_hash(hard):
            raise ValueError(f"{name} hard payload hash mismatch")
        if row.get("probability_payload_sha256") != _record_hash(probability):
            raise ValueError(f"{name} probability payload hash mismatch")
        if (
            row.get("hard_status") != "ok"
            or row.get("hard_error_code") is not None
            or row.get("probability_status") != "ok"
            or row.get("probability_error_code") is not None
        ):
            raise ValueError(f"{name} result stages are incomplete")
        if (
            metadata.get("control") != "full"
            or metadata.get("table_truncated") is not False
        ):
            raise ValueError(f"{name} metadata control or truncation mismatch")
        sample_id = metadata.get("sample_id")
        table_id = metadata.get("table_id")
        gold_label = metadata.get("gold_label")
        hard_label = hard.get("scored_label")
        probability_label = probability.get("scored_label")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or owner_key != f"{sample_id}|full"
            or not isinstance(table_id, str)
            or not table_id
            or gold_label not in TABFACT_LABELS
            or hard_label not in TABFACT_LABELS
            or probability_label not in TABFACT_LABELS
        ):
            raise ValueError(f"{name} record labels or metadata are invalid")
        if hard.get("hard_label_method") != HARD_LABEL_METHOD:
            raise ValueError(f"{name} hard-label method mismatch")
        if (
            probability.get("direct_logit_method")
            != SELECTED_TOKEN_LOGPROB_METHOD
        ):
            raise ValueError(f"{name} selected-logit method mismatch")
        if hard_label != probability_label:
            raise ValueError(f"{name} hard/probability label disagreement")
        output.append(
            TabFactPrediction(
                owner_key=owner_key,
                sample_id=sample_id,
                table_id=table_id,
                gold_label=str(gold_label),
                hard_label=str(hard_label),
                probability_label=str(probability_label),
            )
        )
    if len(output) != 2_000:
        raise ValueError(f"{name} records must contain exactly 2,000 rows")
    table_counts = Counter(record.table_id for record in output)
    if len(table_counts) != 1_434 or max(table_counts.values()) != 2:
        raise ValueError(f"{name} records table population mismatch")
    gold_counts = Counter(record.gold_label for record in output)
    if gold_counts != Counter({"Entailed": 1_003, "Refuted": 997}):
        raise ValueError(f"{name} records gold population mismatch")
    if (
        id_manifest_sha256(record.sample_id for record in output)
        != TABFACT_FORMAL_SPLIT_SHA256
    ):
        raise ValueError(f"{name} records split identity mismatch")
    return tuple(output)


def _validate_aligned_population(
    conditions: Mapping[str, TabFactCondition],
) -> None:
    baseline = {
        row.owner_key: (row.sample_id, row.table_id, row.gold_label)
        for row in conditions[TABFACT_FORMAL_CONDITIONS[0]].records
    }
    for name in TABFACT_FORMAL_CONDITIONS[1:]:
        compared = {
            row.owner_key: (row.sample_id, row.table_id, row.gold_label)
            for row in conditions[name].records
        }
        if compared != baseline:
            raise ValueError(f"{name} owner metadata alignment mismatch")


def _aligned_records(
    conditions: Mapping[str, TabFactCondition],
) -> dict[str, tuple[TabFactPrediction, ...]]:
    indexed = {
        name: {record.owner_key: record for record in condition.records}
        for name, condition in conditions.items()
    }
    keys = set(indexed[TABFACT_FORMAL_CONDITIONS[0]])
    if not keys or any(set(indexed[name]) != keys for name in indexed):
        raise ValueError("formal condition owner alignment mismatch")
    for key in keys:
        metadata = {
            (
                indexed[name][key].sample_id,
                indexed[name][key].table_id,
                indexed[name][key].gold_label,
            )
            for name in indexed
        }
        if len(metadata) != 1:
            raise ValueError("formal condition metadata alignment mismatch")
    return {
        key: tuple(indexed[name][key] for name in TABFACT_FORMAL_CONDITIONS)
        for key in sorted(keys)
    }


def _condition_metrics(
    records: Sequence[TabFactPrediction],
) -> dict[str, Any]:
    confusion = _confusion(records)
    tp, tn, fp, fn = (
        confusion["tp"],
        confusion["tn"],
        confusion["fp"],
        confusion["fn"],
    )
    total = tp + tn + fp + fn
    entailed_recall = tp / (tp + fn)
    refuted_recall = tn / (tn + fp)
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    return {
        "n": total,
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": (entailed_recall + refuted_recall) / 2,
        "mcc": (
            (tp * tn - fp * fn) / denominator if denominator else 0.0
        ),
        "recall": {
            "Entailed": entailed_recall,
            "Refuted": refuted_recall,
        },
        "predicted_entailed_rate": (tp + fp) / total,
        "confusion": confusion,
    }


def _confusion(records: Sequence[TabFactPrediction]) -> dict[str, int]:
    return {
        "tp": sum(
            record.gold_label == "Entailed"
            and record.hard_label == "Entailed"
            for record in records
        ),
        "tn": sum(
            record.gold_label == "Refuted"
            and record.hard_label == "Refuted"
            for record in records
        ),
        "fp": sum(
            record.gold_label == "Refuted"
            and record.hard_label == "Entailed"
            for record in records
        ),
        "fn": sum(
            record.gold_label == "Entailed"
            and record.hard_label == "Refuted"
            for record in records
        ),
    }


def _family_effect(
    f16: TabFactCondition,
    q4: TabFactCondition,
    f16_metrics: Mapping[str, Any],
    q4_metrics: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    left = {record.owner_key: record for record in f16.records}
    right = {record.owner_key: record for record in q4.records}
    if set(left) != set(right):
        raise ValueError("family pair owner alignment mismatch")
    degradation = improvement = flips = both_correct = both_wrong = 0
    for key in left:
        first, second = left[key], right[key]
        if (
            first.sample_id,
            first.table_id,
            first.gold_label,
        ) != (
            second.sample_id,
            second.table_id,
            second.gold_label,
        ):
            raise ValueError("family pair metadata alignment mismatch")
        first_correct = first.hard_label == first.gold_label
        second_correct = second.hard_label == second.gold_label
        flips += first.hard_label != second.hard_label
        degradation += first_correct and not second_correct
        improvement += not first_correct and second_correct
        both_correct += first_correct and second_correct
        both_wrong += not first_correct and not second_correct
    return {
        "accuracy": {
            "f16": f16_metrics["accuracy"],
            "q4_k_m": q4_metrics["accuracy"],
            "difference": q4_metrics["accuracy"] - f16_metrics["accuracy"],
            "cluster_bootstrap_95_ci": bootstrap["accuracy"],
        },
        "balanced_accuracy": {
            "f16": f16_metrics["balanced_accuracy"],
            "q4_k_m": q4_metrics["balanced_accuracy"],
            "difference": (
                q4_metrics["balanced_accuracy"]
                - f16_metrics["balanced_accuracy"]
            ),
            "cluster_bootstrap_95_ci": bootstrap["balanced_accuracy"],
        },
        "mcc": {
            "f16": f16_metrics["mcc"],
            "q4_k_m": q4_metrics["mcc"],
            "difference": q4_metrics["mcc"] - f16_metrics["mcc"],
        },
        "recall": {
            label: {
                "f16": f16_metrics["recall"][label],
                "q4_k_m": q4_metrics["recall"][label],
                "difference": (
                    q4_metrics["recall"][label]
                    - f16_metrics["recall"][label]
                ),
            }
            for label in TABFACT_LABELS
        },
        "predicted_entailed_rate": {
            "f16": f16_metrics["predicted_entailed_rate"],
            "q4_k_m": q4_metrics["predicted_entailed_rate"],
            "difference": (
                q4_metrics["predicted_entailed_rate"]
                - f16_metrics["predicted_entailed_rate"]
            ),
        },
        "transitions": {
            "prediction_flips": int(flips),
            "f16_correct_q4_wrong": int(degradation),
            "f16_wrong_q4_correct": int(improvement),
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            "mcnemar_exact_pvalue": _exact_mcnemar(
                int(degradation), int(improvement)
            ),
        },
    }


def _joint_cluster_bootstrap(
    aligned: Mapping[str, tuple[TabFactPrediction, ...]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_table: dict[str, list[tuple[TabFactPrediction, ...]]] = defaultdict(list)
    for rows in aligned.values():
        by_table[rows[0].table_id].append(rows)
    tables = sorted(by_table)
    contributions: list[tuple[int, ...]] = []
    for table in tables:
        rows = by_table[table]
        n = len(rows)
        positive = sum(item[0].gold_label == "Entailed" for item in rows)
        negative = n - positive
        values: list[int] = [n, positive, negative]
        for condition_index in range(len(TABFACT_FORMAL_CONDITIONS)):
            values.extend(
                [
                    sum(
                        item[condition_index].hard_label
                        == item[condition_index].gold_label
                        for item in rows
                    ),
                    sum(
                        item[condition_index].gold_label == "Entailed"
                        and item[condition_index].hard_label == "Entailed"
                        for item in rows
                    ),
                    sum(
                        item[condition_index].gold_label == "Refuted"
                        and item[condition_index].hard_label == "Refuted"
                        for item in rows
                    ),
                ]
            )
        contributions.append(tuple(values))

    replicates = {
        "qwen_accuracy": [],
        "qwen_balanced_accuracy": [],
        "gemma_accuracy": [],
        "gemma_balanced_accuracy": [],
        "heterogeneity_accuracy": [],
        "heterogeneity_balanced_accuracy": [],
    }
    generator = random.Random(seed)
    cluster_count = len(contributions)
    accepted = 0
    attempts = 0
    maximum_attempts = draws * 100
    while accepted < draws:
        attempts += 1
        if attempts > maximum_attempts:
            raise ValueError(
                "cluster bootstrap cannot draw both gold classes"
            )
        totals = [0] * len(contributions[0])
        for _ in range(cluster_count):
            selected = contributions[generator.randrange(cluster_count)]
            for index, value in enumerate(selected):
                totals[index] += value
        n, positive, negative = totals[:3]
        if positive == 0 or negative == 0:
            continue
        accepted += 1
        metrics: list[tuple[float, float]] = []
        for condition_index in range(len(TABFACT_FORMAL_CONDITIONS)):
            offset = 3 + condition_index * 3
            correct, positive_correct, negative_correct = totals[
                offset : offset + 3
            ]
            metrics.append(
                (
                    correct / n,
                    (
                        positive_correct / positive
                        + negative_correct / negative
                    )
                    / 2,
                )
            )
        qwen_accuracy = metrics[1][0] - metrics[0][0]
        qwen_balanced = metrics[1][1] - metrics[0][1]
        gemma_accuracy = metrics[3][0] - metrics[2][0]
        gemma_balanced = metrics[3][1] - metrics[2][1]
        replicates["qwen_accuracy"].append(qwen_accuracy)
        replicates["qwen_balanced_accuracy"].append(qwen_balanced)
        replicates["gemma_accuracy"].append(gemma_accuracy)
        replicates["gemma_balanced_accuracy"].append(gemma_balanced)
        replicates["heterogeneity_accuracy"].append(
            qwen_accuracy - gemma_accuracy
        )
        replicates["heterogeneity_balanced_accuracy"].append(
            qwen_balanced - gemma_balanced
        )
    intervals = {
        name: _interval(values, draws, seed, cluster_count)
        for name, values in replicates.items()
    }
    return {
        "qwen35_9b": {
            "accuracy": intervals["qwen_accuracy"],
            "balanced_accuracy": intervals["qwen_balanced_accuracy"],
        },
        "gemma4_e4b": {
            "accuracy": intervals["gemma_accuracy"],
            "balanced_accuracy": intervals["gemma_balanced_accuracy"],
        },
        "heterogeneity": {
            "accuracy": intervals["heterogeneity_accuracy"],
            "balanced_accuracy": intervals[
                "heterogeneity_balanced_accuracy"
            ],
        },
    }


def _interval(
    values: list[float], draws: int, seed: int, clusters: int
) -> dict[str, Any]:
    values.sort()
    return {
        "lower": _quantile(values, 0.025),
        "upper": _quantile(values, 0.975),
        "draws": draws,
        "seed": seed,
        "clusters": clusters,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _exact_mcnemar(degradation: int, improvement: int) -> float:
    discordant = degradation + improvement
    if discordant == 0:
        return 1.0
    lower = min(degradation, improvement)
    tail = sum(
        math.comb(discordant, index) for index in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _validate_gate_sidecar(path: Path, gate_sha256: str) -> None:
    fields = path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[0] != gate_sha256:
        raise ValueError("formal gate SHA-256 sidecar mismatch")
    if Path(fields[1]).name != "formal_gate.json":
        raise ValueError("formal gate SHA-256 sidecar filename mismatch")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(
    value: Mapping[str, Any], *, excluded: str | None = None
) -> str:
    payload = {
        key: item for key, item in value.items() if key != excluded
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
