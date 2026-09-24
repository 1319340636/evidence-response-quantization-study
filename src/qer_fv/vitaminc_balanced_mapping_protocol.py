"""Frozen three-model HF balanced label-mapping sensitivity protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .vitaminc import id_manifest_sha256
from .vitaminc_inputs import (
    VitaminCInferenceUnit,
    build_vitaminc_units,
    load_vitaminc_split,
)


BALANCED_MAPPING_PROTOCOL_VERSION = (
    "vitaminc-hf-balanced-mapping-v2-20260905"
)
BALANCED_MAPPING_CONFIG_PATH = (
    "configs/vitaminc_hf_balanced_mapping_v2.json"
)
FORMAL_IDS_SHA256 = (
    "bbe3e66dc7e53e0fa08c656716e113f63b8c5232de8b9fa895de6046b861b9f1"
)
ENGINEERING_IDS_SHA256 = (
    "2aa8f103268c8e0e54def4a5d7afd27f761ba20a2dd7dce93ab54cf21991d1c2"
)
EXPECTED_MAPPINGS = {
    "original": {
        "A": "SUPPORTS",
        "B": "REFUTES",
        "C": "NOT ENOUGH INFO",
    },
    "cycle_1": {
        "A": "REFUTES",
        "B": "NOT ENOUGH INFO",
        "C": "SUPPORTS",
    },
    "cycle_2": {
        "A": "NOT ENOUGH INFO",
        "B": "SUPPORTS",
        "C": "REFUTES",
    },
}
EXPECTED_MODELS = ("qwen35_9b", "ministral3_8b", "olmo3_7b")
EXPECTED_ROUTES = ("FP16", "GPTQ_INT4", "AWQ_INT4")
POPULATIONS = {"engineering": 4, "formal": 1200}


@dataclass(frozen=True)
class BalancedMappingProtocol:
    protocol_version: str
    mappings: Mapping[str, Mapping[str, str]]
    models: tuple[str, ...]
    routes: tuple[str, ...]
    conditions: tuple[str, ...]
    populations: Mapping[str, int]
    prompt_contract_sha256: str
    formal_ids_sha256: str
    config_sha256: str


@dataclass(frozen=True)
class BalancedMappingPopulation:
    population: str
    case_count: int
    unit_count: int
    case_ids_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ids(path: Path) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    if not values or len(values) != len(set(values)) or any(not x for x in values):
        raise ValueError("balanced mapping ID manifest is invalid")
    return values


def load_balanced_mapping_protocol(
    project_root: str | Path,
) -> BalancedMappingProtocol:
    root = Path(project_root).resolve()
    path = root / BALANCED_MAPPING_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("balanced mapping protocol must be a JSON object")
    if raw.get("protocol_version") != BALANCED_MAPPING_PROTOCOL_VERSION:
        raise ValueError("balanced mapping protocol version mismatch")
    if raw.get("mappings") != EXPECTED_MAPPINGS:
        raise ValueError("balanced mapping variants mismatch")
    prompt = raw.get("prompt_contract")
    population = raw.get("population")
    models = raw.get("models")
    if not all(isinstance(item, dict) for item in (prompt, population, models)):
        raise ValueError("balanced mapping protocol identity is incomplete")
    prompt_path = root / str(prompt.get("path", ""))
    prompt_hash = _sha256(prompt_path)
    if prompt.get("sha256") != prompt_hash:
        raise ValueError("balanced mapping prompt contract SHA-256 mismatch")
    if (
        population.get("formal_ids_sha256") != FORMAL_IDS_SHA256
        or population.get("engineering_ids_sha256") != ENGINEERING_IDS_SHA256
        or population.get("engineering_quartets") != 4
        or population.get("formal_quartets") != 1200
        or population.get("formal_pages") != 1078
        or population.get("full_evidence_inputs_per_condition") != 4800
    ):
        raise ValueError("balanced mapping population identity mismatch")
    if tuple(models) != EXPECTED_MODELS:
        raise ValueError("balanced mapping model order mismatch")
    for model in EXPECTED_MODELS:
        spec = models[model]
        if not isinstance(spec, dict) or tuple(spec.get("routes", ())) != EXPECTED_ROUTES:
            raise ValueError("balanced mapping route matrix mismatch")
    conditions = tuple(
        f"{model}_{route}" for model in EXPECTED_MODELS for route in EXPECTED_ROUTES
    )
    return BalancedMappingProtocol(
        protocol_version=BALANCED_MAPPING_PROTOCOL_VERSION,
        mappings={name: dict(mapping) for name, mapping in EXPECTED_MAPPINGS.items()},
        models=EXPECTED_MODELS,
        routes=EXPECTED_ROUTES,
        conditions=conditions,
        populations=dict(POPULATIONS),
        prompt_contract_sha256=prompt_hash,
        formal_ids_sha256=FORMAL_IDS_SHA256,
        config_sha256=_sha256(path),
    )


def load_balanced_mapping_units(
    project_root: str | Path,
    *,
    population: str,
) -> tuple[list[VitaminCInferenceUnit], BalancedMappingPopulation]:
    if population not in POPULATIONS:
        raise ValueError("unsupported balanced mapping population")
    root = Path(project_root).resolve()
    protocol = load_balanced_mapping_protocol(root)
    formal_ids = _read_ids(root / "data/manifests/vitaminc_dose_subset_ids.txt")
    if (
        len(formal_ids) != POPULATIONS["formal"]
        or id_manifest_sha256(formal_ids) != protocol.formal_ids_sha256
    ):
        raise ValueError("balanced mapping formal membership mismatch")
    if population == "engineering":
        case_ids = _read_ids(
            root / "data/manifests/vitaminc_balanced_mapping_smoke_4_ids.txt"
        )
        if case_ids != formal_ids[: POPULATIONS["engineering"]]:
            raise ValueError("balanced mapping engineering membership mismatch")
        if id_manifest_sha256(case_ids) != ENGINEERING_IDS_SHA256:
            raise ValueError("balanced mapping engineering SHA-256 mismatch")
    else:
        case_ids = formal_ids
    quartets = load_vitaminc_split(root, split="dose_subset")
    by_id = {quartet.case_id: quartet for quartet in quartets}
    if set(formal_ids) != set(by_id):
        raise ValueError("balanced mapping quartet identity mismatch")
    selected = [by_id[case_id] for case_id in case_ids]
    units = [
        unit
        for unit in build_vitaminc_units(selected)
        if unit.control == "full"
    ]
    expected_units = len(case_ids) * 4
    if len(units) != expected_units or len({u.owner_key for u in units}) != expected_units:
        raise ValueError("balanced mapping full-evidence unit identity mismatch")
    return units, BalancedMappingPopulation(
        population=population,
        case_count=len(case_ids),
        unit_count=len(units),
        case_ids_sha256=id_manifest_sha256(case_ids),
    )
