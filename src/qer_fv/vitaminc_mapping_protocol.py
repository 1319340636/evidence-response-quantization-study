"""Frozen protocol boundaries for VitaminC label-mapping sensitivity."""

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


D3_PROTOCOL_VERSION = "vitaminc-mapping-sensitivity-v1-20260719"
D3_DISCOVERY_SHA256 = (
    "6d0b47bd5c9c603f2f61f8bad43dd7a55fcbf977525ca701c8a9c717c67ff87c"
)
D3_CONFIG_PATH = "configs/vitaminc_mapping_sensitivity_v1.json"
_EXPECTED_MAPPINGS = {
    "original": {
        "A": "SUPPORTS",
        "B": "REFUTES",
        "C": "NOT ENOUGH INFO",
    },
    "reversed": {
        "A": "NOT ENOUGH INFO",
        "B": "REFUTES",
        "C": "SUPPORTS",
    },
}
_POPULATIONS = {"engineering": 16, "formal": 1000}
_FORBIDDEN_PARTS = ("confirmatory", "recovery")


@dataclass(frozen=True)
class MappingCondition:
    backend: str
    model_key: str
    quantization: str


@dataclass(frozen=True)
class MappingSensitivityProtocol:
    protocol_version: str
    discovery_sha256: str
    prompt_contract_sha256: str
    mappings: Mapping[str, Mapping[str, str]]
    populations: Mapping[str, int]
    controls: tuple[str, ...]
    conditions: tuple[str, ...]
    condition_identities: Mapping[str, MappingCondition]
    routes: Mapping[str, tuple[str, str]]
    config_sha256: str


@dataclass(frozen=True)
class MappingPopulationIdentity:
    population: str
    case_ids: tuple[str, ...]
    case_count: int
    unit_count: int
    case_ids_sha256: str
    file_sha256: str


def require_d3_path(path: str | Path) -> Path:
    """Reject paths whose names could cross the D3 data boundary."""
    supplied = Path(path)
    for part in supplied.parts:
        folded = part.casefold()
        if any(forbidden in folded for forbidden in _FORBIDDEN_PARTS):
            raise ValueError(f"forbidden D3 path: {supplied}")
    return supplied


def load_mapping_protocol(
    project_root: str | Path,
) -> MappingSensitivityProtocol:
    """Load and strictly validate the frozen D3 mapping contract."""
    root = Path(project_root).resolve()
    config_path = require_d3_path(root / D3_CONFIG_PATH)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("D3 mapping protocol must be a JSON object")
    if raw.get("protocol_version") != D3_PROTOCOL_VERSION:
        raise ValueError("D3 mapping protocol version mismatch")
    discovery = raw.get("discovery")
    engineering = raw.get("engineering")
    prompt = raw.get("prompt_contract")
    if not all(
        isinstance(value, dict)
        for value in (discovery, engineering, prompt)
    ):
        raise ValueError("D3 mapping protocol identity is incomplete")
    if (
        discovery.get("id_sha256") != D3_DISCOVERY_SHA256
        or discovery.get("quartets") != _POPULATIONS["formal"]
        or engineering.get("quartets") != _POPULATIONS["engineering"]
    ):
        raise ValueError("D3 mapping population identity mismatch")
    if raw.get("mappings") != _EXPECTED_MAPPINGS:
        raise ValueError("D3 mapping variants mismatch")
    if raw.get("controls") != ["full"]:
        raise ValueError("D3 mapping controls must contain only full")
    prompt_path = require_d3_path(root / str(prompt.get("path", "")))
    prompt_hash = _file_sha256(prompt_path)
    if prompt.get("sha256") != prompt_hash:
        raise ValueError("D3 prompt contract SHA-256 mismatch")
    conditions = raw.get("conditions")
    if (
        not isinstance(conditions, list)
        or len(conditions) != 6
        or len(set(conditions)) != len(conditions)
        or any(not isinstance(item, str) or not item for item in conditions)
    ):
        raise ValueError("D3 condition matrix mismatch")
    identities = raw.get("condition_identities")
    if not isinstance(identities, dict) or set(identities) != set(conditions):
        raise ValueError("D3 condition identities mismatch")
    normalized_identities: dict[str, MappingCondition] = {}
    for condition in conditions:
        identity = identities[condition]
        if not isinstance(identity, dict):
            raise ValueError("D3 condition identity must be an object")
        backend = identity.get("backend")
        model_key = identity.get("model_key")
        quantization = identity.get("quantization")
        if (
            backend not in {"gguf", "hf"}
            or not isinstance(model_key, str)
            or not model_key
            or not isinstance(quantization, str)
            or not quantization
            or set(identity) != {"backend", "model_key", "quantization"}
        ):
            raise ValueError("D3 condition identity is invalid")
        normalized_identities[condition] = MappingCondition(
            backend=backend,
            model_key=model_key,
            quantization=quantization,
        )
    routes = raw.get("routes")
    expected_routes = {"qwen_gguf", "qwen_hf", "gemma_gguf"}
    if not isinstance(routes, dict) or set(routes) != expected_routes:
        raise ValueError("D3 paired routes mismatch")
    normalized_routes: dict[str, tuple[str, str]] = {}
    used_conditions: list[str] = []
    for route in sorted(routes):
        pair = routes[route]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(item not in normalized_identities for item in pair)
        ):
            raise ValueError("D3 paired route identity is invalid")
        high, quantized = pair
        high_identity = normalized_identities[high]
        quantized_identity = normalized_identities[quantized]
        if (
            high_identity.backend != quantized_identity.backend
            or high_identity.model_key != quantized_identity.model_key
            or high_identity.quantization == quantized_identity.quantization
        ):
            raise ValueError("D3 paired route endpoints are incompatible")
        normalized_routes[route] = (high, quantized)
        used_conditions.extend((high, quantized))
    if set(used_conditions) != set(conditions) or len(used_conditions) != 6:
        raise ValueError("D3 paired routes do not cover conditions exactly")
    return MappingSensitivityProtocol(
        protocol_version=D3_PROTOCOL_VERSION,
        discovery_sha256=D3_DISCOVERY_SHA256,
        prompt_contract_sha256=prompt_hash,
        mappings={
            name: dict(mapping)
            for name, mapping in _EXPECTED_MAPPINGS.items()
        },
        populations=dict(_POPULATIONS),
        controls=("full",),
        conditions=tuple(conditions),
        condition_identities=normalized_identities,
        routes=normalized_routes,
        config_sha256=_file_sha256(config_path),
    )


def load_mapping_units(
    project_root: str | Path,
    *,
    population: str,
) -> tuple[list[VitaminCInferenceUnit], MappingPopulationIdentity]:
    """Load only the frozen D3 discovery or derived engineering population."""
    if population not in _POPULATIONS:
        raise ValueError("unsupported D3 mapping population")
    root = Path(project_root).resolve()
    protocol = load_mapping_protocol(root)
    discovery_path = require_d3_path(
        root / "data/manifests/vitaminc_discovery_ids.txt"
    )
    discovery_ids = _read_ids(discovery_path)
    if (
        len(discovery_ids) != protocol.populations["formal"]
        or id_manifest_sha256(discovery_ids) != protocol.discovery_sha256
    ):
        raise ValueError("D3 discovery membership mismatch")
    quartets = load_vitaminc_split(root, split="discovery")
    by_id = {item.case_id: item for item in quartets}
    if set(by_id) != set(discovery_ids):
        raise ValueError("D3 discovery quartet identity mismatch")

    if population == "engineering":
        ids_path = require_d3_path(
            root / "data/manifests/vitaminc_mapping_smoke_16_ids.txt"
        )
        case_ids = _read_ids(ids_path)
        if case_ids != discovery_ids[: _POPULATIONS["engineering"]]:
            raise ValueError("D3 engineering membership is not the frozen prefix")
        expected = "e03dd4ee855dcf1afc28d6e40ec00a4d09aeac45a2f43d6bc34098b50073e91c"
        if id_manifest_sha256(case_ids) != expected:
            raise ValueError("D3 engineering ID SHA-256 mismatch")
    else:
        ids_path = discovery_path
        case_ids = discovery_ids

    selected = [by_id[case_id] for case_id in case_ids]
    units = [
        unit
        for unit in build_vitaminc_units(selected)
        if unit.control == "full"
    ]
    expected_units = len(case_ids) * 4
    if (
        len(units) != expected_units
        or len({unit.owner_key for unit in units}) != expected_units
        or {unit.case_id for unit in units} != set(case_ids)
    ):
        raise ValueError("D3 full-evidence unit identity mismatch")
    identity = MappingPopulationIdentity(
        population=population,
        case_ids=tuple(case_ids),
        case_count=len(case_ids),
        unit_count=len(units),
        case_ids_sha256=id_manifest_sha256(case_ids),
        file_sha256=_file_sha256(ids_path),
    )
    return units, identity


def _read_ids(path: Path) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    if (
        not values
        or any(not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("D3 ID manifest is invalid")
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
