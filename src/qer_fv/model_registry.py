"""Strict, immutable model registry for Experiment B."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


REGISTRY_VERSION = "models-v1-20260712"
_SHA40 = re.compile(r"[0-9a-f]{40}")
_ROOT_FIELDS = {"registry_version", "llama_cpp_commit", "remote_model_root", "models"}
_MODEL_FIELDS = {"key", "family", "capacity", "repo_id", "revision", "source_dir", "variants"}
_VARIANT_FIELDS = {"quantization", "file"}


@dataclass(frozen=True)
class ModelVariant:
    model_key: str
    quantization: str
    relative_file: PurePosixPath
    path: PurePosixPath


@dataclass(frozen=True)
class SourceModel:
    key: str
    family: str
    capacity: str
    repo_id: str
    revision: str
    source_dir: PurePosixPath
    variants: Mapping[str, ModelVariant]


@dataclass(frozen=True)
class ModelRegistry:
    version: str
    llama_cpp_commit: str
    remote_model_root: PurePosixPath
    models: Mapping[str, SourceModel]

    def model(self, key: str) -> SourceModel:
        try:
            return self.models[key]
        except KeyError as error:
            raise ValueError(f"unknown model key: {key}") from error

    def variant(self, model_key: str, quantization: str) -> ModelVariant:
        model = self.model(model_key)
        try:
            return model.variants[quantization]
        except KeyError as error:
            raise ValueError(
                f"unknown quantization for {model_key}: {quantization}"
            ) from error


def _object(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _exact_fields(value: dict, expected: set[str], field: str) -> None:
    extras = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if extras:
        raise ValueError(f"{field} has unknown fields: {extras}")
    if missing:
        raise ValueError(f"{field} is missing fields: {missing}")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _commit(value: object, field: str) -> str:
    text = _text(value, field)
    if _SHA40.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase 40-character commit SHA")
    return text


def _relative(value: object, field: str) -> PurePosixPath:
    text = _text(value, field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return path


def load_model_registry(path: str | Path) -> ModelRegistry:
    payload = _object(json.loads(Path(path).read_text(encoding="utf-8")), "registry")
    _exact_fields(payload, _ROOT_FIELDS, "registry")
    version = _text(payload["registry_version"], "registry_version")
    if version != REGISTRY_VERSION:
        raise ValueError(f"registry version mismatch: expected {REGISTRY_VERSION}, got {version}")
    llama_commit = _commit(payload["llama_cpp_commit"], "llama_cpp_commit")
    root = PurePosixPath(_text(payload["remote_model_root"], "remote_model_root"))
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("remote_model_root must be an absolute safe POSIX path")
    raw_models = payload["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a nonempty list")

    models: dict[str, SourceModel] = {}
    for index, raw_value in enumerate(raw_models):
        raw = _object(raw_value, f"models[{index}]")
        _exact_fields(raw, _MODEL_FIELDS, f"models[{index}]")
        key = _text(raw["key"], f"models[{index}].key")
        if key in models:
            raise ValueError(f"duplicate model key: {key}")
        source_dir = _relative(raw["source_dir"], f"models[{index}].source_dir")
        raw_variants = raw["variants"]
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ValueError(f"models[{index}].variants must be a nonempty list")
        variants: dict[str, ModelVariant] = {}
        for variant_index, raw_variant_value in enumerate(raw_variants):
            raw_variant = _object(raw_variant_value, f"models[{index}].variants[{variant_index}]")
            _exact_fields(raw_variant, _VARIANT_FIELDS, f"models[{index}].variants[{variant_index}]")
            quantization = _text(raw_variant["quantization"], "quantization")
            if quantization in variants:
                raise ValueError(f"duplicate quantization for {key}: {quantization}")
            relative_file = _relative(raw_variant["file"], "variant file")
            variants[quantization] = ModelVariant(
                model_key=key,
                quantization=quantization,
                relative_file=relative_file,
                path=root / relative_file,
            )
        models[key] = SourceModel(
            key=key,
            family=_text(raw["family"], "family"),
            capacity=_text(raw["capacity"], "capacity"),
            repo_id=_text(raw["repo_id"], "repo_id"),
            revision=_commit(raw["revision"], "revision"),
            source_dir=source_dir,
            variants=MappingProxyType(variants),
        )
    return ModelRegistry(version, llama_commit, root, MappingProxyType(models))
