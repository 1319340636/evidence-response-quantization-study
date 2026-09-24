"""Closed registry for the frozen Hugging Face model identities."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


class HFModelIdentityError(ValueError):
    """Raised when an HF route requests an unfrozen model identity."""


@dataclass(frozen=True)
class HFModelIdentity:
    model_key: str
    repo_id: str
    model_type: str
    smoke_protocol: str
    formal_protocol: str
    smoke_gate_protocol: str
    formal_gate_protocol: str


_IDENTITIES: Mapping[str, HFModelIdentity] = MappingProxyType(
    {
        "qwen35_9b": HFModelIdentity(
            model_key="qwen35_9b",
            repo_id="Qwen/Qwen3.5-9B",
            model_type="qwen3_5",
            smoke_protocol="hf-vitaminc-smoke-v2-20260718",
            formal_protocol="hf-vitaminc-formal-v1-20260718",
            smoke_gate_protocol="hf-pair-structural-gate-v2-20260718",
            formal_gate_protocol="hf-pair-formal-gate-v1-20260718",
        ),
        "gemma4_e4b": HFModelIdentity(
            model_key="gemma4_e4b",
            repo_id="google/gemma-4-E4B-it",
            model_type="gemma4",
            smoke_protocol="gemma-hf-vitaminc-smoke-v1-20260720",
            formal_protocol="gemma-hf-vitaminc-formal-v1-20260720",
            smoke_gate_protocol=(
                "gemma-hf-pair-structural-gate-v1-20260720"
            ),
            formal_gate_protocol="gemma-hf-pair-formal-gate-v1-20260720",
        ),
        "ministral3_8b": HFModelIdentity(
            model_key="ministral3_8b",
            repo_id="mistralai/Ministral-3-8B-Instruct-2512-BF16",
            model_type="mistral3",
            smoke_protocol="ministral-hf-triplet-smoke-v1-20260809",
            formal_protocol="ministral-hf-triplet-formal-v1-20260809",
            smoke_gate_protocol=(
                "ministral-hf-triplet-structural-gate-v1-20260809"
            ),
            formal_gate_protocol=(
                "ministral-hf-triplet-formal-gate-v1-20260809"
            ),
        ),
        "olmo3_7b": HFModelIdentity(
            model_key="olmo3_7b",
            repo_id="allenai/OLMo-3-7B-Instruct",
            model_type="olmo3",
            smoke_protocol="olmo3-hf-triplet-smoke-v1-20260901",
            formal_protocol="olmo3-hf-triplet-formal-v1-20260901",
            smoke_gate_protocol=(
                "olmo3-hf-triplet-structural-gate-v1-20260901"
            ),
            formal_gate_protocol=(
                "olmo3-hf-triplet-formal-gate-v1-20260901"
            ),
        ),
    }
)


def hf_model_identity(model_key: str) -> HFModelIdentity:
    """Return one frozen identity and reject all unregistered model keys."""
    try:
        return _IDENTITIES[model_key]
    except (KeyError, TypeError) as error:
        raise HFModelIdentityError(
            f"unsupported HF model identity: {model_key!r}"
        ) from error
