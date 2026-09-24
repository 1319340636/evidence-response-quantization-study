"""Data and prompt provenance gates that deliberately do not certify inference."""

from __future__ import annotations

import hashlib
import string
from pathlib import Path
from typing import Any, Mapping

from .manifests import verify_data_foundation
from .prompts import PromptContract, load_prompt_contract
from .provenance import load_freeze_config


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _required_sha256(value: object, field: str) -> str:
    text = _required_text(value, field)
    if len(text) != 64 or any(character not in string.hexdigits for character in text):
        raise ValueError(f"{field} must be a 64-character SHA-256")
    return text.casefold()


def verify_prompt_provenance(
    project_root: str | Path,
    freeze_config_path: str | Path,
    prompt_contract_path: str | Path,
    contract: PromptContract,
) -> dict[str, str]:
    """Bind one validated prompt contract to the frozen config by path and bytes."""
    root = Path(project_root).resolve()
    config_path = Path(freeze_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_freeze_config(config_path)
    declaration = config.get("prompt_contract")
    if not isinstance(declaration, Mapping):
        raise ValueError("freeze config prompt_contract declaration is required")
    declared_path = declaration.get("path")
    if not isinstance(declared_path, str):
        raise ValueError("freeze config prompt contract path must be a string")
    expected_path = (root / declared_path).resolve()
    actual_path = Path(prompt_contract_path).resolve()
    if actual_path != expected_path:
        raise ValueError(
            f"prompt contract path mismatch: expected {expected_path}, got {actual_path}"
        )
    declared_version = declaration.get("version")
    if declared_version != contract.version:
        raise ValueError(
            f"prompt contract version mismatch: expected {declared_version}, "
            f"got {contract.version}"
        )
    expected_sha256 = _required_sha256(
        declaration.get("sha256"), "freeze config prompt contract sha256"
    )
    actual_sha256 = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"prompt contract SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return {
        "prompt_contract_path": declared_path,
        "prompt_contract_sha256": actual_sha256,
        "prompt_contract_version": contract.version,
    }


def verify_prompt_preflight(
    project_root: str | Path,
    freeze_config_path: str | Path,
    manifest_dir: str | Path,
    prompt_contract_path: str | Path,
) -> dict[str, Any]:
    """Verify frozen data and prompt bytes while keeping inference blocked."""
    data_summary = verify_data_foundation(
        project_root, freeze_config_path, manifest_dir
    )
    contract = load_prompt_contract(prompt_contract_path)
    prompt_summary = verify_prompt_provenance(
        project_root, freeze_config_path, prompt_contract_path, contract
    )
    return {
        "blocking_gate": "runtime_tokenizer_audit",
        "data_bundle_sha256": data_summary["bundle_sha256"],
        "inference_ready": False,
        "prompt_contract_sha256": prompt_summary["prompt_contract_sha256"],
        "prompt_contract_version": prompt_summary["prompt_contract_version"],
    }
