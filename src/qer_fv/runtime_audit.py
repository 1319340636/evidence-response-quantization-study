"""Fail-closed runtime audit before Experiment B inference."""

from __future__ import annotations

import hashlib
import json
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .llama_client import (
    CHOICE_LOGIT_BIAS,
    CHOICE_SCORING_METHOD,
    LlamaServerClient,
)
from .model_registry import ModelRegistry, load_model_registry
from .prompts import PromptInput, load_prompt_contract, render_prompt


AUDIT_PROTOCOL_VERSION = "runtime-audit-v2-20260713"


class RuntimeAuditError(RuntimeError):
    """Raised when a runtime cannot be certified for inference."""


@dataclass(frozen=True)
class RuntimeAuditCertificate:
    protocol_version: str
    registry_version: str
    model_key: str
    quantization: str
    registered_model_path: str
    server_model_path: str
    model_sha256: str
    model_bytes: int
    llama_cpp_commit: str
    server_build_info: str
    prompt_contract_version: str
    prompt_contract_sha256: str
    probe_prompt_sha256: str
    choice_token_ids: Mapping[str, int]
    choice_scoring_method: str
    choice_logit_bias: float
    certificate_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "registry_version": self.registry_version,
            "model_key": self.model_key,
            "quantization": self.quantization,
            "registered_model_path": self.registered_model_path,
            "server_model_path": self.server_model_path,
            "model_sha256": self.model_sha256,
            "model_bytes": self.model_bytes,
            "llama_cpp_commit": self.llama_cpp_commit,
            "server_build_info": self.server_build_info,
            "prompt_contract_version": self.prompt_contract_version,
            "prompt_contract_sha256": self.prompt_contract_sha256,
            "probe_prompt_sha256": self.probe_prompt_sha256,
            "choice_token_ids": dict(self.choice_token_ids),
            "choice_scoring_method": self.choice_scoring_method,
            "choice_logit_bias": self.choice_logit_bias,
            "certificate_sha256": self.certificate_sha256,
        }

    def rehash(self) -> str:
        return _record_hash(self.to_record())


def audit_runtime(
    *,
    project_root: str | Path,
    registry: ModelRegistry,
    prompt_contract_path: str | Path,
    model_key: str,
    quantization: str,
    client: LlamaServerClient,
    model_file_path: str | Path | None = None,
) -> RuntimeAuditCertificate:
    Path(project_root).resolve()
    variant = registry.variant(model_key, quantization)
    registered_model_path = variant.path.as_posix()
    model_path = Path(model_file_path) if model_file_path is not None else Path(registered_model_path)
    if not model_path.exists() or not model_path.is_file():
        raise RuntimeAuditError(f"model file does not exist: {model_path}")
    model_bytes = model_path.stat().st_size
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    props = client.props()
    server_model_path = _required_text(props.get("model_path"), "server model_path")
    expected_server_path = _server_path(model_path, registered_model_path, model_file_path)
    if server_model_path != expected_server_path:
        raise RuntimeAuditError(
            f"server model path mismatch: expected {expected_server_path}, got {server_model_path}"
        )
    server_build_info = _required_text(props.get("build_info"), "server build_info")
    build_commit = _llama_cpp_commit_prefix(server_build_info)
    if build_commit is None or not registry.llama_cpp_commit.startswith(build_commit):
        raise RuntimeAuditError(
            "server build_info does not contain the frozen llama.cpp commit"
        )

    prompt_path = Path(prompt_contract_path)
    prompt_contract_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    contract = load_prompt_contract(prompt_path)
    probe = render_prompt(
        contract,
        PromptInput(
            dataset="vitaminc",
            sample_id="runtime-audit-probe",
            claim="The audit probe has evidence.",
            evidence="The audit probe has evidence.",
        ),
    )
    rendered_text = client.apply_template(probe.messages, enable_thinking=False)
    if _has_thinking_payload(rendered_text):
        raise RuntimeAuditError("chat template leaked thinking content")

    choices = tuple(choice.letter for choice in probe.choices)
    choice_token_ids: dict[str, int] = {}
    for choice in choices:
        tokens = client.tokenize(choice)
        if len(tokens) != 1:
            raise RuntimeAuditError(f"choice {choice} is not a single token")
        choice_token_ids[choice] = tokens[0].id

    completion = client.complete_choice(
        rendered_text,
        choices=choices,
        choice_token_ids=choice_token_ids,
    )
    missing = sorted(set(choices) - set(completion.choice_logprobs))
    if missing:
        raise RuntimeAuditError(f"completion missing constrained logprobs: {missing}")

    record = {
        "protocol_version": AUDIT_PROTOCOL_VERSION,
        "registry_version": registry.version,
        "model_key": model_key,
        "quantization": quantization,
        "registered_model_path": registered_model_path,
        "server_model_path": server_model_path,
        "model_sha256": model_sha256,
        "model_bytes": model_bytes,
        "llama_cpp_commit": registry.llama_cpp_commit,
        "server_build_info": server_build_info,
        "prompt_contract_version": contract.version,
        "prompt_contract_sha256": prompt_contract_sha256,
        "probe_prompt_sha256": probe.prompt_sha256,
        "choice_token_ids": choice_token_ids,
        "choice_scoring_method": CHOICE_SCORING_METHOD,
        "choice_logit_bias": CHOICE_LOGIT_BIAS,
    }
    return RuntimeAuditCertificate(
        **record,
        certificate_sha256=_record_hash(record),
    )


def write_runtime_audit_certificate(
    certificate: RuntimeAuditCertificate, path: str | Path
) -> None:
    Path(path).write_text(
        json.dumps(
            certificate.to_record(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_runtime_audit_certificate(path: str | Path) -> RuntimeAuditCertificate:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeAuditError("runtime audit certificate must be a JSON object")
    certificate = RuntimeAuditCertificate(**payload)
    if certificate.certificate_sha256 != certificate.rehash():
        raise RuntimeAuditError("runtime audit certificate hash mismatch")
    return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--server", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    registry = load_model_registry(root / "configs/models_v1.json")
    certificate = audit_runtime(
        project_root=root,
        registry=registry,
        prompt_contract_path=root / "configs/prompt_contract_v1.json",
        model_key=args.model_key,
        quantization=args.quantization,
        client=LlamaServerClient(args.server),
        model_file_path=args.model_file,
    )
    write_runtime_audit_certificate(certificate, args.output)
    print(json.dumps(certificate.to_record(), ensure_ascii=False, sort_keys=True))
    return 0


def _record_hash(record: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in record.items() if key != "certificate_sha256"}
    payload = json.dumps(
        clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeAuditError(f"{field} must be a nonempty string")
    return value


def _llama_cpp_commit_prefix(build_info: str) -> str | None:
    match = re.search(
        r"(?:^|[-( ])([0-9a-f]{7,40})(?=$|[)\s])", build_info.casefold()
    )
    return None if match is None else match.group(1)


def _has_thinking_payload(rendered_text: str) -> bool:
    without_empty_placeholders = re.sub(
        r"<think>\s*</think>", "", rendered_text, flags=re.IGNORECASE
    )
    return re.search(
        r"</?think(?:\s[^>]*)?>", without_empty_placeholders, flags=re.IGNORECASE
    ) is not None


def _server_path(
    model_path: Path, registered_model_path: str, override: str | Path | None
) -> str:
    if override is None:
        return registered_model_path
    return str(model_path)


if __name__ == "__main__":
    raise SystemExit(main())
