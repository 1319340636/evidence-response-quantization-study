"""Fail-closed VitaminC runtime audit for direct selected-token log-probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .llama_client import (
    CHOICE_LOGIT_BIAS,
    CHOICE_SCORING_METHOD,
    SELECTED_TOKEN_LOGPROB_METHOD,
    LlamaServerClient,
    LlamaServerError,
    SelectedTokenScores,
)
from .model_registry import ModelRegistry, load_model_registry
from .prompts import PromptInput, load_prompt_contract, render_prompt
from .runtime_audit import (
    _has_thinking_payload,
    _llama_cpp_commit_prefix,
    _record_hash,
    _server_path,
)
from .scoring import HARD_LABEL_METHOD, score_choice_argmax


VITAMINC_RUNTIME_AUDIT_V4_PROTOCOL = "vitaminc-runtime-audit-v4-20260716"
VITAMINC_RUNTIME_DUPLICATE_TOLERANCE = 1e-7


class VitaminCRuntimeAuditV4Error(RuntimeError):
    """Raised when the VitaminC formal runtime cannot be certified."""


@dataclass(frozen=True)
class VitaminCRuntimeAuditCertificateV4:
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
    server_patch_path: str
    server_patch_sha256: str
    server_binary_path: str
    server_binary_sha256: str
    server_implementation_path: str
    server_implementation_sha256: str
    prompt_contract_version: str
    prompt_contract_sha256: str
    probe_dataset: str
    probe_control: str
    probe_prompt_sha256: str
    choice_token_ids: Mapping[str, int]
    choice_scoring_method: str
    choice_logit_bias: float
    hard_label_method: str
    direct_logit_method: str
    duplicate_tolerance: float
    duplicate_max_abs_delta: float
    probe_hard_choice_logprobs: Mapping[str, float]
    probe_hard_scored_choice: str
    probe_hard_scored_label: str
    probe_direct_logprobs_first: Mapping[str, float]
    probe_direct_logprobs_second: Mapping[str, float]
    probe_direct_probabilities: Mapping[str, float]
    probe_direct_scored_choice: str
    probe_direct_scored_label: str
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
            "server_patch_path": self.server_patch_path,
            "server_patch_sha256": self.server_patch_sha256,
            "server_binary_path": self.server_binary_path,
            "server_binary_sha256": self.server_binary_sha256,
            "server_implementation_path": self.server_implementation_path,
            "server_implementation_sha256": self.server_implementation_sha256,
            "prompt_contract_version": self.prompt_contract_version,
            "prompt_contract_sha256": self.prompt_contract_sha256,
            "probe_dataset": self.probe_dataset,
            "probe_control": self.probe_control,
            "probe_prompt_sha256": self.probe_prompt_sha256,
            "choice_token_ids": dict(self.choice_token_ids),
            "choice_scoring_method": self.choice_scoring_method,
            "choice_logit_bias": self.choice_logit_bias,
            "hard_label_method": self.hard_label_method,
            "direct_logit_method": self.direct_logit_method,
            "duplicate_tolerance": self.duplicate_tolerance,
            "duplicate_max_abs_delta": self.duplicate_max_abs_delta,
            "probe_hard_choice_logprobs": dict(self.probe_hard_choice_logprobs),
            "probe_hard_scored_choice": self.probe_hard_scored_choice,
            "probe_hard_scored_label": self.probe_hard_scored_label,
            "probe_direct_logprobs_first": dict(self.probe_direct_logprobs_first),
            "probe_direct_logprobs_second": dict(self.probe_direct_logprobs_second),
            "probe_direct_probabilities": dict(self.probe_direct_probabilities),
            "probe_direct_scored_choice": self.probe_direct_scored_choice,
            "probe_direct_scored_label": self.probe_direct_scored_label,
            "certificate_sha256": self.certificate_sha256,
        }

    def rehash(self) -> str:
        return _record_hash(self.to_record())


def audit_vitaminc_runtime_v4(
    *,
    registry: ModelRegistry,
    prompt_contract_path: str | Path,
    model_key: str,
    quantization: str,
    client: LlamaServerClient,
    patch_path: str | Path,
    server_binary_path: str | Path,
    server_implementation_path: str | Path,
    model_file_path: str | Path | None = None,
) -> VitaminCRuntimeAuditCertificateV4:
    variant = registry.variant(model_key, quantization)
    registered_model_path = variant.path.as_posix()
    model_path = (
        Path(model_file_path) if model_file_path is not None else Path(registered_model_path)
    )
    model_sha256, model_bytes = _file_identity(model_path, "model")
    patch = Path(patch_path)
    patch_sha256, _ = _file_identity(patch, "server patch")
    binary = Path(server_binary_path)
    binary_sha256, _ = _file_identity(binary, "server binary")
    implementation = Path(server_implementation_path)
    implementation_sha256, _ = _file_identity(
        implementation, "server implementation library"
    )

    props = client.props()
    server_model_path = _required_text(props.get("model_path"), "server model_path")
    expected_server_path = _server_path(
        model_path, registered_model_path, model_file_path
    )
    if server_model_path != expected_server_path:
        raise VitaminCRuntimeAuditV4Error("server model path mismatch")
    server_build_info = _required_text(props.get("build_info"), "server build_info")
    build_commit = _llama_cpp_commit_prefix(server_build_info)
    if build_commit is None or not registry.llama_cpp_commit.startswith(build_commit):
        raise VitaminCRuntimeAuditV4Error(
            "server build_info does not contain the frozen llama.cpp commit"
        )

    prompt_path = Path(prompt_contract_path)
    prompt_contract_sha256, _ = _file_identity(prompt_path, "prompt contract")
    contract = load_prompt_contract(prompt_path)
    if contract.version != "prompt-v2-20260714":
        raise VitaminCRuntimeAuditV4Error(
            "VitaminC v4 audit requires prompt-v2-20260714"
        )
    probe = render_prompt(
        contract,
        PromptInput(
            dataset="vitaminc",
            sample_id="vitaminc-direct-logit-runtime-audit-probe",
            claim="Water freezes at zero degrees Celsius at standard pressure.",
            evidence="At standard pressure, the freezing point of water is 0 °C.",
            metadata={},
        ),
        control="full",
    )
    rendered_text = client.apply_template(probe.messages, enable_thinking=False)
    if _has_thinking_payload(rendered_text):
        raise VitaminCRuntimeAuditV4Error("chat template leaked thinking content")

    choices = tuple(item.letter for item in probe.choices)
    choice_token_ids: dict[str, int] = {}
    for choice in choices:
        pieces = client.tokenize(choice)
        if len(pieces) != 1:
            raise VitaminCRuntimeAuditV4Error(f"choice {choice} is not a single token")
        choice_token_ids[choice] = pieces[0].id
    if len(set(choice_token_ids.values())) != len(choice_token_ids):
        raise VitaminCRuntimeAuditV4Error("choice token IDs must be unique")

    try:
        hard_completion = client.complete_choice(
            rendered_text,
            choices=choices,
            choice_token_ids=choice_token_ids,
            seed=0,
        )
        hard_score = score_choice_argmax(
            probe.choice_to_label, hard_completion.choice_logprobs
        )
        token_ids = tuple(choice_token_ids.values())
        first = client.score_selected_tokens(rendered_text, token_ids=token_ids, seed=0)
        second = client.score_selected_tokens(rendered_text, token_ids=token_ids, seed=0)
    except (LlamaServerError, ValueError, KeyError, TypeError) as error:
        raise VitaminCRuntimeAuditV4Error("VitaminC direct-logit probe failed") from error

    first_by_choice = _validated_choice_scores(first, choice_token_ids)
    second_by_choice = _validated_choice_scores(second, choice_token_ids)
    max_delta = max(
        abs(first_by_choice[choice] - second_by_choice[choice])
        for choice in choices
    )
    if max_delta > VITAMINC_RUNTIME_DUPLICATE_TOLERANCE:
        raise VitaminCRuntimeAuditV4Error(
            "duplicate direct-logit probe drift exceeds tolerance"
        )
    direct_score = score_choice_argmax(probe.choice_to_label, first_by_choice)
    probabilities = {
        choice: math.exp(first_by_choice[choice]) for choice in choices
    }
    record = {
        "protocol_version": VITAMINC_RUNTIME_AUDIT_V4_PROTOCOL,
        "registry_version": registry.version,
        "model_key": model_key,
        "quantization": quantization,
        "registered_model_path": registered_model_path,
        "server_model_path": server_model_path,
        "model_sha256": model_sha256,
        "model_bytes": model_bytes,
        "llama_cpp_commit": registry.llama_cpp_commit,
        "server_build_info": server_build_info,
        "server_patch_path": patch.as_posix(),
        "server_patch_sha256": patch_sha256,
        "server_binary_path": binary.as_posix(),
        "server_binary_sha256": binary_sha256,
        "server_implementation_path": implementation.as_posix(),
        "server_implementation_sha256": implementation_sha256,
        "prompt_contract_version": contract.version,
        "prompt_contract_sha256": prompt_contract_sha256,
        "probe_dataset": "vitaminc",
        "probe_control": "full",
        "probe_prompt_sha256": probe.prompt_sha256,
        "choice_token_ids": choice_token_ids,
        "choice_scoring_method": CHOICE_SCORING_METHOD,
        "choice_logit_bias": CHOICE_LOGIT_BIAS,
        "hard_label_method": HARD_LABEL_METHOD,
        "direct_logit_method": SELECTED_TOKEN_LOGPROB_METHOD,
        "duplicate_tolerance": VITAMINC_RUNTIME_DUPLICATE_TOLERANCE,
        "duplicate_max_abs_delta": max_delta,
        "probe_hard_choice_logprobs": dict(hard_completion.choice_logprobs),
        "probe_hard_scored_choice": hard_score.choice,
        "probe_hard_scored_label": hard_score.label,
        "probe_direct_logprobs_first": first_by_choice,
        "probe_direct_logprobs_second": second_by_choice,
        "probe_direct_probabilities": probabilities,
        "probe_direct_scored_choice": direct_score.choice,
        "probe_direct_scored_label": direct_score.label,
    }
    return VitaminCRuntimeAuditCertificateV4(
        **record, certificate_sha256=_record_hash(record)
    )


def write_vitaminc_runtime_audit_v4(
    certificate: VitaminCRuntimeAuditCertificateV4, path: str | Path
) -> None:
    Path(path).write_text(
        json.dumps(certificate.to_record(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_vitaminc_runtime_audit_v4(
    path: str | Path,
) -> VitaminCRuntimeAuditCertificateV4:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VitaminCRuntimeAuditV4Error("cannot read VitaminC runtime certificate") from error
    if not isinstance(raw, dict):
        raise VitaminCRuntimeAuditV4Error("VitaminC runtime certificate must be an object")
    if raw.get("protocol_version") != VITAMINC_RUNTIME_AUDIT_V4_PROTOCOL:
        raise VitaminCRuntimeAuditV4Error("VitaminC runtime certificate protocol mismatch")
    record = dict(raw)
    try:
        record["choice_token_ids"] = _integer_mapping(record["choice_token_ids"])
        for field in (
            "probe_hard_choice_logprobs",
            "probe_direct_logprobs_first",
            "probe_direct_logprobs_second",
            "probe_direct_probabilities",
        ):
            record[field] = _float_mapping(record[field])
        certificate = VitaminCRuntimeAuditCertificateV4(**record)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise VitaminCRuntimeAuditV4Error("invalid VitaminC runtime certificate") from error
    if certificate.rehash() != certificate.certificate_sha256:
        raise VitaminCRuntimeAuditV4Error("VitaminC runtime certificate hash mismatch")
    return certificate


def _validated_choice_scores(
    scores: SelectedTokenScores, choice_token_ids: Mapping[str, int]
) -> dict[str, float]:
    expected_ids = tuple(choice_token_ids.values())
    if scores.method != SELECTED_TOKEN_LOGPROB_METHOD:
        raise VitaminCRuntimeAuditV4Error("direct-logit method mismatch")
    if scores.token_ids != expected_ids or tuple(scores.logprobs) != expected_ids:
        raise VitaminCRuntimeAuditV4Error("direct-logit token order mismatch")
    result: dict[str, float] = {}
    for choice, token_id in choice_token_ids.items():
        value = scores.logprobs[token_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value > 0.0
        ):
            raise VitaminCRuntimeAuditV4Error("direct-logit score is invalid")
        result[choice] = float(value)
    return result


def _file_identity(path: Path, label: str) -> tuple[str, int]:
    if not path.is_file():
        raise VitaminCRuntimeAuditV4Error(f"{label} file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VitaminCRuntimeAuditV4Error(f"{field} must be a nonempty string")
    return value


def _integer_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return {str(key): int(item) for key, item in value.items()}


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return {str(key): float(item) for key, item in value.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--server", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--server-implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    certificate = audit_vitaminc_runtime_v4(
        registry=load_model_registry(root / "configs/models_v1.json"),
        prompt_contract_path=root / "configs/prompt_contract_v2.json",
        model_key=args.model_key,
        quantization=args.quantization,
        client=LlamaServerClient(args.server),
        patch_path=args.patch,
        server_binary_path=args.server_binary,
        server_implementation_path=args.server_implementation,
        model_file_path=args.model_file,
    )
    write_vitaminc_runtime_audit_v4(certificate, args.output)
    print(json.dumps(certificate.to_record(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
