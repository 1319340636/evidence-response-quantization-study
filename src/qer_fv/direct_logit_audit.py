"""Fail-closed runtime audit for CUB v4 direct selected-token log-probabilities."""

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


DIRECT_LOGIT_AUDIT_PROTOCOL_VERSION = "cub-runtime-audit-v4-20260715"
DIRECT_LOGIT_DUPLICATE_TOLERANCE = 1e-7


class DirectLogitAuditError(RuntimeError):
    """Raised when a CUB v4 direct-logit runtime cannot be certified."""


@dataclass(frozen=True)
class DirectLogitAuditCertificate:
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


def audit_direct_logit_runtime(
    *,
    project_root: str | Path,
    registry: ModelRegistry,
    prompt_contract_path: str | Path,
    model_key: str,
    quantization: str,
    client: LlamaServerClient,
    patch_path: str | Path,
    server_binary_path: str | Path,
    server_implementation_path: str | Path,
    model_file_path: str | Path | None = None,
) -> DirectLogitAuditCertificate:
    Path(project_root).resolve()
    variant = registry.variant(model_key, quantization)
    registered_model_path = variant.path.as_posix()
    model_path = Path(model_file_path) if model_file_path is not None else Path(registered_model_path)
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
    expected_server_path = _server_path(model_path, registered_model_path, model_file_path)
    if server_model_path != expected_server_path:
        raise DirectLogitAuditError(
            f"server model path mismatch: expected {expected_server_path}, got {server_model_path}"
        )
    server_build_info = _required_text(props.get("build_info"), "server build_info")
    build_commit = _llama_cpp_commit_prefix(server_build_info)
    if build_commit is None or not registry.llama_cpp_commit.startswith(build_commit):
        raise DirectLogitAuditError(
            "server build_info does not contain the frozen llama.cpp commit"
        )

    prompt_path = Path(prompt_contract_path)
    prompt_contract_sha256, _ = _file_identity(prompt_path, "prompt contract")
    contract = load_prompt_contract(prompt_path)
    if contract.version != "prompt-v2-20260714":
        raise DirectLogitAuditError("direct-logit audit requires prompt-v2-20260714")
    probe = render_prompt(
        contract,
        PromptInput(
            dataset="cub_druid",
            sample_id="direct-logit-runtime-audit-probe",
            claim="The audit probe is true.",
            evidence=None,
            metadata={"claimant": "Experiment B audit"},
        ),
        control="query_only",
    )
    rendered_text = client.apply_template(probe.messages, enable_thinking=False)
    if _has_thinking_payload(rendered_text):
        raise DirectLogitAuditError("chat template leaked thinking content")

    choices = tuple(item.letter for item in probe.choices)
    choice_token_ids: dict[str, int] = {}
    for choice in choices:
        pieces = client.tokenize(choice)
        if len(pieces) != 1:
            raise DirectLogitAuditError(f"choice {choice} is not a single token")
        choice_token_ids[choice] = pieces[0].id
    if len(set(choice_token_ids.values())) != len(choice_token_ids):
        raise DirectLogitAuditError("choice token IDs must be unique")

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
        raise DirectLogitAuditError("direct-logit runtime probe failed") from error

    first_by_choice = _validated_choice_scores(first, choice_token_ids)
    second_by_choice = _validated_choice_scores(second, choice_token_ids)
    deltas = {
        choice: abs(first_by_choice[choice] - second_by_choice[choice])
        for choice in choices
    }
    max_delta = max(deltas.values())
    if max_delta > DIRECT_LOGIT_DUPLICATE_TOLERANCE:
        raise DirectLogitAuditError(
            "duplicate direct-logit probe drift exceeds tolerance"
        )
    direct_score = score_choice_argmax(probe.choice_to_label, first_by_choice)
    probabilities = {
        choice: math.exp(first_by_choice[choice]) for choice in choices
    }

    record = {
        "protocol_version": DIRECT_LOGIT_AUDIT_PROTOCOL_VERSION,
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
        "probe_prompt_sha256": probe.prompt_sha256,
        "choice_token_ids": choice_token_ids,
        "choice_scoring_method": CHOICE_SCORING_METHOD,
        "choice_logit_bias": CHOICE_LOGIT_BIAS,
        "hard_label_method": HARD_LABEL_METHOD,
        "direct_logit_method": SELECTED_TOKEN_LOGPROB_METHOD,
        "duplicate_tolerance": DIRECT_LOGIT_DUPLICATE_TOLERANCE,
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
    return DirectLogitAuditCertificate(
        **record,
        certificate_sha256=_record_hash(record),
    )


def write_direct_logit_audit_certificate(
    certificate: DirectLogitAuditCertificate, path: str | Path
) -> None:
    Path(path).write_text(
        json.dumps(certificate.to_record(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_direct_logit_audit_certificate(
    path: str | Path,
) -> DirectLogitAuditCertificate:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DirectLogitAuditError("cannot read direct-logit audit certificate") from error
    if not isinstance(raw, dict):
        raise DirectLogitAuditError("direct-logit audit certificate must be a JSON object")
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
        certificate = DirectLogitAuditCertificate(**record)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise DirectLogitAuditError("invalid direct-logit audit certificate schema") from error
    if certificate.protocol_version != DIRECT_LOGIT_AUDIT_PROTOCOL_VERSION:
        raise DirectLogitAuditError("direct-logit audit certificate protocol is not current")
    if certificate.rehash() != certificate.certificate_sha256:
        raise DirectLogitAuditError("direct-logit audit certificate hash mismatch")
    return certificate


def _validated_choice_scores(
    scores: SelectedTokenScores, choice_token_ids: Mapping[str, int]
) -> dict[str, float]:
    expected_ids = tuple(choice_token_ids.values())
    if scores.method != SELECTED_TOKEN_LOGPROB_METHOD:
        raise DirectLogitAuditError("direct-logit method mismatch")
    if scores.token_ids != expected_ids or tuple(scores.logprobs) != expected_ids:
        raise DirectLogitAuditError("direct-logit token order mismatch")
    result: dict[str, float] = {}
    for choice, token_id in choice_token_ids.items():
        value = scores.logprobs[token_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value > 0.0
        ):
            raise DirectLogitAuditError("direct-logit probe contains an invalid score")
        result[choice] = float(value)
    return result


def _file_identity(path: Path, label: str) -> tuple[str, int]:
    if not path.exists() or not path.is_file():
        raise DirectLogitAuditError(f"{label} file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DirectLogitAuditError(f"{field} must be a nonempty string")
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
    certificate = audit_direct_logit_runtime(
        project_root=root,
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
    write_direct_logit_audit_certificate(certificate, args.output)
    print(json.dumps(certificate.to_record(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
