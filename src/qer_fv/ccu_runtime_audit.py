"""Fail-closed full-vocabulary probability audit for published CUB CCU."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .llama_client import (
    CHOICE_LOGIT_BIAS,
    CHOICE_SCORING_METHOD,
    LlamaServerClient,
    LlamaServerError,
)
from .model_registry import ModelRegistry, load_model_registry
from .probability_probe import (
    BIAS_CANDIDATES,
    LOG_PROBABILITY_TOLERANCE,
    USABLE_PROBABILITY_MAX,
    USABLE_PROBABILITY_MIN,
    StableProbabilityError,
    recover_stable_token_probability,
)
from .prompts import PromptInput, load_prompt_contract, render_prompt
from .scoring import HARD_LABEL_METHOD, score_choice_argmax
from .runtime_audit import (
    _has_thinking_payload,
    _llama_cpp_commit_prefix,
    _record_hash,
    _required_text,
    _server_path,
)


CUB_RUNTIME_AUDIT_PROTOCOL_VERSION = "cub-runtime-audit-v3-20260714"
CCU_AUDIT_PROTOCOL_VERSION = CUB_RUNTIME_AUDIT_PROTOCOL_VERSION
TOKEN_PROBABILITY_METHOD = "single-token-logit-bias-inversion-v1"
class CcuRuntimeAuditError(RuntimeError):
    """Raised when the runtime cannot reproduce full-vocabulary probabilities."""


@dataclass(frozen=True)
class CcuRuntimeAuditCertificate:
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
    hard_label_method: str
    probe_choice_logprobs: Mapping[str, float]
    probe_scored_choice: str
    probe_scored_label: str
    token_probability_method: str
    bias_candidates: tuple[float, ...]
    accepted_biases: Mapping[str, tuple[float, ...]]
    recovered_logprob_ranges: Mapping[str, tuple[float, float]]
    usable_probability_min: float
    usable_probability_max: float
    log_probability_tolerance: float
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
            "hard_label_method": self.hard_label_method,
            "probe_choice_logprobs": dict(self.probe_choice_logprobs),
            "probe_scored_choice": self.probe_scored_choice,
            "probe_scored_label": self.probe_scored_label,
            "token_probability_method": self.token_probability_method,
            "bias_candidates": list(self.bias_candidates),
            "accepted_biases": {
                choice: list(values)
                for choice, values in self.accepted_biases.items()
            },
            "recovered_logprob_ranges": {
                choice: list(values)
                for choice, values in self.recovered_logprob_ranges.items()
            },
            "usable_probability_min": self.usable_probability_min,
            "usable_probability_max": self.usable_probability_max,
            "log_probability_tolerance": self.log_probability_tolerance,
            "certificate_sha256": self.certificate_sha256,
        }

    def rehash(self) -> str:
        return _record_hash(self.to_record())


def audit_ccu_probability_runtime(
    *,
    project_root: str | Path,
    registry: ModelRegistry,
    prompt_contract_path: str | Path,
    model_key: str,
    quantization: str,
    client: LlamaServerClient,
    model_file_path: str | Path | None = None,
) -> CcuRuntimeAuditCertificate:
    Path(project_root).resolve()
    variant = registry.variant(model_key, quantization)
    registered_model_path = variant.path.as_posix()
    model_path = (
        Path(model_file_path)
        if model_file_path is not None
        else Path(registered_model_path)
    )
    if not model_path.exists() or not model_path.is_file():
        raise CcuRuntimeAuditError(f"model file does not exist: {model_path}")
    model_bytes = model_path.stat().st_size
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    props = client.props()
    server_model_path = _required_text(props.get("model_path"), "server model_path")
    expected_server_path = _server_path(
        model_path, registered_model_path, model_file_path
    )
    if server_model_path != expected_server_path:
        raise CcuRuntimeAuditError(
            f"server model path mismatch: expected {expected_server_path}, "
            f"got {server_model_path}"
        )
    server_build_info = _required_text(
        props.get("build_info"), "server build_info"
    )
    build_commit = _llama_cpp_commit_prefix(server_build_info)
    if build_commit is None or not registry.llama_cpp_commit.startswith(build_commit):
        raise CcuRuntimeAuditError(
            "server build_info does not contain the frozen llama.cpp commit"
        )

    prompt_path = Path(prompt_contract_path)
    prompt_contract_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    contract = load_prompt_contract(prompt_path)
    if contract.version != "prompt-v2-20260714":
        raise CcuRuntimeAuditError("CCU audit requires prompt-v2-20260714")
    probe = render_prompt(
        contract,
        PromptInput(
            dataset="cub_druid",
            sample_id="ccu-runtime-audit-probe",
            claim="The audit probe is true.",
            evidence=None,
            metadata={"claimant": "Experiment B audit"},
        ),
        control="query_only",
    )
    rendered_text = client.apply_template(probe.messages, enable_thinking=False)
    if _has_thinking_payload(rendered_text):
        raise CcuRuntimeAuditError("chat template leaked thinking content")

    choice_token_ids: dict[str, int] = {}
    accepted_biases: dict[str, tuple[float, ...]] = {}
    recovered_ranges: dict[str, tuple[float, float]] = {}
    for choice in (item.letter for item in probe.choices):
        tokens = client.tokenize(choice)
        if len(tokens) != 1:
            raise CcuRuntimeAuditError(f"choice {choice} is not a single token")
        token_id = tokens[0].id
        choice_token_ids[choice] = token_id
        try:
            stable = recover_stable_token_probability(
                client, rendered_text, token_id=token_id, seed=0
            )
        except StableProbabilityError as error:
            raise CcuRuntimeAuditError(f"{error} for choice {choice}") from error
        accepted_biases[choice] = tuple(probe.bias for probe in stable.probes)
        recovered_ranges[choice] = stable.recovered_logprob_range

    choices = tuple(probe.choice_to_label)
    try:
        choice_completion = client.complete_choice(
            rendered_text,
            choices=choices,
            choice_token_ids=choice_token_ids,
            seed=0,
        )
        hard_score = score_choice_argmax(
            probe.choice_to_label, choice_completion.choice_logprobs
        )
    except (LlamaServerError, ValueError) as error:
        raise CcuRuntimeAuditError("hard-label choice scoring audit failed") from error

    record = {
        "protocol_version": CCU_AUDIT_PROTOCOL_VERSION,
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
        "hard_label_method": HARD_LABEL_METHOD,
        "probe_choice_logprobs": dict(choice_completion.choice_logprobs),
        "probe_scored_choice": hard_score.choice,
        "probe_scored_label": hard_score.label,
        "token_probability_method": TOKEN_PROBABILITY_METHOD,
        "bias_candidates": BIAS_CANDIDATES,
        "accepted_biases": accepted_biases,
        "recovered_logprob_ranges": recovered_ranges,
        "usable_probability_min": USABLE_PROBABILITY_MIN,
        "usable_probability_max": USABLE_PROBABILITY_MAX,
        "log_probability_tolerance": LOG_PROBABILITY_TOLERANCE,
    }
    return CcuRuntimeAuditCertificate(
        **record,
        certificate_sha256=_record_hash(record),
    )


def write_ccu_runtime_audit_certificate(
    certificate: CcuRuntimeAuditCertificate, path: str | Path
) -> None:
    Path(path).write_text(
        json.dumps(
            certificate.to_record(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )


def load_ccu_runtime_audit_certificate(
    path: str | Path,
) -> CcuRuntimeAuditCertificate:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CcuRuntimeAuditError("cannot read CUB runtime certificate") from error
    if not isinstance(value, dict):
        raise CcuRuntimeAuditError("CUB runtime certificate must be a JSON object")
    record = dict(value)
    try:
        record["bias_candidates"] = tuple(record["bias_candidates"])
        record["choice_token_ids"] = {
            str(key): int(token_id)
            for key, token_id in record["choice_token_ids"].items()
        }
        record["probe_choice_logprobs"] = {
            str(key): float(logprob)
            for key, logprob in record["probe_choice_logprobs"].items()
        }
        record["accepted_biases"] = {
            str(key): tuple(float(item) for item in biases)
            for key, biases in record["accepted_biases"].items()
        }
        record["recovered_logprob_ranges"] = {
            str(key): tuple(float(item) for item in values)
            for key, values in record["recovered_logprob_ranges"].items()
        }
        certificate = CcuRuntimeAuditCertificate(**record)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise CcuRuntimeAuditError("invalid CUB runtime certificate schema") from error
    if certificate.protocol_version != CUB_RUNTIME_AUDIT_PROTOCOL_VERSION:
        raise CcuRuntimeAuditError("CUB runtime certificate protocol is not current")
    if certificate.rehash() != certificate.certificate_sha256:
        raise CcuRuntimeAuditError("CUB runtime certificate hash mismatch")
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
    certificate = audit_ccu_probability_runtime(
        project_root=root,
        registry=load_model_registry(root / "configs/models_v1.json"),
        prompt_contract_path=root / "configs/prompt_contract_v2.json",
        model_key=args.model_key,
        quantization=args.quantization,
        client=LlamaServerClient(args.server),
        model_file_path=args.model_file,
    )
    write_ccu_runtime_audit_certificate(certificate, args.output)
    print(json.dumps(certificate.to_record(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
