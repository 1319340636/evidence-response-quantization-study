"""Audited query-only/context paired runner for CUB BCU and CCU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ccu_runtime_audit import (
    CUB_RUNTIME_AUDIT_PROTOCOL_VERSION,
    TOKEN_PROBABILITY_METHOD,
    CcuRuntimeAuditCertificate,
    CcuRuntimeAuditError,
    load_ccu_runtime_audit_certificate,
)
from .cub_druid import (
    CUB_OUTPUT_LABELS,
    CubRecord,
    cub_binary_context_utilisation,
    cub_continuous_context_utilisation,
)
from .cub_queries import CubQueryGroup, build_cub_query_groups
from .cub_store import CubRunIdentity, CubRunStore
from .llama_client import (
    CHOICE_LOGIT_BIAS,
    CHOICE_SCORING_METHOD,
    LlamaServerClient,
    LlamaServerError,
)
from .probability_probe import (
    BIAS_CANDIDATES,
    LOG_PROBABILITY_TOLERANCE,
    USABLE_PROBABILITY_MAX,
    USABLE_PROBABILITY_MIN,
    StableProbabilityError,
    StableTokenProbability,
    recover_stable_token_probability,
)
from .prompts import PromptInput, load_prompt_contract, render_prompt
from .scoring import HARD_LABEL_METHOD, score_choice_argmax


CUB_RUN_PROTOCOL_VERSION = "cub-paired-run-v1-20260714"
_INFERENCE_ERRORS = (
    LlamaServerError,
    StableProbabilityError,
    ValueError,
    KeyError,
    TypeError,
)


def run_cub_records(
    *,
    project_root: str | Path,
    records: Sequence[CubRecord],
    manifest_sha256: str,
    client: LlamaServerClient,
    audit_certificate_path: str | Path,
    database_path: str | Path,
    export_directory: str | Path | None = None,
) -> dict[str, int]:
    root = Path(project_root).resolve()
    certificate = load_ccu_runtime_audit_certificate(audit_certificate_path)
    contract_path = root / "configs/prompt_contract_v2.json"
    contract = load_prompt_contract(contract_path)
    _verify_runtime(certificate, contract_path, client)
    groups = build_cub_query_groups(
        records, prompt_contract_version=contract.version
    )
    identity = CubRunIdentity(
        protocol_version=CUB_RUN_PROTOCOL_VERSION,
        audit_certificate_sha256=certificate.certificate_sha256,
        manifest_sha256=_required_sha256(manifest_sha256, "manifest_sha256"),
        prompt_contract_sha256=certificate.prompt_contract_sha256,
        model_key=certificate.model_key,
        quantization=certificate.quantization,
        expected_queries=len(groups),
        expected_contexts=len(records),
    )
    with CubRunStore(database_path, identity) as store:
        for group in groups:
            if store.has_query(group.query_identity_sha256):
                query_payload = store.get_query_payload(group.query_identity_sha256)
                query_status = store.get_query_status(group.query_identity_sha256)
            else:
                try:
                    query_payload, query_probes = _run_query(
                        group, contract, certificate, client
                    )
                except _INFERENCE_ERRORS as error:
                    query_payload = _failure_payload(certificate, "query", error)
                    store.put_query_failure(
                        group.query_identity_sha256,
                        error_code="query_inference_error",
                        payload=query_payload,
                    )
                    query_status = "error"
                else:
                    store.put_query_success(
                        group.query_identity_sha256,
                        query_payload,
                        probes=query_probes,
                    )
                    query_status = "ok"
            for record in group.records:
                if store.has_context(record.sample_id):
                    continue
                if query_status != "ok":
                    store.put_context_failure(
                        record.sample_id,
                        group.query_identity_sha256,
                        error_code="query_dependency_error",
                        payload={
                            "analysis_context_type": record.analysis_context_type,
                            "audit_certificate_sha256": certificate.certificate_sha256,
                            "ccu_target_label": (
                                record.target_new
                                if record.context_type != "irrelevant"
                                else query_payload.get("hard_label")
                            ),
                            "context_identity_sha256": record.context_identity_sha256,
                            "stage": "query_dependency",
                            "target_new": record.target_new,
                        },
                    )
                    continue
                try:
                    context_payload, context_probes = _run_context(
                        record,
                        group,
                        query_payload,
                        contract,
                        certificate,
                        client,
                    )
                except _INFERENCE_ERRORS as error:
                    store.put_context_failure(
                        record.sample_id,
                        group.query_identity_sha256,
                        error_code="context_inference_error",
                        payload={
                            **_failure_payload(certificate, "context", error),
                            "analysis_context_type": record.analysis_context_type,
                            "ccu_target_label": (
                                record.target_new
                                if record.context_type != "irrelevant"
                                else query_payload.get("hard_label")
                            ),
                            "context_identity_sha256": record.context_identity_sha256,
                            "target_new": record.target_new,
                        },
                    )
                else:
                    store.put_context_success(
                        record.sample_id,
                        group.query_identity_sha256,
                        context_payload,
                        probes=context_probes,
                    )
        counts = store.counts()
        if export_directory is not None:
            store.export_complete(export_directory)
        return counts


def _run_query(
    group: CubQueryGroup,
    contract,
    certificate: CcuRuntimeAuditCertificate,
    client: LlamaServerClient,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    item = PromptInput(
        dataset="cub_druid",
        sample_id=group.query_identity_sha256,
        claim=group.claim,
        evidence=None,
        metadata={"claimant": group.claimant},
    )
    rendered = render_prompt(contract, item, control="query_only")
    prompt_text = client.apply_template(rendered.messages, enable_thinking=False)
    completion = client.complete_choice(
        prompt_text,
        choices=tuple(rendered.choice_to_label),
        choice_token_ids=certificate.choice_token_ids,
        seed=0,
    )
    hard = score_choice_argmax(
        rendered.choice_to_label, completion.choice_logprobs
    )
    needed_labels = set(group.required_labels)
    if any(record.context_type == "irrelevant" for record in group.records):
        needed_labels.add(hard.label)
    label_to_choice = {
        label: choice for choice, label in rendered.choice_to_label.items()
    }
    probabilities: dict[str, Any] = {}
    probe_records: list[dict[str, Any]] = []
    for label in CUB_OUTPUT_LABELS:
        if label not in needed_labels:
            continue
        choice = label_to_choice[label]
        stable = recover_stable_token_probability(
            client,
            prompt_text,
            token_id=certificate.choice_token_ids[choice],
            seed=0,
        )
        probabilities[label] = _stable_summary(choice, stable)
        probe_records.extend(_probe_records(label, choice, stable))
    payload = {
        "audit_certificate_sha256": certificate.certificate_sha256,
        "claim": group.claim,
        "claimant": group.claimant,
        "prompt_sha256": rendered.prompt_sha256,
        "hard_choice": hard.choice,
        "hard_label": hard.label,
        "choice_logprobs": dict(sorted(completion.choice_logprobs.items())),
        "target_probabilities": probabilities,
    }
    return payload, tuple(probe_records)


def _run_context(
    record: CubRecord,
    group: CubQueryGroup,
    query_payload: Mapping[str, Any],
    contract,
    certificate: CcuRuntimeAuditCertificate,
    client: LlamaServerClient,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    item = PromptInput(
        dataset="cub_druid",
        sample_id=record.sample_id,
        claim=record.claim,
        evidence=record.evidence,
        metadata={"claimant": record.claimant},
    )
    rendered = render_prompt(contract, item, control="full")
    prompt_text = client.apply_template(rendered.messages, enable_thinking=False)
    completion = client.complete_choice(
        prompt_text,
        choices=tuple(rendered.choice_to_label),
        choice_token_ids=certificate.choice_token_ids,
        seed=0,
    )
    hard = score_choice_argmax(
        rendered.choice_to_label, completion.choice_logprobs
    )
    target_label = (
        str(query_payload["hard_label"])
        if record.context_type == "irrelevant"
        else record.target_new
    )
    if target_label not in CUB_OUTPUT_LABELS:
        raise ValueError("CUB context target label is invalid")
    label_to_choice = {
        label: choice for choice, label in rendered.choice_to_label.items()
    }
    target_choice = label_to_choice[target_label]
    stable = recover_stable_token_probability(
        client,
        prompt_text,
        token_id=certificate.choice_token_ids[target_choice],
        seed=0,
    )
    query_probabilities = query_payload.get("target_probabilities")
    if not isinstance(query_probabilities, Mapping):
        raise ValueError("query cache is missing target probabilities")
    query_target = query_probabilities.get(target_label)
    if not isinstance(query_target, Mapping):
        raise ValueError("query cache is missing the context target")
    p0 = float(query_target["probability"])
    p1 = stable.probability
    bcu = cub_binary_context_utilisation(
        context_type=record.context_type,
        prediction_without_context=str(query_payload["hard_label"]),
        prediction_with_context=hard.label,
        target_new=record.target_new,
    )
    ccu = cub_continuous_context_utilisation(
        probability_without_context=p0,
        probability_with_context=p1,
    )
    payload = {
        "analysis_context_type": record.analysis_context_type,
        "audit_certificate_sha256": certificate.certificate_sha256,
        "bcu": bcu,
        "ccu": ccu,
        "ccu_target_choice": target_choice,
        "ccu_target_label": target_label,
        "choice_logprobs": dict(sorted(completion.choice_logprobs.items())),
        "context_identity_sha256": record.context_identity_sha256,
        "context_prediction_choice": hard.choice,
        "context_prediction_label": hard.label,
        "context_type": record.context_type,
        "p_context": p1,
        "p_query": p0,
        "prompt_sha256": rendered.prompt_sha256,
        "query_prediction_label": query_payload["hard_label"],
        "target_new": record.target_new,
        "target_true": record.target_true,
    }
    return payload, _probe_records(target_label, target_choice, stable)


def _stable_summary(
    choice: str, stable: StableTokenProbability
) -> dict[str, Any]:
    return {
        "choice": choice,
        "probability": stable.probability,
        "selected_bias": stable.selected_bias,
        "token_id": stable.token_id,
        "recovered_logprob_range": list(stable.recovered_logprob_range),
    }


def _probe_records(
    label: str, choice: str, stable: StableTokenProbability
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "bias": probe.bias,
            "biased_probability": probe.biased_probability,
            "choice": choice,
            "label": label,
            "token_id": probe.token_id,
            "unbiased_probability": probe.unbiased_probability,
            "validation_ordinal": ordinal,
        }
        for ordinal, probe in enumerate(stable.probes)
    )


def _verify_runtime(
    certificate: CcuRuntimeAuditCertificate,
    contract_path: Path,
    client: LlamaServerClient,
) -> None:
    actual_contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if certificate.protocol_version != CUB_RUNTIME_AUDIT_PROTOCOL_VERSION:
        raise CcuRuntimeAuditError("CUB runtime certificate protocol is not current")
    if certificate.prompt_contract_sha256 != actual_contract_hash:
        raise CcuRuntimeAuditError("CUB prompt contract hash mismatch")
    if certificate.choice_scoring_method != CHOICE_SCORING_METHOD:
        raise CcuRuntimeAuditError("CUB choice scoring method mismatch")
    if certificate.choice_logit_bias != CHOICE_LOGIT_BIAS:
        raise CcuRuntimeAuditError("CUB choice logit bias mismatch")
    if certificate.hard_label_method != HARD_LABEL_METHOD:
        raise CcuRuntimeAuditError("CUB hard-label method mismatch")
    if certificate.token_probability_method != TOKEN_PROBABILITY_METHOD:
        raise CcuRuntimeAuditError("CUB token probability method mismatch")
    if tuple(certificate.bias_candidates) != BIAS_CANDIDATES:
        raise CcuRuntimeAuditError("CUB probability bias grid mismatch")
    if (
        certificate.usable_probability_min != USABLE_PROBABILITY_MIN
        or certificate.usable_probability_max != USABLE_PROBABILITY_MAX
        or certificate.log_probability_tolerance != LOG_PROBABILITY_TOLERANCE
    ):
        raise CcuRuntimeAuditError("CUB probability safety policy mismatch")
    props = client.props()
    if props.get("model_path") != certificate.server_model_path:
        raise CcuRuntimeAuditError("CUB server model path mismatch")
    if props.get("build_info") != certificate.server_build_info:
        raise CcuRuntimeAuditError("CUB server build mismatch")


def _required_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _failure_payload(
    certificate: CcuRuntimeAuditCertificate, stage: str, error: Exception
) -> dict[str, str]:
    return {
        "audit_certificate_sha256": certificate.certificate_sha256,
        "error_type": type(error).__name__,
        "stage": stage,
    }


def load_frozen_cub_records(
    project_root: str | Path, *, sample_ids_path: str | Path | None = None
) -> tuple[tuple[CubRecord, ...], str]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/manifests/cub_test.jsonl"
    source_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    freeze = json.loads((root / "configs/freeze_v1.json").read_text(encoding="utf-8"))
    expected_hash = freeze.get("cub_druid", {}).get("test_records_sha256")
    if source_hash != expected_hash:
        raise ValueError("frozen CUB manifest SHA-256 mismatch")
    records: list[CubRecord] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"CUB manifest line {line_number} is not an object")
        records.append(
            CubRecord(
                sample_id=value["sample_id"],
                claim=value["claim"],
                claimant=value["claimant"],
                evidence=value["evidence"],
                relevant=value["relevant"],
                context_type=value["context_type"],
                target_true=value["target_true"],
                target_new=value["target_new"],
                context_identity_sha256=value["context_identity_sha256"],
            )
        )
    if sample_ids_path is None:
        return tuple(records), source_hash
    sample_ids = [
        line.strip()
        for line in Path(sample_ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("CUB sample-ID selection must be nonempty and unique")
    wanted = set(sample_ids)
    selected = tuple(record for record in records if record.sample_id in wanted)
    found = {record.sample_id for record in selected}
    if found != wanted:
        raise ValueError("CUB sample-ID selection contains unknown IDs")
    selection_payload = json.dumps(
        {"sample_ids": sorted(wanted), "source_manifest_sha256": source_hash},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return selected, hashlib.sha256(selection_payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--server", required=True)
    parser.add_argument("--audit-certificate", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--export-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    records, manifest_hash = load_frozen_cub_records(
        args.project_root, sample_ids_path=args.sample_ids
    )
    summary = run_cub_records(
        project_root=args.project_root,
        records=records,
        manifest_sha256=manifest_hash,
        client=LlamaServerClient(args.server),
        audit_certificate_path=args.audit_certificate,
        database_path=args.database,
        export_directory=args.export_directory,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
