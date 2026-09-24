"""Audited CUB v4 runner with independent hard and direct-logit stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cub_druid import (
    CUB_OUTPUT_LABELS,
    CubRecord,
    cub_binary_context_utilisation,
    cub_continuous_context_utilisation,
)
from .cub_queries import CubQueryGroup, build_cub_query_groups
from .cub_store_v4 import CubRunIdentityV4, CubRunStoreV4
from .direct_logit_audit import (
    DIRECT_LOGIT_AUDIT_PROTOCOL_VERSION,
    DIRECT_LOGIT_DUPLICATE_TOLERANCE,
    DirectLogitAuditCertificate,
    DirectLogitAuditError,
    load_direct_logit_audit_certificate,
)
from .llama_client import (
    CHOICE_LOGIT_BIAS,
    CHOICE_SCORING_METHOD,
    SELECTED_TOKEN_LOGPROB_METHOD,
    LlamaServerClient,
    LlamaServerError,
    SelectedTokenScores,
)
from .prompts import PromptInput, load_prompt_contract, render_prompt
from .scoring import HARD_LABEL_METHOD, score_choice_argmax


CUB_RUN_V4_PROTOCOL_VERSION = "cub-paired-run-v4-20260715"
_INFERENCE_ERRORS = (
    LlamaServerError,
    DirectLogitAuditError,
    ValueError,
    KeyError,
    TypeError,
    OverflowError,
)


def run_cub_records_v4(
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
    certificate = load_direct_logit_audit_certificate(audit_certificate_path)
    contract_path = root / "configs/prompt_contract_v2.json"
    contract = load_prompt_contract(contract_path)
    _verify_runtime(certificate, contract_path, client)
    groups = build_cub_query_groups(records, prompt_contract_version=contract.version)
    identity = CubRunIdentityV4(
        protocol_version=CUB_RUN_V4_PROTOCOL_VERSION,
        audit_certificate_sha256=certificate.certificate_sha256,
        manifest_sha256=_required_sha256(manifest_sha256, "manifest_sha256"),
        prompt_contract_sha256=certificate.prompt_contract_sha256,
        model_key=certificate.model_key,
        quantization=certificate.quantization,
        expected_queries=len(groups),
        expected_contexts=len(records),
    )
    with CubRunStoreV4(database_path, identity) as store:
        for group in groups:
            query_key = group.query_identity_sha256
            if store.stage_status("query", query_key)[0] == "pending":
                try:
                    payload = _run_query_hard(group, contract, certificate, client)
                except _INFERENCE_ERRORS as error:
                    store.put_hard_failure(
                        "query",
                        query_key,
                        error_code="hard_inference_error",
                        payload=_failure_payload(certificate, "query_hard", error),
                    )
                else:
                    store.put_hard_success("query", query_key, payload=payload)

            if store.stage_status("query", query_key)[1] == "pending":
                try:
                    payload = _run_query_probability(
                        group, contract, certificate, client
                    )
                except _INFERENCE_ERRORS as error:
                    store.put_probability_failure(
                        "query",
                        query_key,
                        error_code="direct_logprob_error",
                        payload=_failure_payload(
                            certificate, "query_probability", error
                        ),
                    )
                else:
                    store.put_probability_success("query", query_key, payload=payload)

            query_hard_status, query_probability_status = store.stage_status(
                "query", query_key
            )
            query_hard = (
                store.get_hard_payload("query", query_key)
                if query_hard_status == "ok"
                else None
            )
            query_probability = (
                store.get_probability_payload("query", query_key)
                if query_probability_status == "ok"
                else None
            )

            for record in group.records:
                if store.stage_status("context", record.sample_id)[0] == "pending":
                    if query_hard is None:
                        store.put_hard_failure(
                            "context",
                            record.sample_id,
                            query_identity=query_key,
                            error_code="query_hard_dependency_error",
                            payload=_context_failure_payload(
                                certificate,
                                record,
                                "context_hard_query_dependency",
                                None,
                            ),
                        )
                    else:
                        try:
                            payload = _run_context_hard(
                                record,
                                query_hard,
                                contract,
                                certificate,
                                client,
                            )
                        except _INFERENCE_ERRORS as error:
                            store.put_hard_failure(
                                "context",
                                record.sample_id,
                                query_identity=query_key,
                                error_code="hard_inference_error",
                                payload=_context_failure_payload(
                                    certificate, record, "context_hard", error
                                ),
                            )
                        else:
                            store.put_hard_success(
                                "context",
                                record.sample_id,
                                query_identity=query_key,
                                payload=payload,
                            )

                if store.stage_status("context", record.sample_id)[1] == "pending":
                    target_label = _context_target_label(record, query_hard)
                    if query_probability is None:
                        store.put_probability_failure(
                            "context",
                            record.sample_id,
                            query_identity=query_key,
                            error_code="query_probability_dependency_error",
                            payload=_context_failure_payload(
                                certificate,
                                record,
                                "context_probability_query_dependency",
                                None,
                                target_label=target_label,
                            ),
                        )
                    elif target_label is None:
                        store.put_probability_failure(
                            "context",
                            record.sample_id,
                            query_identity=query_key,
                            error_code="query_hard_dependency_error",
                            payload=_context_failure_payload(
                                certificate,
                                record,
                                "context_probability_target_dependency",
                                None,
                            ),
                        )
                    else:
                        try:
                            payload = _run_context_probability(
                                record,
                                target_label,
                                query_probability,
                                contract,
                                certificate,
                                client,
                            )
                        except _INFERENCE_ERRORS as error:
                            store.put_probability_failure(
                                "context",
                                record.sample_id,
                                query_identity=query_key,
                                error_code="direct_logprob_error",
                                payload=_context_failure_payload(
                                    certificate,
                                    record,
                                    "context_probability",
                                    error,
                                    target_label=target_label,
                                ),
                            )
                        else:
                            store.put_probability_success(
                                "context",
                                record.sample_id,
                                query_identity=query_key,
                                payload=payload,
                            )
        counts = store.counts()
        if export_directory is not None:
            store.export_complete(export_directory)
        return counts


def _run_query_hard(
    group: CubQueryGroup,
    contract,
    certificate: DirectLogitAuditCertificate,
    client: LlamaServerClient,
) -> dict[str, Any]:
    rendered, prompt_text = _render_query(group, contract, client)
    completion = client.complete_choice(
        prompt_text,
        choices=tuple(rendered.choice_to_label),
        choice_token_ids=certificate.choice_token_ids,
        seed=0,
    )
    hard = score_choice_argmax(rendered.choice_to_label, completion.choice_logprobs)
    return {
        "audit_certificate_sha256": certificate.certificate_sha256,
        "claim": group.claim,
        "claimant": group.claimant,
        "prompt_sha256": rendered.prompt_sha256,
        "hard_choice": hard.choice,
        "hard_label": hard.label,
        "choice_logprobs": dict(sorted(completion.choice_logprobs.items())),
    }


def _run_query_probability(
    group: CubQueryGroup,
    contract,
    certificate: DirectLogitAuditCertificate,
    client: LlamaServerClient,
) -> dict[str, Any]:
    rendered, prompt_text = _render_query(group, contract, client)
    direct = _score_direct(rendered.choice_to_label, prompt_text, certificate, client)
    return {
        "audit_certificate_sha256": certificate.certificate_sha256,
        "claim": group.claim,
        "claimant": group.claimant,
        "prompt_sha256": rendered.prompt_sha256,
        **direct,
    }


def _run_context_hard(
    record: CubRecord,
    query_hard: Mapping[str, Any],
    contract,
    certificate: DirectLogitAuditCertificate,
    client: LlamaServerClient,
) -> dict[str, Any]:
    rendered, prompt_text = _render_context(record, contract, client)
    completion = client.complete_choice(
        prompt_text,
        choices=tuple(rendered.choice_to_label),
        choice_token_ids=certificate.choice_token_ids,
        seed=0,
    )
    hard = score_choice_argmax(rendered.choice_to_label, completion.choice_logprobs)
    bcu = cub_binary_context_utilisation(
        context_type=record.context_type,
        prediction_without_context=str(query_hard["hard_label"]),
        prediction_with_context=hard.label,
        target_new=record.target_new,
    )
    return {
        "analysis_context_type": record.analysis_context_type,
        "audit_certificate_sha256": certificate.certificate_sha256,
        "bcu": bcu,
        "choice_logprobs": dict(sorted(completion.choice_logprobs.items())),
        "context_identity_sha256": record.context_identity_sha256,
        "context_prediction_choice": hard.choice,
        "context_prediction_label": hard.label,
        "context_type": record.context_type,
        "prompt_sha256": rendered.prompt_sha256,
        "query_prediction_label": query_hard["hard_label"],
        "target_new": record.target_new,
        "target_true": record.target_true,
    }


def _run_context_probability(
    record: CubRecord,
    target_label: str,
    query_probability: Mapping[str, Any],
    contract,
    certificate: DirectLogitAuditCertificate,
    client: LlamaServerClient,
) -> dict[str, Any]:
    rendered, prompt_text = _render_context(record, contract, client)
    direct = _score_direct(rendered.choice_to_label, prompt_text, certificate, client)
    query_labels = query_probability.get("label_probabilities")
    context_labels = direct.get("label_probabilities")
    if not isinstance(query_labels, Mapping) or not isinstance(context_labels, Mapping):
        raise ValueError("direct-logit payload is missing label probabilities")
    query_target = query_labels.get(target_label)
    context_target = context_labels.get(target_label)
    if not isinstance(query_target, Mapping) or not isinstance(context_target, Mapping):
        raise ValueError("direct-logit payload is missing the context target")
    p0 = float(query_target["probability"])
    p1 = float(context_target["probability"])
    ccu = cub_continuous_context_utilisation(
        probability_without_context=p0,
        probability_with_context=p1,
    )
    label_to_choice = {
        label: choice for choice, label in rendered.choice_to_label.items()
    }
    return {
        "analysis_context_type": record.analysis_context_type,
        "audit_certificate_sha256": certificate.certificate_sha256,
        "ccu": ccu,
        "ccu_target_choice": label_to_choice[target_label],
        "ccu_target_label": target_label,
        "context_identity_sha256": record.context_identity_sha256,
        "context_type": record.context_type,
        "p_context": p1,
        "p_query": p0,
        "prompt_sha256": rendered.prompt_sha256,
        "target_new": record.target_new,
        "target_true": record.target_true,
        **direct,
    }


def _score_direct(
    choice_to_label: Mapping[str, str],
    prompt_text: str,
    certificate: DirectLogitAuditCertificate,
    client: LlamaServerClient,
) -> dict[str, Any]:
    expected_choices = tuple(choice_to_label)
    if tuple(certificate.choice_token_ids) != expected_choices:
        raise DirectLogitAuditError("certificate choice-token order mismatch")
    token_ids = tuple(certificate.choice_token_ids.values())
    scores = client.score_selected_tokens(prompt_text, token_ids=token_ids, seed=0)
    logprobs = _validated_direct_scores(scores, token_ids)
    choice_logprobs = {
        choice: logprobs[certificate.choice_token_ids[choice]]
        for choice in expected_choices
    }
    direct = score_choice_argmax(choice_to_label, choice_logprobs)
    label_probabilities: dict[str, dict[str, Any]] = {}
    for choice, label in choice_to_label.items():
        logprob = choice_logprobs[choice]
        label_probabilities[label] = {
            "choice": choice,
            "token_id": certificate.choice_token_ids[choice],
            "logprob": logprob,
            "probability": math.exp(logprob),
        }
    return {
        "direct_logit_method": SELECTED_TOKEN_LOGPROB_METHOD,
        "direct_scored_choice": direct.choice,
        "direct_scored_label": direct.label,
        "label_probabilities": label_probabilities,
    }


def _validated_direct_scores(
    scores: SelectedTokenScores, expected_ids: tuple[int, ...]
) -> dict[int, float]:
    if scores.method != SELECTED_TOKEN_LOGPROB_METHOD:
        raise DirectLogitAuditError("direct-logit method mismatch")
    if scores.token_ids != expected_ids or tuple(scores.logprobs) != expected_ids:
        raise DirectLogitAuditError("direct-logit token order mismatch")
    result: dict[int, float] = {}
    for token_id in expected_ids:
        value = scores.logprobs[token_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value > 0.0
        ):
            raise DirectLogitAuditError("direct-logit score is invalid")
        result[token_id] = float(value)
    return result


def _render_query(group: CubQueryGroup, contract, client: LlamaServerClient):
    rendered = render_prompt(
        contract,
        PromptInput(
            dataset="cub_druid",
            sample_id=group.query_identity_sha256,
            claim=group.claim,
            evidence=None,
            metadata={"claimant": group.claimant},
        ),
        control="query_only",
    )
    return rendered, client.apply_template(rendered.messages, enable_thinking=False)


def _render_context(record: CubRecord, contract, client: LlamaServerClient):
    rendered = render_prompt(
        contract,
        PromptInput(
            dataset="cub_druid",
            sample_id=record.sample_id,
            claim=record.claim,
            evidence=record.evidence,
            metadata={"claimant": record.claimant},
        ),
        control="full",
    )
    return rendered, client.apply_template(rendered.messages, enable_thinking=False)


def _context_target_label(
    record: CubRecord, query_hard: Mapping[str, Any] | None
) -> str | None:
    if record.context_type == "irrelevant":
        if query_hard is None:
            return None
        target = query_hard.get("hard_label")
    else:
        target = record.target_new
    if target not in CUB_OUTPUT_LABELS:
        raise ValueError("CUB context target label is invalid")
    return str(target)


def _verify_runtime(
    certificate: DirectLogitAuditCertificate,
    contract_path: Path,
    client: LlamaServerClient,
) -> None:
    if certificate.protocol_version != DIRECT_LOGIT_AUDIT_PROTOCOL_VERSION:
        raise DirectLogitAuditError("direct-logit audit protocol is not current")
    if certificate.prompt_contract_sha256 != _file_sha256(contract_path):
        raise DirectLogitAuditError("CUB prompt contract hash mismatch")
    if certificate.choice_scoring_method != CHOICE_SCORING_METHOD:
        raise DirectLogitAuditError("CUB hard-choice scoring method mismatch")
    if certificate.choice_logit_bias != CHOICE_LOGIT_BIAS:
        raise DirectLogitAuditError("CUB hard-choice logit bias mismatch")
    if certificate.hard_label_method != HARD_LABEL_METHOD:
        raise DirectLogitAuditError("CUB hard-label method mismatch")
    if certificate.direct_logit_method != SELECTED_TOKEN_LOGPROB_METHOD:
        raise DirectLogitAuditError("CUB direct-logit method mismatch")
    if certificate.duplicate_tolerance != DIRECT_LOGIT_DUPLICATE_TOLERANCE:
        raise DirectLogitAuditError("CUB direct-logit duplicate policy mismatch")
    if tuple(certificate.choice_token_ids) != ("A", "B", "C"):
        raise DirectLogitAuditError("CUB choice-token order mismatch")
    if len(set(certificate.choice_token_ids.values())) != 3:
        raise DirectLogitAuditError("CUB choice-token IDs are not unique")
    props = client.props()
    if props.get("model_path") != certificate.server_model_path:
        raise DirectLogitAuditError("CUB server model path mismatch")
    if props.get("build_info") != certificate.server_build_info:
        raise DirectLogitAuditError("CUB server build mismatch")
    artifacts = (
        (Path(certificate.server_model_path), certificate.model_sha256, "model"),
        (Path(certificate.server_patch_path), certificate.server_patch_sha256, "patch"),
        (Path(certificate.server_binary_path), certificate.server_binary_sha256, "launcher"),
        (
            Path(certificate.server_implementation_path),
            certificate.server_implementation_sha256,
            "implementation library",
        ),
    )
    for path, expected, label in artifacts:
        if not path.is_file() or _file_sha256(path) != expected:
            raise DirectLogitAuditError(f"CUB {label} hash mismatch")


def _failure_payload(
    certificate: DirectLogitAuditCertificate, stage: str, error: Exception
) -> dict[str, str]:
    return {
        "audit_certificate_sha256": certificate.certificate_sha256,
        "error_type": type(error).__name__,
        "stage": stage,
    }


def _context_failure_payload(
    certificate: DirectLogitAuditCertificate,
    record: CubRecord,
    stage: str,
    error: Exception | None,
    *,
    target_label: str | None = None,
) -> dict[str, Any]:
    return {
        "analysis_context_type": record.analysis_context_type,
        "audit_certificate_sha256": certificate.certificate_sha256,
        "ccu_target_label": target_label,
        "context_identity_sha256": record.context_identity_sha256,
        "error_type": None if error is None else type(error).__name__,
        "stage": stage,
        "target_new": record.target_new,
    }


def _required_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_cub_records_v4(
    project_root: str | Path, *, sample_ids_path: str | Path | None = None
) -> tuple[tuple[CubRecord, ...], str]:
    root = Path(project_root).resolve()
    manifest_path = root / "data/manifests/cub_test.jsonl"
    source_hash = _file_sha256(manifest_path)
    freeze = json.loads((root / "configs/freeze_v1.json").read_text(encoding="utf-8"))
    expected_hash = freeze.get("cub_druid", {}).get("test_records_sha256")
    if source_hash != expected_hash:
        raise ValueError("frozen CUB manifest SHA-256 mismatch")
    records = []
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
    if {record.sample_id for record in selected} != wanted:
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
    records, manifest_hash = load_frozen_cub_records_v4(
        args.project_root, sample_ids_path=args.sample_ids
    )
    summary = run_cub_records_v4(
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
