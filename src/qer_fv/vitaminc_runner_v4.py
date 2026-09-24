"""Resumable VitaminC formal runner with independent hard and direct-logit stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .llama_client import (
    SELECTED_TOKEN_LOGPROB_METHOD,
    LlamaServerClient,
    LlamaServerError,
    SelectedTokenScores,
)
from .prompts import load_prompt_contract, render_prompt
from .scoring import HARD_LABEL_METHOD, score_choice_argmax
from .vitaminc import id_manifest_sha256
from .vitaminc_inputs import (
    VitaminCInferenceUnit,
    build_vitaminc_units,
    load_vitaminc_split,
)
from .vitaminc_runtime_audit_v4 import (
    VitaminCRuntimeAuditV4Error,
    load_vitaminc_runtime_audit_v4,
)
from .vitaminc_store_v4 import VitaminCRunIdentityV4, VitaminCRunStoreV4


VITAMINC_RUN_V4_PROTOCOL_VERSION = "vitaminc-formal-run-v4-20260716"
_INFERENCE_ERRORS = (
    LlamaServerError,
    VitaminCRuntimeAuditV4Error,
    ValueError,
    KeyError,
    TypeError,
    OverflowError,
)


def run_vitaminc_units_v4(
    *,
    project_root: str | Path,
    units: Sequence[VitaminCInferenceUnit],
    split: str,
    split_sha256: str,
    client: LlamaServerClient,
    audit_certificate_path: str | Path,
    database_path: str | Path,
    export_directory: str | Path,
    mapping_variant: str = "original",
    protocol_version: str | None = None,
) -> dict[str, int]:
    """Run already-frozen units without retrying an immutable completed stage."""
    root = Path(project_root).resolve()
    if not units:
        raise ValueError("VitaminC formal run requires at least one unit")
    if len({unit.owner_key for unit in units}) != len(units):
        raise ValueError("VitaminC formal units must have unique owner keys")
    certificate = load_vitaminc_runtime_audit_v4(audit_certificate_path)
    contract_path = root / "configs/prompt_contract_v2.json"
    prompt_sha256 = _file_sha256(contract_path)
    if certificate.prompt_contract_sha256 != prompt_sha256:
        raise ValueError("runtime certificate prompt contract SHA-256 mismatch")
    contract = load_prompt_contract(contract_path)
    if contract.version != certificate.prompt_contract_version:
        raise ValueError("runtime certificate prompt contract version mismatch")
    if certificate.direct_logit_method != SELECTED_TOKEN_LOGPROB_METHOD:
        raise ValueError("runtime certificate direct-logit method mismatch")
    if tuple(certificate.choice_token_ids) != ("A", "B", "C"):
        raise ValueError("runtime certificate choice order mismatch")

    controls = Counter(unit.control for unit in units)
    if set(controls) - {"full", "no_evidence", "knowledge_only"}:
        raise ValueError("VitaminC formal units contain an invalid control")
    if mapping_variant != "original" and protocol_version is None:
        raise ValueError(
            "non-original VitaminC mapping requires an explicit protocol"
        )
    run_protocol = (
        VITAMINC_RUN_V4_PROTOCOL_VERSION
        if protocol_version is None
        else _required_protocol(protocol_version)
    )
    identity = VitaminCRunIdentityV4(
        protocol_version=run_protocol,
        split=split,
        split_sha256=split_sha256,
        audit_certificate_sha256=certificate.certificate_sha256,
        prompt_contract_sha256=prompt_sha256,
        model_key=certificate.model_key,
        quantization=certificate.quantization,
        expected_full=controls["full"],
        expected_no_evidence=controls["no_evidence"],
        expected_knowledge_only=controls["knowledge_only"],
    )
    with VitaminCRunStoreV4(database_path, identity) as store:
        for unit in units:
            pending = store.pending_stages(unit.owner_key)
            if not pending:
                continue
            metadata = _unit_metadata(
                unit,
                mapping_variant=(
                    mapping_variant if protocol_version is not None else None
                ),
            )
            rendered = render_prompt(
                contract,
                unit.prompt_input,
                control=unit.control,
                mapping_variant=mapping_variant,
            )
            try:
                prompt_text = client.apply_template(
                    rendered.messages, enable_thinking=False
                )
            except _INFERENCE_ERRORS as error:
                failure = _failure_payload(error, rendered.prompt_sha256, "template")
                if "hard" in pending:
                    store.put_hard_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="template_error",
                        payload=failure,
                    )
                if "probability" in pending:
                    store.put_probability_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="template_error",
                        payload=failure,
                    )
                continue

            choices = tuple(rendered.choice_to_label)
            if "hard" in pending:
                try:
                    completion = client.complete_choice(
                        prompt_text,
                        choices=choices,
                        choice_token_ids=certificate.choice_token_ids,
                        seed=0,
                    )
                    scored = score_choice_argmax(
                        rendered.choice_to_label, completion.choice_logprobs
                    )
                    store.put_hard_success(
                        unit.owner_key,
                        metadata=metadata,
                        payload={
                            "choice_logprobs": dict(completion.choice_logprobs),
                            "hard_label_method": HARD_LABEL_METHOD,
                            "prompt_sha256": rendered.prompt_sha256,
                            "raw_choice": completion.content,
                            "raw_response": dict(completion.raw_response),
                            "scored_choice": scored.choice,
                            "scored_label": scored.label,
                        },
                    )
                except _INFERENCE_ERRORS as error:
                    store.put_hard_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="hard_inference_error",
                        payload=_failure_payload(
                            error, rendered.prompt_sha256, "hard"
                        ),
                    )

            if "probability" in pending:
                try:
                    scores = client.score_selected_tokens(
                        prompt_text,
                        token_ids=tuple(certificate.choice_token_ids.values()),
                        seed=0,
                    )
                    choice_logprobs = _validate_selected_scores(
                        scores, certificate.choice_token_ids
                    )
                    choice_probabilities = {
                        choice: math.exp(value)
                        for choice, value in choice_logprobs.items()
                    }
                    scored = score_choice_argmax(
                        rendered.choice_to_label, choice_logprobs
                    )
                    store.put_probability_success(
                        unit.owner_key,
                        metadata=metadata,
                        payload={
                            "choice_logprobs": choice_logprobs,
                            "choice_probabilities": choice_probabilities,
                            "direct_logit_method": scores.method,
                            "label_probability_mass": sum(
                                choice_probabilities.values()
                            ),
                            "prompt_sha256": rendered.prompt_sha256,
                            "raw_response": dict(scores.raw_response),
                            "scored_choice": scored.choice,
                            "scored_label": scored.label,
                            "token_ids": dict(certificate.choice_token_ids),
                        },
                    )
                except _INFERENCE_ERRORS as error:
                    store.put_probability_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="direct_logprob_error",
                        payload=_failure_payload(
                            error, rendered.prompt_sha256, "probability"
                        ),
                    )
        summary = store.counts()
        store.export_complete(export_directory)
    return summary


def _validate_selected_scores(
    scores: SelectedTokenScores, choice_token_ids: Mapping[str, int]
) -> dict[str, float]:
    expected_ids = tuple(choice_token_ids.values())
    if scores.method != SELECTED_TOKEN_LOGPROB_METHOD:
        raise ValueError("selected-token method mismatch")
    if scores.token_ids != expected_ids or tuple(scores.logprobs) != expected_ids:
        raise ValueError("selected-token order mismatch")
    result: dict[str, float] = {}
    for choice, token_id in choice_token_ids.items():
        value = scores.logprobs[token_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value > 0.0
        ):
            raise ValueError("selected-token log-probability is invalid")
        result[choice] = float(value)
    return result


def _unit_metadata(
    unit: VitaminCInferenceUnit,
    *,
    mapping_variant: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "case_id": unit.case_id,
        "cell_index": unit.cell_index,
        "claim_index": unit.claim_index,
        "control": unit.control,
        "evidence_index": unit.evidence_index,
        "negative_label": unit.prompt_input.metadata["negative_label"],
        "page": unit.page,
        "sample_id": unit.sample_id,
    }
    if unit.control == "full":
        metadata["gold_label"] = unit.prompt_input.metadata["gold_label"]
    if mapping_variant is not None:
        metadata["mapping_variant"] = mapping_variant
    return metadata


def _required_protocol(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("VitaminC run protocol must be a nonempty string")
    return value


def _failure_payload(
    error: BaseException, prompt_sha256: str, stage: str
) -> dict[str, str]:
    return {
        "error_message": str(error),
        "error_type": type(error).__name__,
        "prompt_sha256": prompt_sha256,
        "stage": stage,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--server", required=True)
    parser.add_argument("--audit-certificate", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("pilot", "discovery", "dose_subset", "confirmatory"),
        required=True,
    )
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    quartets = load_vitaminc_split(root, split=args.split)
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            parser.error("--limit-cases must be positive")
        quartets = quartets[: args.limit_cases]
    summary = run_vitaminc_units_v4(
        project_root=root,
        units=build_vitaminc_units(quartets),
        split=args.split,
        split_sha256=id_manifest_sha256(item.case_id for item in quartets),
        client=LlamaServerClient(args.server),
        audit_certificate_path=args.audit_certificate,
        database_path=args.database,
        export_directory=args.export_directory,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
