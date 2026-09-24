"""Blinded Fresh TabFact smoke runner using audited selected-token logits."""

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
)
from .prompts import load_prompt_contract, render_prompt
from .scoring import HARD_LABEL_METHOD, score_choice_argmax
from .tabfact_inputs import (
    TabFactInferenceUnit,
    TabFactManifestRow,
    build_tabfact_prompt_inputs,
    load_tabfact_manifest,
    tabfact_evidence_manifest_sha256,
)
from .tabfact_store import (
    TabFactFormalRunIdentity,
    TabFactRunIdentity,
    TabFactRunStore,
)
from .vitaminc import id_manifest_sha256
from .vitaminc_runtime_audit_v4 import load_vitaminc_runtime_audit_v4


TABFACT_SMOKE_PROTOCOL_VERSION = "tabfact-smoke-run-v1-20260717"
TABFACT_FORMAL_PROTOCOL_VERSION = "tabfact-formal-run-v1-20260718"
TABFACT_FORMAL_SPLIT_SHA256 = (
    "4318fc178818cdf14f86cdf94d5271a80ba5460ac8ae8c47c3dbb803b2b556a6"
)
TABFACT_FORMAL_MANIFEST_SHA256 = (
    "1123352554c523c89c598b343e2a07f270e4ecab9e5081b1552aa020062cebab"
)
TABFACT_FORMAL_EVIDENCE_SHA256 = (
    "cd78d8b74db92c651374ac75e99302a447b13c326e9f0649e4e6793f8e959610"
)
_ERRORS = (LlamaServerError, ValueError, KeyError, TypeError, OverflowError)


def run_tabfact_smoke(
    *,
    project_root: str | Path,
    units: Sequence[TabFactInferenceUnit],
    split_sha256: str,
    smoke_sha256: str,
    audit_certificate_sha256: str,
    choice_token_ids: Mapping[str, int],
    model_key: str,
    quantization: str,
    client: LlamaServerClient,
    database_path: str | Path,
    export_directory: str | Path,
) -> dict[str, Any]:
    """Run binary full-evidence units and expose structural status only."""
    root = Path(project_root).resolve()
    contract_path = root / "configs/prompt_contract_v2.json"
    identity = TabFactRunIdentity(
        protocol_version=TABFACT_SMOKE_PROTOCOL_VERSION,
        split_sha256=split_sha256,
        smoke_sha256=smoke_sha256,
        audit_certificate_sha256=audit_certificate_sha256,
        prompt_contract_sha256=_file_sha256(contract_path),
        model_key=model_key,
        quantization=quantization,
        expected_units=len(units),
    )
    return _run_tabfact_units(
        project_root=root,
        units=units,
        identity=identity,
        choice_token_ids=choice_token_ids,
        client=client,
        database_path=database_path,
        export_directory=export_directory,
    )


def run_tabfact_formal(
    *,
    project_root: str | Path,
    units: Sequence[TabFactInferenceUnit],
    split_sha256: str,
    manifest_sha256: str,
    evidence_sha256: str,
    audit_certificate_sha256: str,
    choice_token_ids: Mapping[str, int],
    model_key: str,
    quantization: str,
    client: LlamaServerClient,
    database_path: str | Path,
    export_directory: str | Path,
) -> dict[str, Any]:
    """Run formal binary units under a distinct immutable identity."""
    root = Path(project_root).resolve()
    contract_path = root / "configs/prompt_contract_v2.json"
    identity = TabFactFormalRunIdentity(
        protocol_version=TABFACT_FORMAL_PROTOCOL_VERSION,
        split_sha256=split_sha256,
        manifest_sha256=manifest_sha256,
        evidence_sha256=evidence_sha256,
        audit_certificate_sha256=audit_certificate_sha256,
        prompt_contract_sha256=_file_sha256(contract_path),
        model_key=model_key,
        quantization=quantization,
        expected_units=len(units),
    )
    return _run_tabfact_units(
        project_root=root,
        units=units,
        identity=identity,
        choice_token_ids=choice_token_ids,
        client=client,
        database_path=database_path,
        export_directory=export_directory,
    )


def _run_tabfact_units(
    *,
    project_root: Path,
    units: Sequence[TabFactInferenceUnit],
    identity: TabFactRunIdentity | TabFactFormalRunIdentity,
    choice_token_ids: Mapping[str, int],
    client: LlamaServerClient,
    database_path: str | Path,
    export_directory: str | Path,
) -> dict[str, Any]:
    if not units:
        raise ValueError("TabFact run requires at least one unit")
    if len({unit.owner_key for unit in units}) != len(units):
        raise ValueError("TabFact owner keys must be unique")
    if set(choice_token_ids) != {"A", "B"}:
        raise ValueError("TabFact choice token IDs must contain exactly A and B")
    if len(set(choice_token_ids.values())) != 2:
        raise ValueError("TabFact choice token IDs must be unique")
    contract = load_prompt_contract(
        project_root / "configs/prompt_contract_v2.json"
    )
    with TabFactRunStore(database_path, identity) as store:
        for unit in units:
            pending = store.pending_stages(unit.owner_key)
            if not pending:
                continue
            metadata = {
                "control": "full",
                "sample_id": unit.sample_id,
                "table_id": unit.table_id,
                "claim_index": unit.claim_index,
                "gold_label": unit.gold_label,
                "table_rows": unit.table_rows,
                "table_truncated": unit.table_truncated,
            }
            rendered = render_prompt(contract, unit.prompt_input, control="full")
            try:
                prompt = client.apply_template(
                    rendered.messages, enable_thinking=False
                )
            except _ERRORS as error:
                failure = _failure(error, rendered.prompt_sha256, "template")
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

            choices = ("A", "B")
            if "hard" in pending:
                try:
                    completion = client.complete_choice(
                        prompt,
                        choices=choices,
                        choice_token_ids=choice_token_ids,
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
                except _ERRORS as error:
                    store.put_hard_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="hard_inference_error",
                        payload=_failure(error, rendered.prompt_sha256, "hard"),
                    )

            if "probability" in pending:
                try:
                    scores = client.score_selected_tokens(
                        prompt,
                        token_ids=tuple(choice_token_ids.values()),
                        seed=0,
                    )
                    if scores.method != SELECTED_TOKEN_LOGPROB_METHOD:
                        raise ValueError("selected-token method mismatch")
                    logprobs = {
                        choice: float(scores.logprobs[token_id])
                        for choice, token_id in choice_token_ids.items()
                    }
                    if any(
                        not math.isfinite(value) or value > 0.0
                        for value in logprobs.values()
                    ):
                        raise ValueError("selected-token logprob is invalid")
                    scored = score_choice_argmax(
                        rendered.choice_to_label, logprobs
                    )
                    probabilities = {
                        choice: math.exp(value)
                        for choice, value in logprobs.items()
                    }
                    store.put_probability_success(
                        unit.owner_key,
                        metadata=metadata,
                        payload={
                            "choice_logprobs": logprobs,
                            "choice_probabilities": probabilities,
                            "direct_logit_method": scores.method,
                            "label_probability_mass": sum(probabilities.values()),
                            "prompt_sha256": rendered.prompt_sha256,
                            "raw_response": dict(scores.raw_response),
                            "scored_choice": scored.choice,
                            "scored_label": scored.label,
                            "token_ids": dict(choice_token_ids),
                        },
                    )
                except _ERRORS as error:
                    store.put_probability_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="direct_logprob_error",
                        payload=_failure(
                            error, rendered.prompt_sha256, "probability"
                        ),
                    )
        store.export_complete(export_directory)
        return store.structural_report()


def _failure(
    error: BaseException, prompt_sha256: str, stage: str
) -> dict[str, str]:
    return {
        "error_message": str(error),
        "error_type": type(error).__name__,
        "prompt_sha256": prompt_sha256,
        "stage": stage,
    }


def validate_frozen_tabfact_formal_manifest(
    rows: Sequence[TabFactManifestRow], manifest_path: str | Path
) -> dict[str, str]:
    """Fail closed unless rows are the exact frozen Fresh TabFact population."""
    if len(rows) != 2_000:
        raise ValueError("formal TabFact manifest must contain exactly 2,000 rows")
    table_counts = Counter(row.table_id for row in rows)
    if len(table_counts) != 1_434:
        raise ValueError("formal TabFact manifest must contain exactly 1,434 tables")
    if max(table_counts.values()) != 2:
        raise ValueError("formal TabFact manifest max claims per table must be 2")
    labels = Counter(row.label for row in rows)
    if labels != Counter({1: 1_003, 0: 997}):
        raise ValueError("formal TabFact label counts do not match the freeze")
    split_sha256 = id_manifest_sha256(row.sample_id for row in rows)
    if split_sha256 != TABFACT_FORMAL_SPLIT_SHA256:
        raise ValueError("formal TabFact split SHA-256 mismatch")
    manifest_sha256 = _file_sha256(Path(manifest_path))
    if manifest_sha256 != TABFACT_FORMAL_MANIFEST_SHA256:
        raise ValueError("formal TabFact manifest SHA-256 mismatch")
    evidence_sha256 = tabfact_evidence_manifest_sha256(rows)
    if evidence_sha256 != TABFACT_FORMAL_EVIDENCE_SHA256:
        raise ValueError("formal TabFact evidence SHA-256 mismatch")
    return {
        "split_sha256": split_sha256,
        "manifest_sha256": manifest_sha256,
        "evidence_sha256": evidence_sha256,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), default="smoke"
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--audit-certificate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    rows = load_tabfact_manifest(args.manifest, project_root=root)
    certificate = load_vitaminc_runtime_audit_v4(args.audit_certificate)
    prompt_path = root / "configs/prompt_contract_v2.json"
    if certificate.prompt_contract_sha256 != _file_sha256(prompt_path):
        parser.error("audit certificate prompt contract SHA-256 mismatch")
    common = {
        "project_root": root,
        "units": build_tabfact_prompt_inputs(rows),
        "audit_certificate_sha256": certificate.certificate_sha256,
        "choice_token_ids": {
            choice: certificate.choice_token_ids[choice]
            for choice in ("A", "B")
        },
        "model_key": certificate.model_key,
        "quantization": certificate.quantization,
        "client": LlamaServerClient(args.server),
        "database_path": args.database,
        "export_directory": args.export_directory,
    }
    if args.mode == "formal":
        frozen = validate_frozen_tabfact_formal_manifest(rows, args.manifest)
        report = run_tabfact_formal(**common, **frozen)
    else:
        report = run_tabfact_smoke(
            **common,
            split_sha256=id_manifest_sha256(row.sample_id for row in rows),
            smoke_sha256=_file_sha256(args.manifest),
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
