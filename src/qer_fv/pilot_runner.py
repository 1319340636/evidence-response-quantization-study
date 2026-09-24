"""Audited VitaminC pilot runner for Experiment B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .llama_client import (
    CHOICE_LOGIT_BIAS,
    CHOICE_SCORING_METHOD,
    LlamaServerClient,
)
from .parsing import parse_choice
from .prompts import PromptInput, load_prompt_contract, render_prompt
from .provenance import load_freeze_config
from .runtime_audit import (
    AUDIT_PROTOCOL_VERSION,
    RuntimeAuditCertificate,
    RuntimeAuditError,
    load_runtime_audit_certificate,
)
from .scoring import HARD_LABEL_METHOD, score_choice_argmax
from .vitaminc import VitaminCQuartet, build_strict_quartets, read_jsonl


def load_vitaminc_pilot_inputs(
    project_root: str | Path, *, limit_cases: int | None = None
) -> list[PromptInput]:
    root = Path(project_root).resolve()
    config = load_freeze_config(root / "configs/freeze_v1.json")
    vitamin = config.get("vitaminc")
    if not isinstance(vitamin, Mapping) or not isinstance(vitamin.get("dev_path"), str):
        raise ValueError("freeze config vitaminc.dev_path is required")
    pilot_ids = _read_ids(root / "data/manifests/vitaminc_pilot_ids.txt")
    if limit_cases is not None:
        pilot_ids = pilot_ids[:limit_cases]
    quartets = {
        quartet.case_id: quartet
        for quartet in build_strict_quartets(read_jsonl(root / vitamin["dev_path"]))
    }
    missing = [case_id for case_id in pilot_ids if case_id not in quartets]
    if missing:
        raise ValueError(f"pilot IDs missing from strict dev quartets: {missing[:3]}")

    inputs: list[PromptInput] = []
    for case_id in pilot_ids:
        inputs.extend(_quartet_inputs(quartets[case_id]))
    return inputs


def run_vitaminc_pilot(
    *,
    project_root: str | Path,
    client: LlamaServerClient,
    audit_certificate_path: str | Path,
    output_path: str | Path,
    limit_cases: int | None = None,
) -> dict[str, int]:
    root = Path(project_root).resolve()
    certificate = load_runtime_audit_certificate(audit_certificate_path)
    if certificate.protocol_version != AUDIT_PROTOCOL_VERSION:
        raise RuntimeAuditError("runtime audit certificate protocol is not current")
    if certificate.choice_scoring_method != CHOICE_SCORING_METHOD:
        raise RuntimeAuditError(
            "runtime audit certificate scoring method is not current"
        )
    if certificate.choice_logit_bias != CHOICE_LOGIT_BIAS:
        raise RuntimeAuditError("runtime audit certificate choice bias is not current")
    contract_path = root / "configs/prompt_contract_v1.json"
    _verify_prompt_contract(certificate, contract_path)
    contract = load_prompt_contract(contract_path)
    records: list[dict[str, Any]] = []
    valid = 0
    for item in load_vitaminc_pilot_inputs(root, limit_cases=limit_cases):
        rendered = render_prompt(contract, item)
        prompt_text = client.apply_template(rendered.messages, enable_thinking=False)
        completion = client.complete_choice(
            prompt_text,
            choices=tuple(rendered.choice_to_label),
            choice_token_ids=certificate.choice_token_ids,
        )
        parsed = parse_choice(completion.content, rendered.choice_to_label)
        scored = score_choice_argmax(
            rendered.choice_to_label, completion.choice_logprobs
        )
        if parsed.valid:
            valid += 1
        records.append(
            {
                "audit_certificate_sha256": certificate.certificate_sha256,
                "case_id": item.metadata["case_id"],
                "control": rendered.control,
                "dataset": rendered.dataset,
                "gold_label": item.metadata["gold_label"],
                "model_key": certificate.model_key,
                "parsed_choice": parsed.choice,
                "parsed_label": parsed.label,
                "sampled_choice": parsed.choice,
                "sampled_label": parsed.label,
                "scored_choice": scored.choice,
                "scored_label": scored.label,
                "hard_label_method": HARD_LABEL_METHOD,
                "prompt_sha256": rendered.prompt_sha256,
                "quantization": certificate.quantization,
                "raw_choice": completion.content,
                "sample_id": item.sample_id,
                "valid": parsed.valid,
                "choice_logprobs": dict(sorted(completion.choice_logprobs.items())),
            }
        )
    _write_jsonl(Path(output_path), records)
    return {"records": len(records), "valid": valid}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--server", required=True)
    parser.add_argument("--audit-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args(argv)
    client = LlamaServerClient(args.server)
    summary = run_vitaminc_pilot(
        project_root=args.project_root,
        client=client,
        audit_certificate_path=args.audit_certificate,
        output_path=args.output,
        limit_cases=args.limit_cases,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _quartet_inputs(quartet: VitaminCQuartet) -> list[PromptInput]:
    cells = [
        (quartet.claims[0], quartet.evidences[0]),
        (quartet.claims[0], quartet.evidences[1]),
        (quartet.claims[1], quartet.evidences[0]),
        (quartet.claims[1], quartet.evidences[1]),
    ]
    inputs: list[PromptInput] = []
    for row_id, label, (claim, evidence) in zip(quartet.row_ids, quartet.labels, cells):
        inputs.append(
            PromptInput(
                dataset="vitaminc",
                sample_id=row_id,
                claim=claim,
                evidence=evidence,
                metadata={
                    "case_id": quartet.case_id,
                    "gold_label": label,
                    "negative_label": quartet.negative_label,
                    "page": quartet.page,
                },
            )
        )
    return inputs


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def _verify_prompt_contract(
    certificate: RuntimeAuditCertificate, contract_path: Path
) -> None:
    import hashlib

    actual = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if certificate.prompt_contract_sha256 != actual:
        raise RuntimeAuditError(
            "runtime audit certificate prompt contract SHA-256 does not match"
        )


if __name__ == "__main__":
    raise SystemExit(main())
