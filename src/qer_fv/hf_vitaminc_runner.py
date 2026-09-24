"""Paired Hugging Face smoke runner for two frozen VitaminC pilot quartets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .hf_model_identity import hf_model_identity
from .hf_runtime import (
    HF_DIRECT_LOGIT_METHOD,
    HFNextTokenRuntime,
    audit_choice_boundary,
)
from .hf_runtime_audit import (
    HFRuntimeAuditError,
    build_hf_runtime_audit,
    load_hf_runtime_audit,
    require_matching_hf_runtime_audit,
    write_hf_runtime_audit,
)
from .prompts import load_prompt_contract, render_prompt
from .scoring import HARD_LABEL_METHOD, score_choice_argmax
from .vitaminc import id_manifest_sha256
from .vitaminc_inputs import (
    VitaminCInferenceUnit,
    build_vitaminc_units,
    load_vitaminc_split,
)
from .vitaminc_store_v4 import VitaminCRunIdentityV4, VitaminCRunStoreV4


HF_VITAMINC_SMOKE_PROTOCOL = "hf-vitaminc-smoke-v2-20260718"
HF_VITAMINC_FORMAL_PROTOCOL = "hf-vitaminc-formal-v1-20260718"
HF_AWQ_D5_SMOKE_PROTOCOL = "hf-awq-d5-smoke-v1-20260721"
HF_AWQ_D5_FORMAL_PROTOCOL = "hf-awq-d5-formal-v1-20260721"
QWEN_HF_TRIPLET_SMOKE_PROTOCOL = "qwen-hf-triplet-smoke-v1-20260728"
MINISTRAL_HF_TRIPLET_SMOKE_PROTOCOL = (
    "ministral-hf-triplet-smoke-v1-20260809"
)
MINISTRAL_HF_TRIPLET_FORMAL_PROTOCOL = (
    "ministral-hf-triplet-formal-v1-20260809"
)
OLMO3_HF_TRIPLET_SMOKE_PROTOCOL = "olmo3-hf-triplet-smoke-v1-20260901"
OLMO3_HF_TRIPLET_FORMAL_PROTOCOL = "olmo3-hf-triplet-formal-v1-20260901"
HF_BALANCED_MAPPING_SMOKE_PROTOCOL = (
    "vitaminc-hf-balanced-mapping-smoke-v2-20260905"
)
HF_BALANCED_MAPPING_FORMAL_PROTOCOL = (
    "vitaminc-hf-balanced-mapping-formal-v2-20260905"
)
HF_VITAMINC_SMOKE_QUARTETS = 2
_INFERENCE_ERRORS = (
    HFRuntimeAuditError,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    OverflowError,
)


def run_hf_vitaminc_units(
    *,
    project_root: str | Path,
    units: Sequence[VitaminCInferenceUnit],
    split: str,
    split_sha256: str,
    runtime: HFNextTokenRuntime,
    audit_certificate_path: str | Path,
    database_path: str | Path,
    export_directory: str | Path,
    mapping_variant: str = "original",
    protocol_version: str | None = None,
) -> dict[str, int]:
    """Use one last-token forward pass to populate both immutable stages."""
    root = Path(project_root).resolve()
    if not units or len({unit.owner_key for unit in units}) != len(units):
        raise ValueError("HF smoke units must be nonempty and unique")
    certificate = load_hf_runtime_audit(audit_certificate_path)
    if runtime.model_path != certificate.model_path:
        raise ValueError("HF runtime model path does not match audit certificate")
    if runtime.precision != certificate.precision:
        raise ValueError("HF runtime precision does not match audit certificate")
    if dict(runtime.load_policy) != dict(certificate.load_policy):
        raise ValueError("HF runtime load policy does not match audit certificate")
    if certificate.direct_logit_method != HF_DIRECT_LOGIT_METHOD:
        raise ValueError("HF direct-logit method does not match frozen protocol")
    identity = hf_model_identity(certificate.model_key)

    contract_path = root / "configs/prompt_contract_v2.json"
    prompt_sha256 = _file_sha256(contract_path)
    if certificate.prompt_contract_sha256 != prompt_sha256:
        raise ValueError("HF audit prompt contract SHA-256 mismatch")
    contract = load_prompt_contract(contract_path)
    controls = Counter(unit.control for unit in units)
    if set(controls) - {"full", "no_evidence", "knowledge_only"}:
        raise ValueError("HF smoke units contain an invalid control")
    if protocol_version is None:
        if split == "pilot":
            run_protocol = identity.smoke_protocol
        elif split == "dose_subset":
            run_protocol = identity.formal_protocol
        else:
            raise ValueError("HF run split must be pilot or dose_subset")
        if mapping_variant != "original":
            raise ValueError(
                "non-original HF mapping requires an explicit protocol"
            )
    else:
        run_protocol = _required_protocol(protocol_version)
        balanced_protocol_base = run_protocol.partition(":")[0]
        if run_protocol == QWEN_HF_TRIPLET_SMOKE_PROTOCOL:
            if (
                split != "pilot"
                or controls != Counter({"full": 16})
                or mapping_variant != "original"
                or identity.model_key != "qwen35_9b"
                or certificate.precision
                not in {"FP16", "GPTQ_INT4", "AWQ_INT4"}
            ):
                raise ValueError(
                    "Qwen-only triplet smoke requires 16 full inputs per route"
                )
        elif balanced_protocol_base in {
            HF_BALANCED_MAPPING_SMOKE_PROTOCOL,
            HF_BALANCED_MAPPING_FORMAL_PROTOCOL,
        }:
            _validate_balanced_mapping_run(
                run_protocol=run_protocol,
                split=split,
                controls=controls,
                mapping_variant=mapping_variant,
                model_key=identity.model_key,
                precision=certificate.precision,
            )
        elif run_protocol in {
            MINISTRAL_HF_TRIPLET_SMOKE_PROTOCOL,
            MINISTRAL_HF_TRIPLET_FORMAL_PROTOCOL,
        }:
            _validate_ministral_triplet_run(
                run_protocol=run_protocol,
                split=split,
                controls=controls,
                mapping_variant=mapping_variant,
                model_key=identity.model_key,
                precision=certificate.precision,
            )
        elif run_protocol in {
            OLMO3_HF_TRIPLET_SMOKE_PROTOCOL,
            OLMO3_HF_TRIPLET_FORMAL_PROTOCOL,
        }:
            _validate_olmo3_triplet_run(
                run_protocol=run_protocol,
                split=split,
                controls=controls,
                mapping_variant=mapping_variant,
                model_key=identity.model_key,
                precision=certificate.precision,
            )
        else:
            d5_split = {
                HF_AWQ_D5_SMOKE_PROTOCOL: "pilot",
                HF_AWQ_D5_FORMAL_PROTOCOL: "dose_subset",
            }.get(run_protocol)
            if d5_split is not None:
                if (
                    split != d5_split
                    or set(controls) != {"full"}
                    or mapping_variant != "original"
                    or certificate.precision != "AWQ_INT4"
                ):
                    raise ValueError(
                        "D5 AWQ protocol requires matching full-only AWQ units"
                    )
            elif split != "discovery" or set(controls) != {"full"}:
                raise ValueError(
                    "explicit HF mapping protocol requires discovery full-only units"
                )
    balanced_protocol = run_protocol.partition(":")[0] in {
        HF_BALANCED_MAPPING_SMOKE_PROTOCOL,
        HF_BALANCED_MAPPING_FORMAL_PROTOCOL,
    }
    if identity.model_key == "ministral3_8b" and not balanced_protocol:
        _validate_ministral_triplet_run(
            run_protocol=run_protocol,
            split=split,
            controls=controls,
            mapping_variant=mapping_variant,
            model_key=identity.model_key,
            precision=certificate.precision,
        )
    if identity.model_key == "olmo3_7b" and not balanced_protocol:
        _validate_olmo3_triplet_run(
            run_protocol=run_protocol,
            split=split,
            controls=controls,
            mapping_variant=mapping_variant,
            model_key=identity.model_key,
            precision=certificate.precision,
        )
    identity = VitaminCRunIdentityV4(
        protocol_version=run_protocol,
        split=split,
        split_sha256=split_sha256,
        audit_certificate_sha256=certificate.certificate_sha256,
        prompt_contract_sha256=prompt_sha256,
        model_key=certificate.model_key,
        quantization=certificate.precision,
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
                scores = runtime.score_messages(
                    rendered.messages, certificate.choice_token_ids
                )
                scored = score_choice_argmax(
                    rendered.choice_to_label, scores.choice_logprobs
                )
                if scored.choice != scores.scored_choice:
                    raise ValueError("HF runtime and frozen argmax disagree")
                common = {
                    "choice_logprobs": dict(scores.choice_logprobs),
                    "generation_input_ids_sha256": (
                        scores.generation_input_ids_sha256
                    ),
                    "generation_prompt_sha256": scores.generation_prompt_sha256,
                    "prompt_sha256": rendered.prompt_sha256,
                    "scored_choice": scored.choice,
                    "scored_label": scored.label,
                    "token_ids": dict(certificate.choice_token_ids),
                }
                if "hard" in pending:
                    store.put_hard_success(
                        unit.owner_key,
                        metadata=metadata,
                        payload={
                            **common,
                            "hard_label_method": HARD_LABEL_METHOD,
                        },
                    )
                if "probability" in pending:
                    store.put_probability_success(
                        unit.owner_key,
                        metadata=metadata,
                        payload={
                            **common,
                            "choice_probabilities": dict(
                                scores.choice_probabilities
                            ),
                            "direct_logit_method": scores.direct_logit_method,
                            "label_probability_mass": (
                                scores.label_probability_mass
                            ),
                        },
                    )
            except _INFERENCE_ERRORS as error:
                failure = _failure_payload(error, rendered.prompt_sha256)
                if "hard" in pending:
                    store.put_hard_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="hf_inference_error",
                        payload={**failure, "stage": "hard"},
                    )
                if "probability" in pending:
                    store.put_probability_failure(
                        unit.owner_key,
                        metadata=metadata,
                        error_code="hf_inference_error",
                        payload={**failure, "stage": "probability"},
                    )
        summary = store.counts()
        store.export_complete(export_directory)
    return summary


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
        raise ValueError("HF VitaminC run protocol must be a nonempty string")
    return value


def _validate_ministral_triplet_run(
    *,
    run_protocol: str,
    split: str,
    controls: Counter,
    mapping_variant: str,
    model_key: str,
    precision: str,
) -> None:
    requirements = {
        MINISTRAL_HF_TRIPLET_SMOKE_PROTOCOL: ("pilot", 16),
        MINISTRAL_HF_TRIPLET_FORMAL_PROTOCOL: ("dose_subset", 4800),
    }
    expected = requirements.get(run_protocol)
    if expected is None:
        raise ValueError("Ministral run requires its frozen triplet protocol")
    expected_split, expected_full = expected
    if (
        split != expected_split
        or controls != Counter({"full": expected_full})
        or mapping_variant != "original"
        or model_key != "ministral3_8b"
        or precision not in {"FP16", "GPTQ_INT4", "AWQ_INT4"}
    ):
        raise ValueError(
            f"Ministral triplet protocol requires exactly {expected_full} full "
            "inputs per route"
        )


def _validate_olmo3_triplet_run(
    *,
    run_protocol: str,
    split: str,
    controls: Counter,
    mapping_variant: str,
    model_key: str,
    precision: str,
) -> None:
    requirements = {
        OLMO3_HF_TRIPLET_SMOKE_PROTOCOL: ("pilot", 16),
        OLMO3_HF_TRIPLET_FORMAL_PROTOCOL: ("dose_subset", 4800),
    }
    expected = requirements.get(run_protocol)
    if expected is None:
        raise ValueError("OLMo triplet run requires its frozen protocol")
    expected_split, expected_full = expected
    if (
        split != expected_split
        or controls != Counter({"full": expected_full})
        or mapping_variant != "original"
        or model_key != "olmo3_7b"
        or precision not in {"FP16", "GPTQ_INT4", "AWQ_INT4"}
    ):
        raise ValueError(
            f"OLMo triplet protocol requires exactly {expected_full} full "
            "inputs per route"
        )


def _validate_balanced_mapping_run(
    *,
    run_protocol: str,
    split: str,
    controls: Counter,
    mapping_variant: str,
    model_key: str,
    precision: str,
) -> None:
    requirements = {
        HF_BALANCED_MAPPING_SMOKE_PROTOCOL: ("dose_subset", 16),
        HF_BALANCED_MAPPING_FORMAL_PROTOCOL: ("dose_subset", 4800),
    }
    protocol_base, separator, suffix = run_protocol.partition(":")
    expected = requirements.get(protocol_base)
    if expected is None:
        raise ValueError("unknown balanced mapping protocol")
    if separator and suffix != mapping_variant:
        raise ValueError("balanced mapping run identity suffix mismatch")
    expected_split, expected_full = expected
    if (
        split != expected_split
        or controls != Counter({"full": expected_full})
        or mapping_variant not in {"original", "cycle_1", "cycle_2"}
        or model_key not in {"qwen35_9b", "ministral3_8b", "olmo3_7b"}
        or precision not in {"FP16", "GPTQ_INT4", "AWQ_INT4"}
    ):
        raise ValueError(
            "balanced mapping protocol requires one frozen model-route cell, "
            f"one Latin mapping, and exactly {expected_full} full inputs"
        )


def _failure_payload(
    error: BaseException, prompt_sha256: str
) -> dict[str, str]:
    return {
        "error_message": str(error),
        "error_type": type(error).__name__,
        "prompt_sha256": prompt_sha256,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hf_vitaminc_units(
    project_root: str | Path,
    *,
    split: str,
    limit_cases: int | None,
    controls: Sequence[str] | None = None,
) -> tuple[list[VitaminCInferenceUnit], str]:
    """Load only the approved smoke or Stage B formal membership."""
    if split == "pilot":
        qwen_triplet_smoke = (
            limit_cases == 4 and tuple(controls or ()) == ("full",)
        )
        if limit_cases != HF_VITAMINC_SMOKE_QUARTETS and not qwen_triplet_smoke:
            raise ValueError("HF pilot must use exactly 2 quartets")
    elif split == "dose_subset":
        if limit_cases is not None:
            raise ValueError("HF dose_subset must not use a case limit")
    else:
        raise ValueError("HF run split must be pilot or dose_subset")
    quartets = load_vitaminc_split(project_root, split=split)
    if limit_cases is not None:
        quartets = quartets[:limit_cases]
    if controls is not None and tuple(controls) != ("full",):
        raise ValueError("HF D5 controls must contain only full")
    units = build_vitaminc_units(quartets)
    if controls is not None:
        units = [unit for unit in units if unit.control in controls]
    return (
        units,
        id_manifest_sha256(quartet.case_id for quartet in quartets),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-key", default="qwen35_9b")
    parser.add_argument(
        "--precision",
        choices=("FP16", "GPTQ_INT4", "AWQ_INT4"),
        required=True,
    )
    parser.add_argument("--gptq-artifact-audit", type=Path)
    parser.add_argument("--awq-artifact-audit", type=Path)
    parser.add_argument("--expected-audit-certificate", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("pilot", "dose_subset"), required=True
    )
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--controls", choices=("all", "full"), default="all")
    parser.add_argument("--protocol-version")
    parser.add_argument(
        "--mapping-variant",
        choices=("original", "cycle_1", "cycle_2"),
        default="original",
    )
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    model_path = args.model_path.resolve()
    try:
        protocol_base = (args.protocol_version or "").partition(":")[0]
        if protocol_base in {
            HF_BALANCED_MAPPING_SMOKE_PROTOCOL,
            HF_BALANCED_MAPPING_FORMAL_PROTOCOL,
        }:
            if args.split != "dose_subset" or args.limit_cases is not None or args.controls != "full":
                raise ValueError(
                    "balanced mapping runs require dose_subset, no limit, and full controls"
                )
            from .vitaminc_balanced_mapping_protocol import (
                load_balanced_mapping_units,
            )

            population = (
                "engineering"
                if protocol_base == HF_BALANCED_MAPPING_SMOKE_PROTOCOL
                else "formal"
            )
            units, population_identity = load_balanced_mapping_units(
                root, population=population
            )
            split_sha256 = population_identity.case_ids_sha256
        else:
            units, split_sha256 = load_hf_vitaminc_units(
                root,
                split=args.split,
                limit_cases=args.limit_cases,
                controls=None if args.controls == "all" else ("full",),
            )
    except ValueError as error:
        parser.error(str(error))
    contract_path = root / "configs/prompt_contract_v2.json"
    contract = load_prompt_contract(contract_path)
    probe = render_prompt(
        contract,
        units[0].prompt_input,
        control=units[0].control,
        mapping_variant=args.mapping_variant,
    )

    runtime = HFNextTokenRuntime.from_pretrained(
        str(model_path), args.precision, model_key=args.model_key
    )
    token_audit = audit_choice_boundary(
        runtime.tokenizer, probe.messages, tuple(probe.choice_to_label)
    )
    package_names = ["gptqmodel", "torch", "torchvision", "transformers"]
    if args.precision == "AWQ_INT4":
        package_names.append("compressed-tensors")
    package_versions = {
        name: importlib.metadata.version(name) for name in package_names
    }
    certificate = build_hf_runtime_audit(
        model_path=model_path,
        model_key=args.model_key,
        precision=args.precision,
        package_versions=package_versions,
        load_policy=runtime.load_policy,
        prompt_contract_path=contract_path,
        token_audit=token_audit,
        gptq_artifact_audit_path=args.gptq_artifact_audit,
        awq_artifact_audit_path=args.awq_artifact_audit,
    )
    if args.expected_audit_certificate is not None:
        require_matching_hf_runtime_audit(
            certificate, args.expected_audit_certificate
        )
    write_hf_runtime_audit(certificate, args.audit_output)
    summary = run_hf_vitaminc_units(
        project_root=root,
        units=units,
        split=args.split,
        split_sha256=split_sha256,
        runtime=runtime,
        audit_certificate_path=args.audit_output,
        database_path=args.database,
        export_directory=args.export_directory,
        mapping_variant=args.mapping_variant,
        protocol_version=args.protocol_version,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
