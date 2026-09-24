"""Run the frozen three-family, scale-invariant HF heterogeneity analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qer_fv.hf_awq_triplet_gate import _load_export
from qer_fv.hf_runtime_audit import load_hf_runtime_audit
from qer_fv.hf_vitaminc_analysis import _load_condition
from qer_fv.ministral_hf_triplet_analysis import (
    _validate_condition_identities as _validate_ministral_conditions,
    _validate_gate as _validate_ministral_gate,
)
from qer_fv.olmo3_hf_triplet_analysis import (
    _validate_condition_identities as _validate_olmo_conditions,
    _validate_gate as _validate_olmo_gate,
)
from qer_fv.olmo3_preflight import Olmo3PreflightError, verify_detached_sha256
from qer_fv.qwen_hf_triplet_analysis import (
    _validate_condition_identities as _validate_qwen_conditions,
    _validate_gate as _validate_qwen_gate,
)
from qer_fv.three_model_scale_heterogeneity import (
    analyze_three_model_scale_heterogeneity,
    write_three_model_scale_result,
)
from qer_fv.three_model_records import write_text_free_three_model_records


ROUTES = ("fp16", "gptq", "awq")
PRECISIONS = {"fp16": "FP16", "gptq": "GPTQ_INT4", "awq": "AWQ_INT4"}
MODELS = {"qwen": "qwen35_9b", "ministral": "ministral3_8b", "olmo": "olmo3_7b"}


def _verify_gate_file(path: Path, sidecar: Path) -> str:
    try:
        return verify_detached_sha256(path, sha256_path=sidecar)
    except Olmo3PreflightError:
        fields = sidecar.read_text(encoding="ascii").split()
        expected = fields[0] if fields else ""
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(expected) != 64 or expected != actual:
            raise
        return actual


def _protocol(family: str, route: str) -> str:
    if family == "qwen":
        return (
            "hf-awq-d5-formal-v1-20260721"
            if route == "awq"
            else "hf-vitaminc-formal-v1-20260718"
        )
    if family == "ministral":
        return "ministral-hf-triplet-formal-v1-20260809"
    return "olmo3-hf-triplet-formal-v1-20260901"


def _load(root: Path, *, family: str, route: str):
    certificate = load_hf_runtime_audit(root / "audit.json")
    manifest, rows = _load_export(root / "export")
    return _load_condition(
        manifest,
        rows,
        certificate_sha256=certificate.certificate_sha256,
        expected_precision=PRECISIONS[route],
        mode="formal",
        expected_protocol=_protocol(family, route),
        expected_model_key=MODELS[family],
    )


def _aligned_inputs(conditions):
    identities = {
        (condition.split, condition.split_sha256, condition.prompt_contract_sha256)
        for family in conditions.values()
        for condition in family.values()
    }
    if len(identities) != 1:
        raise RuntimeError("nine-condition source or prompt identity drifted")

    reference_metadata = None
    model_values = {}
    for family in ("qwen", "ministral", "olmo"):
        model_values[family] = {}
        for route in ROUTES:
            current = {}
            metadata = {}
            for quartet in conditions[family][route].quartets:
                if quartet.case_id in current or quartet.interaction is None:
                    raise RuntimeError("incomplete or duplicate quartet identity")
                current[quartet.case_id] = float(quartet.interaction)
                metadata[quartet.case_id] = (
                    quartet.page,
                    quartet.negative_label,
                    tuple(quartet.hard_gold_labels),
                )
            if len(current) != 1200:
                raise RuntimeError("formal condition requires 1200 quartets")
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise RuntimeError("nine-condition quartet metadata drifted")
            model_values[family][route] = current

    assert reference_metadata is not None
    pages = {case: metadata[0] for case, metadata in reference_metadata.items()}
    if len(set(pages.values())) != 1078:
        raise RuntimeError("formal analysis requires 1078 page clusters")
    split, split_sha256, prompt_sha256 = next(iter(identities))
    return model_values, pages, {
        "nine_condition_alignment": True,
        "split": split,
        "split_sha256": split_sha256,
        "prompt_contract_sha256": prompt_sha256,
        "complete_hard_coverage": True,
        "complete_probability_coverage": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for family in ("qwen", "ministral", "olmo"):
        for route in ROUTES:
            parser.add_argument(f"--{family}-{route}-root", type=Path, required=True)
        parser.add_argument(f"--{family}-gate", type=Path, required=True)
        parser.add_argument(f"--{family}-gate-sha256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-records-output", type=Path)
    args = parser.parse_args(argv)

    gates = {}
    for family in ("qwen", "ministral", "olmo"):
        gate_path = getattr(args, f"{family}_gate")
        _verify_gate_file(
            gate_path,
            getattr(args, f"{family}_gate_sha256"),
        )
        gates[family] = json.loads(gate_path.read_text(encoding="utf-8"))

    conditions = {
        family: {
            route: _load(
                getattr(args, f"{family}_{route}_root"),
                family=family,
                route=route,
            )
            for route in ROUTES
        }
        for family in ("qwen", "ministral", "olmo")
    }
    _validate_qwen_conditions(conditions["qwen"], formal=True)
    _validate_ministral_conditions(conditions["ministral"], formal=True)
    _validate_olmo_conditions(conditions["olmo"], formal=True)
    _validate_qwen_gate(
        gates["qwen"], formal=True, split_sha256=conditions["qwen"]["fp16"].split_sha256
    )
    _validate_ministral_gate(
        gates["ministral"],
        formal=True,
        split_sha256=conditions["ministral"]["fp16"].split_sha256,
    )
    _validate_olmo_gate(
        gates["olmo"], formal=True, split_sha256=conditions["olmo"]["fp16"].split_sha256
    )
    model_values, pages, input_contract = _aligned_inputs(conditions)
    result = analyze_three_model_scale_heterogeneity(
        model_values=model_values,
        pages=pages,
    )
    result["input_contract"] = input_contract
    result["source_gates"] = {
        family: gates[family]["gate_sha256"] for family in ("qwen", "ministral", "olmo")
    }

    source_files = {}
    for family in ("qwen", "ministral", "olmo"):
        source_files[f"{family}_gate"] = getattr(args, f"{family}_gate")
        source_files[f"{family}_gate_sha256"] = getattr(args, f"{family}_gate_sha256")
        for route in ROUTES:
            root = getattr(args, f"{family}_{route}_root")
            source_files[f"{family}_{route}_audit"] = root / "audit.json"
            source_files[f"{family}_{route}_manifest"] = root / "export" / "manifest.json"
            source_files[f"{family}_{route}_records"] = root / "export" / "records.jsonl"
    write_three_model_scale_result(
        result,
        args.output,
        source_files=source_files,
        formal=True,
    )
    if args.analysis_records_output is not None:
        exports = {
            f"{family}_{route}": getattr(args, f"{family}_{route}_root") / "export"
            for family in ("qwen", "ministral", "olmo")
            for route in ROUTES
        }
        write_text_free_three_model_records(
            exports=exports,
            output_directory=args.analysis_records_output,
            formal=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
