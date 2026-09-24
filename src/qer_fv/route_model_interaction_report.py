"""Hashed reporting for the Qwen × Gemma route-by-model interaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .hf_vitaminc_analysis import load_hf_vitaminc_pair
from .route_model_interaction import analyze_route_model_interaction
from .vitaminc import id_manifest_sha256
from .vitaminc_results import (
    load_vitaminc_condition_export,
    pair_vitaminc_conditions,
)

REPORT_PROTOCOL = "qwen-gemma-route-model-report-v1-20260721"
VALIDATION_PROTOCOL = (
    "qwen-gemma-route-model-report-validation-v1-20260721"
)
FROZEN_CONFIG = {
    "analysis_protocol": (
        "qwen-gemma-route-model-interaction-v1-20260721"
    ),
    "estimand_order": (
        "(Gemma_HF-Gemma_GGUF)-(Qwen_HF-Qwen_GGUF)"
    ),
    "split": "dose_subset",
    "split_sha256": (
        "bbe3e66dc7e53e0fa08c656716e113f63b8c5232de8b9fa895de6046b861b9f1"
    ),
    "quartets": 1200,
    "pages": 1078,
    "bootstrap_draws": 10000,
    "seed": 20260711,
    "cluster": "page",
    "confidence_level": 0.95,
    "inference_scope": "prospectively_frozen_d4_followup_secondary",
    "recovery_access": False,
}


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    """Load the exact pre-outcome four-way analysis contract."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read route-model analysis config") from error
    if not isinstance(value, dict):
        raise ValueError("route-model analysis config must be an object")
    if set(value) != set(FROZEN_CONFIG):
        raise ValueError("analysis config fields do not match frozen contract")
    for key, expected in FROZEN_CONFIG.items():
        if value.get(key) != expected:
            raise ValueError(f"analysis config mismatch at {key}")
    return value


def reject_recovery_paths(paths: list[str | Path]) -> None:
    """Reject any input whose path contains an exact recovery component."""
    for supplied in paths:
        if any(
            part.casefold() == "recovery" for part in Path(supplied).parts
        ):
            raise ValueError("recovery inputs are forbidden")


def parser() -> argparse.ArgumentParser:
    """Build the explicit fail-closed formal analysis CLI."""
    value = argparse.ArgumentParser(description=__doc__)
    for model in ("qwen", "gemma"):
        value.add_argument(
            f"--{model}-hf-fp16-export", type=Path, required=True
        )
        value.add_argument(
            f"--{model}-hf-gptq-export", type=Path, required=True
        )
        value.add_argument(
            f"--{model}-hf-fp16-audit", type=Path, required=True
        )
        value.add_argument(
            f"--{model}-hf-gptq-audit", type=Path, required=True
        )
        value.add_argument(
            f"--{model}-hf-pair-gate", type=Path, required=True
        )
        value.add_argument(
            f"--{model}-gguf-f16-export", type=Path, required=True
        )
        value.add_argument(
            f"--{model}-gguf-q4-export", type=Path, required=True
        )
    value.add_argument("--dose-ids", type=Path, required=True)
    value.add_argument("--analysis-config", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    """Revalidate four frozen routes and write the formal interaction report."""
    args = parser().parse_args(argv)
    input_paths = [
        value
        for key, value in vars(args).items()
        if key != "output_dir" and isinstance(value, Path)
    ]
    reject_recovery_paths(input_paths)
    config = load_analysis_config(args.analysis_config)
    dose_ids = tuple(
        value
        for value in args.dose_ids.read_text(encoding="utf-8").splitlines()
        if value
    )
    if (
        len(dose_ids) != config["quartets"]
        or len(dose_ids) != len(set(dose_ids))
        or id_manifest_sha256(dose_ids) != config["split_sha256"]
    ):
        raise ValueError("frozen dose-ID identity does not match config")

    qwen_hf = load_hf_vitaminc_pair(
        fp16_export=args.qwen_hf_fp16_export,
        gptq_export=args.qwen_hf_gptq_export,
        fp16_audit=args.qwen_hf_fp16_audit,
        gptq_audit=args.qwen_hf_gptq_audit,
        mode="formal",
        model_key="qwen35_9b",
    )
    gemma_hf = load_hf_vitaminc_pair(
        fp16_export=args.gemma_hf_fp16_export,
        gptq_export=args.gemma_hf_gptq_export,
        fp16_audit=args.gemma_hf_fp16_audit,
        gptq_audit=args.gemma_hf_gptq_audit,
        mode="formal",
        model_key="gemma4_e4b",
    )
    for name, path, loaded in (
        ("qwen", args.qwen_hf_pair_gate, qwen_hf),
        ("gemma", args.gemma_hf_pair_gate, gemma_hf),
    ):
        try:
            supplied = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read saved {name} HF pair Gate") from error
        if supplied != dict(loaded.gate):
            raise ValueError(f"saved {name} HF pair Gate does not revalidate")
        if loaded.fp16.split_sha256 != config["split_sha256"]:
            raise ValueError(f"{name} HF split identity mismatch")

    gguf_conditions = {}
    for name, model_key, f16_path, q4_path in (
        (
            "qwen",
            "qwen35_9b",
            args.qwen_gguf_f16_export,
            args.qwen_gguf_q4_export,
        ),
        (
            "gemma",
            "gemma4_e4b",
            args.gemma_gguf_f16_export,
            args.gemma_gguf_q4_export,
        ),
    ):
        f16 = load_vitaminc_condition_export(f16_path)
        q4 = load_vitaminc_condition_export(q4_path)
        if (
            f16.model_key != model_key
            or q4.model_key != model_key
            or f16.quantization != "F16"
            or q4.quantization != "Q4_K_M"
        ):
            raise ValueError(f"{name} GGUF model/precision identity mismatch")
        if f16.prompt_contract_sha256 != q4.prompt_contract_sha256:
            raise ValueError(f"{name} GGUF prompt identity mismatch")
        gguf_conditions[name] = (f16, q4)

    prompt_hashes = {
        qwen_hf.fp16.prompt_contract_sha256,
        gemma_hf.fp16.prompt_contract_sha256,
        gguf_conditions["qwen"][0].prompt_contract_sha256,
        gguf_conditions["gemma"][0].prompt_contract_sha256,
    }
    if len(prompt_hashes) != 1:
        raise ValueError("four-way prompt identity mismatch")

    dose_set = set(dose_ids)
    gguf_pairs = {
        name: tuple(
            pair
            for pair in pair_vitaminc_conditions(f16, q4)
            if pair.case_id in dose_set
        )
        for name, (f16, q4) in gguf_conditions.items()
    }
    expected_cases = set(dose_ids)
    for name, pairs in (
        ("qwen_hf", qwen_hf.pairs),
        ("qwen_gguf", gguf_pairs["qwen"]),
        ("gemma_hf", gemma_hf.pairs),
        ("gemma_gguf", gguf_pairs["gemma"]),
    ):
        if len(pairs) != len(dose_ids) or {
            pair.case_id for pair in pairs
        } != expected_cases:
            raise ValueError(f"{name} dose membership mismatch")

    analysis = analyze_route_model_interaction(
        qwen_hf_pairs=qwen_hf.pairs,
        qwen_gguf_pairs=gguf_pairs["qwen"],
        gemma_hf_pairs=gemma_hf.pairs,
        gemma_gguf_pairs=gguf_pairs["gemma"],
        draws=config["bootstrap_draws"],
        seed=config["seed"],
    )
    if (
        analysis["population"]["quartets"] != config["quartets"]
        or analysis["population"]["pages"] != config["pages"]
    ):
        raise ValueError("analyzed population does not match frozen config")

    sources = {
        "analysis_config": args.analysis_config,
        "dose_ids": args.dose_ids,
    }
    for model in ("qwen", "gemma"):
        for route, precision in (
            ("hf", "fp16"),
            ("hf", "gptq"),
            ("gguf", "f16"),
            ("gguf", "q4"),
        ):
            export = getattr(args, f"{model}_{route}_{precision}_export")
            sources[f"{model}_{route}_{precision}_manifest"] = (
                export / "manifest.json"
            )
            sources[f"{model}_{route}_{precision}_records"] = (
                export / "records.jsonl"
            )
        sources[f"{model}_hf_fp16_audit"] = getattr(
            args, f"{model}_hf_fp16_audit"
        )
        sources[f"{model}_hf_gptq_audit"] = getattr(
            args, f"{model}_hf_gptq_audit"
        )
        sources[f"{model}_hf_pair_gate"] = getattr(
            args, f"{model}_hf_pair_gate"
        )
    manifest = write_route_model_interaction_report(
        analysis=analysis,
        source_paths=sources,
        output_directory=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def write_route_model_interaction_report(
    *,
    analysis: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write deterministic result, validation, and self-hashed manifest files."""
    if not source_paths:
        raise ValueError("route-model report requires source paths")
    sources: dict[str, dict[str, str]] = {}
    for role, supplied in sorted(source_paths.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("route-model source role is invalid")
        path = Path(supplied)
        if any(part.casefold() == "recovery" for part in path.parts):
            raise ValueError("recovery sources are forbidden")
        if not path.is_file():
            raise ValueError(f"route-model source is missing: {role}")
        sources[role] = {"path": str(path), "sha256": _file_sha256(path)}

    output = Path(output_directory)
    if (output / "report_manifest.json").exists():
        raise ValueError("completed report cannot be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "report_protocol": REPORT_PROTOCOL,
        "analysis": dict(analysis),
    }
    _atomic_write_text(output / "results.json", _json_text(results))
    _atomic_write_text(output / "analysis.md", _markdown(results))

    validation = {
        "validation_protocol": VALIDATION_PROTOCOL,
        "status": "valid",
        "source_validation": (
            "Both HF pairs, all four export identities and hashes, common "
            "dose membership, scoring methods, metadata alignment, and the "
            "frozen analysis configuration were revalidated before analysis."
        ),
        "sources": sources,
        "parameters": dict(analysis["parameters"]),
        "output_files": {
            "analysis.md": _file_sha256(output / "analysis.md"),
            "results.json": _file_sha256(output / "results.json"),
        },
    }
    _atomic_write_text(output / "validation.json", _json_text(validation))
    manifest = {
        "manifest_protocol": "hashed-report-manifest-v1",
        "files": {
            name: _file_sha256(output / name)
            for name in ("analysis.md", "results.json", "validation.json")
        },
    }
    manifest["manifest_sha256"] = _record_hash(manifest)
    _atomic_write_text(output / "report_manifest.json", _json_text(manifest))
    return manifest


def _markdown(results: Mapping[str, Any]) -> str:
    analysis = results["analysis"]
    components = analysis["components"]
    interaction = analysis["interaction"]
    interval = interaction["page_bootstrap"]
    wald = interaction["page_wald"]
    directions = analysis["direction_diagnostic"]
    labels = (
        ("Qwen HF", "qwen_hf"),
        ("Qwen GGUF", "qwen_gguf"),
        ("Qwen route", "qwen_route"),
        ("Gemma HF", "gemma_hf"),
        ("Gemma GGUF", "gemma_gguf"),
        ("Gemma route", "gemma_route"),
    )
    return "\n".join(
        [
            "# Qwen × Gemma route-by-model interaction",
            "",
            "## Frozen four-way interaction",
            "",
            (
                "- Gemma route − Qwen route: "
                f"{interval['estimate']:.4f} "
                f"[{interval['lower']:.4f}, {interval['upper']:.4f}]."
            ),
            f"- Page-level Wald p-value: {wald['pvalue']:.6g}.",
            (
                "- Quartet-weighted sensitivity: "
                f"{interaction['quartet_weighted_sensitivity']:.4f}."
            ),
            "",
            "## Components",
            "",
            "| Component | Mean | 95% page-bootstrap CI |",
            "|---|---:|---:|",
            *[
                (
                    f"| {label} | {components[key]['estimate']:.4f} | "
                    f"[{components[key]['lower']:.4f}, "
                    f"{components[key]['upper']:.4f}] |"
                )
                for label, key in labels
            ],
            "",
            "## Direction diagnostic",
            "",
            (
                "- Qwen route negative pages: "
                f"{directions['qwen_route']['negative']}/"
                f"{directions['qwen_route']['clusters']}."
            ),
            (
                "- Gemma route negative pages: "
                f"{directions['gemma_route']['negative']}/"
                f"{directions['gemma_route']['clusters']}."
            ),
            (
                "- Paired direction McNemar p-value: "
                f"{directions['paired_mcnemar']['pvalue']:.6g}."
            ),
            "",
            "## Interpretation boundary",
            "",
            f"- {analysis['causal_boundary']}",
            "- The direction diagnostic is secondary and unadjusted.",
            "- This is not a pure quantizer, bit-width, backend, or architecture effect.",
            "",
        ]
    )


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            key: value
            for key, value in record.items()
            if key != "manifest_sha256"
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
