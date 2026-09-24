"""Generate the hashed Qwen HF/GPTQ VitaminC formal report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .hf_vitaminc_analysis import (
    analyze_hf_vitaminc_pair,
    analyze_route_heterogeneity,
    load_hf_vitaminc_pair,
)
from .vitaminc import id_manifest_sha256
from .vitaminc_results import (
    load_vitaminc_condition_export,
    pair_vitaminc_conditions,
)


HF_VITAMINC_REPORT_PROTOCOL = "qwen-hf-formal-report-v1-20260718"
HF_VITAMINC_VALIDATION_PROTOCOL = (
    "qwen-hf-formal-report-validation-v1-20260718"
)
_REPORT_IDENTITIES = {
    "qwen35_9b": {
        "report_protocol": HF_VITAMINC_REPORT_PROTOCOL,
        "validation_protocol": HF_VITAMINC_VALIDATION_PROTOCOL,
        "title": "Qwen3.5-9B HF FP16 versus GPTQ INT4",
    },
    "gemma4_e4b": {
        "report_protocol": "gemma-hf-d4-report-v1-20260721",
        "validation_protocol": (
            "gemma-hf-d4-report-validation-v1-20260721"
        ),
        "title": "Gemma 4 E4B HF FP16 versus GPTQ INT4",
    },
}


def hf_report_identity(model_key: str) -> dict[str, str]:
    """Return immutable report metadata for one registered HF analysis."""
    try:
        return dict(_REPORT_IDENTITIES[model_key])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unsupported HF report identity: {model_key!r}"
        ) from error


def write_hf_vitaminc_report(
    *,
    analysis: Mapping[str, Any],
    route_heterogeneity: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
    output_directory: str | Path,
    report_protocol: str = HF_VITAMINC_REPORT_PROTOCOL,
    validation_protocol: str = HF_VITAMINC_VALIDATION_PROTOCOL,
    report_title: str = "Qwen3.5-9B HF FP16 versus GPTQ INT4",
) -> dict[str, Any]:
    """Write deterministic report artifacts and bind them with SHA-256."""
    if not source_paths:
        raise ValueError("HF formal report requires source paths")
    sources: dict[str, dict[str, str]] = {}
    for role, supplied_path in sorted(source_paths.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("HF formal report source role is invalid")
        path = Path(supplied_path)
        if not path.is_file():
            raise ValueError(f"HF formal report source is missing: {role}")
        sources[role] = {
            "path": str(path),
            "sha256": _file_sha256(path),
        }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "report_protocol": report_protocol,
        "confirmatory_hf": dict(analysis),
        "exploratory_route_heterogeneity": dict(route_heterogeneity),
    }
    results_text = _json_text(results)
    markdown_text = _markdown(results, title=report_title)
    _atomic_write_text(output / "results.json", results_text)
    _atomic_write_text(output / "analysis.md", markdown_text)

    validation = {
        "validation_protocol": validation_protocol,
        "status": "valid",
        "source_validation": (
            "The HF pair gate, runtime certificates, export self-hashes, "
            "records hashes, owner alignment, scoring method, coverage, and "
            "frozen dose identity were revalidated before analysis."
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
    _atomic_write_text(
        output / "report_manifest.json", _json_text(manifest)
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16-export", type=Path, required=True)
    parser.add_argument("--gptq-export", type=Path, required=True)
    parser.add_argument("--fp16-audit", type=Path, required=True)
    parser.add_argument("--gptq-audit", type=Path, required=True)
    parser.add_argument("--pair-gate", type=Path, required=True)
    parser.add_argument("--gguf-f16-export", type=Path, required=True)
    parser.add_argument("--gguf-q4-export", type=Path, required=True)
    parser.add_argument("--dose-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--ba-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--model-key", choices=tuple(_REPORT_IDENTITIES), default="qwen35_9b"
    )
    args = parser.parse_args(argv)
    report_identity = hf_report_identity(args.model_key)

    loaded = load_hf_vitaminc_pair(
        fp16_export=args.fp16_export,
        gptq_export=args.gptq_export,
        fp16_audit=args.fp16_audit,
        gptq_audit=args.gptq_audit,
        mode="formal",
        model_key=args.model_key,
    )
    supplied_gate = _load_json_object(args.pair_gate)
    if supplied_gate != dict(loaded.gate):
        raise ValueError("saved HF formal pair gate does not match revalidation")
    analysis = analyze_hf_vitaminc_pair(
        loaded,
        draws=args.draws,
        ba_draws=args.ba_draws,
        seed=args.seed,
    )

    dose_ids = tuple(
        value
        for value in args.dose_ids.read_text(
            encoding="utf-8"
        ).splitlines()
        if value
    )
    if (
        len(dose_ids) != len(set(dose_ids))
        or id_manifest_sha256(dose_ids) != loaded.fp16.split_sha256
    ):
        raise ValueError("frozen dose-ID identity does not match HF run")
    gguf_fp16 = load_vitaminc_condition_export(args.gguf_f16_export)
    gguf_q4 = load_vitaminc_condition_export(args.gguf_q4_export)
    dose_set = set(dose_ids)
    gguf_pairs = tuple(
        pair
        for pair in pair_vitaminc_conditions(gguf_fp16, gguf_q4)
        if pair.case_id in dose_set
    )
    if (
        len(gguf_pairs) != len(dose_ids)
        or {pair.case_id for pair in gguf_pairs}
        != {pair.case_id for pair in loaded.pairs}
    ):
        raise ValueError("GGUF and HF dose-subset memberships differ")
    route = analyze_route_heterogeneity(
        loaded.pairs,
        gguf_pairs,
        draws=args.draws,
        seed=args.seed,
    )
    source_paths = {
        "dose_ids": args.dose_ids,
        "fp16_audit": args.fp16_audit,
        "fp16_manifest": args.fp16_export / "manifest.json",
        "fp16_records": args.fp16_export / "records.jsonl",
        "gguf_f16_manifest": args.gguf_f16_export / "manifest.json",
        "gguf_f16_records": args.gguf_f16_export / "records.jsonl",
        "gguf_q4_manifest": args.gguf_q4_export / "manifest.json",
        "gguf_q4_records": args.gguf_q4_export / "records.jsonl",
        "gptq_audit": args.gptq_audit,
        "gptq_manifest": args.gptq_export / "manifest.json",
        "gptq_records": args.gptq_export / "records.jsonl",
        "pair_gate": args.pair_gate,
    }
    manifest = write_hf_vitaminc_report(
        analysis=analysis,
        route_heterogeneity=route,
        source_paths=source_paths,
        output_directory=args.output_dir,
        report_protocol=report_identity["report_protocol"],
        validation_protocol=report_identity["validation_protocol"],
        report_title=report_identity["title"],
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _markdown(results: Mapping[str, Any], *, title: str) -> str:
    analysis = results["confirmatory_hf"]
    route = results["exploratory_route_heterogeneity"]
    interaction = analysis["interaction"]["page_bootstrap"]
    hard = analysis["hard_labels"]
    accuracy = hard["accuracy"]
    balanced = hard["balanced_accuracy"]["page_bootstrap"]
    direction = analysis["interaction"]["page_direction"]
    route_difference = route["effect_difference_hf_minus_gguf"][
        "page_bootstrap"
    ]
    recalls_fp16 = hard["recall"]["fp16"]
    recalls_gptq = hard["recall"]["gptq_int4"]
    knowledge = analysis["controls"]["knowledge_only"]
    no_evidence = analysis["controls"]["no_evidence"]
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Confirmatory second-route result",
            "",
            (
                f"- Page-balanced ΔI (GPTQ INT4 − FP16): "
                f"{interaction['estimate']:.4f} "
                f"[{interaction['lower']:.4f}, "
                f"{interaction['upper']:.4f}]."
            ),
            (
                f"- Negative page fraction: "
                f"{100 * direction['negative_fraction']:.2f}% "
                f"({direction['negative']}/{direction['clusters']})."
            ),
            (
                f"- Accuracy: {100 * accuracy['fp16']:.2f}% → "
                f"{100 * accuracy['gptq_int4']:.2f}% "
                f"({100 * accuracy['difference']:.2f} pp)."
            ),
            (
                f"- Balanced accuracy: {100 * balanced['f16']:.2f}% → "
                f"{100 * balanced['quantized']:.2f}% "
                f"({100 * balanced['difference']:.2f} pp; "
                f"95% CI [{100 * balanced['lower']:.2f}, "
                f"{100 * balanced['upper']:.2f}] pp)."
            ),
            "",
            "### Per-class recall",
            "",
            "| Label | FP16 | GPTQ INT4 | Difference |",
            "|---|---:|---:|---:|",
            *[
                (
                    f"| {label} | {100 * recalls_fp16[label]:.2f}% | "
                    f"{100 * recalls_gptq[label]:.2f}% | "
                    f"{100 * (recalls_gptq[label] - recalls_fp16[label]):.2f} pp |"
                )
                for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
            ],
            "",
            "### Controls",
            "",
            (
                f"- No-evidence agreement: "
                f"{100 * no_evidence['agreement']:.2f}%."
            ),
            (
                f"- Knowledge-only agreement: "
                f"{100 * knowledge['agreement']:.2f}%."
            ),
            "",
            "## Exploratory route heterogeneity",
            "",
            (
                f"- Matched route-effect difference "
                f"(HF GPTQ effect − GGUF Q4 effect): "
                f"{route_difference['estimate']:.4f} "
                f"[{route_difference['lower']:.4f}, "
                f"{route_difference['upper']:.4f}]."
            ),
            f"- Causal boundary: {route['causal_boundary']}",
            "",
            "## Interpretation limits",
            "",
            "- The HF FP16/GPTQ comparison is the frozen second-route analysis.",
            "- The direct GGUF-versus-HF route contrast is exploratory.",
            "- Behavioral changes do not directly identify an internal neural mechanism.",
            "",
        ]
    )


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(
            handle, "w", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


if __name__ == "__main__":
    raise SystemExit(main())
