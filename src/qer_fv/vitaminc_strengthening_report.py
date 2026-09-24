"""Write deterministic hashed VitaminC D1/D2 strengthening reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .hf_vitaminc_analysis import load_hf_vitaminc_pair
from .vitaminc_backend_comparability import (
    compare_high_precision_backends,
    derive_condition_subset,
)
from .vitaminc_edit_strength import (
    analyze_edit_strength,
    extract_edit_features,
)
from .vitaminc_inputs import load_vitaminc_split
from .vitaminc import id_manifest_sha256
from .vitaminc_results import (
    load_vitaminc_condition_export,
    pair_vitaminc_conditions,
)


STRENGTHENING_REPORT_PROTOCOL = (
    "vitaminc-strengthening-report-v1-20260719"
)
STRENGTHENING_VALIDATION_PROTOCOL = (
    "vitaminc-strengthening-validation-v1-20260719"
)
_OUTPUT_NAMES = (
    "analysis.md",
    "results.json",
    "validation.json",
    "report_manifest.json",
)


def write_vitaminc_strengthening_report(
    *,
    edit_strength: Mapping[str, Any],
    backend_comparability: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Bind D1/D2 analyses and immutable sources into a hashed report."""
    if (
        edit_strength.get("inference_scope")
        != "post_confirmatory_edit_robustness"
    ):
        raise ValueError("edit-strength inference scope mismatch")
    if (
        backend_comparability.get("inference_scope")
        != "descriptive_backend_comparability"
    ):
        raise ValueError("backend comparability inference scope mismatch")
    if not source_paths:
        raise ValueError("VitaminC strengthening report requires source paths")

    sources: dict[str, dict[str, str]] = {}
    for role, supplied in sorted(source_paths.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("VitaminC strengthening source role is invalid")
        path = Path(supplied)
        if _mentions_recovery(role) or any(
            _mentions_recovery(part) for part in path.parts
        ):
            raise ValueError(
                "VitaminC strengthening report forbids recovery sources"
            )
        if not path.is_file():
            raise ValueError(
                f"VitaminC strengthening source is missing: {role}"
            )
        sources[role] = {
            "path": str(path),
            "sha256": _file_sha256(path),
        }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in _OUTPUT_NAMES if (output / name).exists()]
    if existing:
        raise ValueError(
            "VitaminC strengthening output directory is not immutable-empty"
        )

    results = {
        "report_protocol": STRENGTHENING_REPORT_PROTOCOL,
        "edit_strength_robustness": dict(edit_strength),
        "backend_comparability": dict(backend_comparability),
    }
    _atomic_write_text(output / "results.json", _json_text(results))
    _atomic_write_text(output / "analysis.md", _markdown(results))

    validation = {
        "validation_protocol": STRENGTHENING_VALIDATION_PROTOCOL,
        "status": "valid",
        "source_validation": (
            "All declared D1/D2 inputs were present and SHA-256 bound. "
            "Recovery paths and roles were rejected before report creation."
        ),
        "scientific_boundary": (
            "D1 is post-confirmatory robustness. D2 is descriptive backend "
            "comparability and is not a statistical equivalence test."
        ),
        "sources": sources,
        "parameters": {
            "edit_strength": dict(edit_strength["bootstrap_contract"]),
            "backend_comparability": dict(
                backend_comparability["parameters"]
            ),
        },
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
    manifest["manifest_sha256"] = _record_hash(
        manifest, excluded="manifest_sha256"
    )
    _atomic_write_text(
        output / "report_manifest.json", _json_text(manifest)
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--gguf-f16-export", type=Path, required=True)
    parser.add_argument("--gguf-q4-export", type=Path, required=True)
    parser.add_argument("--hf-fp16-export", type=Path, required=True)
    parser.add_argument("--hf-gptq-export", type=Path, required=True)
    parser.add_argument("--hf-fp16-audit", type=Path, required=True)
    parser.add_argument("--hf-gptq-audit", type=Path, required=True)
    parser.add_argument("--hf-pair-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--page-weight-cap", type=int, default=5)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    quartets = tuple(
        load_vitaminc_split(project_root, split="dose_subset")
    )
    features = tuple(extract_edit_features(item) for item in quartets)
    ordered_ids = tuple(item.case_id for item in quartets)
    expected_ids = set(ordered_ids)
    dose_split_sha256 = id_manifest_sha256(ordered_ids)

    gguf_f16_source = load_vitaminc_condition_export(
        args.gguf_f16_export
    )
    gguf_q4_source = load_vitaminc_condition_export(
        args.gguf_q4_export
    )
    gguf_f16 = derive_condition_subset(
        gguf_f16_source,
        case_ids=ordered_ids,
        split="dose_subset",
        split_sha256=dose_split_sha256,
    )
    gguf_q4 = derive_condition_subset(
        gguf_q4_source,
        case_ids=ordered_ids,
        split="dose_subset",
        split_sha256=dose_split_sha256,
    )
    gguf_pairs = pair_vitaminc_conditions(gguf_f16, gguf_q4)
    if (
        len(gguf_pairs) != len(expected_ids)
        or {pair.case_id for pair in gguf_pairs} != expected_ids
    ):
        raise ValueError(
            "GGUF strengthening pair does not match frozen dose subset"
        )

    loaded_hf = load_hf_vitaminc_pair(
        fp16_export=args.hf_fp16_export,
        gptq_export=args.hf_gptq_export,
        fp16_audit=args.hf_fp16_audit,
        gptq_audit=args.hf_gptq_audit,
        mode="formal",
    )
    saved_pair_gate = _load_json_object(args.hf_pair_gate)
    if saved_pair_gate != dict(loaded_hf.gate):
        raise ValueError(
            "saved HF formal pair gate does not match revalidation"
        )
    if {pair.case_id for pair in loaded_hf.pairs} != expected_ids:
        raise ValueError(
            "HF strengthening pair does not match frozen dose subset"
        )

    edit = analyze_edit_strength(
        gguf_pairs,
        features,
        draws=args.draws,
        seed=args.seed,
        page_weight_cap=args.page_weight_cap,
    )
    backend = compare_high_precision_backends(
        hf_fp16=loaded_hf.fp16,
        gguf_f16=gguf_f16,
        draws=args.draws,
        seed=args.seed,
    )
    sources = {
        "dose_subset_ids": (
            project_root
            / "data/manifests/vitaminc_dose_subset_ids.txt"
        ),
        "vitaminc_splits_summary": (
            project_root
            / "data/manifests/vitaminc_splits_summary.json"
        ),
        "gguf_f16_manifest": args.gguf_f16_export / "manifest.json",
        "gguf_f16_records": args.gguf_f16_export / "records.jsonl",
        "gguf_q4_manifest": args.gguf_q4_export / "manifest.json",
        "gguf_q4_records": args.gguf_q4_export / "records.jsonl",
        "hf_fp16_manifest": args.hf_fp16_export / "manifest.json",
        "hf_fp16_records": args.hf_fp16_export / "records.jsonl",
        "hf_gptq_manifest": args.hf_gptq_export / "manifest.json",
        "hf_gptq_records": args.hf_gptq_export / "records.jsonl",
        "hf_fp16_audit": args.hf_fp16_audit,
        "hf_gptq_audit": args.hf_gptq_audit,
        "hf_pair_gate": args.hf_pair_gate,
    }
    manifest = write_vitaminc_strengthening_report(
        edit_strength=edit,
        backend_comparability=backend,
        source_paths=sources,
        output_directory=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _markdown(results: Mapping[str, Any]) -> str:
    edit = results["edit_strength_robustness"]
    backend = results["backend_comparability"]
    edit_interval = edit["overall"]["page_bootstrap"]
    sensitivity = edit["high_volume_page_sensitivity"]
    leave_one_out = sensitivity["leave_one_page_out"]
    flags = edit["claim_narrowing_flags"]
    identity = backend["identity"]
    hard = backend["hard_labels"]
    interaction = backend["interaction"]
    endpoint_interval = interaction["hf_minus_gguf"]
    correlation = interaction["per_page_pearson_correlation"]
    correlation_text = (
        "undefined"
        if correlation is None
        else f"{correlation:.4f}"
    )
    return "\n".join(
        [
            "# VitaminC D1/D2 Strengthening Audit",
            "",
            "## Edit-strength robustness",
            "",
            (
                f"- Population: {edit['population']['quartets']:,} quartets "
                f"from {edit['population']['pages']:,} pages."
            ),
            (
                f"- Page-balanced effect: {edit_interval['estimate']:.4f} "
                f"[{edit_interval['lower']:.4f}, "
                f"{edit_interval['upper']:.4f}]."
            ),
            (
                f"- Capped page-weighted effect (cap "
                f"{sensitivity['page_weight_cap']}): "
                f"{sensitivity['capped_page_weighted_estimate']:.4f}."
            ),
            (
                f"- Leave-one-page-out range: "
                f"[{leave_one_out['minimum']:.4f}, "
                f"{leave_one_out['maximum']:.4f}]; crosses zero: "
                f"{str(leave_one_out['crosses_zero']).lower()}."
            ),
            (
                "- Claim-narrowing flags: "
                + ", ".join(
                    f"{name}={str(value).lower()}"
                    for name, value in sorted(flags.items())
                )
                + "."
            ),
            f"- Interpretation boundary: {edit['interpretation_boundary']}",
            "",
            "## Backend comparability",
            "",
            (
                f"- Model: {identity['model_key']}; "
                f"{identity['quartets']:,} quartets / "
                f"{identity['pages']:,} pages."
            ),
            (
                f"- Cell-level hard-label agreement: "
                f"{100 * hard['agreement']:.2f}%."
            ),
            (
                f"- Accuracy, HF FP16 versus GGUF F16: "
                f"{100 * hard['accuracy']['hf_fp16']:.2f}% versus "
                f"{100 * hard['accuracy']['gguf_f16']:.2f}%."
            ),
            (
                f"- Page-balanced interaction difference "
                f"(HF FP16 − GGUF F16): "
                f"{endpoint_interval['estimate']:.4f} "
                f"[{endpoint_interval['lower']:.4f}, "
                f"{endpoint_interval['upper']:.4f}]."
            ),
            (
                f"- Per-page interaction correlation: "
                f"{correlation_text}."
            ),
            f"- Interpretation boundary: {backend['interpretation_boundary']}",
            f"- Route boundary: {backend['route_claim_boundary']}",
            "",
            "## Decision boundary",
            "",
            (
                "- These analyses may narrow the manuscript claim, but they "
                "cannot replace the original confirmatory result."
            ),
            (
                "- Discovery label-mapping sensitivity and the Gemma second "
                "route remain gated on this report."
            ),
            "",
        ]
    )


def _mentions_recovery(value: str) -> bool:
    return "recovery" in value.casefold()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(
    value: Mapping[str, Any], *, excluded: str | None = None
) -> str:
    payload = {
        key: item for key, item in value.items() if key != excluded
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
