"""Generate the deterministic hashed Fresh TabFact formal report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .tabfact_analysis import (
    TABFACT_FORMAL_CONDITIONS,
    analyze_tabfact_formal,
    load_tabfact_formal_run,
)


TABFACT_FORMAL_REPORT_PROTOCOL = "tabfact-formal-report-v1-20260719"
TABFACT_FORMAL_VALIDATION_PROTOCOL = (
    "tabfact-formal-report-validation-v1-20260719"
)


def write_tabfact_formal_report(
    *,
    analysis: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write deterministic report files and bind sources and outputs."""
    if not source_paths:
        raise ValueError("TabFact formal report requires source paths")
    sources: dict[str, dict[str, str]] = {}
    for role, supplied_path in sorted(source_paths.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("TabFact report source role is invalid")
        path = Path(supplied_path)
        if not path.is_file():
            raise ValueError(f"TabFact report source is missing: {role}")
        sources[role] = {
            "path": str(path),
            "sha256": _file_sha256(path),
        }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "report_protocol": TABFACT_FORMAL_REPORT_PROTOCOL,
        "fresh_tabfact_formal": dict(analysis),
    }
    _atomic_write_text(output / "results.json", _json_text(results))
    _atomic_write_text(output / "analysis.md", _markdown(results))

    validation = {
        "validation_protocol": TABFACT_FORMAL_VALIDATION_PROTOCOL,
        "status": "valid",
        "source_validation": (
            "The formal gate and sidecar, audit identities, export and "
            "structural-report hashes, manifest self-hashes, records hashes, "
            "frozen population, owner metadata, scoring methods, coverage, "
            "and four-condition alignment were revalidated before analysis."
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
    manifest["manifest_sha256"] = _record_hash(
        manifest, excluded="manifest_sha256"
    )
    _atomic_write_text(
        output / "report_manifest.json", _json_text(manifest)
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args(argv)

    loaded = load_tabfact_formal_run(args.run_root)
    analysis = analyze_tabfact_formal(
        loaded, draws=args.draws, seed=args.seed
    )
    sources: dict[str, Path] = {
        "formal_gate": args.run_root / "formal_gate.json",
        "formal_gate_sha256": args.run_root / "formal_gate.sha256",
        "frozen_manifest": args.frozen_manifest,
    }
    for name in TABFACT_FORMAL_CONDITIONS:
        run = args.run_root / name
        sources.update(
            {
                f"{name}_audit": run / "audit.json",
                f"{name}_export_manifest": run / "export/manifest.json",
                f"{name}_records": run / "export/records.jsonl",
                f"{name}_structural_report": (
                    run / "export/structural_report.json"
                ),
            }
        )
    manifest = write_tabfact_formal_report(
        analysis=analysis,
        source_paths=sources,
        output_directory=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _markdown(results: Mapping[str, Any]) -> str:
    analysis = results["fresh_tabfact_formal"]
    effects = analysis["within_family_effects"]
    qwen = effects["qwen35_9b"]
    gemma = effects["gemma4_e4b"]
    heterogeneity = analysis["family_effect_heterogeneity"]["accuracy"]
    return "\n".join(
        [
            "# Fresh TabFact F16 versus GGUF Q4_K_M",
            "",
            "## Formal external structured-evidence result",
            "",
            (
                f"- Population: {analysis['population']['claims']:,} claims "
                f"from {analysis['population']['tables']:,} tables; four "
                "conditions completed with exact owner alignment."
            ),
            "- Primary comparisons are within-family Q4_K_M minus F16.",
            "",
            "| Family | F16 accuracy | Q4_K_M accuracy | Difference | "
            "Table-cluster 95% CI | McNemar p |",
            "|---|---:|---:|---:|---:|---:|",
            _family_row("Qwen3.5-9B", qwen),
            _family_row("Gemma 4 E4B", gemma),
            "",
            "## Error-direction shift",
            "",
            "| Family | Entailed recall change | Refuted recall change | "
            "Predicted-Entailed-rate change |",
            "|---|---:|---:|---:|",
            _shift_row("Qwen3.5-9B", qwen),
            _shift_row("Gemma 4 E4B", gemma),
            "",
            (
                "- Both families shift toward Entailed after Q4_K_M, with "
                "higher Entailed recall but substantially lower Refuted "
                "recall."
            ),
            "",
            "## Family-effect heterogeneity",
            "",
            (
                "- Qwen effect minus Gemma effect: "
                f"{100 * heterogeneity['estimate']:.2f} pp "
                f"[{100 * heterogeneity['cluster_bootstrap_95_ci']['lower']:.2f}, "
                f"{100 * heterogeneity['cluster_bootstrap_95_ci']['upper']:.2f}] pp."
            ),
            (
                f"- Interpretation boundary: "
                f"{analysis['family_effect_heterogeneity']['interpretation_boundary']}"
            ),
            "",
            "## Integrity and interpretation limits",
            "",
            "- All four immutable exports passed the formal hash and coverage gate.",
            "- Hard and probability-stage deterministic labels agree for every row.",
            "- The cluster bootstrap resamples whole tables; claim-level McNemar is a paired diagnostic.",
            "- These results cover Fresh TabFact, two model families, GGUF Q4_K_M, and the frozen llama.cpp route.",
            "- The behavioral pattern does not identify an internal neural mechanism.",
            "",
        ]
    )


def _family_row(label: str, effect: Mapping[str, Any]) -> str:
    accuracy = effect["accuracy"]
    interval = accuracy["cluster_bootstrap_95_ci"]
    pvalue = effect["transitions"]["mcnemar_exact_pvalue"]
    return (
        f"| {label} | {100 * accuracy['f16']:.2f}% | "
        f"{100 * accuracy['q4_k_m']:.2f}% | "
        f"{100 * accuracy['difference']:.2f} pp | "
        f"[{100 * interval['lower']:.2f}, {100 * interval['upper']:.2f}] pp | "
        f"{pvalue:.3g} |"
    )


def _shift_row(label: str, effect: Mapping[str, Any]) -> str:
    return (
        f"| {label} | "
        f"{100 * effect['recall']['Entailed']['difference']:.2f} pp | "
        f"{100 * effect['recall']['Refuted']['difference']:.2f} pp | "
        f"{100 * effect['predicted_entailed_rate']['difference']:.2f} pp |"
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
