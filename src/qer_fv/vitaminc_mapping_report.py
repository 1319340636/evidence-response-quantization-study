"""Write deterministic, source-bound VitaminC D3 reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .vitaminc_mapping_analysis import (
    D3_ANALYSIS_PROTOCOL,
    analyze_mapping_run,
)
from .vitaminc_mapping_protocol import (
    load_mapping_protocol,
    require_d3_path,
)


D3_REPORT_PROTOCOL = "vitaminc-mapping-report-v1-20260719"
_OUTPUTS = (
    "analysis.md",
    "results.json",
    "validation.json",
    "report_manifest.json",
)


def write_mapping_report(
    *,
    analysis: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
    output_directory: str | Path,
) -> dict[str, Any]:
    if analysis.get("analysis_protocol") != D3_ANALYSIS_PROTOCOL:
        raise ValueError("D3 analysis protocol mismatch")
    if analysis.get("inference_scope") != "post_hoc_prompt_robustness":
        raise ValueError("D3 inference scope mismatch")
    if not source_paths:
        raise ValueError("D3 report requires sources")
    sources: dict[str, dict[str, str]] = {}
    for role, supplied in sorted(source_paths.items()):
        require_d3_path(role)
        path = require_d3_path(supplied)
        if not path.is_file():
            raise ValueError(f"D3 report source is missing: {role}")
        sources[role] = {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
    output = require_d3_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in _OUTPUTS):
        raise ValueError("D3 report output is not immutable-empty")
    results = {
        "report_protocol": D3_REPORT_PROTOCOL,
        **dict(analysis),
    }
    _atomic_write(output / "results.json", _json_text(results))
    _atomic_write(output / "analysis.md", _markdown(results))
    validation = {
        "validation_protocol": (
            "vitaminc-mapping-validation-v1-20260719"
        ),
        "status": "valid",
        "scientific_boundary": (
            "D3 is post-hoc prompt robustness. It cannot select the "
            "primary mapping or rewrite confirmatory H1."
        ),
        "sources": sources,
        "parameters": dict(analysis["parameters"]),
        "output_files": {
            "analysis.md": _file_sha256(output / "analysis.md"),
            "results.json": _file_sha256(output / "results.json"),
        },
    }
    _atomic_write(output / "validation.json", _json_text(validation))
    manifest = {
        "manifest_protocol": "hashed-report-manifest-v1",
        "files": {
            name: _file_sha256(output / name)
            for name in ("analysis.md", "results.json", "validation.json")
        },
    }
    manifest["manifest_sha256"] = _record_hash(manifest)
    _atomic_write(
        output / "report_manifest.json", _json_text(manifest)
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--engineering-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args(argv)
    analysis = analyze_mapping_run(
        project_root=args.project_root,
        run_root=args.run_root,
        draws=args.draws,
        seed=args.seed,
    )
    frozen = load_mapping_protocol(args.project_root)
    sources: dict[str, Path] = {
        "mapping_config": (
            args.project_root
            / "configs/vitaminc_mapping_sensitivity_v1.json"
        ),
        "discovery_ids": (
            args.project_root
            / "data/manifests/vitaminc_discovery_ids.txt"
        ),
        "engineering_gate": args.engineering_gate,
    }
    for condition in frozen.conditions:
        for mapping in ("original", "reversed"):
            prefix = f"{condition}__{mapping}"
            cell = args.run_root / prefix
            sources[f"{prefix}_manifest"] = cell / "export/manifest.json"
            sources[f"{prefix}_records"] = cell / "export/records.jsonl"
            sources[f"{prefix}_audit"] = cell / "audit.json"
    manifest = write_mapping_report(
        analysis=analysis,
        source_paths=sources,
        output_directory=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _markdown(results: Mapping[str, Any]) -> str:
    lines = [
        "# VitaminC D3 Label-Mapping Sensitivity",
        "",
        "## Route results",
        "",
    ]
    for route, result in sorted(results["routes"].items()):
        original = result["original"]["page_bootstrap"]
        reversed_value = result["reversed"]["page_bootstrap"]
        difference = result["reversed_minus_original"]["page_bootstrap"]
        lines.extend(
            [
                f"### {route}",
                "",
                (
                    f"- Original mapping: {original['estimate']:.4f} "
                    f"[{original['lower']:.4f}, {original['upper']:.4f}]."
                ),
                (
                    f"- Reversed mapping: {reversed_value['estimate']:.4f} "
                    f"[{reversed_value['lower']:.4f}, "
                    f"{reversed_value['upper']:.4f}]."
                ),
                (
                    f"- Reversed minus original: "
                    f"{difference['estimate']:.4f} "
                    f"[{difference['lower']:.4f}, "
                    f"{difference['upper']:.4f}]."
                ),
                f"- Direction stable: {str(result['direction_stable']).lower()}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "- Mapping sensitivity cannot select the primary mapping or "
                "rewrite the original confirmatory hypothesis."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())

