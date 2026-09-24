"""Run one frozen VitaminC D3 mapping-sensitivity cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hf_runtime import HFNextTokenRuntime
from .hf_vitaminc_runner import run_hf_vitaminc_units
from .llama_client import LlamaServerClient
from .vitaminc_mapping_protocol import (
    D3_PROTOCOL_VERSION,
    load_mapping_protocol,
    load_mapping_units,
)
from .vitaminc_runner_v4 import run_vitaminc_units_v4


_BACKENDS = frozenset({"gguf", "hf"})
_MAPPINGS = frozenset({"original", "reversed"})
_POPULATIONS = frozenset({"engineering", "formal"})


def validate_mapping_cell_request(
    *,
    backend: str,
    mapping_variant: str,
    population: str,
) -> None:
    if (
        backend not in _BACKENDS
        or mapping_variant not in _MAPPINGS
        or population not in _POPULATIONS
    ):
        raise ValueError("unsupported D3 mapping cell request")


def build_mapping_run_protocol(
    *,
    backend: str,
    mapping_variant: str,
    mapping_config_sha256: str,
) -> str:
    validate_mapping_cell_request(
        backend=backend,
        mapping_variant=mapping_variant,
        population="engineering",
    )
    if (
        not isinstance(mapping_config_sha256, str)
        or len(mapping_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in mapping_config_sha256
        )
    ):
        raise ValueError("D3 mapping config SHA-256 is invalid")
    return (
        f"{D3_PROTOCOL_VERSION}:{backend}:{mapping_variant}:"
        f"{mapping_config_sha256}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--backend", choices=sorted(_BACKENDS), required=True)
    parser.add_argument(
        "--mapping-variant", choices=sorted(_MAPPINGS), required=True
    )
    parser.add_argument(
        "--population", choices=sorted(_POPULATIONS), required=True
    )
    parser.add_argument("--audit-certificate", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path, required=True)
    parser.add_argument("--server")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--precision", choices=("FP16", "GPTQ_INT4"))
    args = parser.parse_args(argv)

    validate_mapping_cell_request(
        backend=args.backend,
        mapping_variant=args.mapping_variant,
        population=args.population,
    )
    root = args.project_root.resolve()
    frozen = load_mapping_protocol(root)
    units, population = load_mapping_units(
        root, population=args.population
    )
    run_protocol = build_mapping_run_protocol(
        backend=args.backend,
        mapping_variant=args.mapping_variant,
        mapping_config_sha256=frozen.config_sha256,
    )

    if args.backend == "gguf":
        if not args.server or args.model_path is not None or args.precision:
            parser.error(
                "GGUF D3 requires --server and forbids HF model arguments"
            )
        summary = run_vitaminc_units_v4(
            project_root=root,
            units=units,
            split="discovery",
            split_sha256=population.case_ids_sha256,
            client=LlamaServerClient(args.server),
            audit_certificate_path=args.audit_certificate,
            database_path=args.database,
            export_directory=args.export_directory,
            mapping_variant=args.mapping_variant,
            protocol_version=run_protocol,
        )
    else:
        if (
            args.server is not None
            or args.model_path is None
            or args.precision is None
        ):
            parser.error(
                "HF D3 requires --model-path and --precision and forbids --server"
            )
        runtime = HFNextTokenRuntime.from_pretrained(
            str(args.model_path.resolve()), args.precision
        )
        summary = run_hf_vitaminc_units(
            project_root=root,
            units=units,
            split="discovery",
            split_sha256=population.case_ids_sha256,
            runtime=runtime,
            audit_certificate_path=args.audit_certificate,
            database_path=args.database,
            export_directory=args.export_directory,
            mapping_variant=args.mapping_variant,
            protocol_version=run_protocol,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

