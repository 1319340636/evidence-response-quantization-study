"""Re-estimate the three-model scale-invariant family from text-free records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from qer_fv.three_model_records import load_text_free_three_model_inputs
from qer_fv.three_model_scale_heterogeneity import (
    analyze_three_model_scale_heterogeneity,
)


CORE_KEYS = (
    "analysis_protocol",
    "estimand",
    "population",
    "bootstrap_contract",
    "model_order",
    "pair_order",
    "contrast_order",
    "endpoint_order",
    "within_model_effects",
    "interactions",
    "holm_family",
    "causal_boundary",
)


def _equivalent(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _equivalent(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _equivalent(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, float):
        if not math.isfinite(actual) or not math.isfinite(expected):
            return actual == expected
        if actual == 0.0 or expected == 0.0:
            return actual == expected
        return math.isclose(actual, expected, rel_tol=5e-15, abs_tol=0.0)
    return actual == expected


def _verified_reference(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = path.with_name(path.name + ".sha256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fields = sidecar.read_text(encoding="ascii").split()
    if not fields or fields[0] != digest:
        raise ValueError("three-model reference integrity mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("three-model reference analysis is missing")
    return analysis, digest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-records", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    model_values, pages = load_text_free_three_model_inputs(
        args.analysis_records,
        formal=True,
    )
    reestimated = analyze_three_model_scale_heterogeneity(
        model_values=model_values,
        pages=pages,
    )
    reference, reference_sha256 = _verified_reference(args.reference_result)
    expected = {key: reference[key] for key in CORE_KEYS}
    if not _equivalent(reestimated, expected):
        raise RuntimeError(
            "three-model text-free re-estimation differs beyond machine precision"
        )

    manifest_path = args.analysis_records / "analysis_records_manifest.json"
    record = {
        "analysis": reestimated,
        "reference_result_sha256": reference_sha256,
        "analysis_records_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    record["analysis_record_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output.with_name(args.output.name + ".sha256").write_text(
        hashlib.sha256(args.output.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
