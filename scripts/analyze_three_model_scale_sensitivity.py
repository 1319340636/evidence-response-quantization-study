"""Run post-review three-model scale and weighting sensitivity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from qer_fv.three_model_records import load_text_free_three_model_inputs
from qer_fv.three_model_scale_sensitivity import (
    analyze_three_model_scale_sensitivity,
    write_three_model_scale_sensitivity,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args(argv)

    model_values, pages = load_text_free_three_model_inputs(
        args.analysis_records,
        formal=True,
    )
    result = analyze_three_model_scale_sensitivity(
        model_values=model_values,
        pages=pages,
        draws=args.draws,
        seed=args.seed,
    )
    manifest = args.analysis_records / "analysis_records_manifest.json"
    records = args.analysis_records / "analysis_records.jsonl"
    write_three_model_scale_sensitivity(
        result,
        args.output,
        source_files={
            "analysis_records_manifest": manifest,
            "analysis_records": records,
        },
        formal=args.draws == 10_000 and args.seed == 20260711,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
