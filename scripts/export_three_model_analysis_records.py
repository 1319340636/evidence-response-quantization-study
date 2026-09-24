"""Export the frozen nine-condition HF release without benchmark text."""

from __future__ import annotations

import argparse
from pathlib import Path

from qer_fv.three_model_records import write_text_free_three_model_records


RELATIVE_EXPORTS = {
    "qwen_fp16": "runs/qwen_hf_formal_v1/qwen35_9b_FP16/export",
    "qwen_gptq": "runs/qwen_hf_formal_v1/qwen35_9b_GPTQ_INT4/export",
    "qwen_awq": "runs/qwen_hf_triplet_formal_v1/qwen35_9b_AWQ_INT4/export",
    "ministral_fp16": (
        "runs/ministral_hf_triplet_formal_v1/ministral3_8b_FP16/export"
    ),
    "ministral_gptq": (
        "runs/ministral_hf_triplet_formal_v1/ministral3_8b_GPTQ_INT4/export"
    ),
    "ministral_awq": (
        "runs/ministral_hf_triplet_formal_v1/ministral3_8b_AWQ_INT4/export"
    ),
    "olmo_fp16": "runs/olmo3_hf_triplet_formal_v1/olmo3_7b_FP16/export",
    "olmo_gptq": "runs/olmo3_hf_triplet_formal_v1/olmo3_7b_GPTQ_INT4/export",
    "olmo_awq": "runs/olmo3_hf_triplet_formal_v1/olmo3_7b_AWQ_INT4/export",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    write_text_free_three_model_records(
        exports={key: root / path for key, path in RELATIVE_EXPORTS.items()},
        output_directory=args.output,
        formal=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
