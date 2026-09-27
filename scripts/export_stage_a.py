"""Export safe Stage A records from the eight already-completed GGUF runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from qer_fv.stage_a_release import write_stage_a_text_free


RELATIVE_EXPORTS = {
    "gemma_f16": "vitaminc_crossfamily_extension_v4/formal/gemma4_e4b_F16/export",
    "gemma_q4": "vitaminc_crossfamily_extension_v4/formal/gemma4_e4b_Q4_K_M/export",
    "olmo_f16": "vitaminc_crossfamily_extension_v4/formal/olmo3_7b_F16/export",
    "olmo_q4": "vitaminc_crossfamily_extension_v4/formal/olmo3_7b_Q4_K_M/export",
    "ministral_f16": "vitaminc_crossfamily_v4/ministral3_8b/ministral3_8b_F16/export",
    "ministral_q4": "vitaminc_crossfamily_v4/ministral3_8b/ministral3_8b_Q4_K_M/export",
    "qwen_f16": "vitaminc_main_v4/qwen35_9b_F16/export",
    "qwen_q4": "vitaminc_main_v4/qwen35_9b_Q4_K_M/export",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--subset-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = write_stage_a_text_free(
        {key: args.runs_root / value for key, value in RELATIVE_EXPORTS.items()},
        args.subset_ids,
        args.output,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
