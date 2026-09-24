"""Rescore immutable pilot JSONL artifacts with deterministic argmax labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .prompts import load_prompt_contract
from .scoring import HARD_LABEL_METHOD, score_choice_argmax


def rescore_pilot_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    prompt_contract_path: str | Path,
) -> dict[str, int]:
    """Validate and rescore a JSONL artifact without modifying the source."""
    source = Path(input_path)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    contract = load_prompt_contract(prompt_contract_path)
    rescored: list[dict[str, Any]] = []
    disagreements = 0
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"line {line_number} must contain a JSON object")
            dataset = row.get("dataset")
            control = row.get("control")
            if not isinstance(dataset, str) or not isinstance(control, str):
                raise ValueError(
                    f"line {line_number} requires string dataset and control"
                )
            choice_logprobs = row.get("choice_logprobs")
            if not isinstance(choice_logprobs, Mapping):
                raise ValueError(
                    f"line {line_number} choice_logprobs must be an object"
                )

            choice_to_label = contract.choice_map(dataset, control=control)
            scored = score_choice_argmax(choice_to_label, choice_logprobs)
            sampled_choice = row.get("sampled_choice", row.get("parsed_choice"))
            sampled_label = row.get("sampled_label", row.get("parsed_label"))
            if sampled_choice != scored.choice:
                disagreements += 1
            updated = dict(row)
            updated.update(
                {
                    "sampled_choice": sampled_choice,
                    "sampled_label": sampled_label,
                    "scored_choice": scored.choice,
                    "scored_label": scored.label,
                    "hard_label_method": HARD_LABEL_METHOD,
                }
            )
            rescored.append(updated)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for record in rescored:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "records": len(rescored),
        "sampled_scored_disagreements": disagreements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-contract", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = rescore_pilot_jsonl(
        input_path=args.input,
        output_path=args.output,
        prompt_contract_path=args.prompt_contract,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

