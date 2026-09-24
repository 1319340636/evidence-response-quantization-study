"""Shared, auditable calibration rendering for both OLMo 3 INT4 routes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .awq_route import normalize_chat_template_token_ids
from .prompts import PromptInput, load_prompt_contract, render_prompt


def render_olmo3_calibration(
    project_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Render the frozen 256 rows once for both GPTQ and AWQ."""
    contract = load_prompt_contract(Path(project_root) / "configs/prompt_contract_v2.json")
    text_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for row in rows:
        unique_id = row.get("unique_id")
        if not isinstance(unique_id, str) or not unique_id:
            raise ValueError("calibration unique_id is invalid")
        rendered = render_prompt(
            contract,
            PromptInput(
                dataset="vitaminc",
                sample_id=unique_id,
                claim=str(row.get("claim", "")),
                evidence=str(row.get("evidence", "")),
            ),
            control="full",
        )
        kwargs = {"add_generation_prompt": False, "enable_thinking": False}
        text = tokenizer.apply_chat_template(
            rendered.messages, tokenize=False, **kwargs
        )
        input_ids = normalize_chat_template_token_ids(
            tokenizer.apply_chat_template(
                rendered.messages, tokenize=True, **kwargs
            )
        )
        if not isinstance(text, str) or not text or not input_ids:
            raise ValueError("OLMo calibration chat-template rendering is invalid")
        text_rows.append({"unique_id": unique_id, "text": text})
        token_rows.append({"unique_id": unique_id, "input_ids": input_ids})
        texts.append(text)
    return text_rows, token_rows, texts


def write_jsonl_atomic(
    path: str | Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
