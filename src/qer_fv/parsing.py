"""Strict parsing for frozen single-letter fact-verification outputs."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ParseResult:
    raw_text: Any
    normalized_text: str | None
    valid: bool
    choice: str | None
    label: str | None
    failure_reason: str | None


def _validate_choice_map(choice_to_label: Mapping[str, str]) -> None:
    if not isinstance(choice_to_label, Mapping) or not choice_to_label:
        raise ValueError("choice map must be a nonempty mapping")
    for choice, label in choice_to_label.items():
        if (
            not isinstance(choice, str)
            or len(choice) != 1
            or choice not in string.ascii_uppercase
            or not isinstance(label, str)
            or not label.strip()
        ):
            raise ValueError("choice map must contain uppercase ASCII letters and labels")


def parse_choice(
    raw_text: Any, choice_to_label: Mapping[str, str]
) -> ParseResult:
    """Parse exactly one allowed ASCII choice after case/whitespace normalization."""
    _validate_choice_map(choice_to_label)
    if not isinstance(raw_text, str):
        return ParseResult(
            raw_text, None, False, None, None, "invalid_response_type"
        )
    normalized = raw_text.strip().upper()
    if not normalized:
        return ParseResult(
            raw_text, normalized, False, None, None, "empty_response"
        )
    if len(normalized) != 1:
        return ParseResult(
            raw_text, normalized, False, None, None, "not_single_choice"
        )
    if normalized not in choice_to_label:
        return ParseResult(
            raw_text, normalized, False, None, None, "disallowed_choice"
        )
    return ParseResult(
        raw_text=raw_text,
        normalized_text=normalized,
        valid=True,
        choice=normalized,
        label=choice_to_label[normalized],
        failure_reason=None,
    )
