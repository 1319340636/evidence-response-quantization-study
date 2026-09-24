"""Deterministic hard-label scoring over frozen choice log-probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


HARD_LABEL_METHOD = "choice-logprob-argmax-v1"


@dataclass(frozen=True)
class HardLabelScore:
    """One deterministic closed-set decision."""

    choice: str
    label: str


def score_choice_argmax(
    choice_to_label: Mapping[str, str],
    choice_logprobs: Mapping[str, float],
) -> HardLabelScore:
    """Return the first maximum in the supplied frozen choice order."""
    choices = tuple(choice_to_label)
    if not choices or set(choice_logprobs) != set(choices):
        raise ValueError("choice_logprobs must match allowed choices exactly")

    values: list[float] = []
    for choice in choices:
        value = choice_logprobs[choice]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("choice log-probabilities must be finite real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("choice log-probabilities must be finite real numbers")
        values.append(number)

    winner = max(range(len(choices)), key=values.__getitem__)
    choice = choices[winner]
    return HardLabelScore(choice=choice, label=choice_to_label[choice])

