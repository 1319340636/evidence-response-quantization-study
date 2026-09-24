"""Multiplicity corrections used by frozen Experiment B families."""

from __future__ import annotations

import math
from typing import Mapping


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    if not pvalues:
        raise ValueError("pvalues must be nonempty")
    ordered = []
    for key, value in pvalues.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("pvalues must be finite numbers in [0, 1]")
        ordered.append((str(key), float(value)))
    ordered.sort(key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (key, value) in enumerate(ordered):
        running = max(running, (count - index) * value)
        adjusted[key] = min(1.0, running)
    return adjusted
