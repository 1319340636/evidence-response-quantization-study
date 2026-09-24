"""Frozen float32-safe probability recovery shared by CUB audit and runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .llama_client import LlamaServerError, TokenProbabilityProbe


BIAS_CANDIDATES = (0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0)
LOG_PROBABILITY_TOLERANCE = 1e-4
USABLE_PROBABILITY_MIN = 1e-4
USABLE_PROBABILITY_MAX = 0.999


class StableProbabilityError(RuntimeError):
    """Raised when two reliable probability probes cannot be recovered."""


class TokenProbabilityClient(Protocol):
    def probe_token_probability(
        self, prompt: str, *, token_id: int, bias: float, seed: int = 0
    ) -> TokenProbabilityProbe: ...


@dataclass(frozen=True)
class StableTokenProbability:
    token_id: int
    probability: float
    selected_bias: float
    probes: tuple[TokenProbabilityProbe, TokenProbabilityProbe]
    recovered_logprob_range: tuple[float, float]


def recover_stable_token_probability(
    client: TokenProbabilityClient,
    prompt: str,
    *,
    token_id: int,
    seed: int = 0,
) -> StableTokenProbability:
    """Recover one point estimate and an independent validation replicate."""
    usable: list[TokenProbabilityProbe] = []
    for bias in BIAS_CANDIDATES:
        try:
            probe = client.probe_token_probability(
                prompt, token_id=token_id, bias=bias, seed=seed
            )
        except LlamaServerError:
            continue
        if probe.token_id != token_id or probe.bias != bias:
            raise StableProbabilityError("probability probe identity mismatch")
        if not (
            USABLE_PROBABILITY_MIN
            <= probe.biased_probability
            <= USABLE_PROBABILITY_MAX
        ):
            continue
        probability = probe.unbiased_probability
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            continue
        usable.append(probe)
        if len(usable) == 2:
            break

    if len(usable) < 2:
        raise StableProbabilityError("fewer than two usable probability probes")
    log_values = [math.log(probe.unbiased_probability) for probe in usable]
    recovered_range = (min(log_values), max(log_values))
    if recovered_range[1] - recovered_range[0] > LOG_PROBABILITY_TOLERANCE:
        raise StableProbabilityError("inconsistent recovered log-probability")
    first, second = usable
    return StableTokenProbability(
        token_id=token_id,
        probability=first.unbiased_probability,
        selected_bias=first.bias,
        probes=(first, second),
        recovered_logprob_range=recovered_range,
    )
