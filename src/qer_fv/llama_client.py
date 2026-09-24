"""Small standard-library client for llama-server audit runs."""

from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class LlamaServerError(RuntimeError):
    """Raised when llama-server cannot provide an auditable response."""


CHOICE_LOGIT_BIAS = 100.0
CHOICE_SCORING_METHOD = "equal-logit-bias-v1"
SELECTED_TOKEN_LOGPROB_METHOD = (
    "pre-sampling-full-vocabulary-double-logsumexp-v1"
)


@dataclass(frozen=True)
class TokenPiece:
    id: int
    piece: Any


@dataclass(frozen=True)
class ChoiceCompletion:
    content: str
    choice_logprobs: Mapping[str, float]
    raw_response: Mapping[str, Any]


@dataclass(frozen=True)
class TokenProbabilityProbe:
    token_id: int
    bias: float
    biased_probability: float
    unbiased_probability: float
    raw_response: Mapping[str, Any]


@dataclass(frozen=True)
class SelectedTokenScores:
    token_ids: tuple[int, ...]
    logprobs: Mapping[int, float]
    method: str
    raw_response: Mapping[str, Any]


def recover_unbiased_probability(
    biased_probability: float, bias: float
) -> float:
    """Invert a known additive logit bias applied to one vocabulary token."""
    if (
        isinstance(biased_probability, bool)
        or not isinstance(biased_probability, (int, float))
        or not math.isfinite(biased_probability)
        or not 0.0 < biased_probability < 1.0
    ):
        raise ValueError("biased_probability must be finite and in (0, 1)")
    if (
        isinstance(bias, bool)
        or not isinstance(bias, (int, float))
        or not math.isfinite(bias)
    ):
        raise ValueError("bias must be finite")
    probability = float(biased_probability)
    logit = math.log(probability) - math.log1p(-probability) - float(bias)
    if logit >= 0.0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def choice_grammar(choices: Sequence[str]) -> str:
    normalized = tuple(choices)
    if not normalized:
        raise ValueError("choices must be nonempty")
    for choice in normalized:
        if (
            not isinstance(choice, str)
            or len(choice) != 1
            or choice not in string.ascii_uppercase
        ):
            raise ValueError("choices must be uppercase ASCII single letters")
    return "root ::= " + " | ".join(json.dumps(choice) for choice in normalized)


class LlamaServerClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a nonempty string")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def props(self) -> Mapping[str, Any]:
        return self._get("/props")

    def tokenize(
        self,
        content: str,
        *,
        add_special: bool = False,
        parse_special: bool = True,
        with_pieces: bool = True,
    ) -> list[TokenPiece]:
        response = self._post(
            "/tokenize",
            {
                "content": content,
                "add_special": add_special,
                "parse_special": parse_special,
                "with_pieces": with_pieces,
            },
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise LlamaServerError("/tokenize response missing tokens")
        result: list[TokenPiece] = []
        for token in tokens:
            if not isinstance(token, Mapping) or "id" not in token or "piece" not in token:
                raise LlamaServerError("/tokenize token missing id or piece")
            result.append(TokenPiece(id=int(token["id"]), piece=token["piece"]))
        return result

    def apply_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> str:
        response = self._post(
            "/apply-template",
            {
                "messages": [dict(message) for message in messages],
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
        )
        prompt = response.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise LlamaServerError("/apply-template response missing prompt")
        return prompt

    def complete_choice(
        self,
        prompt: str,
        *,
        choices: Sequence[str],
        choice_token_ids: Mapping[str, int],
        seed: int = 0,
    ) -> ChoiceCompletion:
        logit_bias = _choice_logit_bias(choices, choice_token_ids)
        response = self._post(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "stream": False,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "seed": seed,
                "cache_prompt": False,
                "grammar": choice_grammar(choices),
                "n_probs": len(tuple(choices)),
                "post_sampling_probs": True,
                "return_tokens": True,
                "samplers": ["temperature"],
                "logit_bias": logit_bias,
            },
        )
        content = response.get("content")
        if not isinstance(content, str):
            raise LlamaServerError("/completion response missing content")
        return ChoiceCompletion(
            content=content,
            choice_logprobs=_choice_logprobs(response, choices),
            raw_response=response,
        )

    def probe_token_probability(
        self,
        prompt: str,
        *,
        token_id: int,
        bias: float,
        seed: int = 0,
    ) -> TokenProbabilityProbe:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("token_id must be a non-negative integer")
        if (
            isinstance(bias, bool)
            or not isinstance(bias, (int, float))
            or not math.isfinite(bias)
        ):
            raise ValueError("bias must be finite")
        normalized_bias = float(bias)
        response = self._post(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "stream": False,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "seed": seed,
                "cache_prompt": False,
                "n_probs": 5,
                "post_sampling_probs": True,
                "return_tokens": True,
                "samplers": ["temperature"],
                "logit_bias": [[token_id, normalized_bias]],
            },
        )
        biased_probability = _token_probability(response, token_id)
        return TokenProbabilityProbe(
            token_id=token_id,
            bias=normalized_bias,
            biased_probability=biased_probability,
            unbiased_probability=recover_unbiased_probability(
                biased_probability, normalized_bias
            ),
            raw_response=response,
        )

    def score_selected_tokens(
        self,
        prompt: str,
        *,
        token_ids: Sequence[int],
        seed: int = 0,
    ) -> SelectedTokenScores:
        normalized = _selected_token_ids(token_ids)
        response = self._post(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "stream": False,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "seed": seed,
                "cache_prompt": False,
                "backend_sampling": False,
                "post_sampling_probs": False,
                "selected_token_ids": list(normalized),
            },
        )
        method = response.get("selected_token_logprob_method")
        if method != SELECTED_TOKEN_LOGPROB_METHOD:
            raise LlamaServerError("selected token logprob method mismatch")
        entries = response.get("selected_token_logprobs")
        if not isinstance(entries, list) or len(entries) != len(normalized):
            raise LlamaServerError("selected token logprob response order mismatch")
        logprobs: dict[int, float] = {}
        for expected_id, entry in zip(normalized, entries, strict=True):
            if (
                not isinstance(entry, Mapping)
                or isinstance(entry.get("id"), bool)
                or entry.get("id") != expected_id
            ):
                raise LlamaServerError(
                    "selected token logprob response order mismatch"
                )
            value = entry.get("logprob")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise LlamaServerError("selected token logprob must be finite")
            logprob = float(value)
            if logprob > 0.0:
                raise LlamaServerError(
                    "selected token logprob must be non-positive"
                )
            logprobs[expected_id] = logprob
        return SelectedTokenScores(
            token_ids=normalized,
            logprobs=logprobs,
            method=method,
            raw_response=response,
        )

    def _get(self, path: str) -> Mapping[str, Any]:
        return self._json("GET", path, None)

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._json("POST", path, payload)

    def _json(
        self, method: str, path: str, payload: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            raise LlamaServerError(f"llama-server request failed: {path}") from error
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LlamaServerError(f"llama-server returned non-JSON: {path}") from error
        if not isinstance(value, Mapping):
            raise LlamaServerError(f"llama-server returned non-object JSON: {path}")
        return value


def _choice_logit_bias(
    choices: Sequence[str], choice_token_ids: Mapping[str, int]
) -> list[list[int | float]]:
    wanted = tuple(choices)
    if set(choice_token_ids) != set(wanted):
        raise ValueError("choice_token_ids must match choices exactly")
    token_ids = [choice_token_ids[choice] for choice in wanted]
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError("choice token IDs must be non-negative integers")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("choice token IDs must be unique")
    return [[token_id, CHOICE_LOGIT_BIAS] for token_id in token_ids]


def _selected_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)):
        raise ValueError("token_ids must be a nonempty sequence of integers")
    try:
        normalized = tuple(token_ids)
    except TypeError as error:
        raise ValueError(
            "token_ids must be a nonempty sequence of integers"
        ) from error
    if not normalized or any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in normalized
    ):
        raise ValueError("token_ids must be a nonempty sequence of integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("token_ids must be unique")
    return normalized


def _choice_logprobs(
    response: Mapping[str, Any], choices: Sequence[str]
) -> dict[str, float]:
    wanted = tuple(choices)
    probabilities = response.get("completion_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise LlamaServerError("/completion response missing completion_probabilities")
    first = probabilities[0]
    if not isinstance(first, Mapping):
        raise LlamaServerError("/completion probability entry must be an object")
    top_probs = first.get("top_probs")
    if not isinstance(top_probs, list):
        raise LlamaServerError("/completion response missing top_probs")
    found: dict[str, float] = {}
    for item in top_probs:
        if not isinstance(item, Mapping):
            continue
        token = str(item.get("token", "")).strip().upper()
        if token not in wanted:
            continue
        probability = float(item.get("prob", 0.0))
        if probability <= 0.0:
            raise LlamaServerError(f"non-positive probability for choice {token}")
        found[token] = math.log(probability)
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise LlamaServerError(f"missing constrained probabilities for choices: {missing}")
    return found


def _token_probability(response: Mapping[str, Any], token_id: int) -> float:
    probabilities = response.get("completion_probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        raise LlamaServerError("/completion response missing completion_probabilities")
    first = probabilities[0]
    if not isinstance(first, Mapping):
        raise LlamaServerError("/completion probability entry must be an object")
    top_probs = first.get("top_probs")
    if not isinstance(top_probs, list):
        raise LlamaServerError("/completion response missing top_probs")
    matches = [
        item
        for item in top_probs
        if isinstance(item, Mapping)
        and not isinstance(item.get("id"), bool)
        and item.get("id") == token_id
    ]
    if len(matches) != 1:
        raise LlamaServerError(
            f"expected one probability for target token ID {token_id}, got {len(matches)}"
        )
    try:
        probability = float(matches[0]["prob"])
    except (KeyError, TypeError, ValueError) as error:
        raise LlamaServerError(
            f"invalid probability for target token ID {token_id}"
        ) from error
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise LlamaServerError(
            f"target token ID {token_id} probability must be finite and in (0, 1)"
        )
    return probability
