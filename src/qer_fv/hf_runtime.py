"""Dependency-light Hugging Face chat-template and next-token runtime."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


HF_DIRECT_LOGIT_METHOD = "hf-full-vocabulary-next-token-logsoftmax-v1"
HF_GPTQ_BACKEND = "gptq_torch"
HF_AWQ_BACKEND = "transformers_compressed_tensors"
GEMMA4_GPTQ_DEVICE_MAP = {
    "model.language_model": "cuda:0",
    "lm_head": "cuda:0",
    "model.vision_tower": "cuda:1",
    "model.audio_tower": "cuda:1",
    "model.embed_vision": "cuda:1",
    "model.embed_audio": "cuda:1",
}


def _qwen35_awq_device_map() -> dict[str, str]:
    """Keep the unused vision tower off GPU and split text layers evenly."""
    device_map = {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cuda:0",
        "model.language_model.rotary_emb": "cuda:0",
        "model.language_model.norm": "cuda:1",
        "lm_head": "cuda:1",
    }
    device_map.update(
        {
            f"model.language_model.layers.{index}": (
                "cuda:0" if index < 16 else "cuda:1"
            )
            for index in range(32)
        }
    )
    return device_map


def _ministral3_awq_device_map() -> dict[str, str]:
    """Keep unused multimodal modules off GPU and split text layers evenly."""
    device_map = {
        "model.vision_tower": "cpu",
        "model.multi_modal_projector": "cpu",
        "model.language_model.embed_tokens": "cuda:0",
        "model.language_model.rotary_emb": "cuda:0",
        "model.language_model.norm": "cuda:1",
        "lm_head": "cuda:1",
    }
    device_map.update(
        {
            f"model.language_model.layers.{index}": (
                "cuda:0" if index < 17 else "cuda:1"
            )
            for index in range(34)
        }
    )
    return device_map


def _olmo3_awq_device_map() -> dict[str, str]:
    """Split OLMo3 while reserving decompression headroom on the output GPU."""
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.rotary_emb": "cuda:0",
        "model.norm": "cuda:1",
        "lm_head": "cuda:1",
    }
    device_map.update(
        {
            f"model.layers.{index}": (
                "cuda:0" if index < 17 else "cuda:1"
            )
            for index in range(32)
        }
    )
    return device_map


@dataclass(frozen=True)
class HFTokenAudit:
    choice_token_ids: Mapping[str, int]
    generation_input_ids_sha256: str
    generation_prompt_sha256: str
    chat_template_sha256: str
    direct_logit_method: str = HF_DIRECT_LOGIT_METHOD

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "choice_token_ids", MappingProxyType(dict(self.choice_token_ids))
        )


@dataclass(frozen=True)
class HFNextTokenScores:
    choice_logprobs: Mapping[str, float]
    choice_probabilities: Mapping[str, float]
    label_probability_mass: float
    scored_choice: str
    generation_input_ids_sha256: str
    generation_prompt_sha256: str
    direct_logit_method: str = HF_DIRECT_LOGIT_METHOD

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "choice_logprobs", MappingProxyType(dict(self.choice_logprobs))
        )
        object.__setattr__(
            self,
            "choice_probabilities",
            MappingProxyType(dict(self.choice_probabilities)),
        )


class HFNextTokenRuntime:
    """Model-independent chat rendering plus full-vocabulary log-softmax."""

    def __init__(
        self,
        *,
        tokenizer: object,
        forward_logits: Callable[[tuple[int, ...]], Sequence[float]],
        model_path: str,
        precision: str,
        loaded_model: object | None = None,
        load_policy: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("HF model_path must be nonempty")
        if not isinstance(precision, str) or not precision:
            raise ValueError("HF precision must be nonempty")
        self.tokenizer = tokenizer
        self._forward_logits = forward_logits
        self.model_path = model_path
        self.precision = precision
        self._loaded_model = loaded_model
        self.load_policy = MappingProxyType(
            dict(
                frozen_hf_load_policy(precision)
                if load_policy is None
                else load_policy
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        precision: str,
        *,
        model_key: str | None = None,
    ) -> "HFNextTokenRuntime":
        """Load one frozen HF condition without importing ML dependencies locally."""
        if precision not in {"FP16", "GPTQ_INT4", "AWQ_INT4"}:
            raise ValueError("precision must be FP16, GPTQ_INT4, or AWQ_INT4")
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoTokenizer,
        )

        model_class = AutoModelForImageTextToText
        if model_key == "olmo3_7b":
            from transformers import AutoModelForCausalLM

            model_class = AutoModelForCausalLM

        tokenizer_kwargs: dict[str, Any] = {"local_files_only": True}
        if model_key == "ministral3_8b":
            from .ministral_calibration import ministral_tokenizer_load_kwargs

            tokenizer_kwargs = ministral_tokenizer_load_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        if model_key == "ministral3_8b":
            from transformers import AutoConfig

            source_config = AutoConfig.from_pretrained(
                model_path, local_files_only=True
            )
            if getattr(source_config, "model_type", None) != "mistral3":
                raise ValueError(
                    "Ministral source model_type does not match the frozen identity"
                )
        if precision == "FP16":
            loaded_model = model_class.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="balanced",
                max_memory=_fp16_max_memory(torch),
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            forward_model = loaded_model
        elif precision == "GPTQ_INT4":
            from gptqmodel import GPTQModel
            from gptqmodel.utils.backend import BACKEND

            if model_key == "olmo3_7b":
                from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS

                from .gptq_adapters import register_olmo3_gptq_adapter

                register_olmo3_gptq_adapter(MODEL_MAP, SUPPORTED_MODELS)

            load_policy = frozen_hf_load_policy(
                precision, model_key=model_key
            )
            loaded_model = GPTQModel.load(
                model_path,
                device_map=load_policy["device_map"],
                backend=BACKEND.GPTQ_TORCH,
            )
            forward_model = getattr(loaded_model, "model", loaded_model)
        else:
            load_policy = frozen_hf_load_policy(
                precision, model_key=model_key
            )
            loaded_model = model_class.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map=load_policy["device_map"],
                max_memory=_fp16_max_memory(torch),
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            forward_model = loaded_model
        if not callable(forward_model):
            raise RuntimeError("loaded HF model is not callable")
        if hasattr(forward_model, "eval"):
            forward_model.eval()
        return cls(
            tokenizer=tokenizer,
            forward_logits=_build_torch_forward_logits(forward_model, torch),
            model_path=model_path,
            precision=precision,
            loaded_model=loaded_model,
            load_policy=frozen_hf_load_policy(
                precision, model_key=model_key
            ),
        )

    def score_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        choice_token_ids: Mapping[str, int],
    ) -> HFNextTokenScores:
        if not choice_token_ids:
            raise ValueError("choice_token_ids must be nonempty")
        generation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = normalize_input_ids(generation)
        generation_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(generation_prompt, str) or not generation_prompt:
            raise ValueError("generation prompt must be nonempty text")
        raw_logits = self._forward_logits(input_ids)
        try:
            logits = tuple(raw_logits)
        except TypeError as error:
            raise ValueError("full-vocabulary logits must be nonempty") from error
        if not logits:
            raise ValueError("full-vocabulary logits must be nonempty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in logits
        ):
            raise ValueError("full-vocabulary logits must be finite")
        normalized_ids: dict[str, int] = {}
        for choice, token_id in choice_token_ids.items():
            if (
                not isinstance(choice, str)
                or not choice
                or isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise ValueError("choice token IDs are invalid")
            if token_id >= len(logits):
                raise ValueError("choice token ID is outside vocabulary")
            normalized_ids[choice] = token_id
        if len(set(normalized_ids.values())) != len(normalized_ids):
            raise ValueError("choice token IDs must be unique")

        maximum = max(float(value) for value in logits)
        log_z = maximum + math.log(
            math.fsum(math.exp(float(value) - maximum) for value in logits)
        )
        choice_logprobs = {
            choice: float(logits[token_id]) - log_z
            for choice, token_id in normalized_ids.items()
        }
        probabilities = {
            choice: math.exp(value)
            for choice, value in choice_logprobs.items()
        }
        scored_choice = max(
            normalized_ids, key=lambda choice: choice_logprobs[choice]
        )
        return HFNextTokenScores(
            choice_logprobs=choice_logprobs,
            choice_probabilities=probabilities,
            label_probability_mass=math.fsum(probabilities.values()),
            scored_choice=scored_choice,
            generation_input_ids_sha256=_ids_sha256(input_ids),
            generation_prompt_sha256=_text_sha256(generation_prompt),
        )


def normalize_input_ids(value: object) -> tuple[int, ...]:
    """Normalize legacy token lists and Transformers 5 BatchEncoding objects."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("input_ids are missing")
        value = value["input_ids"]
    if isinstance(value, (str, bytes)):
        raise ValueError("input_ids must be an integer sequence")
    try:
        normalized = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("input_ids must be an integer sequence") from error
    if not normalized or any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in normalized
    ):
        raise ValueError("input_ids must be nonempty nonnegative integers")
    return normalized


def audit_choice_boundary(
    tokenizer: object,
    messages: Sequence[Mapping[str, str]],
    choices: Sequence[str],
) -> HFTokenAudit:
    """Prove each allowed answer is one exact token after the chat prefix."""
    normalized_choices = tuple(choices)
    if not normalized_choices or len(set(normalized_choices)) != len(
        normalized_choices
    ):
        raise ValueError("choices must be unique and nonempty")
    generation_value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    generation_ids = normalize_input_ids(generation_value)
    generation_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(generation_prompt, str) or not generation_prompt:
        raise ValueError("generation prompt must be nonempty text")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer chat template must be nonempty")

    token_ids: dict[str, int] = {}
    for choice in normalized_choices:
        if not isinstance(choice, str) or len(choice) != 1:
            raise ValueError("choices must be one-character strings")
        standalone = normalize_input_ids(
            tokenizer.encode(choice, add_special_tokens=False)
        )
        if len(standalone) != 1:
            raise ValueError(f"choice {choice} must be a single token")
        completed = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": choice}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        completed_ids = normalize_input_ids(completed)
        prefix_length = len(generation_ids)
        if (
            completed_ids[:prefix_length] != generation_ids
            or len(completed_ids) <= prefix_length
        ):
            raise ValueError(f"choice {choice} chat-template prefix mismatch")
        if completed_ids[prefix_length] != standalone[0]:
            raise ValueError(f"choice {choice} answer-boundary token mismatch")
        token_ids[choice] = standalone[0]
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("choice token IDs must be unique")
    return HFTokenAudit(
        choice_token_ids=token_ids,
        generation_input_ids_sha256=_ids_sha256(generation_ids),
        generation_prompt_sha256=_text_sha256(generation_prompt),
        chat_template_sha256=_text_sha256(chat_template),
    )


def _ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        list(token_ids), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_torch_forward_logits(
    model: object, torch_module: object
) -> Callable[[tuple[int, ...]], Sequence[float]]:
    """Build the only permitted one-forward, last-token logits callback."""

    def forward(input_ids: tuple[int, ...]) -> Sequence[float]:
        embeddings = model.get_input_embeddings()
        device = embeddings.weight.device
        tensor = torch_module.tensor([list(input_ids)], dtype=torch_module.long)
        tensor = tensor.to(device)
        with torch_module.inference_mode():
            output = model(input_ids=tensor, use_cache=False)
        logits = output.logits[0, -1]
        return logits.detach().float().cpu().tolist()

    return forward


def _fp16_max_memory(torch_module: object) -> dict[int | str, str]:
    """Reserve two GiB per 2080 Ti for activations and transient tensors."""
    cuda = torch_module.cuda
    if cuda.device_count() != 2:
        raise RuntimeError("FP16 paired smoke requires exactly two GPUs")
    minimum_total = 10 * 1024**3
    if any(
        cuda.get_device_properties(index).total_memory < minimum_total
        for index in range(2)
    ):
        raise RuntimeError("FP16 paired smoke requires at least 10 GiB per GPU")
    return {0: "9GiB", 1: "9GiB", "cpu": "32GiB"}


def frozen_hf_load_policy(
    precision: str, *, model_key: str | None = None
) -> dict[str, Any]:
    """Return the protocol-level load policy recorded in runtime audits."""
    if precision == "FP16":
        return {
            "backend": "transformers",
            "cpu_max_memory": "32GiB",
            "device_map": "balanced",
            "gpu_max_memory": {"0": "9GiB", "1": "9GiB"},
        }
    if precision == "GPTQ_INT4":
        device_map: str | dict[str, str] = "balanced"
        if model_key == "gemma4_e4b":
            device_map = dict(GEMMA4_GPTQ_DEVICE_MAP)
        return {
            "backend": HF_GPTQ_BACKEND,
            "cpu_max_memory": None,
            "device_map": device_map,
            "gpu_max_memory": {},
        }
    if precision == "AWQ_INT4":
        policy: dict[str, Any] = {
            "backend": HF_AWQ_BACKEND,
            "cpu_max_memory": "32GiB",
            "device_map": "balanced",
            "gpu_max_memory": {"0": "9GiB", "1": "9GiB"},
            "quantization_format": "compressed-tensors",
        }
        if model_key == "qwen35_9b":
            policy["device_map"] = _qwen35_awq_device_map()
            policy["device_map_rationale"] = (
                "CPU-offload unused vision tower; split 32 text layers 16+16 "
                "to avoid compressed-size placement bias during W4A16 decompression"
            )
        elif model_key == "ministral3_8b":
            policy["device_map"] = _ministral3_awq_device_map()
            policy["device_map_rationale"] = (
                "CPU-offload unused vision tower and multimodal projector; split 34 "
                "text layers 17+17 to avoid compressed-size placement bias during "
                "W4A16 decompression"
            )
        elif model_key == "olmo3_7b":
            policy["device_map"] = _olmo3_awq_device_map()
            policy["device_map_rationale"] = (
                "Split 32 text layers 17+15 so the output GPU retains W4A16 "
                "decompression headroom for final norm and lm_head"
            )
        return policy
    raise ValueError("precision must be FP16, GPTQ_INT4, or AWQ_INT4")
