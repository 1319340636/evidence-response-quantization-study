"""Process-local GPTQModel adapters for frozen model architectures."""

from __future__ import annotations

from typing import MutableMapping, MutableSequence, Type


OLMO3_GPTQ_MODULE_TREE = [
    "model",
    "layers",
    "#",
    {
        "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
        "post_attention_layernorm": ("post_attention_layernorm:!",),
        "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        "post_feedforward_layernorm": ("post_feedforward_layernorm:!",),
    },
]


def register_olmo3_gptq_adapter(
    model_map: MutableMapping[str, type],
    supported_models: MutableSequence[str],
    *,
    base_class: Type | None = None,
) -> type:
    """Register the OLMo3 decoder tree in both GPTQModel registries.

    GPTQModel 7.1.0 falls back to automatic tree detection unless the model
    type is present in ``SUPPORTED_MODELS`` as well as ``MODEL_MAP``.  Its
    automatic loader cannot reconstruct the OLMo3 quantized checkpoint.
    """
    if base_class is None:
        from gptqmodel.models.definitions.llama import LlamaQModel

        base_class = LlamaQModel

    class Olmo3QModel(base_class):
        module_tree = OLMO3_GPTQ_MODULE_TREE

    Olmo3QModel.__name__ = "Olmo3QModel"
    Olmo3QModel.__qualname__ = "Olmo3QModel"
    model_map["olmo3"] = Olmo3QModel
    if "olmo3" not in supported_models:
        supported_models.append("olmo3")
    return Olmo3QModel
