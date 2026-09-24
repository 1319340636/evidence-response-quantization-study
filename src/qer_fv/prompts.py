"""Frozen prompt contracts and deterministic prompt construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


CONTRACT_VERSION = "prompt-v1-20260712"
CONTRACT_V2_VERSION = "prompt-v2-20260714"
EXPECTED_CHOICE_MAPS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "vitaminc": MappingProxyType(
            {"A": "SUPPORTS", "B": "REFUTES", "C": "NOT ENOUGH INFO"}
        ),
        "cub_druid": MappingProxyType(
            {"A": "True", "B": "False", "C": "None"}
        ),
        "tabfact": MappingProxyType({"A": "Entailed", "B": "Refuted"}),
    }
)
V1_CONTROLS = frozenset({"full", "no_evidence", "knowledge_only"})
V2_CONTROLS = V1_CONTROLS | {"query_only"}
EXPECTED_CONTROLS_BY_VERSION = {
    CONTRACT_VERSION: V1_CONTROLS,
    CONTRACT_V2_VERSION: V2_CONTROLS,
}
KNOWLEDGE_ONLY_CHOICE_MAP: Mapping[str, str] = MappingProxyType(
    {"A": "True", "B": "False", "C": "Unknown"}
)
MAPPING_VARIANTS = frozenset(
    {"original", "reversed", "cycle_1", "cycle_2"}
)
VITAMINC_REVERSED_CHOICE_MAP: Mapping[str, str] = MappingProxyType(
    {"A": "NOT ENOUGH INFO", "B": "REFUTES", "C": "SUPPORTS"}
)
VITAMINC_BALANCED_CHOICE_MAPS: Mapping[str, Mapping[str, str]] = (
    MappingProxyType(
        {
            "cycle_1": MappingProxyType(
                {
                    "A": "REFUTES",
                    "B": "NOT ENOUGH INFO",
                    "C": "SUPPORTS",
                }
            ),
            "cycle_2": MappingProxyType(
                {
                    "A": "NOT ENOUGH INFO",
                    "B": "SUPPORTS",
                    "C": "REFUTES",
                }
            ),
        }
    )
)


@dataclass(frozen=True)
class Choice:
    letter: str
    label: str


@dataclass(frozen=True)
class DatasetPromptSpec:
    task_instruction: str
    choices: tuple[Choice, ...]


@dataclass(frozen=True)
class ControlPromptSpec:
    task_instruction: str
    choices: tuple[Choice, ...] | None


@dataclass(frozen=True)
class PromptContract:
    version: str
    system_instruction: str
    controls: Mapping[str, ControlPromptSpec]
    datasets: Mapping[str, DatasetPromptSpec]

    def choice_map(self, dataset: str, *, control: str = "full") -> dict[str, str]:
        try:
            dataset_choices = self.datasets[dataset].choices
        except KeyError as error:
            raise ValueError(f"unsupported prompt dataset: {dataset}") from error
        try:
            control_choices = self.controls[control].choices
        except KeyError as error:
            raise ValueError(f"unsupported prompt control: {control}") from error
        choices = dataset_choices if control_choices is None else control_choices
        return {choice.letter: choice.label for choice in choices}


@dataclass(frozen=True)
class PromptInput:
    dataset: str
    sample_id: str
    claim: str
    evidence: str | None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise ValueError("prompt metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class RenderedPrompt:
    contract_version: str
    dataset: str
    sample_id: str
    control: str
    mapping_variant: str
    messages: tuple[Mapping[str, str], ...]
    choices: tuple[Choice, ...]
    prompt_sha256: str

    @property
    def choice_to_label(self) -> dict[str, str]:
        return {choice.letter: choice.label for choice in self.choices}


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prompt contract {field} must be a nonempty string")
    return value


def load_prompt_contract(path: str | Path) -> PromptContract:
    """Load one exact, immutable Experiment B prompt contract."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("prompt contract root must be a JSON object")

    version = _required_text(payload.get("contract_version"), "contract_version")
    if version not in EXPECTED_CONTROLS_BY_VERSION:
        raise ValueError(
            f"unsupported prompt contract version: {version}"
        )
    system_instruction = _required_text(
        payload.get("system_instruction"), "system_instruction"
    )

    controls_raw = payload.get("controls")
    expected_controls = EXPECTED_CONTROLS_BY_VERSION[version]
    if not isinstance(controls_raw, dict) or set(controls_raw) != expected_controls:
        raise ValueError("prompt contract control set mismatch")
    controls: dict[str, ControlPromptSpec] = {}
    for name in sorted(expected_controls):
        raw_control = controls_raw[name]
        if not isinstance(raw_control, dict):
            raise ValueError(f"prompt contract controls.{name} must be an object")
        instruction = _required_text(
            raw_control.get("task_instruction"), f"controls.{name}.task_instruction"
        )
        raw_choices = raw_control.get("choices")
        if name == "knowledge_only":
            if raw_choices != dict(KNOWLEDGE_ONLY_CHOICE_MAP):
                raise ValueError("knowledge_only frozen choice mapping mismatch")
            choices = tuple(
                Choice(letter, label)
                for letter, label in KNOWLEDGE_ONLY_CHOICE_MAP.items()
            )
        else:
            if raw_choices is not None:
                raise ValueError(f"prompt contract controls.{name} cannot override choices")
            choices = None
        controls[name] = ControlPromptSpec(instruction, choices)

    datasets_raw = payload.get("datasets")
    if not isinstance(datasets_raw, dict):
        raise ValueError("prompt contract datasets must be a JSON object")
    if set(datasets_raw) != set(EXPECTED_CHOICE_MAPS):
        raise ValueError("prompt contract dataset set mismatch")

    datasets: dict[str, DatasetPromptSpec] = {}
    for dataset, expected_map in EXPECTED_CHOICE_MAPS.items():
        raw = datasets_raw[dataset]
        if not isinstance(raw, dict):
            raise ValueError(f"prompt contract datasets.{dataset} must be an object")
        task_instruction = _required_text(
            raw.get("task_instruction"),
            f"datasets.{dataset}.task_instruction",
        )
        choices_raw = raw.get("choices")
        if not isinstance(choices_raw, dict) or choices_raw != dict(expected_map):
            raise ValueError(f"{dataset} frozen choice mapping mismatch")
        choices = tuple(
            Choice(letter=letter, label=label)
            for letter, label in expected_map.items()
        )
        datasets[dataset] = DatasetPromptSpec(task_instruction, choices)

    return PromptContract(
        version=version,
        system_instruction=system_instruction,
        controls=MappingProxyType(controls),
        datasets=MappingProxyType(datasets),
    )


def _prompt_hash(
    *,
    contract_version: str,
    dataset: str,
    sample_id: str,
    control: str,
    messages: tuple[Mapping[str, str], ...],
    choices: tuple[Choice, ...],
) -> str:
    record = {
        "choices": [
            {"label": choice.label, "letter": choice.letter} for choice in choices
        ],
        "contract_version": contract_version,
        "control": control,
        "dataset": dataset,
        "messages": [dict(message) for message in messages],
        "sample_id": sample_id,
    }
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_metadata(item: PromptInput, key: str) -> str:
    value = item.metadata.get(key)
    return _required_text(value, f"{item.dataset}.{item.sample_id}.{key}")


def render_prompt(
    contract: PromptContract,
    item: PromptInput,
    *,
    control: str = "full",
    mapping_variant: str = "original",
) -> RenderedPrompt:
    """Render one prediction-free prompt under the frozen shared scaffold."""
    if not isinstance(item, PromptInput):
        raise ValueError("item must be a PromptInput")
    if mapping_variant not in MAPPING_VARIANTS:
        raise ValueError("unsupported prompt mapping variant")
    sample_id = _required_text(item.sample_id, "sample_id")
    claim = _required_text(item.claim, f"{item.dataset}.{sample_id}.claim")
    if control not in contract.controls:
        raise ValueError(f"unsupported prompt control: {control}")
    if mapping_variant != "original" and not (
        item.dataset == "vitaminc" and control == "full"
    ):
        raise ValueError(
            f"{mapping_variant} mapping is restricted to VitaminC full evidence"
        )
    if control == "query_only" and item.dataset != "cub_druid":
        raise ValueError("query_only control is restricted to cub_druid")
    try:
        spec = contract.datasets[item.dataset]
    except KeyError as error:
        raise ValueError(f"unsupported prompt dataset: {item.dataset}") from error

    control_spec = contract.controls[control]
    choices = spec.choices if control_spec.choices is None else control_spec.choices
    if mapping_variant == "reversed":
        choices = tuple(
            Choice(letter=letter, label=label)
            for letter, label in VITAMINC_REVERSED_CHOICE_MAP.items()
        )
    elif mapping_variant in VITAMINC_BALANCED_CHOICE_MAPS:
        choices = tuple(
            Choice(letter=letter, label=label)
            for letter, label in VITAMINC_BALANCED_CHOICE_MAPS[
                mapping_variant
            ].items()
        )
    task_instruction = control_spec.task_instruction
    if control != "knowledge_only":
        task_instruction = spec.task_instruction + " " + task_instruction
    sections = [
        "Task:\n" + task_instruction,
        "Choices:\n"
        + "\n".join(
            f"{choice.letter} = {choice.label}" for choice in choices
        ),
    ]
    if item.dataset == "cub_druid":
        sections.append("Claimant:\n" + _validated_metadata(item, "claimant"))
    elif item.dataset == "tabfact":
        sections.append("Caption:\n" + _validated_metadata(item, "caption"))
    sections.append("Claim:\n" + claim)

    if control not in {"knowledge_only", "query_only"}:
        if control == "no_evidence":
            evidence = "[NO_EVIDENCE]"
        else:
            evidence = _required_text(
                item.evidence, f"{item.dataset}.{sample_id}.evidence"
            )
        sections.append("Evidence:\n" + evidence)

    allowed = ", ".join(choice.letter for choice in choices)
    sections.append("Answer:\nReturn exactly one of: " + allowed)
    messages = (
        MappingProxyType(
            {"role": "system", "content": contract.system_instruction}
        ),
        MappingProxyType({"role": "user", "content": "\n\n".join(sections)}),
    )
    prompt_sha256 = _prompt_hash(
        contract_version=contract.version,
        dataset=item.dataset,
        sample_id=sample_id,
        control=control,
        messages=messages,
        choices=choices,
    )
    return RenderedPrompt(
        contract_version=contract.version,
        dataset=item.dataset,
        sample_id=sample_id,
        control=control,
        mapping_variant=mapping_variant,
        messages=messages,
        choices=choices,
        prompt_sha256=prompt_sha256,
    )
