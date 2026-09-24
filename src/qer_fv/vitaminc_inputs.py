"""Frozen VitaminC split loading and deterministic formal inference units."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .prompts import PromptInput
from .provenance import load_freeze_config
from .vitaminc import (
    VitaminCQuartet,
    build_strict_quartets,
    id_manifest_sha256,
    read_jsonl,
)


_RELEASED_SPLITS = {
    "pilot": ("dev_path", "vitaminc_pilot_ids.txt"),
    "discovery": ("dev_path", "vitaminc_discovery_ids.txt"),
    "dose_subset": ("test_path", "vitaminc_dose_subset_ids.txt"),
    "confirmatory": ("test_path", "vitaminc_confirmatory_ids.txt"),
}


@dataclass(frozen=True)
class VitaminCInferenceUnit:
    owner_key: str
    case_id: str
    page: str
    control: str
    cell_index: int | None
    claim_index: int
    evidence_index: int | None
    sample_id: str
    prompt_input: PromptInput


def load_vitaminc_split(
    project_root: str | Path, *, split: str
) -> list[VitaminCQuartet]:
    """Load one explicitly released split and revalidate its frozen identity."""
    if split not in _RELEASED_SPLITS:
        raise ValueError(f"split is not a released VitaminC split: {split}")
    root = Path(project_root).resolve()
    config = load_freeze_config(root / "configs/freeze_v1.json")
    vitamin = config.get("vitaminc")
    if not isinstance(vitamin, Mapping):
        raise ValueError("freeze config vitaminc section is required")
    source_key, manifest_name = _RELEASED_SPLITS[split]
    source = vitamin.get(source_key)
    frozen = vitamin.get(split)
    if not isinstance(source, str) or not isinstance(frozen, Mapping):
        raise ValueError(f"missing frozen VitaminC definition for {split}")

    ids = _read_ids(root / "data/manifests" / manifest_name)
    expected_count = frozen.get("examples")
    expected_hash = frozen.get("id_sha256")
    if expected_count != len(ids) or expected_hash != id_manifest_sha256(ids):
        raise ValueError(f"frozen VitaminC {split} ID manifest mismatch")

    all_quartets = {
        quartet.case_id: quartet
        for quartet in build_strict_quartets(read_jsonl(root / source))
    }
    missing = [case_id for case_id in ids if case_id not in all_quartets]
    if missing:
        raise ValueError(f"VitaminC {split} IDs missing from strict quartets")
    quartets = [all_quartets[case_id] for case_id in ids]
    if id_manifest_sha256(item.case_id for item in quartets) != expected_hash:
        raise ValueError(f"frozen VitaminC {split} quartet identity mismatch")

    summary_path = root / "data/manifests/vitaminc_splits_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_summary = summary.get(split)
    if not isinstance(split_summary, Mapping):
        raise ValueError(f"missing VitaminC {split} split summary")
    if split_summary.get("groups") != len(quartets):
        raise ValueError(f"VitaminC {split} group count mismatch")
    if split_summary.get("pages") != len({item.page for item in quartets}):
        raise ValueError(f"VitaminC {split} page count mismatch")
    if split_summary.get("id_manifest_sha256") != expected_hash:
        raise ValueError(f"VitaminC {split} summary hash mismatch")
    return quartets


def build_vitaminc_units(
    quartets: Iterable[VitaminCQuartet],
) -> list[VitaminCInferenceUnit]:
    """Materialize four full cells and two claim-level units per control."""
    ordered = sorted(quartets, key=lambda item: item.case_id)
    if len({item.case_id for item in ordered}) != len(ordered):
        raise ValueError("VitaminC quartets must have unique case IDs")
    units: list[VitaminCInferenceUnit] = []
    for quartet in ordered:
        for cell_index, (row_id, label) in enumerate(
            zip(quartet.row_ids, quartet.labels, strict=True)
        ):
            claim_index, evidence_index = divmod(cell_index, 2)
            metadata = {
                "case_id": quartet.case_id,
                "cell_index": str(cell_index),
                "claim_index": str(claim_index),
                "evidence_index": str(evidence_index),
                "gold_label": label,
                "negative_label": quartet.negative_label,
                "page": quartet.page,
            }
            units.append(
                VitaminCInferenceUnit(
                    owner_key=f"full:{row_id}",
                    case_id=quartet.case_id,
                    page=quartet.page,
                    control="full",
                    cell_index=cell_index,
                    claim_index=claim_index,
                    evidence_index=evidence_index,
                    sample_id=row_id,
                    prompt_input=PromptInput(
                        dataset="vitaminc",
                        sample_id=row_id,
                        claim=quartet.claims[claim_index],
                        evidence=quartet.evidences[evidence_index],
                        metadata=metadata,
                    ),
                )
            )
        for control in ("no_evidence", "knowledge_only"):
            for claim_index, claim in enumerate(quartet.claims):
                owner_key = f"{control}:{quartet.case_id}:claim:{claim_index}"
                metadata = {
                    "case_id": quartet.case_id,
                    "claim_index": str(claim_index),
                    "negative_label": quartet.negative_label,
                    "page": quartet.page,
                }
                units.append(
                    VitaminCInferenceUnit(
                        owner_key=owner_key,
                        case_id=quartet.case_id,
                        page=quartet.page,
                        control=control,
                        cell_index=None,
                        claim_index=claim_index,
                        evidence_index=None,
                        sample_id=owner_key,
                        prompt_input=PromptInput(
                            dataset="vitaminc",
                            sample_id=owner_key,
                            claim=claim,
                            evidence=None,
                            metadata=metadata,
                        ),
                    )
                )
    owner_keys = [item.owner_key for item in units]
    if len(set(owner_keys)) != len(owner_keys):
        raise ValueError("VitaminC inference owner keys must be unique")
    return units


def _read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("VitaminC split manifest contains invalid IDs")
    return ids
