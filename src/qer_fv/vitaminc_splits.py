"""Frozen, metadata-only VitaminC split construction."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, TypeVar


class PageBound(Protocol):
    case_id: str
    page: str


QuartetT = TypeVar("QuartetT", bound=PageBound)


class LabelledPageBound(PageBound, Protocol):
    negative_label: str


LabelledQuartetT = TypeVar("LabelledQuartetT", bound=LabelledPageBound)


@dataclass(frozen=True)
class OldOverlap:
    """Old evaluation exposure measured at row, group, and page levels."""

    rows: int
    group_ids: frozenset[str]
    pages: frozenset[str]


def extract_old_overlap(
    old_rows: Iterable[Mapping[str, Any]],
    raw_test_rows: Iterable[Mapping[str, Any]],
) -> OldOverlap:
    """Resolve pages for old group IDs from the raw VitaminC test metadata."""
    old_list = list(old_rows)
    group_ids = frozenset(
        group_id
        for row in old_list
        if isinstance((group_id := row.get("group_id")), str) and group_id
    )
    pages_by_group: dict[str, set[str]] = defaultdict(set)
    for row in raw_test_rows:
        case_id = row.get("case_id")
        page = row.get("page")
        if isinstance(case_id, str) and isinstance(page, str):
            pages_by_group[case_id].add(page)
    pages = frozenset(
        page for group_id in group_ids for page in pages_by_group.get(group_id, ())
    )
    return OldOverlap(rows=len(old_list), group_ids=group_ids, pages=pages)


def fresh_page_disjoint(
    quartets: Iterable[QuartetT], overlap: OldOverlap
) -> list[QuartetT]:
    """Exclude every strict quartet on any page exposed by the old evaluation."""
    return [quartet for quartet in quartets if quartet.page not in overlap.pages]


def stable_rank(seed: str, namespace: str, value: str) -> str:
    """Return the preregistered SHA-256 ordering key."""
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()


def split_confirmatory_recovery(
    quartets: Iterable[QuartetT],
    seed: str,
    recovery_fraction: float = 0.20,
) -> tuple[list[QuartetT], list[QuartetT]]:
    """Assign whole pages to recovery by frozen SHA rank; retain the rest."""
    quartet_list = list(quartets)
    pages = sorted(
        {quartet.page for quartet in quartet_list},
        key=lambda page: stable_rank(seed, "recovery-page", page),
    )
    recovery_count = round(len(pages) * recovery_fraction)
    recovery_pages = set(pages[:recovery_count])
    recovery = [q for q in quartet_list if q.page in recovery_pages]
    confirmatory = [q for q in quartet_list if q.page not in recovery_pages]
    return confirmatory, recovery


def label_pair(quartet: LabelledPageBound) -> str:
    """Map the raw negative label to the frozen pair stratum name."""
    if quartet.negative_label == "REFUTES":
        return "S/R"
    if quartet.negative_label == "NOT ENOUGH INFO":
        return "S/NEI"
    raise ValueError(f"unsupported negative label: {quartet.negative_label}")


def _ranked_pair(
    quartets: Iterable[LabelledQuartetT], pair: str, seed: str, namespace: str
) -> list[LabelledQuartetT]:
    return sorted(
        (quartet for quartet in quartets if label_pair(quartet) == pair),
        key=lambda quartet: stable_rank(seed, namespace, quartet.case_id),
    )


def select_dev_pilot_discovery(
    quartets: Iterable[LabelledQuartetT],
    seed: str,
    pilot_quotas: Mapping[str, int] | None = None,
    discovery_quotas: Mapping[str, int] | None = None,
) -> tuple[list[LabelledQuartetT], list[LabelledQuartetT]]:
    """Select frozen label-stratified dev sets with page isolation."""
    quartet_list = list(quartets)
    pilot_targets = dict(pilot_quotas or {"S/R": 83, "S/NEI": 17})
    discovery_targets = dict(
        discovery_quotas or {"S/R": 827, "S/NEI": 173}
    )
    pilot: list[LabelledQuartetT] = []
    for pair in ("S/R", "S/NEI"):
        ranked = _ranked_pair(quartet_list, pair, seed, "pilot")
        quota = pilot_targets[pair]
        if len(ranked) < quota:
            raise ValueError(f"insufficient {pair} quartets for pilot quota {quota}")
        pilot.extend(ranked[:quota])

    pilot_pages = {quartet.page for quartet in pilot}
    discovery_pool = [q for q in quartet_list if q.page not in pilot_pages]
    discovery: list[LabelledQuartetT] = []
    for pair in ("S/R", "S/NEI"):
        ranked = _ranked_pair(discovery_pool, pair, seed, "discovery")
        quota = discovery_targets[pair]
        if len(ranked) < quota:
            raise ValueError(f"insufficient {pair} quartets for discovery quota {quota}")
        discovery.extend(ranked[:quota])
    return pilot, discovery


def select_dose_subset(
    quartets: Iterable[LabelledQuartetT],
    seed: str,
    total: int = 1_200,
) -> list[LabelledQuartetT]:
    """Select the frozen page-first, pair-stratified severity subset."""
    quartet_list = list(quartets)
    if total > len(quartet_list):
        raise ValueError(f"dose size {total} exceeds population {len(quartet_list)}")
    pair_counts = {
        pair: sum(label_pair(quartet) == pair for quartet in quartet_list)
        for pair in ("S/R", "S/NEI")
    }
    nei_quota = round(total * pair_counts["S/NEI"] / len(quartet_list))
    quotas = {"S/NEI": nei_quota, "S/R": total - nei_quota}
    selected: list[LabelledQuartetT] = []

    for pair in ("S/NEI", "S/R"):
        by_page: dict[str, list[LabelledQuartetT]] = defaultdict(list)
        for quartet in quartet_list:
            if label_pair(quartet) == pair:
                by_page[quartet.page].append(quartet)
        page_order = sorted(
            by_page,
            key=lambda page: stable_rank(seed, "dose-page", f"{pair}|{page}"),
        )
        for page in page_order:
            by_page[page].sort(
                key=lambda quartet: stable_rank(seed, "dose-case", quartet.case_id)
            )
        pair_selected = 0
        pass_number = 0
        while pair_selected < quotas[pair]:
            progressed = False
            for page in page_order:
                if pass_number < len(by_page[page]):
                    selected.append(by_page[page][pass_number])
                    pair_selected += 1
                    progressed = True
                    if pair_selected >= quotas[pair]:
                        break
            if not progressed:
                raise ValueError(f"insufficient {pair} quartets for dose quota")
            pass_number += 1
    return selected
