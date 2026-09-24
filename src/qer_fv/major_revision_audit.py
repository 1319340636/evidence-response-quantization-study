"""Deterministic audits required by the CI manuscript major revision."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _text_sha256(value: object) -> str:
    return hashlib.sha256(_normalized_text(value).encode("utf-8")).hexdigest()


def _values(records: Iterable[Mapping[str, Any]], key: str) -> set[str]:
    return {
        str(record[key])
        for record in records
        if record.get(key) not in (None, "")
    }


def _text_hashes(
    records: Iterable[Mapping[str, Any]], key: str
) -> set[str]:
    return {
        _text_sha256(record[key])
        for record in records
        if record.get(key) not in (None, "")
    }


def _intersection(left: set[str], right: set[str]) -> dict[str, Any]:
    values = sorted(left & right)
    return {"status": "assessed", "count": len(values), "values": values}


def audit_calibration_overlap(
    calibration_records: Iterable[Mapping[str, Any]],
    evaluation_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare calibration and evaluation records at every available level."""

    calibration = list(calibration_records)
    evaluation = list(evaluation_records)
    return {
        "calibration_records": len(calibration),
        "evaluation_records": len(evaluation),
        "levels": {
            "record_id": _intersection(
                _values(calibration, "unique_id"),
                _values(evaluation, "unique_id"),
            ),
            "case_id": _intersection(
                _values(calibration, "case_id"),
                _values(evaluation, "case_id"),
            ),
            "claim_text_sha256": _intersection(
                _text_hashes(calibration, "claim"),
                _text_hashes(evaluation, "claim"),
            ),
            "evidence_text_sha256": _intersection(
                _text_hashes(calibration, "evidence"),
                _text_hashes(evaluation, "evidence"),
            ),
            "source_page": _intersection(
                {_normalized_text(value) for value in _values(calibration, "page")},
                {_normalized_text(value) for value in _values(evaluation, "page")},
            ),
            "entity_id": {"status": "not_assessable"},
        },
    }


def _population_summary(
    quartets: Iterable[Mapping[str, Any]], ids: set[str]
) -> dict[str, Any]:
    rows = [row for row in quartets if str(row["case_id"]) in ids]
    labels = Counter(str(row["negative_label"]) for row in rows)
    return {
        "quartets": len(rows),
        "pages": len({str(row["page"]) for row in rows}),
        "negative_labels": dict(sorted(labels.items())),
    }


def build_population_flow(
    quartets: Iterable[Mapping[str, Any]],
    confirmatory_ids: set[str],
    dose_subset_ids: set[str],
) -> dict[str, Any]:
    """Describe the frozen route-blind confirmatory-to-dose population flow."""

    if not dose_subset_ids <= confirmatory_ids:
        raise ValueError("dose population must be a subset of confirmatory")
    rows = list(quartets)
    available = {str(row["case_id"]) for row in rows}
    missing = confirmatory_ids - available
    if missing:
        raise ValueError(f"confirmatory manifest has {len(missing)} missing quartets")
    excluded_ids = confirmatory_ids - dose_subset_ids
    return {
        "confirmatory": _population_summary(rows, confirmatory_ids),
        "retained": _population_summary(rows, dose_subset_ids),
        "excluded": _population_summary(rows, excluded_ids),
        "membership": {
            "dose_is_subset": True,
            "retained_fraction": len(dose_subset_ids) / len(confirmatory_ids),
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_local_major_revision_audit(project_root: str | Path) -> dict[str, Any]:
    """Build the local overlap and population audits from frozen manifests."""

    root = Path(project_root)
    paths = {
        "calibration_records": root
        / "data/manifests/gptq_calibration_vitaminc_dev_256.jsonl",
        "confirmatory_ids": root / "data/manifests/vitaminc_confirmatory_ids.txt",
        "dose_subset_ids": root / "data/manifests/vitaminc_dose_subset_ids.txt",
        "evaluation_records": root / "data/raw/vitaminc/test.jsonl",
        "strict_quartets": root / "data/manifests/vitaminc_strict_quartets.jsonl",
    }
    dose_ids = _load_ids(paths["dose_subset_ids"])
    evaluation = [
        record
        for record in _load_jsonl(paths["evaluation_records"])
        if str(record.get("case_id", "")) in dose_ids
    ]
    return {
        "audit_protocol": "ci-major-revision-local-audit-v1-20260829",
        "calibration_overlap": audit_calibration_overlap(
            _load_jsonl(paths["calibration_records"]), evaluation
        ),
        "population_flow": build_population_flow(
            _load_jsonl(paths["strict_quartets"]),
            _load_ids(paths["confirmatory_ids"]),
            dose_ids,
        ),
        "source_sha256": {
            name: _file_sha256(path) for name, path in sorted(paths.items())
        },
    }


def _render_local_audit(result: Mapping[str, Any]) -> str:
    overlap = result["calibration_overlap"]
    flow = result["population_flow"]
    lines = [
        "# CI Major-Revision Local Audit",
        "",
        f"Protocol: `{result['audit_protocol']}`",
        "",
        "## Calibration overlap",
        "",
        "| Level | Status | Overlap count |",
        "|---|---:|---:|",
    ]
    for name, value in overlap["levels"].items():
        lines.append(
            f"| {name} | {value['status']} | {value.get('count', 'NA')} |"
        )
    lines.extend(
        [
            "",
            "## Confirmatory-to-common-support population flow",
            "",
            "| Population | Quartets | Pages | Negative labels |",
            "|---|---:|---:|---|",
        ]
    )
    for name in ("confirmatory", "retained", "excluded"):
        value = flow[name]
        labels = ", ".join(
            f"{label}={count}"
            for label, count in value["negative_labels"].items()
        )
        lines.append(
            f"| {name} | {value['quartets']} | {value['pages']} | {labels} |"
        )
    lines.extend(
        [
            "",
            f"- Dose subset is contained in confirmatory: `{flow['membership']['dose_is_subset']}`",
            f"- Retained fraction: `{flow['membership']['retained_fraction']:.6f}`",
            "",
            "## Source hashes",
            "",
        ]
    )
    for name, value in result["source_sha256"].items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def write_major_revision_audit(
    result: Mapping[str, Any], output_directory: str | Path
) -> dict[str, str]:
    """Write deterministic local-audit artifacts and detached hashes."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "local_audit.json"
    markdown_path = output / "local_audit.md"
    sha_path = output / "local_audit.sha256"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_local_audit(result), encoding="utf-8")
    sha_path.write_text(
        "\n".join(
            [
                f"{_file_sha256(json_path)} *{json_path.name}",
                f"{_file_sha256(markdown_path)} *{markdown_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "sha256": str(sha_path),
    }
