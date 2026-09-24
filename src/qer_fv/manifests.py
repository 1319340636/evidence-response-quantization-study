"""Build deterministic, prediction-free manifests for Experiment B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cub_druid import load_cub_test
from .provenance import load_freeze_config, verify_frozen_files
from .tabfact import (
    TabFactClaim,
    extract_old_table_ids,
    fresh_eligible,
    load_tabfact_examples,
    select_fresh_subset,
)
from .vitaminc import (
    VitaminCExclusion,
    VitaminCQuartet,
    build_strict_quartet_audit,
    build_strict_quartets,
    canonical_id_manifest,
    id_manifest_sha256,
    read_jsonl,
)
from .vitaminc_splits import (
    OldOverlap,
    extract_old_overlap,
    fresh_page_disjoint,
    select_dev_pilot_discovery,
    select_dose_subset,
    split_confirmatory_recovery,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(value) for value in values)


def _manifest_index_payload(
    directory: Path, protocol_version: Any, freeze_config_sha256: str
) -> bytes:
    files = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative == "manifest_index.json":
            continue
        payload = path.read_bytes()
        files.append(
            {
                "bytes": len(payload),
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return _json_bytes(
        {
            "bundle_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
            "files": files,
            "freeze_config_sha256": freeze_config_sha256,
            "protocol_version": protocol_version,
        }
    )


def _publish_manifest_bundle(staging: Path, destination: Path) -> None:
    """Publish payloads first and the bundle commit marker last."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging)
        if relative.as_posix() == "manifest_index.json":
            continue
        _atomic_write(destination / relative, path.read_bytes())
    _atomic_write(
        destination / "manifest_index.json",
        (staging / "manifest_index.json").read_bytes(),
    )


def _required_manifest_paths(config: Mapping[str, Any]) -> set[str]:
    required = {
        "raw_provenance.json",
        "vitaminc_strict_exclusions.jsonl",
        "vitaminc_strict_exclusions_summary.json",
        "vitaminc_strict_ids.txt",
        "vitaminc_strict_quartets.jsonl",
        "vitaminc_strict_summary.json",
    }
    vitamin = config.get("vitaminc")
    if isinstance(vitamin, Mapping) and all(
        isinstance(vitamin.get(key), str)
        for key in ("dev_path", "old_evaluation_path")
    ):
        required.update(
            {
                "vitaminc_old_group_ids.txt",
                "vitaminc_old_overlap.json",
                "vitaminc_old_pages.txt",
                "vitaminc_split_assignments.jsonl",
                "vitaminc_splits_summary.json",
                *{
                    f"vitaminc_{name}_ids.txt"
                    for name in (
                        "fresh",
                        "confirmatory",
                        "recovery",
                        "pilot",
                        "discovery",
                        "dose_subset",
                    )
                },
            }
        )
    tabfact = config.get("tabfact")
    if isinstance(tabfact, Mapping) and all(
        isinstance(tabfact.get(key), str)
        for key in ("test_path", "tables_path", "old_evaluation_path")
    ):
        required.update(
            {
                "tabfact_fresh_claims.jsonl",
                "tabfact_fresh_ids.txt",
                "tabfact_old_table_ids.txt",
                "tabfact_summary.json",
            }
        )
    cub = config.get("cub_druid")
    if isinstance(cub, Mapping) and isinstance(cub.get("path"), str):
        required.update(
            {"cub_files.json", "cub_summary.json", "cub_test.jsonl", "cub_test_ids.txt"}
        )
    return required


def verify_data_foundation(
    project_root: str | Path,
    config_path: str | Path,
    manifest_dir: str | Path,
) -> dict[str, Any]:
    """Reject any raw/config/manifest state not matching one committed bundle."""
    root = Path(project_root).resolve()
    frozen_path = Path(config_path)
    if not frozen_path.is_absolute():
        frozen_path = root / frozen_path
    config = load_freeze_config(frozen_path)
    verify_frozen_files(root, config)
    cub = config.get("cub_druid")
    if isinstance(cub, Mapping) and cub.get("artifact_gate") != "closed":
        raise ValueError("CUB artifact gate is not closed")

    directory = Path(manifest_dir)
    index_path = directory / "manifest_index.json"
    if not index_path.is_file():
        raise ValueError("manifest bundle commit marker is missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, Mapping):
        raise ValueError("manifest index must be a JSON object")
    if index.get("freeze_config_sha256") != hashlib.sha256(
        frozen_path.read_bytes()
    ).hexdigest():
        raise ValueError("freeze config SHA-256 mismatch")
    if index.get("protocol_version") != config.get("protocol_version"):
        raise ValueError("manifest protocol version mismatch")
    files = index.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest index files must be a list")
    if index.get("bundle_sha256") != hashlib.sha256(_json_bytes(files)).hexdigest():
        raise ValueError("manifest bundle SHA-256 mismatch")

    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("manifest file entry must be a JSON object")
        relative_raw = entry.get("path")
        if not isinstance(relative_raw, str):
            raise ValueError("manifest file path must be a string")
        relative = Path(relative_raw)
        if relative.is_absolute() or ".." in relative.parts or relative_raw in seen:
            raise ValueError(f"unsafe or duplicate manifest path: {relative_raw}")
        seen.add(relative_raw)
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"manifest payload is missing: {relative_raw}")
        payload = path.read_bytes()
        if len(payload) != entry.get("bytes"):
            raise ValueError(f"manifest payload byte count mismatch: {relative_raw}")
        if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            raise ValueError(f"manifest payload SHA-256 mismatch: {relative_raw}")
    missing = sorted(_required_manifest_paths(config) - seen)
    if missing:
        raise ValueError(
            "required manifest payload missing from index: " + ", ".join(missing)
        )
    return {
        "bundle_sha256": index.get("bundle_sha256"),
        "files": len(files),
        "protocol_version": index.get("protocol_version"),
    }


def _quartet_record(quartet: VitaminCQuartet) -> dict[str, Any]:
    return {
        "case_id": quartet.case_id,
        "negative_label": quartet.negative_label,
        "page": quartet.page,
        "row_ids": list(quartet.row_ids),
    }


def _exclusion_record(exclusion: VitaminCExclusion) -> dict[str, Any]:
    return {
        "case_id": exclusion.case_id,
        "reason": exclusion.reason,
        "row_count": exclusion.row_count,
    }


def _strict_summary(quartets: list[VitaminCQuartet]) -> dict[str, Any]:
    return {
        "groups": len(quartets),
        "id_manifest_sha256": id_manifest_sha256(
            quartet.case_id for quartet in quartets
        ),
        "negative_labels": dict(
            sorted(Counter(quartet.negative_label for quartet in quartets).items())
        ),
        "pages": len({quartet.page for quartet in quartets}),
    }


def _verify_strict_freeze(summary: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    vitamin = config.get("vitaminc")
    if not isinstance(vitamin, Mapping):
        return
    frozen = vitamin.get("strict")
    if not isinstance(frozen, Mapping):
        return
    comparisons = (
        ("groups", "examples", "strict quartet count"),
        ("pages", "pages", "strict page count"),
        ("id_manifest_sha256", "id_sha256", "strict ID manifest SHA-256"),
    )
    for actual_key, expected_key, label in comparisons:
        if expected_key in frozen and summary[actual_key] != frozen[expected_key]:
            raise ValueError(
                f"{label} mismatch: expected {frozen[expected_key]}, "
                f"got {summary[actual_key]}"
            )


def _verify_named_split(
    name: str, summary: Mapping[str, Any], vitamin: Mapping[str, Any]
) -> None:
    frozen = vitamin.get(name)
    if not isinstance(frozen, Mapping):
        raise ValueError(f"missing frozen definition for vitaminc.{name}")
    comparisons = (
        ("groups", "examples", "group count"),
        ("pages", "pages", "page count"),
        ("id_manifest_sha256", "id_sha256", "ID manifest SHA-256"),
    )
    for actual_key, expected_key, label in comparisons:
        if expected_key in frozen and summary[actual_key] != frozen[expected_key]:
            raise ValueError(
                f"vitaminc.{name} {label} mismatch: expected "
                f"{frozen[expected_key]}, got {summary[actual_key]}"
            )


def _verify_old_overlap(overlap: OldOverlap, vitamin: Mapping[str, Any]) -> None:
    frozen = vitamin.get("old_overlap")
    if not isinstance(frozen, Mapping):
        raise ValueError("missing frozen definition for vitaminc.old_overlap")
    actual = {
        "rows": overlap.rows,
        "groups": len(overlap.group_ids),
        "pages": len(overlap.pages),
    }
    for key, value in actual.items():
        if frozen.get(key) != value:
            raise ValueError(
                f"vitaminc.old_overlap {key} mismatch: expected "
                f"{frozen.get(key)}, got {value}"
            )


def _write_vitaminc_splits(
    destination: Path,
    vitamin: Mapping[str, Any],
    seed: str,
    raw_test: list[dict[str, Any]],
    strict_test: list[VitaminCQuartet],
    root: Path,
) -> dict[str, dict[str, Any]]:
    old_path = vitamin.get("old_evaluation_path")
    dev_path = vitamin.get("dev_path")
    if not isinstance(old_path, str) or not isinstance(dev_path, str):
        return {}
    old_rows = read_jsonl(root / old_path)
    strict_dev = build_strict_quartets(read_jsonl(root / dev_path))
    overlap = extract_old_overlap(old_rows, raw_test)
    _verify_old_overlap(overlap, vitamin)
    fresh = fresh_page_disjoint(strict_test, overlap)
    confirmatory, recovery = split_confirmatory_recovery(fresh, seed)
    pilot, discovery = select_dev_pilot_discovery(strict_dev, seed)
    dose_subset = select_dose_subset(confirmatory, seed)
    splits = {
        "fresh": fresh,
        "confirmatory": confirmatory,
        "recovery": recovery,
        "pilot": pilot,
        "discovery": discovery,
        "dose_subset": dose_subset,
    }
    summaries = {name: _strict_summary(values) for name, values in splits.items()}
    for name, summary in summaries.items():
        _verify_named_split(name, summary, vitamin)

    overlap_summary = {
        "groups": len(overlap.group_ids),
        "pages": len(overlap.pages),
        "rows": overlap.rows,
    }
    _atomic_write(
        destination / "vitaminc_old_overlap.json", _json_bytes(overlap_summary)
    )
    _atomic_write(
        destination / "vitaminc_old_group_ids.txt",
        canonical_id_manifest(overlap.group_ids),
    )
    _atomic_write(
        destination / "vitaminc_old_pages.txt", canonical_id_manifest(overlap.pages)
    )
    for name, values in splits.items():
        _atomic_write(
            destination / f"vitaminc_{name}_ids.txt",
            canonical_id_manifest(quartet.case_id for quartet in values),
        )
    _atomic_write(
        destination / "vitaminc_splits_summary.json", _json_bytes(summaries)
    )

    assignments: dict[str, dict[str, Any]] = {}
    split_order = tuple(splits)
    for name in split_order:
        for quartet in splits[name]:
            record = assignments.setdefault(
                quartet.case_id,
                {"case_id": quartet.case_id, "page": quartet.page, "splits": []},
            )
            record["splits"].append(name)
    ordered_assignments = []
    for case_id in sorted(assignments):
        record = assignments[case_id]
        record["splits"] = [name for name in split_order if name in record["splits"]]
        ordered_assignments.append(record)
    _atomic_write(
        destination / "vitaminc_split_assignments.jsonl",
        _jsonl_bytes(ordered_assignments),
    )
    return summaries


def _require_frozen_counts(
    label: str, actual: Mapping[str, Any], frozen: Any, key_map: Mapping[str, str]
) -> None:
    if not isinstance(frozen, Mapping):
        raise ValueError(f"missing frozen definition for {label}")
    for actual_key, frozen_key in key_map.items():
        if actual[actual_key] != frozen.get(frozen_key):
            raise ValueError(
                f"{label} {actual_key} mismatch: expected {frozen.get(frozen_key)}, "
                f"got {actual[actual_key]}"
            )


def _tabfact_record(claim: TabFactClaim, tables_path: str) -> dict[str, Any]:
    return {
        "caption": claim.caption,
        "claim": claim.claim,
        "claim_index": claim.claim_index,
        "label": claim.label,
        "sample_id": claim.sample_id,
        "table_id": claim.table_id,
        "table_path": (Path(tables_path) / claim.table_id).as_posix(),
    }


def _write_tabfact(
    destination: Path,
    tabfact: Any,
    seed: str,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(tabfact, Mapping):
        return {}
    required = ("test_path", "tables_path", "old_evaluation_path")
    if any(not isinstance(tabfact.get(key), str) for key in required):
        return {}
    claims = load_tabfact_examples(
        root / tabfact["test_path"], root / tabfact["tables_path"]
    )
    old_rows = read_jsonl(root / tabfact["old_evaluation_path"])
    overlap = extract_old_table_ids(old_rows)
    eligible = fresh_eligible(claims, overlap)
    selected = select_fresh_subset(eligible, seed)
    per_table = Counter(row.table_id for row in selected)
    summary = {
        "official": {
            "claims": len(claims),
            "tables": len({row.table_id for row in claims}),
        },
        "old": {"claims": overlap.rows, "tables": len(overlap.table_ids)},
        "eligible": {
            "claims": len(eligible),
            "tables": len({row.table_id for row in eligible}),
        },
        "selected": {
            "claims": len(selected),
            "id_manifest_sha256": id_manifest_sha256(
                row.sample_id for row in selected
            ),
            "max_per_table": max(per_table.values()),
            "negative": sum(row.label == 0 for row in selected),
            "positive": sum(row.label == 1 for row in selected),
            "tables": len(per_table),
        },
    }
    _require_frozen_counts(
        "tabfact.official_test",
        summary["official"],
        tabfact.get("official_test"),
        {"claims": "examples", "tables": "tables"},
    )
    _require_frozen_counts(
        "tabfact.old_used",
        summary["old"],
        tabfact.get("old_used"),
        {"claims": "examples", "tables": "tables"},
    )
    _require_frozen_counts(
        "tabfact.fresh_eligible",
        summary["eligible"],
        tabfact.get("fresh_eligible"),
        {"claims": "examples", "tables": "tables"},
    )
    _require_frozen_counts(
        "tabfact.frozen_subset",
        summary["selected"],
        tabfact.get("frozen_subset"),
        {
            "claims": "examples",
            "tables": "tables",
            "positive": "positive",
            "negative": "negative",
            "max_per_table": "max_per_table",
            "id_manifest_sha256": "id_sha256",
        },
    )
    _atomic_write(
        destination / "tabfact_old_table_ids.txt",
        canonical_id_manifest(overlap.table_ids),
    )
    _atomic_write(
        destination / "tabfact_fresh_ids.txt",
        canonical_id_manifest(row.sample_id for row in selected),
    )
    ordered = sorted(selected, key=lambda row: row.sample_id)
    _atomic_write(
        destination / "tabfact_fresh_claims.jsonl",
        _jsonl_bytes(_tabfact_record(row, tabfact["tables_path"]) for row in ordered),
    )
    _atomic_write(destination / "tabfact_summary.json", _json_bytes(summary))
    return summary


def _write_cub(
    destination: Path,
    cub: Any,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(cub, Mapping) or not isinstance(cub.get("path"), str):
        return {}
    repository = root / cub["path"]
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != cub.get("commit"):
        raise ValueError(
            f"CUB repository commit mismatch: expected {cub.get('commit')}, got {commit}"
        )
    source_files = []
    for path in sorted(item for item in repository.iterdir() if item.is_file()):
        source_files.append(
            {
                "bytes": path.stat().st_size,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    files_payload = _json_bytes({"commit": commit, "files": source_files})
    files_digest = hashlib.sha256(files_payload).hexdigest()
    artifact = load_cub_test(
        repository,
        expected_variant_count=int(cub["variant_files"]),
        expected_rows_per_file=int(cub["test_rows_per_file"]),
    )
    records_payload = _jsonl_bytes(
        record.to_manifest_record() for record in artifact.records
    )
    ids_payload = canonical_id_manifest(record.sample_id for record in artifact.records)
    records_digest = hashlib.sha256(records_payload).hexdigest()
    ids_digest = hashlib.sha256(ids_payload).hexdigest()
    context_types = dict(
        sorted(Counter(record.context_type for record in artifact.records).items())
    )
    analysis_context_types = dict(
        sorted(
            Counter(record.analysis_context_type for record in artifact.records).items()
        )
    )
    summary = {
        "analysis_context_types": analysis_context_types,
        "context_types": context_types,
        "files_manifest_sha256": files_digest,
        "prediction_fields_included": False,
        "repository_commit": commit,
        "rows_across_variants": len(artifact.records) * len(artifact.variant_files),
        "source_files": len(source_files),
        "test_ids_sha256": ids_digest,
        "test_records_sha256": records_digest,
        "unique_contexts": len(
            {record.context_identity_sha256 for record in artifact.records}
        ),
        "unique_samples": len({record.sample_id for record in artifact.records}),
        "variant_files": len(artifact.variant_files),
    }
    expected = {
        "analysis_context_types": cub.get("analysis_context_types"),
        "context_types": cub.get("context_types"),
        "files_manifest_sha256": cub.get("files_manifest_sha256"),
        "repository_commit": cub.get("commit"),
        "source_files": cub.get("source_files"),
        "test_ids_sha256": cub.get("test_ids_sha256"),
        "test_records_sha256": cub.get("test_records_sha256"),
        "unique_contexts": cub.get("unique_contexts"),
        "unique_samples": cub.get("unique_samples"),
        "variant_files": cub.get("variant_files"),
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(
                f"cub_druid {key} mismatch: expected {value}, got {summary[key]}"
            )
    _atomic_write(destination / "cub_files.json", files_payload)
    _atomic_write(destination / "cub_test.jsonl", records_payload)
    _atomic_write(destination / "cub_test_ids.txt", ids_payload)
    _atomic_write(destination / "cub_summary.json", _json_bytes(summary))
    return summary


def _build_data_foundation_to_directory(
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify raw inputs and write strict VitaminC foundation manifests."""
    root = Path(project_root).resolve()
    config = load_freeze_config(config_path)
    verified = verify_frozen_files(root, config)
    vitamin = config.get("vitaminc")
    if not isinstance(vitamin, Mapping) or not isinstance(vitamin.get("test_path"), str):
        raise ValueError("vitaminc.test_path is required")
    rows = read_jsonl(root / vitamin["test_path"])
    audit = build_strict_quartet_audit(rows)
    quartets = list(audit.quartets)
    quartets.sort(key=lambda quartet: quartet.case_id)
    summary = _strict_summary(quartets)
    _verify_strict_freeze(summary, config)

    destination = Path(output_dir)
    _atomic_write(
        destination / "raw_provenance.json",
        _json_bytes(
            {
                "files": dict(sorted(verified.items())),
                "protocol_version": config.get("protocol_version"),
            }
        ),
    )
    _atomic_write(
        destination / "vitaminc_strict_ids.txt",
        canonical_id_manifest(quartet.case_id for quartet in quartets),
    )
    _atomic_write(
        destination / "vitaminc_strict_quartets.jsonl",
        _jsonl_bytes(_quartet_record(quartet) for quartet in quartets),
    )
    _atomic_write(
        destination / "vitaminc_strict_exclusions.jsonl",
        _jsonl_bytes(_exclusion_record(item) for item in audit.exclusions),
    )
    exclusion_summary = {
        "excluded_records": len(audit.exclusions),
        "reasons": dict(
            sorted(Counter(item.reason for item in audit.exclusions).items())
        ),
    }
    _atomic_write(
        destination / "vitaminc_strict_exclusions_summary.json",
        _json_bytes(exclusion_summary),
    )
    _atomic_write(destination / "vitaminc_strict_summary.json", _json_bytes(summary))
    split_summaries = _write_vitaminc_splits(
        destination,
        vitamin,
        str(config.get("seed", "")),
        rows,
        quartets,
        root,
    )
    tabfact_summary = _write_tabfact(
        destination, config.get("tabfact"), str(config.get("seed", "")), root
    )
    cub_summary = _write_cub(destination, config.get("cub_druid"), root)
    return {
        "strict_quartets": summary["groups"],
        **summary,
        "splits": {name: value["groups"] for name, value in split_summaries.items()},
        "tabfact_fresh": tabfact_summary.get("selected", {}).get("claims"),
        "cub_unique_samples": cub_summary.get("unique_samples"),
    }


def build_data_foundation(
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build in staging, then publish one verifiable manifest bundle."""
    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve()
    frozen_path = Path(config_path)
    if not frozen_path.is_absolute():
        frozen_path = root / frozen_path
    config = load_freeze_config(frozen_path)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.build-", dir=destination.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        summary = _build_data_foundation_to_directory(
            root, frozen_path, staging
        )
        _atomic_write(
            staging / "manifest_index.json",
            _manifest_index_payload(
                staging,
                config.get("protocol_version"),
                hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            ),
        )
        _publish_manifest_bundle(staging, destination)
    verify_data_foundation(root, frozen_path, destination)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/freeze_v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests")
    )
    arguments = parser.parse_args(argv)
    root = arguments.project_root.resolve()
    config = arguments.config if arguments.config.is_absolute() else root / arguments.config
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    summary = build_data_foundation(root, config, output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0
