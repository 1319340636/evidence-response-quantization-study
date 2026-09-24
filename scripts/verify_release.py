"""Write or verify the candidate release's file and record integrity."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"
FORBIDDEN = (
    b'"raw_response":', b'"prompt":', b'"content":',
    b'"claim":', b'"evidence":', b'"text":',
    b'"api_key":', b'"access_token":', b'"password":',
    b'/root/', b'C:\\Users\\', b'D:\\',
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def release_files() -> dict[str, Path]:
    result = {}
    for path in ROOT.rglob("*"):
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.relative_to(ROOT).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink forbidden: {path}")
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT).as_posix()
        result[relative] = path
    return result


def write_manifest() -> None:
    lines = [f"{digest(path)}  {name}\n" for name, path in sorted(release_files().items())]
    MANIFEST.write_text("".join(lines), encoding="ascii")


def verify_manifest() -> None:
    expected = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        sha, name = line.split("  ", 1)
        if name in expected or len(sha) != 64:
            raise ValueError("duplicate or malformed manifest entry")
        expected[name] = sha
    actual = release_files()
    if set(expected) != set(actual):
        raise ValueError("release file set differs from SHA256SUMS.txt")
    for name, path in sorted(actual.items()):
        if digest(path) != expected[name]:
            raise ValueError(f"SHA-256 mismatch: {name}")


def verify_archive(path: Path, *, files: int, rows_each: int) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            parts = PurePosixPath(name).parts
            if not member.isfile() or not parts or any(part == ".." for part in parts):
                raise ValueError(f"unsafe archive member: {member.name}")
            if name in members:
                raise ValueError(f"duplicate archive member: {name}")
            members[name] = member
        records = sorted(name for name in members if name.endswith("/export/records.jsonl"))
        if len(records) != files or len(members) != 2 * files:
            raise ValueError(f"unexpected archive file count: {path}")
        for name in records:
            manifest_name = name.removesuffix("records.jsonl") + "manifest.json"
            if manifest_name not in members:
                raise ValueError(f"missing manifest: {name}")
            manifest_file = archive.extractfile(members[manifest_name])
            assert manifest_file is not None
            manifest = json.load(manifest_file)
            record_file = archive.extractfile(members[name])
            assert record_file is not None
            sha = hashlib.sha256()
            count = 0
            for line in record_file:
                sha.update(line)
                if any(marker in line for marker in FORBIDDEN):
                    raise ValueError(f"text, credential, or host path in: {name}")
                row = json.loads(line)
                if (row.get("control") != "full"
                    or row.get("hard_status") != "ok"
                    or row.get("probability_status") != "ok"
                    or row.get("hard_error_code") is not None
                    or row.get("probability_error_code") is not None):
                    raise ValueError(f"noncomplete row in: {name}")
                count += 1
            if count != rows_each or sha.hexdigest() != manifest.get("records_jsonl_sha256"):
                raise ValueError(f"count or embedded hash mismatch: {name}")


def verify_confirmatory(root: Path) -> None:
    """Reconstruct the frozen primary estimate and hard-label metric from safe rows."""

    data = root / "data/confirmatory"
    records_path = data / "analysis_records.jsonl"
    manifest_path = data / "analysis_records_manifest.json"
    sidecar_path = data / "analysis_records_manifest.json.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sidecar_path.read_text(encoding="utf-8").split() != [
        digest(manifest_path), manifest_path.name,
    ]:
        raise ValueError("confirmatory detached manifest hash mismatch")
    canonical = json.dumps(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("manifest_sha256"):
        raise ValueError("confirmatory manifest content hash mismatch")
    if digest(records_path) != manifest.get("records_jsonl_sha256"):
        raise ValueError("confirmatory records hash mismatch")
    if manifest.get("dataset_text_included") is not False:
        raise ValueError("confirmatory text-free flag missing")

    by_condition: dict[str, dict[str, dict]] = {"f16": {}, "q4": {}}
    for line in records_path.open("rb"):
        if any(marker in line for marker in FORBIDDEN):
            raise ValueError("confirmatory record includes text, credential, or host path")
        row = json.loads(line)
        condition = row.get("condition")
        if condition not in by_condition:
            raise ValueError("confirmatory unexpected condition")
        if row.get("model_key") != "qwen35_9b" or row.get("precision") != {
            "f16": "F16", "q4": "Q4_K_M",
        }[condition]:
            raise ValueError("confirmatory model or precision mismatch")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in by_condition[condition]:
            raise ValueError("confirmatory duplicate or invalid quartet")
        cells = row.get("cells")
        if not isinstance(cells, list) or len(cells) != 4:
            raise ValueError("confirmatory quartet cells incomplete")
        negative_choice = {"REFUTES": "B", "NOT ENOUGH INFO": "C"}.get(
            row.get("negative_label")
        )
        if negative_choice is None or [c.get("cell_index") for c in cells] != [0, 1, 2, 3]:
            raise ValueError("confirmatory quartet cell identity mismatch")
        margins = []
        for cell in cells:
            scores = cell.get("choice_logprobs")
            if not isinstance(scores, dict) or set(scores) != {"A", "B", "C"}:
                raise ValueError("confirmatory choice scores incomplete")
            margins.append(scores["A"] - scores[negative_choice])
        interaction = (margins[0] - margins[1] - margins[2] + margins[3]) / 2
        if interaction != row.get("quartet_interaction"):
            raise ValueError("confirmatory quartet interaction mismatch")
        by_condition[condition][case_id] = row

    f16, q4 = by_condition["f16"], by_condition["q4"]
    if len(f16) != 2483 or set(f16) != set(q4):
        raise ValueError("confirmatory paired quartet count mismatch")
    if manifest.get("conditions") != {"f16": 2483, "q4": 2483}:
        raise ValueError("confirmatory manifest condition count mismatch")
    page_values: dict[str, list[float]] = defaultdict(list)
    all_effects: list[float] = []
    for case_id, f16_row in f16.items():
        q4_row = q4[case_id]
        if (f16_row["page"], f16_row["negative_label"]) != (
            q4_row["page"], q4_row["negative_label"]
        ):
            raise ValueError("confirmatory quartet metadata mismatch")
        effect = q4_row["quartet_interaction"] - f16_row["quartet_interaction"]
        page_values[f16_row["page"]].append(effect)
        all_effects.append(effect)
    if len(page_values) != 1158 or manifest.get("paired_pages") != 1158:
        raise ValueError("confirmatory page count mismatch")
    primary = json.loads((root / "results/confirmatory/primary_results.json").read_text())
    estimate = statistics.fmean(statistics.fmean(v) for v in page_values.values())
    quartet_estimate = statistics.fmean(all_effects)
    if abs(estimate - primary["primary"]["estimate"]) > 1e-12:
        raise ValueError("confirmatory primary estimate mismatch")
    if abs(quartet_estimate - primary["primary"]["quartet_weighted_sensitivity"]) > 1e-12:
        raise ValueError("confirmatory quartet-weighted estimate mismatch")
    hard = json.loads((root / "results/confirmatory/hard_metrics.json").read_text())
    for condition, key in (("f16", "f16"), ("q4", "q4")):
        cells = [cell for row in by_condition[condition].values() for cell in row["cells"]]
        labels = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
        balanced_accuracy = statistics.fmean(
            sum(cell["hard_scored_label"] == label for cell in cells if cell["gold_label"] == label)
            / sum(cell["gold_label"] == label for cell in cells)
            for label in labels
        )
        if abs(balanced_accuracy - hard["balanced_accuracy"][key]) > 1e-12:
            raise ValueError(f"confirmatory {condition} Balanced Accuracy mismatch")


def verify_tabfact(root: Path) -> None:
    """Check source-bound safe rows and recalculate the four frozen diagnostics."""

    data = root / "data/tabfact"
    records_path = data / "analysis_records.jsonl"
    manifest_path = data / "analysis_records_manifest.json"
    sidecar = data / "analysis_records_manifest.json.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sidecar.read_text(encoding="utf-8").split() != [digest(manifest_path), manifest_path.name]:
        raise ValueError("TabFact detached manifest hash mismatch")
    canonical = json.dumps(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("manifest_sha256"):
        raise ValueError("TabFact manifest content hash mismatch")
    if digest(records_path) != manifest.get("records_jsonl_sha256"):
        raise ValueError("TabFact records hash mismatch")
    if manifest.get("dataset_text_included") is not False:
        raise ValueError("TabFact text-free flag missing")
    conditions = {
        "qwen_f16": "qwen35_9b_F16",
        "qwen_q4": "qwen35_9b_Q4_K_M",
        "gemma_f16": "gemma4_e4b_F16",
        "gemma_q4": "gemma4_e4b_Q4_K_M",
    }
    by_condition: dict[str, dict[str, dict]] = {name: {} for name in conditions}
    for line in records_path.open("rb"):
        if any(marker in line for marker in FORBIDDEN) or b'"table_rows":' in line:
            raise ValueError("TabFact record includes source text, credential, or host path")
        row = json.loads(line)
        condition = row.get("condition")
        if condition not in by_condition:
            raise ValueError("TabFact unexpected condition")
        if f"{row.get('model_key')}_{row.get('precision')}" != conditions[condition]:
            raise ValueError("TabFact model or precision mismatch")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_condition[condition]:
            raise ValueError("TabFact duplicate or invalid sample")
        if row.get("gold_label") not in ("Entailed", "Refuted"):
            raise ValueError("TabFact invalid gold label")
        if row.get("hard_scored_label") not in ("Entailed", "Refuted"):
            raise ValueError("TabFact invalid scored label")
        if set(row.get("choice_logprobs", {})) != {"A", "B"}:
            raise ValueError("TabFact choice scores incomplete")
        by_condition[condition][sample_id] = row
    if manifest.get("conditions") != {name: 2000 for name in conditions}:
        raise ValueError("TabFact manifest condition count mismatch")
    reference: dict[str, tuple[str, int, str]] | None = None
    metrics = json.loads((root / "results/tabfact/results.json").read_text())["fresh_tabfact_formal"]["condition_metrics"]
    for condition, published_name in conditions.items():
        records = by_condition[condition]
        if len(records) != 2000:
            raise ValueError("TabFact sample count mismatch")
        metadata = {sample_id: (row["table_id"], row["claim_index"], row["gold_label"])
                    for sample_id, row in records.items()}
        if reference is None:
            reference = metadata
        elif metadata != reference:
            raise ValueError("TabFact paired membership mismatch")
        rows = list(records.values())
        accuracy = sum(r["gold_label"] == r["hard_scored_label"] for r in rows) / len(rows)
        balanced_accuracy = statistics.fmean(
            sum(r["hard_scored_label"] == label for r in rows if r["gold_label"] == label)
            / sum(r["gold_label"] == label for r in rows)
            for label in ("Entailed", "Refuted")
        )
        frozen = metrics[published_name]
        if abs(accuracy - frozen["accuracy"]) > 1e-12:
            raise ValueError(f"TabFact {condition} accuracy mismatch")
        if abs(balanced_accuracy - frozen["balanced_accuracy"]) > 1e-12:
            raise ValueError(f"TabFact {condition} Balanced Accuracy mismatch")


def verify_data() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from qer_fv.three_model_records import load_text_free_three_model_inputs

    verify_confirmatory(ROOT)
    verify_tabfact(ROOT)
    values, pages = load_text_free_three_model_inputs(ROOT / "data/three_model", formal=True)
    if len(values) != 3 or sum(map(len, values.values())) != 9 or len(set(pages.values())) != 1078:
        raise ValueError("three-model release structure mismatch")
    verify_archive(ROOT / "data/mapping/mapping_exports.tar.gz", files=27, rows_each=4800)
    verify_archive(ROOT / "data/repeatability/repeatability_exports.tar.gz", files=81, rows_each=480)
    gate = json.loads((ROOT / "results/mapping/formal_gate_v2.json").read_text())
    for name in ("results.json", "scale_sensitivity.json"):
        result = json.loads((ROOT / "results/mapping" / name).read_text())
        if not gate.get("gate_passed") or result.get("formal_gate_sha256") != gate.get("gate_sha256"):
            raise ValueError("mapping formal gate binding mismatch")
    done = json.loads((ROOT / "results/repeatability/done.json").read_text())
    if done.get("report_sha256") != digest(ROOT / "results/repeatability/repeatability_report.json"):
        raise ValueError("repeatability report binding mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        verify_data()
        write_manifest()
    else:
        verify_manifest()
        verify_data()
    print("Release candidate integrity verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
