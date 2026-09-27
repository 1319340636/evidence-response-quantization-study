"""Write or verify the candidate release's file and record integrity."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
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
        parts = path.relative_to(ROOT).parts
        if (any(part in {".git", "__pycache__", ".pytest_cache", ".venv"}
                or part.endswith(".egg-info") for part in parts)
            or parts == ("uv.lock",)):
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


def verify_stage_a(root: Path) -> dict[str, int]:
    """Recompute four-model Stage A point estimates from source-bound safe rows."""

    data = root / "data/stage_a"
    records_path = data / "analysis_records.jsonl"
    manifest_path = data / "analysis_records_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (data / "analysis_records_manifest.json.sha256").read_text(encoding="utf-8").split() != [
        digest(manifest_path), manifest_path.name,
    ]:
        raise ValueError("Stage A detached manifest hash mismatch")
    canonical = json.dumps(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("manifest_sha256"):
        raise ValueError("Stage A manifest content hash mismatch")
    if manifest.get("dataset_text_included") is not False or digest(records_path) != manifest.get("records_jsonl_sha256"):
        raise ValueError("Stage A safe record hash mismatch")
    ids_path = root / "data/manifests/vitaminc_dose_subset_ids.txt"
    if digest(ids_path) != manifest.get("source_sha256", {}).get("subset_ids"):
        raise ValueError("Stage A subset manifest hash mismatch")
    selected = set(ids_path.read_text(encoding="utf-8").splitlines())
    if len(selected) != 1200 or manifest.get("paired_quartets") != 1200 or manifest.get("paired_pages") != 1078:
        raise ValueError("Stage A frozen population mismatch")
    expected = {
        "gemma_f16": ("gemma4_e4b", "F16"),
        "gemma_q4": ("gemma4_e4b", "Q4_K_M"),
        "ministral_f16": ("ministral3_8b", "F16"),
        "ministral_q4": ("ministral3_8b", "Q4_K_M"),
        "olmo_f16": ("olmo3_7b", "F16"),
        "olmo_q4": ("olmo3_7b", "Q4_K_M"),
        "qwen_f16": ("qwen35_9b", "F16"),
        "qwen_q4": ("qwen35_9b", "Q4_K_M"),
    }
    if manifest.get("conditions") != {condition: 1200 for condition in expected}:
        raise ValueError("Stage A condition counts mismatch")
    rows: dict[str, dict[str, dict]] = {condition: {} for condition in expected}
    for line in records_path.open("rb"):
        if any(marker in line for marker in FORBIDDEN):
            raise ValueError("Stage A safe record includes text, credential, or host path")
        row = json.loads(line)
        condition = row.get("condition")
        if condition not in expected:
            raise ValueError("Stage A unexpected condition")
        if (row.get("model_key"), row.get("precision")) != expected[condition]:
            raise ValueError("Stage A model or precision mismatch")
        case_id = row.get("case_id")
        if case_id not in selected or case_id in rows[condition]:
            raise ValueError("Stage A duplicate or unexpected quartet")
        cells = row.get("cells")
        negative_choice = {"REFUTES": "B", "NOT ENOUGH INFO": "C"}.get(row.get("negative_label"))
        if (not isinstance(cells, list) or len(cells) != 4 or negative_choice is None
            or [cell.get("cell_index") for cell in cells] != [0, 1, 2, 3]):
            raise ValueError("Stage A quartet cells incomplete")
        margins = [cell["choice_logprobs"]["A"] - cell["choice_logprobs"][negative_choice] for cell in cells]
        value = (margins[0] - margins[1] - margins[2] + margins[3]) / 2
        if not math.isclose(value, row.get("quartet_interaction"), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Stage A quartet interaction mismatch")
        rows[condition][case_id] = row
    if any(set(group) != selected for group in rows.values()):
        raise ValueError("Stage A cross-condition membership mismatch")
    pages = {case_id: rows["gemma_f16"][case_id]["page"] for case_id in selected}
    if len(set(pages.values())) != 1078:
        raise ValueError("Stage A common page count mismatch")
    for case_id in selected:
        reference = rows["gemma_f16"][case_id]
        if any((group[case_id]["page"], group[case_id]["negative_label"])
               != (reference["page"], reference["negative_label"])
               for group in rows.values()):
            raise ValueError("Stage A cross-condition metadata mismatch")
    report = json.loads((root / "reports/vitaminc_stage_a_v4/results.json").read_text(encoding="utf-8"))
    for short, model in (("gemma", "gemma4_e4b"), ("ministral", "ministral3_8b"),
                         ("olmo", "olmo3_7b"), ("qwen", "qwen35_9b")):
        f16, q4 = rows[f"{short}_f16"], rows[f"{short}_q4"]
        page_effects: dict[str, list[float]] = defaultdict(list)
        for case_id in selected:
            page_effects[pages[case_id]].append(
                q4[case_id]["quartet_interaction"] - f16[case_id]["quartet_interaction"]
            )
        estimate = statistics.fmean(statistics.fmean(values) for values in page_effects.values())
        target = report["families"][model]
        if not math.isclose(estimate, target["page_effect"]["estimate"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Stage A {model} page effect mismatch")
        for condition, key in ((f16, "f16"), (q4, "quantized")):
            cells = [cell for row in condition.values() for cell in row["cells"]]
            accuracy = sum(cell["hard_scored_label"] == cell["gold_label"] for cell in cells) / len(cells)
            balanced = statistics.fmean(
                sum(cell["hard_scored_label"] == label for cell in cells if cell["gold_label"] == label)
                / sum(cell["gold_label"] == label for cell in cells)
                for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
            )
            if not math.isclose(accuracy, target[f"hard_accuracy_{key}"], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Stage A {model} hard accuracy mismatch")
            if not math.isclose(balanced, target["balanced_accuracy"][key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Stage A {model} balanced accuracy mismatch")
    return {"conditions": len(expected), "quartets": len(selected), "pages": len(set(pages.values()))}


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


def verify_cub(root: Path) -> None:
    """Recompute the frozen CUB/DRUID point estimates from paired safe rows."""

    sys.path.insert(0, str(root / "src"))
    from qer_fv.cub_statistics import PairedCubObservation, cluster_effects

    data = root / "data/cub_druid"
    records_path = data / "analysis_records.jsonl"
    manifest_path = data / "analysis_records_manifest.json"
    sidecar = data / "analysis_records_manifest.json.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sidecar.read_text(encoding="utf-8").split() != [digest(manifest_path), manifest_path.name]:
        raise ValueError("CUB detached manifest hash mismatch")
    canonical = json.dumps(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("manifest_sha256"):
        raise ValueError("CUB manifest content hash mismatch")
    if digest(records_path) != manifest.get("records_jsonl_sha256"):
        raise ValueError("CUB records hash mismatch")
    if manifest.get("dataset_text_included") is not False:
        raise ValueError("CUB text-free flag missing")
    comparisons = {
        "ministral3_3b_q4": ("ministral3_3b", "Q4_K_M"),
        "ministral3_8b_q4": ("ministral3_8b", "Q4_K_M"),
        "qwen35_4b_q4": ("qwen35_4b", "Q4_K_M"),
        "qwen35_9b_q4": ("qwen35_9b", "Q4_K_M"),
        "qwen35_9b_q5": ("qwen35_9b", "Q5_K_M"),
        "qwen35_9b_q8": ("qwen35_9b", "Q8_0"),
    }
    by_comparison: dict[str, list[PairedCubObservation]] = {name: [] for name in comparisons}
    allowed = set(PairedCubObservation.__dataclass_fields__) | {
        "comparison", "model_key", "quantization",
    }
    for line in records_path.open("rb"):
        if any(marker in line for marker in FORBIDDEN) or b'"claimant":' in line:
            raise ValueError("CUB record includes source text, credential, or host path")
        row = json.loads(line)
        if set(row) != allowed:
            raise ValueError("CUB record field allowlist mismatch")
        comparison = row.pop("comparison")
        if comparison not in comparisons:
            raise ValueError("CUB comparison identity mismatch")
        model, precision = comparisons[comparison]
        if (row.pop("model_key"), row.pop("quantization")) != (model, precision):
            raise ValueError("CUB model or precision mismatch")
        by_comparison[comparison].append(PairedCubObservation(**row))
    if manifest.get("comparisons") != {name: 4302 for name in comparisons}:
        raise ValueError("CUB manifest count mismatch")
    for comparison, rows in by_comparison.items():
        if len(rows) != 4302 or len({row.sample_id for row in rows}) != 4302:
            raise ValueError("CUB sample count or identity mismatch")
        report_name = f"{comparison}.json" if comparison.startswith("qwen35_9b_") else f"{comparisons[comparison][0]}.json"
        report = json.loads((root / "results/cub_druid" / report_name).read_text())
        if (report.get("model_key"), report.get("quantization")) != comparisons[comparison]:
            raise ValueError("CUB frozen report identity mismatch")
        for context, expected_count in (("gold", 1872), ("conflicting", 2413)):
            selected = [row for row in rows if row.analysis_context_type == context]
            if len(selected) != expected_count:
                raise ValueError("CUB context count mismatch")
            for metric in ("bcu", "ccu"):
                complete = selected if metric == "bcu" else [
                    row for row in selected
                    if row.ccu_f16 is not None and row.ccu_quantized is not None
                ]
                effects = cluster_effects(complete, metric=metric, context_type=context)
                estimate = statistics.fmean(effects.values())
                if abs(estimate - report[context][metric]["estimate"]) > 1e-12:
                    raise ValueError(f"CUB {comparison} {context} {metric} estimate mismatch")
        irrelevant = [row for row in rows if row.analysis_context_type == "irrelevant"]
        if len(irrelevant) != 17:
            raise ValueError("CUB irrelevant context count mismatch")
        bcu = statistics.fmean(float(row.bcu_quantized) - float(row.bcu_f16) for row in irrelevant)
        ccu = statistics.fmean(
            row.ccu_quantized - row.ccu_f16 for row in irrelevant
            if row.ccu_f16 is not None and row.ccu_quantized is not None
        )
        if (abs(bcu - report["irrelevant"]["bcu_effect"]) > 1e-12
            or abs(ccu - report["irrelevant"]["ccu_effect"]) > 1e-12):
            raise ValueError(f"CUB {comparison} irrelevant estimate mismatch")


def resolve_code_source_key(path: Path, key: str) -> object:
    """Read frozen GPTQ factory literals without importing or running route code."""

    parts = key.split(".")
    if len(parts) != 3 or parts[0] != "GPTQRouteConfig" or parts[2] not in {"bits", "group_size"}:
        raise ValueError(f"unsupported code source key: {key}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == parts[0]:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == parts[1]:
                    for expression in ast.walk(method):
                        if (isinstance(expression, ast.Return)
                            and isinstance(expression.value, ast.Call)
                            and isinstance(expression.value.func, ast.Name)
                            and expression.value.func.id == "cls"):
                            for keyword in expression.value.keywords:
                                if keyword.arg == parts[2]:
                                    return ast.literal_eval(keyword.value)
    raise ValueError(f"code source key not found: {key}")


def resolve_json_source_key(source: object, key: str) -> object:
    if key.endswith("/units,/agreement"):
        # Legacy table metadata denotes a derived changed-label count, not a JSON pointer.
        parent = resolve_json_source_key(source, key.removesuffix("/units,/agreement"))
        return round(parent["units"] * (1 - parent["agreement"]))
    if key.startswith("/"):
        segments = [part.replace("~1", "/").replace("~0", "~") for part in key[1:].split("/")]
        value = source
    else:
        match = re.fullmatch(r"rows\[row_id=([0-9a-f]+)\]\.(.+)", key)
        if not match or not isinstance(source, dict):
            raise ValueError(f"unsupported JSON source key: {key}")
        rows = [row for row in source["rows"] if row["row_id"] == match.group(1)]
        if len(rows) != 1:
            raise ValueError(f"source row identity not unique: {key}")
        value = rows[0]
        segments = match.group(2).split(".")
    for segment in segments:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)(?:\[([0-9]+)\])?", segment)
        if not key.startswith("/") and match:
            value = value[match.group(1)]
            if match.group(2) is not None:
                value = value[int(match.group(2))]
        elif isinstance(value, list):
            value = value[int(segment)]
        else:
            value = value[segment]
    return value


def _source_nodes(node: object):
    if isinstance(node, dict):
        if "source_path" in node and "source_sha256" in node:
            yield node
        for value in node.values():
            yield from _source_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _source_nodes(value)


def verify_manuscript_source_links(root: Path) -> dict[str, int]:
    """Resolve published figure/table source paths, hashes, and marked values."""

    sources: dict[str, tuple[Path, object | None]] = {}
    references = 0
    values = 0
    for directory in (root / "manuscript_source/figures", root / "manuscript_source/tables"):
        for artifact in sorted(directory.glob("*.json")):
            if "manifest" in artifact.name:
                continue
            for node in _source_nodes(json.loads(artifact.read_text(encoding="utf-8"))):
                relative = node["source_path"]
                parts = PurePosixPath(relative).parts
                if (not parts or PurePosixPath(relative).is_absolute()
                    or any(part in {".", ".."} for part in parts)
                    or "\\" in relative or ":" in relative):
                    raise ValueError(f"unsafe manuscript source path: {relative}")
                path = root.joinpath(*parts)
                if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError(f"manuscript source missing: {relative}")
                if relative not in sources:
                    if digest(path) != node["source_sha256"]:
                        raise ValueError(f"source SHA-256 mismatch: {relative}")
                    content = None if path.suffix == ".py" else json.loads(path.read_text(encoding="utf-8"))
                    sources[relative] = (path, content)
                elif digest(path) != node["source_sha256"]:
                    raise ValueError(f"source SHA-256 mismatch: {relative}")
                references += 1
                for field in ("value", "lower", "upper"):
                    source_key = "source_key" if field == "value" else f"{field}_source_key"
                    if source_key not in node:
                        continue
                    if field not in node:
                        raise ValueError(f"marked field missing: {artifact.name}:{field}")
                    actual = (resolve_code_source_key(path, node[source_key]) if path.suffix == ".py"
                              else resolve_json_source_key(sources[relative][1], node[source_key]))
                    expected = node[field]
                    equal = (math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                             if isinstance(actual, (int, float)) and isinstance(expected, (int, float))
                             else actual == expected)
                    if not equal:
                        raise ValueError(f"source value mismatch: {artifact.name}:{node[source_key]}")
                    values += 1
    return {"source_references": references, "distinct_sources": len(sources), "value_links": values}


def verify_mapping_display_sources(root: Path) -> dict[str, int]:
    """Match every balanced-mapping plot row and main-table cell to the frozen report."""

    report = json.loads((root / "results/mapping/scale_sensitivity.json").read_text(encoding="utf-8"))
    analysis = report["analysis"]
    csv_path = root / "manuscript_source/figures/fig_balanced_mapping.csv"
    seen = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["mapping"], row["pair"], row["contrast"], row["endpoint"])
            if key in seen:
                raise ValueError(f"duplicate mapping figure row: {key}")
            seen.add(key)
            source = (analysis[key[0]]["weighting_sensitivities"]["page_balanced"]
                      ["interactions"][key[1]][key[2]]["endpoints"][key[3]])
            for field in ("estimate", "lower", "upper", "holm_adjusted_pvalue"):
                if not math.isclose(float(row[field]), source[field], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"mapping figure value mismatch: {key}:{field}")
    if len(seen) != 36:
        raise ValueError("mapping figure row count mismatch")

    pair_names = {
        "Ministral--Qwen": "ministral_minus_qwen",
        "OLMo--Qwen": "olmo_minus_qwen",
        "OLMo--Ministral": "olmo_minus_ministral",
    }
    contrast_names = {"AWQ--FP16": "awq_minus_fp16", "AWQ--GPTQ": "awq_minus_gptq"}
    table_cells = 0
    table_path = root / "manuscript_source/tables/tab_balanced_mapping.tex"
    for line in table_path.read_text(encoding="utf-8").splitlines():
        if not any(line.startswith(name + " &") for name in pair_names):
            continue
        parts = [part.strip() for part in line.removesuffix("\\\\").split("&")]
        if len(parts) != 5 or parts[0] not in pair_names or parts[1] not in contrast_names:
            raise ValueError("balanced mapping table layout mismatch")
        pair, contrast = pair_names[parts[0]], contrast_names[parts[1]]
        for mapping, cell in zip(("original", "cycle_1", "cycle_2"), parts[2:]):
            match = re.fullmatch(r"\$(-?\d+\.\d{3})\\;\((-?\d+\.\d{3})\)\$", cell)
            if not match:
                raise ValueError("balanced mapping table cell format mismatch")
            endpoints = analysis[mapping]["weighting_sensitivities"]["page_balanced"]["interactions"][pair][contrast]["endpoints"]
            expected = (
                f"{endpoints['reference_standardized_route_effect']['estimate']:.3f}",
                f"{endpoints['directional_common_language_effect']['estimate']:.3f}",
            )
            if match.groups() != expected:
                raise ValueError(f"balanced mapping table value mismatch: {mapping}:{pair}:{contrast}")
            table_cells += 1
    if table_cells != 18:
        raise ValueError("balanced mapping table cell count mismatch")
    return {"figure_rows": len(seen), "table_cells": table_cells}


def verify_data() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from qer_fv.three_model_records import load_text_free_three_model_inputs

    verify_confirmatory(ROOT)
    verify_stage_a(ROOT)
    verify_tabfact(ROOT)
    verify_cub(ROOT)
    verify_manuscript_source_links(ROOT)
    verify_mapping_display_sources(ROOT)
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
