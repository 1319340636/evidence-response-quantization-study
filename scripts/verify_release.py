"""Write or verify the candidate release's file and record integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
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
        if any(part in {".git", "__pycache__"} for part in path.relative_to(ROOT).parts):
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


def verify_data() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from qer_fv.three_model_records import load_text_free_three_model_inputs

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
