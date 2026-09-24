"""Immutable two-stage storage for blinded Fresh TabFact smoke runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


TABFACT_STORE_SCHEMA_VERSION = "tabfact-store-v1-20260717"
_STAGES = ("hard", "probability")


class TabFactStoreError(RuntimeError):
    """Raised when immutable TabFact run state is invalid."""


@dataclass(frozen=True)
class TabFactRunIdentity:
    protocol_version: str
    split_sha256: str
    smoke_sha256: str
    audit_certificate_sha256: str
    prompt_contract_sha256: str
    model_key: str
    quantization: str
    expected_units: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TabFactFormalRunIdentity:
    protocol_version: str
    split_sha256: str
    manifest_sha256: str
    evidence_sha256: str
    audit_certificate_sha256: str
    prompt_contract_sha256: str
    model_key: str
    quantization: str
    expected_units: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class TabFactRunStore:
    def __init__(
        self,
        path: str | Path,
        identity: TabFactRunIdentity | TabFactFormalRunIdentity,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._validate_identity()
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._open_or_create()
            self._bind_identity()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> "TabFactRunStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def pending_stages(self, owner_key: str) -> tuple[str, ...]:
        key = _required_text(owner_key, "owner key")
        pending: list[str] = []
        for stage in _STAGES:
            row = self._connection.execute(
                f"SELECT status FROM {stage}_results WHERE owner_key = ?", (key,)
            ).fetchone()
            if row is None:
                pending.append(stage)
        return tuple(pending)

    def put_hard_success(
        self, owner_key: str, *, metadata: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> None:
        self._put("hard", owner_key, metadata, "ok", payload, None)

    def put_hard_failure(
        self,
        owner_key: str,
        *,
        metadata: Mapping[str, Any],
        error_code: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._put("hard", owner_key, metadata, "error", payload, error_code)

    def put_probability_success(
        self, owner_key: str, *, metadata: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> None:
        self._put("probability", owner_key, metadata, "ok", payload, None)

    def put_probability_failure(
        self,
        owner_key: str,
        *,
        metadata: Mapping[str, Any],
        error_code: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._put(
            "probability", owner_key, metadata, "error", payload, error_code
        )

    def counts(self) -> dict[str, int]:
        owners = self._connection.execute(
            "SELECT COUNT(*) AS n FROM owners"
        ).fetchone()
        hard = self._connection.execute(
            "SELECT COUNT(*) AS n, SUM(status = 'error') AS failures "
            "FROM hard_results"
        ).fetchone()
        probability = self._connection.execute(
            "SELECT COUNT(*) AS n, SUM(status = 'error') AS failures "
            "FROM probability_results"
        ).fetchone()
        return {
            "owners": int(owners["n"] or 0),
            "hard_completed": int(hard["n"] or 0),
            "hard_failures": int(hard["failures"] or 0),
            "probability_completed": int(probability["n"] or 0),
            "probability_failures": int(probability["failures"] or 0),
        }

    def structural_report(self) -> dict[str, Any]:
        counts = self.counts()
        return {
            "schema_version": (
                "tabfact-structural-formal-v1-20260718"
                if isinstance(self.identity, TabFactFormalRunIdentity)
                else "tabfact-structural-smoke-v1-20260717"
            ),
            "run_identity": self.identity.to_record(),
            **counts,
            "gate_passed": (
                counts["owners"] == self.identity.expected_units
                and counts["hard_completed"] == self.identity.expected_units
                and counts["probability_completed"] == self.identity.expected_units
                and counts["hard_failures"] == 0
                and counts["probability_failures"] == 0
            ),
        }

    def export_complete(self, directory: str | Path) -> dict[str, Any]:
        counts = self.counts()
        if (
            counts["owners"] != self.identity.expected_units
            or counts["hard_completed"] != self.identity.expected_units
            or counts["probability_completed"] != self.identity.expected_units
        ):
            raise TabFactStoreError("run is incomplete")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        records_path = root / "records.jsonl"
        _atomic_write(records_path, self._export_rows())
        manifest: dict[str, Any] = {
            "schema_version": TABFACT_STORE_SCHEMA_VERSION,
            "run_identity": self.identity.to_record(),
            **counts,
            "records_jsonl_sha256": _file_sha256(records_path),
        }
        manifest["manifest_sha256"] = _record_hash(manifest)
        _atomic_write(root / "manifest.json", _canonical_json(manifest) + "\n")
        report = self.structural_report()
        report["export_manifest_sha256"] = manifest["manifest_sha256"]
        report["report_sha256"] = _record_hash(report)
        _atomic_write(
            root / "structural_report.json", _canonical_json(report) + "\n"
        )
        return manifest

    def _validate_identity(self) -> None:
        common_fields = (
            "protocol_version",
            "split_sha256",
            "audit_certificate_sha256",
            "prompt_contract_sha256",
            "model_key",
            "quantization",
        )
        if isinstance(self.identity, TabFactRunIdentity):
            membership_fields = ("smoke_sha256",)
        elif isinstance(self.identity, TabFactFormalRunIdentity):
            membership_fields = ("manifest_sha256", "evidence_sha256")
        else:
            raise TabFactStoreError("unsupported TabFact run identity")
        for field in common_fields + membership_fields:
            _required_text(getattr(self.identity, field), field)
        if (
            type(self.identity.expected_units) is not int
            or self.identity.expected_units < 1
        ):
            raise TabFactStoreError("expected_units must be positive")

    def _open_or_create(self) -> None:
        existing = {
            str(row["name"])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        expected = {
            "schema_metadata",
            "run_identity",
            "owners",
            "hard_results",
            "probability_results",
        }
        if existing:
            if existing != expected:
                raise TabFactStoreError("database is not a TabFact store")
            version = self._connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
            ).fetchone()
            if version is None or version["schema_version"] != TABFACT_STORE_SCHEMA_VERSION:
                raise TabFactStoreError("database is not a TabFact store")
            return
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE schema_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL
                );
                CREATE TABLE run_identity (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL
                );
                CREATE TABLE owners (
                    owner_key TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    metadata_sha256 TEXT NOT NULL
                );
                CREATE TABLE hard_results (
                    owner_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    error_code TEXT,
                    FOREIGN KEY (owner_key) REFERENCES owners(owner_key)
                );
                CREATE TABLE probability_results (
                    owner_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    error_code TEXT,
                    FOREIGN KEY (owner_key) REFERENCES owners(owner_key)
                );
                """
            )
            self._connection.execute(
                "INSERT INTO schema_metadata VALUES (1, ?)",
                (TABFACT_STORE_SCHEMA_VERSION,),
            )

    def _bind_identity(self) -> None:
        payload = _canonical_json(self.identity.to_record())
        digest = _text_sha256(payload)
        row = self._connection.execute(
            "SELECT payload_json, payload_sha256 FROM run_identity WHERE singleton = 1"
        ).fetchone()
        if row is None:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO run_identity VALUES (1, ?, ?)", (payload, digest)
                )
            return
        if row["payload_json"] != payload or row["payload_sha256"] != digest:
            raise TabFactStoreError("run identity mismatch")

    def _put(
        self,
        stage: str,
        owner_key: str,
        metadata: Mapping[str, Any],
        status: str,
        payload: Mapping[str, Any],
        error_code: str | None,
    ) -> None:
        key = _required_text(owner_key, "owner key")
        if stage not in _STAGES:
            raise TabFactStoreError("unknown result stage")
        metadata_json = _canonical_json(metadata)
        metadata_record = json.loads(metadata_json)
        if metadata_record.get("control") != "full":
            raise TabFactStoreError("TabFact control must be full")
        metadata_sha = _text_sha256(metadata_json)
        payload_json = _canonical_json(payload)
        payload_sha = _text_sha256(payload_json)
        if status == "error":
            error_code = _required_text(error_code, "error code")
        elif error_code is not None:
            raise TabFactStoreError("successful stage cannot have an error code")

        owner = self._connection.execute(
            "SELECT metadata_json, metadata_sha256 FROM owners WHERE owner_key = ?",
            (key,),
        ).fetchone()
        if owner is not None and (
            owner["metadata_json"] != metadata_json
            or owner["metadata_sha256"] != metadata_sha
        ):
            raise TabFactStoreError("immutable owner metadata mismatch")
        result = self._connection.execute(
            f"SELECT status, payload_json, payload_sha256, error_code "
            f"FROM {stage}_results WHERE owner_key = ?",
            (key,),
        ).fetchone()
        if result is not None:
            if (
                result["status"] == status
                and result["payload_json"] == payload_json
                and result["payload_sha256"] == payload_sha
                and result["error_code"] == error_code
            ):
                return
            raise TabFactStoreError(f"immutable {stage} payload mismatch")
        with self._connection:
            if owner is None:
                self._connection.execute(
                    "INSERT INTO owners VALUES (?, ?, ?)",
                    (key, metadata_json, metadata_sha),
                )
            self._connection.execute(
                f"INSERT INTO {stage}_results VALUES (?, ?, ?, ?, ?)",
                (key, status, payload_json, payload_sha, error_code),
            )

    def _export_rows(self) -> str:
        rows = self._connection.execute(
            "SELECT o.owner_key, o.metadata_json, o.metadata_sha256, "
            "h.status AS hard_status, h.payload_json AS hard_payload_json, "
            "h.payload_sha256 AS hard_payload_sha256, h.error_code AS hard_error_code, "
            "p.status AS probability_status, p.payload_json AS probability_payload_json, "
            "p.payload_sha256 AS probability_payload_sha256, "
            "p.error_code AS probability_error_code "
            "FROM owners o JOIN hard_results h USING (owner_key) "
            "JOIN probability_results p USING (owner_key) ORDER BY o.owner_key"
        ).fetchall()
        output: list[str] = []
        for row in rows:
            output.append(
                _canonical_json(
                    {
                        "owner_key": row["owner_key"],
                        "metadata": json.loads(row["metadata_json"]),
                        "metadata_sha256": row["metadata_sha256"],
                        "hard_status": row["hard_status"],
                        "hard_error_code": row["hard_error_code"],
                        "hard_payload": json.loads(row["hard_payload_json"]),
                        "hard_payload_sha256": row["hard_payload_sha256"],
                        "probability_status": row["probability_status"],
                        "probability_error_code": row["probability_error_code"],
                        "probability_payload": json.loads(
                            row["probability_payload_json"]
                        ),
                        "probability_payload_sha256": row[
                            "probability_payload_sha256"
                        ],
                    }
                )
                + "\n"
            )
        return "".join(output)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TabFactStoreError(f"{label} must be nonempty")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TabFactStoreError("payload must be an object")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TabFactStoreError("payload is not canonical JSON") from error


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(value: Mapping[str, Any]) -> str:
    return _text_sha256(_canonical_json(value))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
