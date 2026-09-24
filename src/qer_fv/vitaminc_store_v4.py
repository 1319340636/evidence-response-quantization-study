"""Immutable two-stage storage for VitaminC formal direct-logit runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


VITAMINC_STORE_V4_SCHEMA_VERSION = "vitaminc-store-v4-20260716"
_CONTROLS = ("full", "no_evidence", "knowledge_only")
_STAGES = ("hard", "probability")
_EXPECTED_TABLES = {
    "schema_metadata",
    "run_identity",
    "owners",
    "hard_results",
    "probability_results",
}


class VitaminCStoreV4Error(RuntimeError):
    """Raised when immutable VitaminC run state is invalid."""


@dataclass(frozen=True)
class VitaminCRunIdentityV4:
    protocol_version: str
    split: str
    split_sha256: str
    audit_certificate_sha256: str
    prompt_contract_sha256: str
    model_key: str
    quantization: str
    expected_full: int
    expected_no_evidence: int
    expected_knowledge_only: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def expected_total(self) -> int:
        return (
            self.expected_full
            + self.expected_no_evidence
            + self.expected_knowledge_only
        )


class VitaminCRunStoreV4:
    def __init__(self, path: str | Path, identity: VitaminCRunIdentityV4) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._validate_identity()
            self._open_or_create_schema()
            self._bind_identity()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> "VitaminCRunStoreV4":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def stage_status(self, owner_key: str) -> tuple[str, str]:
        key = _required_text(owner_key, "owner key")
        statuses: list[str] = []
        for stage in _STAGES:
            row = self._connection.execute(
                f"SELECT status FROM {stage}_results WHERE owner_key = ?", (key,)
            ).fetchone()
            statuses.append("pending" if row is None else str(row["status"]))
        return statuses[0], statuses[1]

    def pending_stages(self, owner_key: str) -> tuple[str, ...]:
        statuses = self.stage_status(owner_key)
        return tuple(
            stage
            for stage, status in zip(_STAGES, statuses, strict=True)
            if status == "pending"
        )

    def get_hard_payload(self, owner_key: str) -> dict[str, Any]:
        return self._get_success_payload("hard", owner_key)

    def get_probability_payload(self, owner_key: str) -> dict[str, Any]:
        return self._get_success_payload("probability", owner_key)

    def put_hard_success(
        self, owner_key: str, *, metadata: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> None:
        self._put_stage("hard", owner_key, metadata, "ok", payload, None)

    def put_hard_failure(
        self,
        owner_key: str,
        *,
        metadata: Mapping[str, Any],
        error_code: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._put_stage("hard", owner_key, metadata, "error", payload, error_code)

    def put_probability_success(
        self, owner_key: str, *, metadata: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> None:
        self._put_stage("probability", owner_key, metadata, "ok", payload, None)

    def put_probability_failure(
        self,
        owner_key: str,
        *,
        metadata: Mapping[str, Any],
        error_code: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._put_stage(
            "probability", owner_key, metadata, "error", payload, error_code
        )

    def counts(self) -> dict[str, int]:
        owners = self._connection.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(control = 'full') AS full, "
            "SUM(control = 'no_evidence') AS no_evidence, "
            "SUM(control = 'knowledge_only') AS knowledge_only FROM owners"
        ).fetchone()
        hard = self._connection.execute(
            "SELECT COUNT(*) AS n, SUM(status = 'error') AS failures FROM hard_results"
        ).fetchone()
        probability = self._connection.execute(
            "SELECT COUNT(*) AS n, SUM(status = 'error') AS failures "
            "FROM probability_results"
        ).fetchone()
        return {
            "units_registered": int(owners["total"] or 0),
            "full_registered": int(owners["full"] or 0),
            "no_evidence_registered": int(owners["no_evidence"] or 0),
            "knowledge_only_registered": int(owners["knowledge_only"] or 0),
            "hard_completed": int(hard["n"] or 0),
            "hard_failures": int(hard["failures"] or 0),
            "probability_completed": int(probability["n"] or 0),
            "probability_failures": int(probability["failures"] or 0),
        }

    def export_complete(self, directory: str | Path) -> dict[str, Any]:
        counts = self.counts()
        if (
            counts["units_registered"] != self.identity.expected_total
            or counts["full_registered"] != self.identity.expected_full
            or counts["no_evidence_registered"]
            != self.identity.expected_no_evidence
            or counts["knowledge_only_registered"]
            != self.identity.expected_knowledge_only
            or counts["hard_completed"] != self.identity.expected_total
            or counts["probability_completed"] != self.identity.expected_total
        ):
            raise VitaminCStoreV4Error("run is incomplete")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        records_path = root / "records.jsonl"
        _atomic_write(records_path, self._export_rows())
        manifest: dict[str, Any] = {
            "schema_version": VITAMINC_STORE_V4_SCHEMA_VERSION,
            "run_identity": self.identity.to_record(),
            **counts,
            "records_jsonl_sha256": _file_sha256(records_path),
        }
        manifest["manifest_sha256"] = _record_hash(manifest)
        _atomic_write(
            root / "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
        )
        return manifest

    def _validate_identity(self) -> None:
        for field in (
            "protocol_version",
            "split",
            "split_sha256",
            "audit_certificate_sha256",
            "prompt_contract_sha256",
            "model_key",
            "quantization",
        ):
            _required_text(getattr(self.identity, field), field)
        expected = (
            self.identity.expected_full,
            self.identity.expected_no_evidence,
            self.identity.expected_knowledge_only,
        )
        if any(type(value) is not int or value < 0 for value in expected):
            raise VitaminCStoreV4Error("expected control counts must be nonnegative integers")
        if self.identity.expected_total < 1:
            raise VitaminCStoreV4Error("expected run must contain at least one unit")

    def _open_or_create_schema(self) -> None:
        existing = {
            str(row["name"])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if existing:
            if existing != _EXPECTED_TABLES:
                raise VitaminCStoreV4Error("database is not a VitaminC v4 store")
            row = self._connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
            ).fetchone()
            if row is None or row["schema_version"] != VITAMINC_STORE_V4_SCHEMA_VERSION:
                raise VitaminCStoreV4Error("database is not a VitaminC v4 store")
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
                    control TEXT NOT NULL CHECK (
                        control IN ('full', 'no_evidence', 'knowledge_only')
                    ),
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
                (VITAMINC_STORE_V4_SCHEMA_VERSION,),
            )

    def _bind_identity(self) -> None:
        payload_json = _canonical_json(self.identity.to_record())
        payload_sha256 = _text_sha256(payload_json)
        row = self._connection.execute(
            "SELECT payload_json, payload_sha256 FROM run_identity WHERE singleton = 1"
        ).fetchone()
        if row is None:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO run_identity VALUES (1, ?, ?)",
                    (payload_json, payload_sha256),
                )
            return
        if row["payload_json"] != payload_json or row["payload_sha256"] != payload_sha256:
            raise VitaminCStoreV4Error("run identity mismatch")

    def _put_stage(
        self,
        stage: str,
        owner_key: str,
        metadata: Mapping[str, Any],
        status: str,
        payload: Mapping[str, Any],
        error_code: str | None,
    ) -> None:
        if stage not in _STAGES:
            raise VitaminCStoreV4Error("unknown result stage")
        key = _required_text(owner_key, "owner key")
        metadata_json = _canonical_json(metadata)
        metadata_sha256 = _text_sha256(metadata_json)
        metadata_record = json.loads(metadata_json)
        control = metadata_record.get("control")
        if control not in _CONTROLS:
            raise VitaminCStoreV4Error("owner metadata has an invalid control")
        payload_json = _canonical_json(payload)
        payload_sha256 = _text_sha256(payload_json)
        if status == "error":
            error_code = _required_text(error_code, "error code")
        elif error_code is not None:
            raise VitaminCStoreV4Error("successful stage cannot have an error code")

        owner = self._connection.execute(
            "SELECT control, metadata_json, metadata_sha256 FROM owners "
            "WHERE owner_key = ?",
            (key,),
        ).fetchone()
        if owner is not None and (
            owner["control"] != control
            or owner["metadata_json"] != metadata_json
            or owner["metadata_sha256"] != metadata_sha256
        ):
            raise VitaminCStoreV4Error("immutable owner metadata mismatch")

        table = f"{stage}_results"
        result = self._connection.execute(
            f"SELECT status, payload_json, payload_sha256, error_code FROM {table} "
            "WHERE owner_key = ?",
            (key,),
        ).fetchone()
        if result is not None:
            if (
                result["status"] == status
                and result["payload_json"] == payload_json
                and result["payload_sha256"] == payload_sha256
                and result["error_code"] == error_code
            ):
                return
            raise VitaminCStoreV4Error(f"immutable {stage} payload mismatch")
        try:
            with self._connection:
                if owner is None:
                    self._connection.execute(
                        "INSERT INTO owners VALUES (?, ?, ?, ?)",
                        (key, control, metadata_json, metadata_sha256),
                    )
                self._connection.execute(
                    f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?)",
                    (key, status, payload_json, payload_sha256, error_code),
                )
        except sqlite3.IntegrityError as error:
            raise VitaminCStoreV4Error(f"failed to store {stage} transaction") from error

    def _get_success_payload(self, stage: str, owner_key: str) -> dict[str, Any]:
        key = _required_text(owner_key, "owner key")
        row = self._connection.execute(
            f"SELECT status, payload_json FROM {stage}_results WHERE owner_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            raise VitaminCStoreV4Error(f"{stage} stage is pending")
        if row["status"] != "ok":
            raise VitaminCStoreV4Error(f"{stage} stage is not successful")
        value = json.loads(row["payload_json"])
        if not isinstance(value, dict):
            raise VitaminCStoreV4Error("stored payload is not a JSON object")
        return value

    def _export_rows(self) -> str:
        rows = self._connection.execute(
            "SELECT o.owner_key, o.control, o.metadata_json, o.metadata_sha256, "
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
            record = {
                "owner_key": row["owner_key"],
                "control": row["control"],
                "metadata": json.loads(row["metadata_json"]),
                "metadata_sha256": row["metadata_sha256"],
                "hard_status": row["hard_status"],
                "hard_error_code": row["hard_error_code"],
                "hard_payload": json.loads(row["hard_payload_json"]),
                "hard_payload_sha256": row["hard_payload_sha256"],
                "probability_status": row["probability_status"],
                "probability_error_code": row["probability_error_code"],
                "probability_payload": json.loads(row["probability_payload_json"]),
                "probability_payload_sha256": row["probability_payload_sha256"],
            }
            output.append(_canonical_json(record) + "\n")
        return "".join(output)


def _canonical_json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise VitaminCStoreV4Error("payload must be a canonical JSON object")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise VitaminCStoreV4Error("payload is not canonical JSON") from error


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VitaminCStoreV4Error(f"{label} must be a nonempty string")
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(value: Mapping[str, Any]) -> str:
    return _text_sha256(_canonical_json(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
