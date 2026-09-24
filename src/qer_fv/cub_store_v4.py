"""Transactional CUB v4 storage with independent hard and probability stages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


CUB_STORE_V4_SCHEMA_VERSION = "cub-store-v4-20260715"
_EXPECTED_TABLES = {
    "schema_metadata",
    "run_identity",
    "owners",
    "hard_results",
    "probability_results",
}
_STAGES = ("hard", "probability")


class CubStoreV4Error(RuntimeError):
    """Raised when a CUB v4 store violates schema or immutable state."""


@dataclass(frozen=True)
class CubRunIdentityV4:
    protocol_version: str
    audit_certificate_sha256: str
    manifest_sha256: str
    prompt_contract_sha256: str
    model_key: str
    quantization: str
    expected_queries: int
    expected_contexts: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class CubRunStoreV4:
    def __init__(self, path: str | Path, identity: CubRunIdentityV4) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._open_or_create_schema()
            self._bind_identity()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> "CubRunStoreV4":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def has_owner(self, owner_type: str, owner_key: str) -> bool:
        owner_type, owner_key = _owner_identity(owner_type, owner_key)
        row = self._connection.execute(
            "SELECT 1 FROM owners WHERE owner_type = ? AND owner_key = ?",
            (owner_type, owner_key),
        ).fetchone()
        return row is not None

    def stage_status(self, owner_type: str, owner_key: str) -> tuple[str, str]:
        owner_type, owner_key = _owner_identity(owner_type, owner_key)
        values = []
        for stage in _STAGES:
            row = self._connection.execute(
                f"SELECT status FROM {stage}_results "
                "WHERE owner_type = ? AND owner_key = ?",
                (owner_type, owner_key),
            ).fetchone()
            values.append("pending" if row is None else str(row["status"]))
        return values[0], values[1]

    def pending_stages(self, owner_type: str, owner_key: str) -> tuple[str, ...]:
        statuses = self.stage_status(owner_type, owner_key)
        return tuple(
            stage for stage, status in zip(_STAGES, statuses, strict=True)
            if status == "pending"
        )

    def get_hard_payload(self, owner_type: str, owner_key: str) -> dict[str, Any]:
        return self._get_success_payload("hard", owner_type, owner_key)

    def get_probability_payload(
        self, owner_type: str, owner_key: str
    ) -> dict[str, Any]:
        return self._get_success_payload("probability", owner_type, owner_key)

    def put_hard_success(
        self,
        owner_type: str,
        owner_key: str,
        *,
        payload: Mapping[str, Any],
        query_identity: str | None = None,
    ) -> None:
        self._put_stage(
            "hard", owner_type, owner_key, query_identity, "ok", payload, None
        )

    def put_hard_failure(
        self,
        owner_type: str,
        owner_key: str,
        *,
        error_code: str,
        payload: Mapping[str, Any],
        query_identity: str | None = None,
    ) -> None:
        self._put_stage(
            "hard",
            owner_type,
            owner_key,
            query_identity,
            "error",
            payload,
            error_code,
        )

    def put_probability_success(
        self,
        owner_type: str,
        owner_key: str,
        *,
        payload: Mapping[str, Any],
        query_identity: str | None = None,
    ) -> None:
        self._put_stage(
            "probability", owner_type, owner_key, query_identity, "ok", payload, None
        )

    def put_probability_failure(
        self,
        owner_type: str,
        owner_key: str,
        *,
        error_code: str,
        payload: Mapping[str, Any],
        query_identity: str | None = None,
    ) -> None:
        self._put_stage(
            "probability",
            owner_type,
            owner_key,
            query_identity,
            "error",
            payload,
            error_code,
        )

    def counts(self) -> dict[str, int]:
        owner_counts = self._connection.execute(
            "SELECT "
            "SUM(owner_type = 'query') AS queries, "
            "SUM(owner_type = 'context') AS contexts FROM owners"
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
            "queries_registered": int(owner_counts["queries"] or 0),
            "contexts_registered": int(owner_counts["contexts"] or 0),
            "hard_completed": int(hard["n"]),
            "hard_failures": int(hard["failures"] or 0),
            "probability_completed": int(probability["n"]),
            "probability_failures": int(probability["failures"] or 0),
        }

    def export_complete(self, directory: str | Path) -> dict[str, Any]:
        counts = self.counts()
        expected_total = self.identity.expected_queries + self.identity.expected_contexts
        if (
            counts["queries_registered"] != self.identity.expected_queries
            or counts["contexts_registered"] != self.identity.expected_contexts
            or counts["hard_completed"] != expected_total
            or counts["probability_completed"] != expected_total
        ):
            raise CubStoreV4Error("run is incomplete")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        query_path = root / "queries.jsonl"
        context_path = root / "contexts.jsonl"
        _atomic_write(query_path, self._export_rows("query"))
        _atomic_write(context_path, self._export_rows("context"))
        manifest = {
            "schema_version": CUB_STORE_V4_SCHEMA_VERSION,
            "run_identity": self.identity.to_record(),
            **counts,
            "queries_jsonl_sha256": _file_sha256(query_path),
            "contexts_jsonl_sha256": _file_sha256(context_path),
        }
        manifest["manifest_sha256"] = _record_hash(manifest)
        _atomic_write(
            root / "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )
        return manifest

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
                raise CubStoreV4Error("database is not a v4 store schema")
            row = self._connection.execute(
                "SELECT schema_version FROM schema_metadata WHERE singleton = 1"
            ).fetchone()
            if row is None or row["schema_version"] != CUB_STORE_V4_SCHEMA_VERSION:
                raise CubStoreV4Error("database is not a v4 store schema")
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
                    owner_type TEXT NOT NULL CHECK (owner_type IN ('query', 'context')),
                    owner_key TEXT NOT NULL,
                    query_identity TEXT,
                    PRIMARY KEY (owner_type, owner_key),
                    CHECK (
                        (owner_type = 'query' AND query_identity IS NULL) OR
                        (owner_type = 'context' AND query_identity IS NOT NULL)
                    )
                );
                CREATE TABLE hard_results (
                    owner_type TEXT NOT NULL,
                    owner_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    error_code TEXT,
                    PRIMARY KEY (owner_type, owner_key),
                    FOREIGN KEY (owner_type, owner_key)
                        REFERENCES owners(owner_type, owner_key)
                );
                CREATE TABLE probability_results (
                    owner_type TEXT NOT NULL,
                    owner_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    error_code TEXT,
                    PRIMARY KEY (owner_type, owner_key),
                    FOREIGN KEY (owner_type, owner_key)
                        REFERENCES owners(owner_type, owner_key)
                );
                """
            )
            self._connection.execute(
                "INSERT INTO schema_metadata VALUES (1, ?)",
                (CUB_STORE_V4_SCHEMA_VERSION,),
            )

    def _bind_identity(self) -> None:
        payload_json = _canonical_json(self.identity.to_record())
        payload_hash = _text_sha256(payload_json)
        existing = self._connection.execute(
            "SELECT payload_json, payload_sha256 FROM run_identity WHERE singleton = 1"
        ).fetchone()
        if existing is None:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO run_identity VALUES (1, ?, ?)",
                    (payload_json, payload_hash),
                )
            return
        if (
            existing["payload_json"] != payload_json
            or existing["payload_sha256"] != payload_hash
        ):
            raise CubStoreV4Error("run identity mismatch")

    def _put_stage(
        self,
        stage: str,
        owner_type: str,
        owner_key: str,
        query_identity: str | None,
        status: str,
        payload: Mapping[str, Any],
        error_code: str | None,
    ) -> None:
        if stage not in _STAGES:
            raise CubStoreV4Error("unknown result stage")
        owner_type, owner_key = _owner_identity(owner_type, owner_key)
        query_identity = _query_relation(owner_type, query_identity)
        if status == "error":
            error_code = _required_identifier(error_code, "error code")
        elif error_code is not None:
            raise CubStoreV4Error("successful stage cannot have an error code")
        payload_json = _canonical_json(payload)
        payload_hash = _text_sha256(payload_json)
        table = f"{stage}_results"
        existing_owner = self._connection.execute(
            "SELECT query_identity FROM owners WHERE owner_type = ? AND owner_key = ?",
            (owner_type, owner_key),
        ).fetchone()
        if existing_owner is not None and existing_owner["query_identity"] != query_identity:
            raise CubStoreV4Error("immutable owner query identity mismatch")
        existing = self._connection.execute(
            f"SELECT status, payload_json, payload_sha256, error_code FROM {table} "
            "WHERE owner_type = ? AND owner_key = ?",
            (owner_type, owner_key),
        ).fetchone()
        if existing is not None:
            if (
                existing["status"] == status
                and existing["payload_json"] == payload_json
                and existing["payload_sha256"] == payload_hash
                and existing["error_code"] == error_code
            ):
                return
            raise CubStoreV4Error(f"immutable {stage} payload mismatch")

        try:
            with self._connection:
                if existing_owner is None:
                    if owner_type == "context":
                        query = self._connection.execute(
                            "SELECT 1 FROM owners "
                            "WHERE owner_type = 'query' AND owner_key = ?",
                            (query_identity,),
                        ).fetchone()
                        if query is None:
                            raise CubStoreV4Error("query identity does not exist")
                    self._connection.execute(
                        "INSERT INTO owners VALUES (?, ?, ?)",
                        (owner_type, owner_key, query_identity),
                    )
                self._connection.execute(
                    f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        owner_type,
                        owner_key,
                        status,
                        payload_json,
                        payload_hash,
                        error_code,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise CubStoreV4Error(f"failed to store {stage} transaction") from error

    def _get_success_payload(
        self, stage: str, owner_type: str, owner_key: str
    ) -> dict[str, Any]:
        owner_type, owner_key = _owner_identity(owner_type, owner_key)
        row = self._connection.execute(
            f"SELECT status, payload_json FROM {stage}_results "
            "WHERE owner_type = ? AND owner_key = ?",
            (owner_type, owner_key),
        ).fetchone()
        if row is None:
            raise CubStoreV4Error(f"{stage} stage is pending")
        if row["status"] != "ok":
            raise CubStoreV4Error(f"{stage} stage is not successful")
        value = json.loads(row["payload_json"])
        if not isinstance(value, dict):
            raise CubStoreV4Error("stored payload is not a JSON object")
        return value

    def _export_rows(self, owner_type: str) -> str:
        rows = self._connection.execute(
            "SELECT o.owner_key, o.query_identity, "
            "h.status AS hard_status, h.payload_json AS hard_payload_json, "
            "h.payload_sha256 AS hard_payload_sha256, h.error_code AS hard_error_code, "
            "p.status AS probability_status, p.payload_json AS probability_payload_json, "
            "p.payload_sha256 AS probability_payload_sha256, "
            "p.error_code AS probability_error_code "
            "FROM owners o "
            "JOIN hard_results h USING (owner_type, owner_key) "
            "JOIN probability_results p USING (owner_type, owner_key) "
            "WHERE o.owner_type = ? ORDER BY o.owner_key",
            (owner_type,),
        ).fetchall()
        output = []
        for row in rows:
            record = {
                "hard_status": row["hard_status"],
                "hard_error_code": row["hard_error_code"],
                "hard_payload": json.loads(row["hard_payload_json"]),
                "hard_payload_sha256": row["hard_payload_sha256"],
                "probability_status": row["probability_status"],
                "probability_error_code": row["probability_error_code"],
                "probability_payload": json.loads(row["probability_payload_json"]),
                "probability_payload_sha256": row["probability_payload_sha256"],
            }
            if owner_type == "query":
                record["query_identity_sha256"] = row["owner_key"]
            else:
                record["sample_id"] = row["owner_key"]
                record["query_identity_sha256"] = row["query_identity"]
            output.append(_canonical_json(record) + "\n")
        return "".join(output)


def _owner_identity(owner_type: str, owner_key: str) -> tuple[str, str]:
    if owner_type not in {"query", "context"}:
        raise CubStoreV4Error("owner type must be query or context")
    return owner_type, _required_identifier(owner_key, "owner key")


def _query_relation(owner_type: str, query_identity: str | None) -> str | None:
    if owner_type == "query":
        if query_identity is not None:
            raise CubStoreV4Error("query owner cannot reference another query")
        return None
    return _required_identifier(query_identity, "query identity")


def _canonical_json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise CubStoreV4Error("payload must be a canonical JSON object")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CubStoreV4Error("payload is not canonical JSON") from error


def _required_identifier(value: str | None, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CubStoreV4Error(f"{label} must be a nonempty string")
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    return _text_sha256(_canonical_json(record))


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
