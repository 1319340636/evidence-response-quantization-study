"""Transactional, immutable storage for paired CUB runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class CubStoreError(RuntimeError):
    """Raised when a CUB run store violates its frozen identity or schema."""


@dataclass(frozen=True)
class CubRunIdentity:
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


class CubRunStore:
    def __init__(self, path: str | Path, identity: CubRunIdentity) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._create_schema()
            self._bind_identity()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> "CubRunStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def has_query(self, query_identity: str) -> bool:
        return self._has_owner("queries", "query_identity", query_identity)

    def has_context(self, sample_id: str) -> bool:
        return self._has_owner("contexts", "sample_id", sample_id)

    def get_query_payload(self, query_identity: str) -> dict[str, Any]:
        return self._get_owner_payload("queries", "query_identity", query_identity)

    def get_context_payload(self, sample_id: str) -> dict[str, Any]:
        return self._get_owner_payload("contexts", "sample_id", sample_id)

    def get_query_status(self, query_identity: str) -> str:
        return self._get_owner_status("queries", "query_identity", query_identity)

    def get_context_status(self, sample_id: str) -> str:
        return self._get_owner_status("contexts", "sample_id", sample_id)

    def put_query_success(
        self,
        query_identity: str,
        payload: Mapping[str, Any],
        *,
        probes: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._put_owner("query", query_identity, None, "ok", payload, probes, None)

    def put_query_failure(
        self,
        query_identity: str,
        *,
        error_code: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._put_owner(
            "query", query_identity, None, "error", payload, (), error_code
        )

    def put_context_success(
        self,
        sample_id: str,
        query_identity: str,
        payload: Mapping[str, Any],
        *,
        probes: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._require_query(query_identity)
        self._put_owner(
            "context", sample_id, query_identity, "ok", payload, probes, None
        )

    def put_context_failure(
        self,
        sample_id: str,
        query_identity: str,
        *,
        error_code: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._require_query(query_identity)
        self._put_owner(
            "context",
            sample_id,
            query_identity,
            "error",
            payload,
            (),
            error_code,
        )

    def counts(self) -> dict[str, int]:
        queries = self._connection.execute(
            "SELECT COUNT(*) AS n, SUM(status = 'error') AS failures FROM queries"
        ).fetchone()
        contexts = self._connection.execute(
            "SELECT COUNT(*) AS n, SUM(status = 'error') AS failures FROM contexts"
        ).fetchone()
        return {
            "queries_completed": int(queries["n"]),
            "query_failures": int(queries["failures"] or 0),
            "contexts_completed": int(contexts["n"]),
            "context_failures": int(contexts["failures"] or 0),
        }

    def export_complete(self, directory: str | Path) -> dict[str, Any]:
        counts = self.counts()
        if (
            counts["queries_completed"] != self.identity.expected_queries
            or counts["contexts_completed"] != self.identity.expected_contexts
        ):
            raise CubStoreError("run is incomplete")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        query_path = root / "queries.jsonl"
        context_path = root / "contexts.jsonl"
        _atomic_write(query_path, self._export_rows("query"))
        _atomic_write(context_path, self._export_rows("context"))
        manifest = {
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

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_identity (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS queries (
                query_identity TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                error_code TEXT
            );
            CREATE TABLE IF NOT EXISTS contexts (
                sample_id TEXT PRIMARY KEY,
                query_identity TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                error_code TEXT,
                FOREIGN KEY (query_identity) REFERENCES queries(query_identity)
            );
            CREATE TABLE IF NOT EXISTS probes (
                owner_type TEXT NOT NULL CHECK (owner_type IN ('query', 'context')),
                owner_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                PRIMARY KEY (owner_type, owner_key, ordinal)
            );
            CREATE TABLE IF NOT EXISTS failures (
                owner_type TEXT NOT NULL CHECK (owner_type IN ('query', 'context')),
                owner_key TEXT NOT NULL,
                error_code TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                PRIMARY KEY (owner_type, owner_key)
            );
            """
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
            raise CubStoreError("run identity mismatch")

    def _put_owner(
        self,
        owner_type: str,
        owner_key: str,
        query_identity: str | None,
        status: str,
        payload: Mapping[str, Any],
        probes: Sequence[Mapping[str, Any]],
        error_code: str | None,
    ) -> None:
        _required_identifier(owner_key, "owner key")
        if status == "error":
            _required_identifier(error_code, "error code")
        payload_json = _canonical_json(payload)
        payload_hash = _text_sha256(payload_json)
        encoded_probes = [
            (_canonical_json(probe),) for probe in probes
        ]
        encoded_probes = [
            (value[0], _text_sha256(value[0])) for value in encoded_probes
        ]
        table = "queries" if owner_type == "query" else "contexts"
        key_column = "query_identity" if owner_type == "query" else "sample_id"
        existing = self._connection.execute(
            f"SELECT status, payload_json, payload_sha256, error_code FROM {table} "
            f"WHERE {key_column} = ?",
            (owner_key,),
        ).fetchone()
        if existing is not None:
            if self._owner_matches(
                owner_type,
                owner_key,
                existing,
                status,
                payload_json,
                payload_hash,
                error_code,
                encoded_probes,
            ):
                return
            raise CubStoreError(f"immutable {owner_type} payload mismatch")

        try:
            with self._connection:
                if owner_type == "query":
                    self._connection.execute(
                        "INSERT INTO queries VALUES (?, ?, ?, ?, ?)",
                        (owner_key, status, payload_json, payload_hash, error_code),
                    )
                else:
                    self._connection.execute(
                        "INSERT INTO contexts VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            owner_key,
                            query_identity,
                            status,
                            payload_json,
                            payload_hash,
                            error_code,
                        ),
                    )
                for ordinal, (probe_json, probe_hash) in enumerate(encoded_probes):
                    self._connection.execute(
                        "INSERT INTO probes VALUES (?, ?, ?, ?, ?)",
                        (owner_type, owner_key, ordinal, probe_json, probe_hash),
                    )
                if error_code is not None:
                    self._connection.execute(
                        "INSERT INTO failures VALUES (?, ?, ?, ?, ?)",
                        (
                            owner_type,
                            owner_key,
                            error_code,
                            payload_json,
                            payload_hash,
                        ),
                    )
        except sqlite3.IntegrityError as error:
            raise CubStoreError(f"failed to store {owner_type} transaction") from error

    def _owner_matches(
        self,
        owner_type: str,
        owner_key: str,
        existing: sqlite3.Row,
        status: str,
        payload_json: str,
        payload_hash: str,
        error_code: str | None,
        encoded_probes: Sequence[tuple[str, str]],
    ) -> bool:
        if (
            existing["status"] != status
            or existing["payload_json"] != payload_json
            or existing["payload_sha256"] != payload_hash
            or existing["error_code"] != error_code
        ):
            return False
        stored = self._connection.execute(
            "SELECT payload_json, payload_sha256 FROM probes "
            "WHERE owner_type = ? AND owner_key = ? ORDER BY ordinal",
            (owner_type, owner_key),
        ).fetchall()
        return [(row["payload_json"], row["payload_sha256"]) for row in stored] == list(
            encoded_probes
        )

    def _require_query(self, query_identity: str) -> None:
        _required_identifier(query_identity, "query identity")
        row = self._connection.execute(
            "SELECT 1 FROM queries WHERE query_identity = ?", (query_identity,)
        ).fetchone()
        if row is None:
            raise CubStoreError("query identity does not exist")

    def _has_owner(self, table: str, key_column: str, key: str) -> bool:
        _required_identifier(key, key_column)
        row = self._connection.execute(
            f"SELECT 1 FROM {table} WHERE {key_column} = ?", (key,)
        ).fetchone()
        return row is not None

    def _get_owner_payload(
        self, table: str, key_column: str, key: str
    ) -> dict[str, Any]:
        _required_identifier(key, key_column)
        row = self._connection.execute(
            f"SELECT payload_json FROM {table} WHERE {key_column} = ?", (key,)
        ).fetchone()
        if row is None:
            raise CubStoreError(f"{key_column} does not exist")
        value = json.loads(row["payload_json"])
        if not isinstance(value, dict):
            raise CubStoreError("stored payload is not a JSON object")
        return value

    def _get_owner_status(self, table: str, key_column: str, key: str) -> str:
        _required_identifier(key, key_column)
        row = self._connection.execute(
            f"SELECT status FROM {table} WHERE {key_column} = ?", (key,)
        ).fetchone()
        if row is None:
            raise CubStoreError(f"{key_column} does not exist")
        return str(row["status"])

    def _export_rows(self, owner_type: str) -> str:
        if owner_type == "query":
            rows = self._connection.execute(
                "SELECT query_identity AS owner_key, NULL AS query_identity, "
                "status, payload_json, error_code FROM queries ORDER BY query_identity"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT sample_id AS owner_key, query_identity, status, payload_json, "
                "error_code FROM contexts ORDER BY sample_id"
            ).fetchall()
        output = []
        for row in rows:
            record = json.loads(row["payload_json"])
            record.update(
                {
                    "status": row["status"],
                    "error_code": row["error_code"],
                }
            )
            if owner_type == "query":
                record["query_identity_sha256"] = row["owner_key"]
            else:
                record["sample_id"] = row["owner_key"]
                record["query_identity_sha256"] = row["query_identity"]
            probes = self._connection.execute(
                "SELECT payload_json FROM probes WHERE owner_type = ? "
                "AND owner_key = ? ORDER BY ordinal",
                (owner_type, row["owner_key"]),
            ).fetchall()
            record["probes"] = [json.loads(probe["payload_json"]) for probe in probes]
            output.append(_canonical_json(record) + "\n")
        return "".join(output)


def _canonical_json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise CubStoreError("payload must be a canonical JSON object")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CubStoreError("payload is not canonical JSON") from error


def _required_identifier(value: str | None, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CubStoreError(f"{label} must be a nonempty string")
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(record: Mapping[str, Any]) -> str:
    return _text_sha256(_canonical_json(record))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
