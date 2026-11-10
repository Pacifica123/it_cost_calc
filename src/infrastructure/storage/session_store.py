"""Canonical SQLite persistence for resumable IT Cost Calc sessions.

A session is a self-contained snapshot of mutable domain/runtime state.  The
SQLite database is the single source of truth for sessions, calculation state
and generated export artifacts. JSON remains only for fixtures, interchange
and one-time legacy import.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping
from uuid import uuid4

SESSION_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1

_DEFAULT_ELECTRICITY_PROFILE = {
    "hours_per_day": 8.0,
    "working_days": 22.0,
    "cost_per_kwh": 1.0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Unsupported session payload value: {type(value).__name__}")


def _dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    )


def _loads(payload: str) -> Any:
    return json.loads(payload)


def default_session_state(*, entities: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "entities": deepcopy(dict(entities or {})),
        "workflow": {
            "active_route": "software",
            "steps": {"software": "data", "technical": "data"},
        },
        "candidate_pools": {},
        "ga_results": {},
        "ahp_results": {},
        "pareto_results": {},
        "hybrid_results": {},
        "electricity_profile": deepcopy(_DEFAULT_ELECTRICITY_PROFILE),
        "electricity_report": {"total_cost": 0.0, "items": []},
        "npv_basis": {},
        "npv_inputs": {},
        "npv_report": {},
        "derived_meta": {},
    }


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    name: str
    context_type: str
    created_at: str
    updated_at: str
    revision: int
    input_revision: int
    active: bool = False


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    artifact_id: str
    session_id: str
    kind: str
    format: str
    name: str
    created_at: str
    session_revision: int
    source_hash: str
    size_bytes: int
    metadata: dict[str, Any]


class SessionStore:
    """Transactional canonical store for global reusable data and sessions."""

    def __init__(
        self,
        path: str | Path,
        *,
        legacy_runtime_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.legacy_runtime_path = Path(legacy_runtime_path) if legacy_runtime_path else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_schema()
        self._ensure_global_state()
        self._migrate_legacy_runtime_once()
        self._ensure_initial_session()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _migrate_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            current_row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
            current = int(current_row["version"] if current_row else 0)
            if current > SESSION_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Session DB schema {current} is newer than supported {SESSION_SCHEMA_VERSION}."
                )
            if current < 1:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS global_state (
                        id INTEGER PRIMARY KEY CHECK(id = 1),
                        revision INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        context_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        input_revision INTEGER NOT NULL,
                        state_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
                    CREATE TABLE IF NOT EXISTS export_artifacts (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        kind TEXT NOT NULL,
                        format TEXT NOT NULL,
                        name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        session_revision INTEGER NOT NULL,
                        source_hash TEXT NOT NULL,
                        payload BLOB NOT NULL,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_exports_session_time
                        ON export_artifacts(session_id, created_at DESC);
                    CREATE TABLE IF NOT EXISTS legacy_imports (
                        source_key TEXT PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        source_hash TEXT NOT NULL,
                        imported_at TEXT NOT NULL,
                        session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, _utc_now()),
                )

    def _ensure_global_state(self) -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT id FROM global_state WHERE id = 1").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO global_state(id, revision, payload_json, updated_at) "
                    "VALUES (1, 0, ?, ?)",
                    (_dumps({"entities": {}}), _utc_now()),
                )

    def _legacy_source_key(self) -> str | None:
        if self.legacy_runtime_path is None:
            return None
        return str(self.legacy_runtime_path.expanduser().resolve())

    def _migrate_legacy_runtime_once(self) -> None:
        path = self.legacy_runtime_path
        source_key = self._legacy_source_key()
        if path is None or source_key is None or not path.is_file():
            return
        with self._connect() as connection:
            imported = connection.execute(
                "SELECT source_key FROM legacy_imports WHERE source_key = ?", (source_key,)
            ).fetchone()
            if imported is not None:
                return

        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        entities = payload.get("entities", payload) if isinstance(payload, Mapping) else {}
        if not isinstance(entities, Mapping):
            raise ValueError(f"Legacy runtime payload must be a mapping: {path}")
        normalized_entities = {
            str(name): [deepcopy(dict(row)) for row in rows if isinstance(row, Mapping)]
            for name, rows in entities.items()
            if isinstance(rows, list)
        }
        source_hash = hashlib.sha256(raw).hexdigest()
        now = _utc_now()
        session_id = uuid4().hex
        state = default_session_state(entities=normalized_entities)
        state["migration"] = {
            "source": "legacy_runtime_json",
            "source_path": source_key,
            "source_hash": source_hash,
        }

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_sessions = int(
                connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            )
            connection.execute(
                "UPDATE global_state SET revision = revision + 1, payload_json = ?, updated_at = ? "
                "WHERE id = 1",
                (_dumps({"entities": normalized_entities}), now),
            )
            connection.execute(
                "INSERT INTO sessions(id, name, context_type, created_at, updated_at, revision, "
                "input_revision, state_json) VALUES (?, ?, ?, ?, ?, 1, 1, ?)",
                (
                    session_id,
                    "Мигрированная рабочая сессия",
                    "legacy-import",
                    now,
                    now,
                    _dumps(state),
                ),
            )
            if existing_sessions == 0:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('active_session_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (session_id,),
                )
            connection.execute(
                "INSERT INTO legacy_imports(source_key, source_path, source_hash, imported_at, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (source_key, source_key, source_hash, now, session_id),
            )

    def _ensure_initial_session(self) -> None:
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        if count:
            if not self.active_session_id():
                first = self.list_sessions()[0]
                self.switch_session(first.session_id)
            return
        self.create_session("Рабочая сессия", context_type="work", from_global=True, activate=True)

    def active_session_id(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = 'active_session_id'"
            ).fetchone()
        return str(row[0]) if row else ""

    def list_sessions(self) -> list[SessionInfo]:
        active_id = self.active_session_id()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, context_type, created_at, updated_at, revision, input_revision "
                "FROM sessions ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [
            SessionInfo(
                session_id=str(row["id"]),
                name=str(row["name"]),
                context_type=str(row["context_type"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                revision=int(row["revision"]),
                input_revision=int(row["input_revision"]),
                active=str(row["id"]) == active_id,
            )
            for row in rows
        ]

    def active_session(self) -> SessionInfo:
        active_id = self.active_session_id()
        for info in self.list_sessions():
            if info.session_id == active_id:
                return info
        raise RuntimeError("Active session is missing.")

    def session_state(self, session_id: str | None = None) -> dict[str, Any]:
        requested = session_id or self.active_session_id()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM sessions WHERE id = ?", (requested,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown session: {requested}")
        state = _loads(str(row["state_json"]))
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid session state for {requested}")
        return deepcopy(state)

    def create_session(
        self,
        name: str,
        *,
        context_type: str = "discussion",
        from_global: bool = True,
        activate: bool = True,
    ) -> SessionInfo:
        safe_name = str(name or "Новая сессия").strip() or "Новая сессия"
        entities = self.global_entities() if from_global else {}
        state = default_session_state(entities=entities)
        session_id = uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO sessions(id, name, context_type, created_at, updated_at, revision, "
                "input_revision, state_json) VALUES (?, ?, ?, ?, ?, 0, 0, ?)",
                (session_id, safe_name, str(context_type or "discussion"), now, now, _dumps(state)),
            )
            if activate:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('active_session_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (session_id,),
                )
        return next(item for item in self.list_sessions() if item.session_id == session_id)

    def switch_session(self, session_id: str) -> SessionInfo:
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if exists is None:
                raise KeyError(f"Unknown session: {session_id}")
            connection.execute(
                "INSERT INTO meta(key, value) VALUES ('active_session_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (session_id,),
            )
        return self.active_session()

    def rename_session(self, session_id: str, name: str) -> SessionInfo:
        safe_name = str(name).strip()
        if not safe_name:
            raise ValueError("Session name cannot be empty.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                (safe_name, _utc_now(), session_id),
            )
            if connection.total_changes == 0:
                raise KeyError(f"Unknown session: {session_id}")
        return next(item for item in self.list_sessions() if item.session_id == session_id)

    def duplicate_session(
        self,
        session_id: str,
        *,
        name: str | None = None,
        activate: bool = True,
    ) -> SessionInfo:
        source = next((item for item in self.list_sessions() if item.session_id == session_id), None)
        if source is None:
            raise KeyError(f"Unknown session: {session_id}")
        state = self.session_state(session_id)
        new_id = uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO sessions(id, name, context_type, created_at, updated_at, revision, "
                "input_revision, state_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    str(name or f"{source.name} — копия"),
                    source.context_type,
                    now,
                    now,
                    source.revision,
                    source.input_revision,
                    _dumps(state),
                ),
            )
            export_rows = connection.execute(
                "SELECT kind, format, name, created_at, session_revision, source_hash, payload, metadata_json "
                "FROM export_artifacts WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
            for export_row in export_rows:
                connection.execute(
                    "INSERT INTO export_artifacts(id, session_id, kind, format, name, created_at, "
                    "session_revision, source_hash, payload, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        uuid4().hex,
                        new_id,
                        export_row["kind"],
                        export_row["format"],
                        export_row["name"],
                        export_row["created_at"],
                        export_row["session_revision"],
                        export_row["source_hash"],
                        export_row["payload"],
                        export_row["metadata_json"],
                    ),
                )
            if activate:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('active_session_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (new_id,),
                )
        return next(item for item in self.list_sessions() if item.session_id == new_id)

    def import_session(
        self,
        name: str,
        state: Mapping[str, Any],
        *,
        context_type: str = "discussion",
        activate: bool = True,
    ) -> SessionInfo:
        """Create a new session from a full state (backup restore or project template)."""
        if not isinstance(state, Mapping):
            raise ValueError("Session state must be an object.")
        base = default_session_state()
        merged = {**base, **deepcopy(dict(state))}
        merged["state_schema_version"] = STATE_SCHEMA_VERSION
        if not isinstance(merged.get("entities"), Mapping):
            merged["entities"] = {}
        safe_name = str(name or "Импортированная сессия").strip() or "Импортированная сессия"
        session_id = uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO sessions(id, name, context_type, created_at, updated_at, revision, "
                "input_revision, state_json) VALUES (?, ?, ?, ?, ?, 0, 0, ?)",
                (session_id, safe_name, str(context_type or "discussion"), now, now, _dumps(merged)),
            )
            if activate:
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('active_session_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (session_id,),
                )
        return next(item for item in self.list_sessions() if item.session_id == session_id)

    def delete_session(self, session_id: str) -> None:
        sessions = self.list_sessions()
        if len(sessions) <= 1:
            raise ValueError("Нельзя удалить единственную сессию.")
        active_id = self.active_session_id()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            if connection.total_changes == 0:
                raise KeyError(f"Unknown session: {session_id}")
            if session_id == active_id:
                replacement = connection.execute(
                    "SELECT id FROM sessions ORDER BY updated_at DESC, created_at DESC LIMIT 1"
                ).fetchone()
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES ('active_session_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(replacement[0]),),
                )

    def mutate_active_state(
        self,
        mutator: Callable[[dict[str, Any]], Any],
        *,
        input_change: bool = False,
        stale_keys: tuple[str, ...] = (),
    ) -> Any:
        session_id = self.active_session_id()
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision, input_revision, state_json FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Active session is missing.")
            state = _loads(str(row["state_json"]))
            result = mutator(state)
            new_revision = int(row["revision"]) + 1
            new_input_revision = int(row["input_revision"]) + (1 if input_change else 0)
            if input_change:
                self._mark_stale_in_state(state, tuple(state.get("derived_meta", {}).keys()))
            elif stale_keys:
                self._mark_stale_in_state(state, stale_keys)
            connection.execute(
                "UPDATE sessions SET revision = ?, input_revision = ?, state_json = ?, updated_at = ? "
                "WHERE id = ?",
                (new_revision, new_input_revision, _dumps(state), now, session_id),
            )
        return deepcopy(result)

    @staticmethod
    def _mark_stale_in_state(state: dict[str, Any], keys: tuple[str, ...]) -> None:
        meta = state.setdefault("derived_meta", {})
        for key in keys:
            entry = meta.get(key)
            if isinstance(entry, dict) and entry.get("validity") == "current":
                entry["validity"] = "stale"
                entry["stale_at"] = _utc_now()

    def mark_artifact_current(
        self,
        state: dict[str, Any],
        key: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        info = self.active_session()
        state.setdefault("derived_meta", {})[key] = {
            "validity": "current",
            "input_revision": info.input_revision,
            "produced_from_session_revision": info.revision,
            "produced_at": _utc_now(),
            "parameters": deepcopy(dict(parameters or {})),
            "provenance": deepcopy(dict(provenance or {})),
        }

    @staticmethod
    def artifact_is_current(state: Mapping[str, Any], key: str) -> bool:
        meta = state.get("derived_meta") if isinstance(state, Mapping) else None
        entry = meta.get(key) if isinstance(meta, Mapping) else None
        return isinstance(entry, Mapping) and entry.get("validity") == "current"

    def global_entities(self) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM global_state WHERE id = 1").fetchone()
        payload = _loads(str(row["payload_json"])) if row else {"entities": {}}
        entities = payload.get("entities", {}) if isinstance(payload, Mapping) else {}
        return deepcopy(dict(entities)) if isinstance(entities, Mapping) else {}

    def replace_global_entities(self, entities: Mapping[str, Any]) -> None:
        payload = {"entities": deepcopy(dict(entities))}
        with self._connect() as connection:
            connection.execute(
                "UPDATE global_state SET revision = revision + 1, payload_json = ?, updated_at = ? "
                "WHERE id = 1",
                (_dumps(payload), _utc_now()),
            )

    def resync_active_from_global(self) -> None:
        entities = self.global_entities()

        def apply(state: dict[str, Any]) -> None:
            state["entities"] = deepcopy(entities)

        self.mutate_active_state(apply, input_change=True)

    def record_export(
        self,
        *,
        kind: str,
        format: str,
        name: str,
        payload: bytes,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExportArtifact:
        session = self.active_session()
        artifact_id = uuid4().hex
        created_at = _utc_now()
        source_hash = hashlib.sha256(payload).hexdigest()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO export_artifacts(id, session_id, kind, format, name, created_at, "
                "session_revision, source_hash, payload, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact_id,
                    session.session_id,
                    str(kind),
                    str(format),
                    str(name),
                    created_at,
                    session.revision,
                    source_hash,
                    sqlite3.Binary(payload),
                    _dumps(dict(metadata or {})),
                ),
            )
        return ExportArtifact(
            artifact_id=artifact_id,
            session_id=session.session_id,
            kind=str(kind),
            format=str(format),
            name=str(name),
            created_at=created_at,
            session_revision=session.revision,
            source_hash=source_hash,
            size_bytes=len(payload),
            metadata=deepcopy(dict(metadata or {})),
        )

    def list_exports(self, session_id: str | None = None) -> list[ExportArtifact]:
        requested = session_id or self.active_session_id()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, session_id, kind, format, name, created_at, session_revision, "
                "source_hash, length(payload) AS size_bytes, metadata_json "
                "FROM export_artifacts WHERE session_id = ? ORDER BY created_at DESC, rowid DESC",
                (requested,),
            ).fetchall()
        return [
            ExportArtifact(
                artifact_id=str(row["id"]),
                session_id=str(row["session_id"]),
                kind=str(row["kind"]),
                format=str(row["format"]),
                name=str(row["name"]),
                created_at=str(row["created_at"]),
                session_revision=int(row["session_revision"]),
                source_hash=str(row["source_hash"]),
                size_bytes=int(row["size_bytes"] or 0),
                metadata=dict(_loads(str(row["metadata_json"]))),
            )
            for row in rows
        ]

    def rematerialize_export(self, artifact_id: str, target: str | Path) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, name FROM export_artifacts WHERE id = ? AND session_id = ?",
                (artifact_id, self.active_session_id()),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown export artifact for active session: {artifact_id}")
        destination = Path(target)
        if destination.exists() and destination.is_dir():
            destination = destination / str(row["name"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bytes(row["payload"]))
        return destination

    def legacy_import_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM legacy_imports").fetchone()[0])


class SessionEntityRepository:
    """EntityRepository adapter backed by the active session state."""

    def __init__(self, store: SessionStore):
        self.store = store

    @property
    def entities(self) -> dict[str, list[dict[str, Any]]]:
        state = self.store.session_state()
        entities = state.get("entities", {})
        return deepcopy(dict(entities)) if isinstance(entities, Mapping) else {}

    def save(self) -> Path:
        return self.store.path

    def list(self, entity_name: str) -> list[dict[str, Any]]:
        return deepcopy(self.entities.get(str(entity_name), []))

    def get(self, entity_name: str, index: int) -> dict[str, Any]:
        return deepcopy(self.list(entity_name)[index])

    def add(self, entity_name: str, data: Any) -> dict[str, Any]:
        row = deepcopy(_loads(_dumps(data)))

        def apply(state: dict[str, Any]) -> dict[str, Any]:
            state.setdefault("entities", {}).setdefault(str(entity_name), []).append(row)
            return row

        return self.store.mutate_active_state(apply, input_change=True)

    def update(self, entity_name: str, index: int, data: Any) -> dict[str, Any]:
        row = deepcopy(_loads(_dumps(data)))

        def apply(state: dict[str, Any]) -> dict[str, Any]:
            state.setdefault("entities", {}).setdefault(str(entity_name), [])[index] = row
            return row

        return self.store.mutate_active_state(apply, input_change=True)

    def delete(self, entity_name: str, index: int) -> dict[str, Any]:
        def apply(state: dict[str, Any]) -> dict[str, Any]:
            return state.setdefault("entities", {}).setdefault(str(entity_name), []).pop(index)

        return self.store.mutate_active_state(apply, input_change=True)

    def replace_all(self, entities: dict[str, list[dict[str, Any]]]) -> None:
        snapshot = deepcopy(entities)

        def apply(state: dict[str, Any]) -> None:
            state["entities"] = snapshot

        self.store.mutate_active_state(apply, input_change=True)

    def clear(self) -> None:
        self.replace_all({})


__all__ = [
    "ExportArtifact",
    "SESSION_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "SessionEntityRepository",
    "SessionInfo",
    "SessionStore",
    "default_session_state",
]
