"""SQLite read model for large catalog-staging screens.

The canonical staging state remains JSON because it is portable, inspectable and
used by computation/import workflows.  This companion database is a disposable
projection optimised for UI reads: page, search, summary and one-record lookup.
It can always be rebuilt from canonical staging state and is never committed as
project data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

READ_MODEL_SCHEMA_VERSION = 1


class CatalogStagingReadModel:
    """Disposable SQLite projection for catalog staging."""

    def __init__(self, staging_path: str | Path, path: str | Path | None = None) -> None:
        self.staging_path = Path(staging_path)
        self.path = Path(path) if path is not None else self.default_path(self.staging_path)

    @staticmethod
    def default_path(staging_path: str | Path) -> Path:
        source = Path(staging_path)
        return source.with_name(f"{source.stem}.ui.sqlite3")

    def is_fresh(self) -> bool:
        stamp = self._source_stamp()
        if stamp is None or not self.path.is_file():
            return False
        try:
            with self._connect(readonly=True) as connection:
                meta = dict(connection.execute("SELECT key, value FROM meta"))
        except (sqlite3.Error, OSError):
            return False
        return (
            meta.get("schema_version") == str(READ_MODEL_SCHEMA_VERSION)
            and meta.get("source_mtime_ns") == str(stamp[0])
            and meta.get("source_size") == str(stamp[1])
        )

    def rebuild(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        compact_records: Sequence[Mapping[str, Any]] | None = None,
    ) -> Path:
        """Atomically rebuild the projection from in-memory staging records.

        ``rows`` are fully upgraded records used to derive UI/search columns.
        ``compact_records`` can be supplied by the staging writer so one-record
        lookup can reconstruct an item without reading the complete JSON file.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(self.path.name + ".tmp")
        temp_path.unlink(missing_ok=True)
        compact_iter = iter(compact_records) if compact_records is not None else None
        try:
            try:
                with sqlite3.connect(temp_path) as connection:
                    connection.execute("PRAGMA journal_mode=OFF")
                    connection.execute("PRAGMA synchronous=OFF")
                    connection.execute("PRAGMA temp_store=MEMORY")
                    self._create_schema(connection)
                    batch: list[tuple[Any, ...]] = []
                    for ordinal, record in enumerate(rows):
                        compact = next(compact_iter) if compact_iter is not None else record
                        batch.append(self._row_tuple(ordinal, record, compact))
                        if len(batch) >= 1000:
                            connection.executemany(self._insert_sql(), batch)
                            batch.clear()
                    if batch:
                        connection.executemany(self._insert_sql(), batch)
                    stamp = self._source_stamp()
                    meta = {
                        "schema_version": str(READ_MODEL_SCHEMA_VERSION),
                        "source_mtime_ns": str(stamp[0] if stamp else 0),
                        "source_size": str(stamp[1] if stamp else 0),
                    }
                    connection.executemany(
                        "INSERT INTO meta(key, value) VALUES (?, ?)",
                        tuple(meta.items()),
                    )
                    connection.commit()
                temp_path.replace(self.path)
            except sqlite3.Error as exc:
                raise RuntimeError(f"SQLite read-model error: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)
        logger.info("Catalog staging UI projection rebuilt: %s", self.path)
        return self.path

    def page(
        self,
        *,
        status_filter: str = "all",
        query: str = "",
        offset: int = 0,
        limit: int = 250,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return only lightweight fields required by the Qt table."""

        where: list[str] = []
        params: list[Any] = []
        if status_filter != "all":
            where.append("status = ?")
            params.append(str(status_filter))
        needle = str(query or "").strip().casefold()
        if needle:
            where.append("instr(search_text, ?) > 0")
            params.append(needle)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        start = max(0, int(offset))
        page_limit = max(1, min(5000, int(limit)))
        with self._connect(readonly=True) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM records{clause}",  # noqa: S608 - clause is internal
                    tuple(params),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                "SELECT staging_id, status, readiness, title, source, source_count, "
                "category, target_category, price, issues "
                f"FROM records{clause} ORDER BY ordinal LIMIT ? OFFSET ?",  # noqa: S608
                (*params, page_limit, start),
            )
            rows = [
                {
                    "staging_id": row[0],
                    "status": row[1],
                    "readiness": row[2],
                    "title": row[3],
                    "source": row[4],
                    "source_count": int(row[5] or 0),
                    "category": row[6],
                    "target_category": row[7],
                    "price": float(row[8] or 0.0),
                    "issues": int(row[9] or 0),
                }
                for row in cursor.fetchall()
            ]
        return rows, total

    def summary_counts(self) -> dict[str, int]:
        with self._connect(readonly=True) as connection:
            status_rows = connection.execute(
                "SELECT status, COUNT(*) FROM records GROUP BY status"
            ).fetchall()
            total, ready = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN readiness IN ('import_ready','ga_ready') "
                "THEN 1 ELSE 0 END) FROM records"
            ).fetchone()
        counts = {str(status): int(count) for status, count in status_rows}
        counts["total"] = int(total or 0)
        counts["ready"] = int(ready or 0)
        return counts

    def compact_record(self, staging_id: str) -> dict[str, Any] | None:
        with self._connect(readonly=True) as connection:
            row = connection.execute(
                "SELECT compact_json FROM records WHERE staging_id = ?",
                (str(staging_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(str(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _source_stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.staging_path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            return sqlite3.connect(uri, uri=True, timeout=2.0)
        return sqlite3.connect(self.path, timeout=5.0)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE records (
                staging_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL,
                status TEXT NOT NULL,
                readiness TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                source_count INTEGER NOT NULL,
                category TEXT NOT NULL,
                target_category TEXT NOT NULL,
                price REAL NOT NULL,
                issues INTEGER NOT NULL,
                search_text TEXT NOT NULL,
                compact_json TEXT NOT NULL
            );
            CREATE INDEX idx_catalog_stage_ordinal ON records(ordinal);
            CREATE INDEX idx_catalog_stage_status_ordinal ON records(status, ordinal);
            """
        )

    @staticmethod
    def _insert_sql() -> str:
        return (
            "INSERT INTO records("
            "staging_id, ordinal, status, readiness, title, source, source_count, "
            "category, target_category, price, issues, search_text, compact_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

    @staticmethod
    def _row_tuple(
        ordinal: int,
        record: Mapping[str, Any],
        compact: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        item = record.get("catalog_item") if isinstance(record.get("catalog_item"), Mapping) else {}
        offer = item.get("offer") if isinstance(item.get("offer"), Mapping) else {}
        identity = item.get("identity") if isinstance(item.get("identity"), Mapping) else {}
        federation = item.get("federation") if isinstance(item.get("federation"), Mapping) else {}
        title = str(item.get("title") or "")
        source = str(item.get("source") or "")
        category = str(item.get("category") or "")
        target_category = str(record.get("target_category") or "")
        errors = record.get("validation_errors") if isinstance(record.get("validation_errors"), list) else []
        warnings = record.get("validation_warnings") if isinstance(record.get("validation_warnings"), list) else []
        source_count = int(federation.get("source_count") or 0)
        status = str(record.get("status") or "")
        if errors:
            readiness = "blocked"
        elif status == "imported":
            readiness = "stale" if record.get("source_changed_since_review") else "ga_ready"
        elif status == "approved":
            readiness = "import_ready"
        else:
            readiness = "review"
        search_text = " ".join(
            (
                title,
                source,
                category,
                str(identity.get("brand") or ""),
                str(identity.get("model") or ""),
                str(identity.get("mpn") or ""),
                str(identity.get("gtin") or ""),
            )
        ).casefold()
        try:
            price = float(offer.get("price") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        return (
            str(record.get("staging_id") or ""),
            ordinal,
            status,
            readiness,
            title,
            source,
            source_count,
            category,
            target_category,
            price,
            len(errors) + len(warnings),
            search_text,
            json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        )


__all__ = ["CatalogStagingReadModel", "READ_MODEL_SCHEMA_VERSION"]
