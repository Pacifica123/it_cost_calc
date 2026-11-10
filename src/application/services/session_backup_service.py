"""Backups of work sessions: one session as JSON, all sessions as a ZIP, restore from either.

A backup holds the whole session state (entities, candidate pools, GA/AHP/Pareto/Hybrid,
NPV, plan settings), so a restored session continues exactly where it was saved.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKUP_FORMAT = "it-cost-calc-session"
BACKUP_VERSION = 1


@dataclass(frozen=True)
class SessionBackup:
    name: str
    context_type: str
    state: dict[str, Any]
    exported_at: str = ""


def build_backup_payload(name: str, context_type: str, state: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    moment = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "name": str(name),
        "context_type": str(context_type),
        "exported_at": moment,
        "state": json.loads(json.dumps(dict(state), ensure_ascii=False, default=str)),
    }


def parse_backup_payload(payload: Any) -> SessionBackup:
    if not isinstance(payload, Mapping) or payload.get("format") != BACKUP_FORMAT:
        raise ValueError("Это не резервная копия сессии IT Cost Calc.")
    version = int(payload.get("version", 0) or 0)
    if version > BACKUP_VERSION:
        raise ValueError(f"Копия сделана более новой версией программы (формат {version}).")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("В копии нет состояния сессии.")
    return SessionBackup(
        name=str(payload.get("name") or "Восстановленная сессия"),
        context_type=str(payload.get("context_type") or "discussion"),
        state=dict(state),
        exported_at=str(payload.get("exported_at") or ""),
    )


def safe_file_stem(name: str) -> str:
    stem = re.sub(r"[^\w\-. ]+", "_", str(name), flags=re.UNICODE).strip(" ._") or "session"
    return stem[:60]


def write_session_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def write_sessions_zip(path: str | Path, payloads: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, payload in enumerate(payloads, start=1):
            stem = f"{index:02d}_{safe_file_stem(str(payload.get('name', 'session')))}"
            while stem in used:
                stem += "_"
            used.add(stem)
            archive.writestr(f"{stem}.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return target


def read_backups(path: str | Path) -> list[SessionBackup]:
    """Reads a .json with one session or a .zip with several."""
    source = Path(path)
    if zipfile.is_zipfile(source):
        result = []
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith(".json") and not name.startswith("__MACOSX"):
                    result.append(parse_backup_payload(json.loads(archive.read(name).decode("utf-8"))))
        if not result:
            raise ValueError("В архиве нет резервных копий сессий.")
        return result
    return [parse_backup_payload(json.loads(source.read_text(encoding="utf-8")))]
