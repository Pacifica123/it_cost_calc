"""Project templates: starter sets of alternatives for typical IT projects.

Built-in templates are shipped in ``data/templates/project_templates.json``; the user can save
the entities of the current session as an own template (kept next to the session database).
A template creates a new session with its entities, so nothing in the current work is touched.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectTemplate:
    template_id: str
    title: str
    description: str
    entities: dict[str, list[dict[str, Any]]]
    builtin: bool = True

    @property
    def summary(self) -> str:
        counts = [f"{name}: {len(rows)}" for name, rows in self.entities.items() if rows]
        return ", ".join(counts)


def _parse(payload: Any, *, builtin: bool) -> list[ProjectTemplate]:
    items = payload.get("templates", []) if isinstance(payload, Mapping) else []
    result = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        template_id = str(item.get("id") or "").strip()
        entities = item.get("entities")
        if not template_id or not isinstance(entities, Mapping):
            continue
        clean = {str(name): [dict(row) for row in rows if isinstance(row, Mapping)] for name, rows in entities.items() if isinstance(rows, list)}
        result.append(
            ProjectTemplate(
                template_id=template_id,
                title=str(item.get("title") or template_id),
                description=str(item.get("description") or ""),
                entities=clean,
                builtin=builtin,
            )
        )
    return result


class ProjectTemplateService:
    def __init__(self, builtin_path: Path, user_path: Path) -> None:
        self.builtin_path = Path(builtin_path)
        self.user_path = Path(user_path)

    def _read(self, path: Path) -> Any:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def list_templates(self) -> list[ProjectTemplate]:
        return _parse(self._read(self.builtin_path), builtin=True) + _parse(self._read(self.user_path), builtin=False)

    def get(self, template_id: str) -> ProjectTemplate:
        for template in self.list_templates():
            if template.template_id == template_id:
                return template
        raise ValueError(f"Неизвестный шаблон: {template_id}")

    def save_user_template(self, title: str, description: str, entities: Mapping[str, Any]) -> ProjectTemplate:
        title = str(title).strip()
        if not title:
            raise ValueError("Название шаблона не может быть пустым.")
        payload = self._read(self.user_path)
        templates = list(payload.get("templates", [])) if isinstance(payload, Mapping) else []
        slug = re.sub(r"[^\w]+", "-", title.lower(), flags=re.UNICODE).strip("-") or "template"
        template_id = f"user-{slug}"
        existing = {str(item.get("id")) for item in templates if isinstance(item, Mapping)}
        suffix = 2
        while template_id in existing:
            template_id = f"user-{slug}-{suffix}"
            suffix += 1
        clean = {str(name): deepcopy(list(rows)) for name, rows in entities.items() if isinstance(rows, list)}
        templates.append(
            {
                "id": template_id,
                "title": title,
                "description": str(description or ""),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "entities": clean,
            }
        )
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_path.write_text(json.dumps({"schema_version": 1, "templates": templates}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self.get(template_id)

    def delete_user_template(self, template_id: str) -> None:
        payload = self._read(self.user_path)
        templates = list(payload.get("templates", [])) if isinstance(payload, Mapping) else []
        kept = [item for item in templates if not (isinstance(item, Mapping) and item.get("id") == template_id)]
        if len(kept) == len(templates):
            raise ValueError("Встроенный шаблон удалить нельзя." if any(t.template_id == template_id for t in self.list_templates()) else f"Неизвестный шаблон: {template_id}")
        self.user_path.write_text(json.dumps({"schema_version": 1, "templates": kept}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
