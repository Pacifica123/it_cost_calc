from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from application.services.catalog_staging_service import (
    CatalogStagingService,
    STAGING_APPROVED,
    STAGING_BLOCKED,
    STAGING_IMPORTED,
    STAGING_PENDING,
    STAGING_REJECTED,
    catalog_item_to_runtime_row,
    target_for_catalog_category,
)
from shared.constants import TECHNICAL_CAPITAL_CATEGORIES
from ui_qt.presenters.app_presenter import QtAppPresenter

_STATUS_LABELS = {
    STAGING_PENDING: "Ожидает",
    STAGING_APPROVED: "Подтверждено",
    STAGING_REJECTED: "Отклонено",
    STAGING_BLOCKED: "Заблокировано",
    STAGING_IMPORTED: "Импортировано",
}
_TARGET_LABELS = {
    "server": "Серверы",
    "client": "Клиенты",
    "network": "Сеть",
}


@dataclass(frozen=True)
class CatalogStagingSummary:
    total: int
    pending: int
    approved: int
    blocked: int
    imported: int


class CatalogStagingPresenter:
    """Qt-facing orchestration for review and explicit catalog import."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        *,
        staging_path: str | Path | None = None,
        service: CatalogStagingService | None = None,
    ) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        path = staging_path or (
            self.app_presenter.paths.repo_root
            / "data"
            / "generated"
            / "catalog"
            / "catalog_staging.json"
        )
        self.service = service or CatalogStagingService(path, self.app_presenter.storage)

    def stage_file(self, path: str | Path) -> list[dict[str, Any]]:
        return self.service.stage_file(path)

    def records(self) -> list[dict[str, Any]]:
        return self.service.list_records()

    def table_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in self.records():
            item = dict(record.get("catalog_item") or {})
            offer = dict(item.get("offer") or {})
            target = target_for_catalog_category(str(item.get("category") or ""))
            rows.append(
                {
                    "status": _STATUS_LABELS.get(str(record.get("status")), str(record.get("status"))),
                    "name": str(item.get("title") or ""),
                    "source": str(item.get("source") or ""),
                    "category": str(item.get("category") or ""),
                    "target": _TARGET_LABELS.get(target[0], target[0]) if target else "—",
                    "price": float(offer.get("price") or 0.0),
                    "issues": len(record.get("validation_errors") or [])
                    + len(record.get("validation_warnings") or []),
                }
            )
        return rows

    def summary(self) -> CatalogStagingSummary:
        records = self.records()
        statuses = [str(record.get("status")) for record in records]
        return CatalogStagingSummary(
            total=len(records),
            pending=statuses.count(STAGING_PENDING),
            approved=statuses.count(STAGING_APPROVED),
            blocked=statuses.count(STAGING_BLOCKED),
            imported=statuses.count(STAGING_IMPORTED),
        )

    def record_at(self, index: int) -> dict[str, Any]:
        return self.records()[index]

    def approve(self, index: int) -> dict[str, Any]:
        record = self.record_at(index)
        return self.service.set_status(str(record["staging_id"]), STAGING_APPROVED)

    def reject(self, index: int) -> dict[str, Any]:
        record = self.record_at(index)
        return self.service.set_status(str(record["staging_id"]), STAGING_REJECTED)

    def reset_pending(self, index: int) -> dict[str, Any]:
        record = self.record_at(index)
        return self.service.set_status(str(record["staging_id"]), STAGING_PENDING)

    def import_approved(self) -> dict[str, int]:
        approved = self.service.approved_records()
        if not approved:
            return {"imported": 0, "skipped": 0}

        entities = self.app_presenter.entities_snapshot()
        existing_ids = {
            str(row.get("catalog_item_id"))
            for category in TECHNICAL_CAPITAL_CATEGORIES
            for row in entities.get(category, [])
            if row.get("catalog_item_id")
        }
        imported_ids: list[str] = []
        imported = 0
        skipped = 0
        for record in approved:
            target_category, row = catalog_item_to_runtime_row(record)
            catalog_item_id = str(row.get("catalog_item_id") or "")
            imported_ids.append(str(record.get("staging_id")))
            if catalog_item_id in existing_ids:
                skipped += 1
                continue
            entities.setdefault(target_category, []).append(row)
            existing_ids.add(catalog_item_id)
            imported += 1

        if imported:
            self.app_presenter.replace_entities(entities)
        self.service.mark_imported(imported_ids)
        return {"imported": imported, "skipped": skipped}

    def details_text(self, index: int) -> str:
        record = self.record_at(index)
        item = dict(record.get("catalog_item") or {})
        identity = dict(item.get("identity") or {})
        offer = dict(item.get("offer") or {})
        metrics = dict(item.get("attributes") or {})
        errors = list(record.get("validation_errors") or [])
        warnings = list(record.get("validation_warnings") or [])
        lines = [
            f"Статус: {_STATUS_LABELS.get(str(record.get('status')), record.get('status'))}",
            f"Источник: {item.get('source') or '—'}",
            f"Категория: {item.get('category') or '—'}",
            f"Цена: {offer.get('price') or 0:g} {offer.get('currency') or 'RUB'}",
            f"Модель: {identity.get('model') or identity.get('mpn') or '—'}",
            f"Метрики: {metrics or 'нет'}",
        ]
        if errors:
            lines.append("Блокирует:")
            lines.extend(f"  - {message}" for message in errors)
        if warnings:
            lines.append("Проверить:")
            lines.extend(f"  - {message}" for message in warnings)
        return "\n".join(lines)


__all__ = ["CatalogStagingPresenter", "CatalogStagingSummary"]
