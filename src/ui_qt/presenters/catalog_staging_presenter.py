from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

from application.services.catalog_staging_service import (
    CatalogStagingService,
    STAGING_APPROVED,
    STAGING_BLOCKED,
    STAGING_IMPORTED,
    STAGING_PENDING,
    STAGING_REJECTED,
    catalog_metric_fields,
    catalog_target_options,
    catalog_item_to_runtime_row,
    staging_record_readiness,
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
_COMPONENT_TYPE_LABELS = {
    "server": "Сервер",
    "workstation": "Рабочая станция",
    "peripheral": "Периферия",
    "network_device": "Сетевое устройство",
}
_READINESS_LABELS = {
    "blocked": "Заблокировано",
    "review": "Нужна проверка",
    "import_ready": "Готово к импорту",
    "ga_ready": "В ТО / готово к GA",
    "stale": "Источник обновлён",
}


@dataclass(frozen=True)
class CatalogStagingSummary:
    total: int
    pending: int
    approved: int
    blocked: int
    imported: int
    rejected: int
    ready: int


@dataclass(frozen=True)
class CatalogFilterOption:
    value: str
    label: str


@dataclass(frozen=True)
class DnsCatalogJobSpec:
    program: str
    arguments: tuple[str, ...]
    working_directory: Path
    output_path: Path
    snapshot_path: Path


_DNS_CATEGORY_OPTIONS = (
    ("routers", "Роутеры"),
    ("switches", "Коммутаторы"),
    ("prebuilt_pcs", "Готовые ПК"),
    ("servers", "Серверы"),
)
_DNS_BROWSER_ENGINES = {"firefox", "chromium"}
_DNS_BROWSER_URLS = {
    "routers": "https://www.dns-shop.ru/catalog/17a8aa1c16404e77/wi-fi-routery/",
    "switches": "https://www.dns-shop.ru/catalog/17a9dc3716404e77/kommutatory/",
    "prebuilt_pcs": "https://www.dns-shop.ru/catalog/17a8932c16404e77/personalnye-komputery/",
    "servers": "https://www.dns-shop.ru/catalog/17a8939816404e77/servery/",
}


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

    def filter_options(self) -> list[CatalogFilterOption]:
        return [
            CatalogFilterOption("all", "Все"),
            CatalogFilterOption(STAGING_PENDING, "Ожидают"),
            CatalogFilterOption(STAGING_APPROVED, "Подтверждены"),
            CatalogFilterOption(STAGING_BLOCKED, "Заблокированы"),
            CatalogFilterOption(STAGING_REJECTED, "Отклонены"),
            CatalogFilterOption(STAGING_IMPORTED, "Импортированы"),
        ]

    def target_category_options(self) -> list[tuple[str, str]]:
        return [(value, _TARGET_LABELS[value]) for value in catalog_target_options()]

    def component_type_options(self, target_category: str) -> list[tuple[str, str]]:
        return [
            (value, _COMPONENT_TYPE_LABELS.get(value, value))
            for value in catalog_target_options().get(target_category, ())
        ]

    def metric_fields(self) -> tuple[str, ...]:
        return catalog_metric_fields()

    def dns_category_options(self) -> tuple[tuple[str, str], ...]:
        return _DNS_CATEGORY_OPTIONS

    def dns_browser_url(self, category: str) -> str:
        try:
            return _DNS_BROWSER_URLS[category]
        except KeyError as exc:
            raise ValueError("Неизвестная DNS-категория.") from exc

    def build_dns_job(
        self,
        *,
        categories: list[str],
        per_category_limit: int,
        time_limit_seconds: int,
        visible_browser: bool,
        browser_engine: str = "firefox",
        region: str = "",
    ) -> DnsCatalogJobSpec:
        allowed = {value for value, _label in _DNS_CATEGORY_OPTIONS}
        selected = tuple(dict.fromkeys(str(value) for value in categories))
        if not selected or any(value not in allowed for value in selected):
            raise ValueError("Выберите хотя бы одну поддерживаемую DNS-категорию.")
        if not 1 <= per_category_limit <= 50:
            raise ValueError("Лимит карточек должен быть от 1 до 50.")
        if not 30 <= time_limit_seconds <= 1800:
            raise ValueError("Таймаут должен быть от 30 до 1800 секунд.")
        if browser_engine not in _DNS_BROWSER_ENGINES:
            raise ValueError("Выберите Firefox или Chromium.")

        repo_root = self.app_presenter.paths.repo_root
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_root = repo_root / "data" / "generated" / "catalog" / "dns_runs" / timestamp
        output_path = run_root / "equipment_catalog.json"
        snapshot_path = run_root / "snapshot"
        profile_path = (
            repo_root
            / "data"
            / "generated"
            / "catalog"
            / "dns_browser_profiles"
            / browser_engine
        )
        arguments = [
            "-u",
            str(repo_root / "scripts" / "update_equipment_catalog.py"),
            "--mode",
            "dns-live",
            "--categories",
            ",".join(selected),
            "--browser-engine",
            browser_engine,
            "--limit",
            str(per_category_limit),
            "--time-limit",
            str(time_limit_seconds),
            "--browser-wait",
            "8",
            "--snapshot-output",
            str(snapshot_path),
            "--profile",
            str(profile_path),
            "--region",
            str(region).strip(),
            "--output",
            str(output_path),
        ]
        if not visible_browser:
            arguments.append("--headless")
        return DnsCatalogJobSpec(
            program=sys.executable,
            arguments=tuple(arguments),
            working_directory=repo_root,
            output_path=output_path,
            snapshot_path=snapshot_path,
        )

    def build_dns_capture_job(self, path: str | Path, *, region: str = "") -> DnsCatalogJobSpec:
        capture_path = Path(path).resolve()
        suffix = capture_path.suffix.lower()
        if not capture_path.is_file() or suffix not in {".har", ".html", ".htm"}:
            raise ValueError("Выберите HAR или сохранённый HTML-файл DNS.")
        repo_root = self.app_presenter.paths.repo_root
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_root = repo_root / "data" / "generated" / "catalog" / "dns_imports" / timestamp
        output_path = run_root / "equipment_catalog.json"
        mode = "dns-har" if suffix == ".har" else "dns-html"
        return DnsCatalogJobSpec(
            program=sys.executable,
            arguments=(
                "-u",
                str(repo_root / "scripts" / "update_equipment_catalog.py"),
                "--mode",
                mode,
                "--input",
                str(capture_path),
                "--region",
                str(region).strip(),
                "--output",
                str(output_path),
            ),
            working_directory=repo_root,
            output_path=output_path,
            snapshot_path=run_root / "capture",
        )

    def table_rows(self, status_filter: str = "all") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in self.records():
            if status_filter != "all" and record.get("status") != status_filter:
                continue
            item = dict(record.get("catalog_item") or {})
            offer = dict(item.get("offer") or {})
            target_category = str(record.get("target_category") or "")
            readiness = staging_record_readiness(record)
            rows.append(
                {
                    "staging_id": str(record.get("staging_id") or ""),
                    "status": _STATUS_LABELS.get(str(record.get("status")), str(record.get("status"))),
                    "readiness": _READINESS_LABELS.get(readiness, readiness),
                    "name": str(item.get("title") or ""),
                    "source": str(item.get("source") or ""),
                    "category": str(item.get("category") or ""),
                    "target": _TARGET_LABELS.get(target_category, target_category or "—"),
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
            rejected=statuses.count(STAGING_REJECTED),
            ready=sum(
                staging_record_readiness(record) in {"import_ready", "ga_ready"}
                for record in records
            ),
        )

    def record_at(self, index: int) -> dict[str, Any]:
        return self.records()[index]

    def record_by_id(self, staging_id: str) -> dict[str, Any]:
        for record in self.records():
            if record.get("staging_id") == staging_id:
                return record
        raise KeyError(staging_id)

    def approve(self, index: int) -> dict[str, Any]:
        record = self.record_at(index)
        return self.service.set_status(str(record["staging_id"]), STAGING_APPROVED)

    def reject(self, index: int) -> dict[str, Any]:
        record = self.record_at(index)
        return self.service.set_status(str(record["staging_id"]), STAGING_REJECTED)

    def reset_pending(self, index: int) -> dict[str, Any]:
        record = self.record_at(index)
        return self.service.set_status(str(record["staging_id"]), STAGING_PENDING)

    def approve_ids(self, staging_ids: list[str]) -> dict[str, int]:
        return self.service.set_status_many(staging_ids, STAGING_APPROVED)

    def reject_ids(self, staging_ids: list[str]) -> dict[str, int]:
        return self.service.set_status_many(staging_ids, STAGING_REJECTED)

    def edit_values(self, staging_id: str) -> dict[str, Any]:
        record = self.record_by_id(staging_id)
        item = dict(record.get("catalog_item") or {})
        offer = dict(item.get("offer") or {})
        runtime_inputs = dict(record.get("runtime_inputs") or {})
        return {
            "title": str(item.get("title") or ""),
            "category": str(item.get("category") or ""),
            "price": _format_number(offer.get("price")),
            "currency": str(offer.get("currency") or "RUB"),
            "target_category": str(record.get("target_category") or ""),
            "target_component_type": str(record.get("target_component_type") or ""),
            "quantity": _format_number(runtime_inputs.get("quantity")),
            "client_seats": _format_number(runtime_inputs.get("client_seats")),
            "attributes": dict(item.get("attributes") or {}),
            "locked": record.get("status") == STAGING_IMPORTED,
        }

    def save_edits(self, staging_id: str, values: dict[str, Any]) -> dict[str, Any]:
        return self.service.update_record(staging_id, values)

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
        return self.details_text_by_id(str(self.record_at(index).get("staging_id")))

    def details_text_by_id(self, staging_id: str) -> str:
        record = self.record_by_id(staging_id)
        item = dict(record.get("catalog_item") or {})
        source_item = dict(record.get("source_catalog_item") or {})
        identity = dict(item.get("identity") or {})
        offer = dict(item.get("offer") or {})
        metrics = dict(item.get("attributes") or {})
        errors = list(record.get("validation_errors") or [])
        warnings = list(record.get("validation_warnings") or [])
        readiness = staging_record_readiness(record)
        lines = [
            f"Статус: {_STATUS_LABELS.get(str(record.get('status')), record.get('status'))}",
            f"Готовность: {_READINESS_LABELS.get(readiness, readiness)}",
            f"Источник: {item.get('source') or '—'}",
            f"Категория: {item.get('category') or '—'}",
            "Назначение: "
            f"{_TARGET_LABELS.get(str(record.get('target_category')), record.get('target_category') or '—')} / "
            f"{_COMPONENT_TYPE_LABELS.get(str(record.get('target_component_type')), record.get('target_component_type') or '—')}",
            f"Цена: {offer.get('price') or 0:g} {offer.get('currency') or 'RUB'}",
            f"Модель: {identity.get('model') or identity.get('mpn') or '—'}",
            f"Метрики: {metrics or 'нет'}",
        ]
        changes = _manual_change_lines(source_item, record)
        if changes:
            lines.append("Исправлено вручную:")
            lines.extend(f"  - {message}" for message in changes)
        if errors:
            lines.append("Блокирует:")
            lines.extend(f"  - {message}" for message in errors)
        if warnings:
            lines.append("Проверить:")
            lines.extend(f"  - {message}" for message in warnings)
        return "\n".join(lines)


def _manual_change_lines(
    source_item: dict[str, Any],
    record: dict[str, Any],
) -> list[str]:
    item = dict(record.get("catalog_item") or {})
    changes: list[str] = []
    labels = {"title": "Название", "category": "Категория"}
    for field, label in labels.items():
        if source_item.get(field) != item.get(field):
            changes.append(f"{label}: {source_item.get(field) or '—'} → {item.get(field) or '—'}")

    source_offer = dict(source_item.get("offer") or {})
    offer = dict(item.get("offer") or {})
    for field, label in (("price", "Цена"), ("currency", "Валюта")):
        if source_offer.get(field) != offer.get(field):
            changes.append(
                f"{label}: {source_offer.get(field) or '—'} → {offer.get(field) or '—'}"
            )

    source_metrics = dict(source_item.get("attributes") or {})
    metrics = dict(item.get("attributes") or {})
    for field in catalog_metric_fields():
        if source_metrics.get(field) != metrics.get(field):
            changes.append(
                f"{field}: {source_metrics.get(field, '—')} → {metrics.get(field, '—')}"
            )

    source_target = target_for_catalog_category(str(source_item.get("category") or ""))
    source_category = source_target[0] if source_target else ""
    source_type = source_target[1] if source_target else ""
    if source_category != record.get("target_category"):
        changes.append(
            f"Раздел ТО: {_TARGET_LABELS.get(source_category, source_category or '—')} → "
            f"{_TARGET_LABELS.get(str(record.get('target_category')), record.get('target_category') or '—')}"
        )
    if source_type != record.get("target_component_type"):
        changes.append(
            f"Тип: {_COMPONENT_TYPE_LABELS.get(source_type, source_type or '—')} → "
            f"{_COMPONENT_TYPE_LABELS.get(str(record.get('target_component_type')), record.get('target_component_type') or '—')}"
        )
    runtime_inputs = dict(record.get("runtime_inputs") or {})
    default_quantity = 1.0
    default_seats = 1.0 if record.get("target_component_type") == "workstation" else 0.0
    if runtime_inputs.get("quantity") != default_quantity:
        changes.append(f"Количество: {default_quantity:g} → {runtime_inputs.get('quantity')}")
    if runtime_inputs.get("client_seats") != default_seats:
        changes.append(f"Рабочие места: {default_seats:g} → {runtime_inputs.get('client_seats')}")
    return changes


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:g}"


__all__ = [
    "CatalogFilterOption",
    "CatalogStagingPresenter",
    "CatalogStagingSummary",
    "DnsCatalogJobSpec",
]
